use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::calculate_cost;
use crate::registry::ApiProvider;
use crate::simple_options::{build_base_options, clamp_reasoning};

// ---------- Tool call counter ----------

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------- GoogleOptions ----------

#[derive(Debug, Clone)]
pub struct GoogleOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<String>, // "auto" | "none" | "any"
    pub thinking_enabled: bool,
    pub thinking_budget_tokens: Option<i64>, // -1 for dynamic, 0 to disable
    pub thinking_level: Option<String>,      // "MINIMAL" | "LOW" | "MEDIUM" | "HIGH"
}

// ---------- Google thinking level type ----------

/// Gemini 3 thinking levels: MINIMAL, LOW, MEDIUM, HIGH
pub type GoogleThinkingLevel = String;

// ---------- Thought signature helpers (from google-shared.ts) ----------

/// Determines whether a streamed Gemini Part should be treated as "thinking".
///
/// Protocol note:
/// - `thought: true` is the definitive marker for thinking content.
/// - `thoughtSignature` is an encrypted representation of internal reasoning
///   used for multi-turn replay; it can appear on ANY part type and does NOT
///   mean the part itself is thinking content.
fn is_thinking_part(part: &Value) -> bool {
    part.get("thought")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Retain thought signatures during streaming.
/// Preserves the last non-empty signature for the current block.
fn retain_thought_signature(existing: Option<&str>, incoming: Option<&str>) -> Option<String> {
    if let Some(s) = incoming {
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    existing.map(|s| s.to_string())
}

/// Check if a thought signature is valid base64 for Google APIs (TYPE_BYTES).
fn is_valid_thought_signature(signature: &str) -> bool {
    if signature.is_empty() {
        return false;
    }
    if signature.len() % 4 != 0 {
        return false;
    }
    lazy_static_regex().is_match(signature)
}

fn lazy_static_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9+/]+={0,2}$").unwrap())
}

/// Only keep signatures from the same provider/model and with valid base64.
fn resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: Option<&str>,
) -> Option<String> {
    match signature {
        Some(sig) if is_same_provider_and_model && is_valid_thought_signature(sig) => {
            Some(sig.to_string())
        }
        _ => None,
    }
}

/// Models via Google APIs that require explicit tool call IDs in function calls/responses.
fn requires_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

// ---------- Map stop reason ----------

/// Map Gemini FinishReason enum to our StopReason.
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        // All safety/error related reasons
        "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "SAFETY"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION"
        | "IMAGE_OTHER"
        | "RECITATION"
        | "FINISH_REASON_UNSPECIFIED"
        | "OTHER"
        | "LANGUAGE"
        | "MALFORMED_FUNCTION_CALL"
        | "UNEXPECTED_TOOL_CALL"
        | "NO_IMAGE" => StopReason::Error,
        unknown => {
            tracing::warn!("Unknown Google stop reason: {}", unknown);
            StopReason::Error
        }
    }
}

/// Map Gemini FinishReason string (from raw API response) to our StopReason.
/// This handles string representations that may come directly from the API.
pub fn map_stop_reason_string(reason: &str) -> StopReason {
    map_stop_reason(reason)
}

/// Map tool choice string to Gemini FunctionCallingConfigMode string.
fn map_tool_choice(choice: &str) -> &'static str {
    match choice {
        "auto" => "AUTO",
        "none" => "NONE",
        "any" => "ANY",
        _ => "AUTO",
    }
}

// ---------- Normalize tool call ID ----------

fn normalize_tool_call_id(id: &str) -> String {
    let normalized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if normalized.len() > 64 {
        normalized[..64].to_string()
    } else {
        normalized
    }
}

fn normalize_tool_call_id_for_transform(
    id: &str,
    _model: &Model,
    _source: &AssistantMessage,
) -> String {
    normalize_tool_call_id(id)
}

// ---------- Convert messages to Gemini Content[] format ----------

fn convert_messages(model: &Model, context: &Context) -> Vec<Value> {
    let transformed = transform_messages(
        &context.messages,
        model,
        Some(&normalize_tool_call_id_for_transform),
    );
    let mut contents: Vec<Value> = Vec::new();

    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                match &user_msg.content {
                    UserContent::Text(text) => {
                        contents.push(json!({
                            "role": "user",
                            "parts": [{ "text": sanitize_surrogates(text) }]
                        }));
                    }
                    UserContent::Blocks(blocks) => {
                        let parts: Vec<Value> = blocks
                            .iter()
                            .filter_map(|item| match item {
                                ContentBlock::Text(t) => {
                                    Some(json!({ "text": sanitize_surrogates(&t.text) }))
                                }
                                ContentBlock::Image(img) => {
                                    if model.input.contains(&"image".to_string()) {
                                        Some(json!({
                                            "inlineData": {
                                                "mimeType": img.mime_type,
                                                "data": img.data
                                            }
                                        }))
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            })
                            .collect();

                        // Filter out images if model doesn't support them
                        let filtered_parts: Vec<Value> =
                            if !model.input.contains(&"image".to_string()) {
                                parts
                                    .into_iter()
                                    .filter(|p| p.get("text").is_some())
                                    .collect()
                            } else {
                                parts
                            };

                        if !filtered_parts.is_empty() {
                            contents.push(json!({
                                "role": "user",
                                "parts": filtered_parts
                            }));
                        }
                    }
                }
            }
            Message::Assistant(assistant_msg) => {
                let mut parts: Vec<Value> = Vec::new();
                let is_same_provider_and_model =
                    assistant_msg.provider == model.provider && assistant_msg.model == model.id;

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            // Skip empty text blocks
                            if t.text.trim().is_empty() {
                                continue;
                            }
                            let thought_signature = resolve_thought_signature(
                                is_same_provider_and_model,
                                t.text_signature.as_deref(),
                            );
                            let mut part = json!({ "text": sanitize_surrogates(&t.text) });
                            if let Some(sig) = thought_signature {
                                part["thoughtSignature"] = json!(sig);
                            }
                            parts.push(part);
                        }
                        ContentBlock::Thinking(t) => {
                            // Skip empty thinking blocks
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_provider_and_model {
                                let thought_signature = resolve_thought_signature(
                                    is_same_provider_and_model,
                                    t.thinking_signature.as_deref(),
                                );
                                let mut part = json!({
                                    "thought": true,
                                    "text": sanitize_surrogates(&t.thinking)
                                });
                                if let Some(sig) = thought_signature {
                                    part["thoughtSignature"] = json!(sig);
                                }
                                parts.push(part);
                            } else {
                                // Cross-model: convert to plain text
                                parts.push(json!({
                                    "text": sanitize_surrogates(&t.thinking)
                                }));
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            let thought_signature = resolve_thought_signature(
                                is_same_provider_and_model,
                                tc.thought_signature.as_deref(),
                            );

                            // Gemini 3 requires thoughtSignature on all function calls
                            // when thinking mode is enabled.
                            let is_gemini3 = model.id.to_lowercase().contains("gemini-3");
                            if is_gemini3 && thought_signature.is_none() {
                                // Convert unsigned function calls to text for Gemini 3
                                let args_str = serde_json::to_string_pretty(&tc.arguments)
                                    .unwrap_or_else(|_| "{}".to_string());
                                parts.push(json!({
                                    "text": format!(
                                        "[Historical context: a different model called tool \"{}\" with arguments: {}. Do not mimic this format - use proper function calling.]",
                                        tc.name,
                                        args_str
                                    )
                                }));
                            } else {
                                let mut fc = json!({
                                    "name": tc.name,
                                    "args": tc.arguments
                                });
                                if requires_tool_call_id(&model.id) {
                                    fc["id"] = json!(tc.id);
                                }
                                let mut part = json!({ "functionCall": fc });
                                if let Some(sig) = thought_signature {
                                    part["thoughtSignature"] = json!(sig);
                                }
                                parts.push(part);
                            }
                        }
                        _ => {}
                    }
                }

                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
            }
            Message::ToolResult(tr) => {
                // Extract text and image content
                let text_content: Vec<&TextContent> =
                    tr.content.iter().filter_map(|c| c.as_text()).collect();
                let text_result: String = text_content
                    .iter()
                    .map(|c| c.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let image_content: Vec<&ImageContent> =
                    if model.input.contains(&"image".to_string()) {
                        tr.content.iter().filter_map(|c| c.as_image()).collect()
                    } else {
                        Vec::new()
                    };

                let has_text = !text_result.is_empty();
                let has_images = !image_content.is_empty();

                // Gemini 3 supports multimodal function responses
                let supports_multimodal_fn_response = model.id.contains("gemini-3");

                let response_value = if has_text {
                    sanitize_surrogates(&text_result)
                } else if has_images {
                    "(see attached image)".to_string()
                } else {
                    String::new()
                };

                let image_parts: Vec<Value> = image_content
                    .iter()
                    .map(|img| {
                        json!({
                            "inlineData": {
                                "mimeType": img.mime_type,
                                "data": img.data
                            }
                        })
                    })
                    .collect();

                let include_id = requires_tool_call_id(&model.id);

                let response_obj = if tr.is_error {
                    json!({ "error": response_value })
                } else {
                    json!({ "output": response_value })
                };

                let mut fn_response = json!({
                    "name": tr.tool_name,
                    "response": response_obj
                });

                // Nest images inside functionResponse.parts for Gemini 3
                if has_images && supports_multimodal_fn_response {
                    fn_response["parts"] = json!(image_parts);
                }
                if include_id {
                    fn_response["id"] = json!(tr.tool_call_id);
                }

                let function_response_part = json!({ "functionResponse": fn_response });

                // Merge consecutive function responses into a single user turn
                let last_content = contents.last_mut();
                let should_merge = if let Some(last) = &last_content {
                    last.get("role").and_then(|r| r.as_str()) == Some("user")
                        && last
                            .get("parts")
                            .and_then(|p| p.as_array())
                            .map(|arr| arr.iter().any(|p| p.get("functionResponse").is_some()))
                            .unwrap_or(false)
                } else {
                    false
                };

                if should_merge {
                    if let Some(last) = contents.last_mut() {
                        if let Some(parts) = last.get_mut("parts").and_then(|p| p.as_array_mut()) {
                            parts.push(function_response_part);
                        }
                    }
                } else {
                    contents.push(json!({
                        "role": "user",
                        "parts": [function_response_part]
                    }));
                }

                // For older models, add images in a separate user message
                if has_images && !supports_multimodal_fn_response {
                    let mut img_parts = vec![json!({ "text": "Tool result image:" })];
                    img_parts.extend(image_parts);
                    contents.push(json!({
                        "role": "user",
                        "parts": img_parts
                    }));
                }
            }
        }
    }

    contents
}

// ---------- Convert tools to Gemini function declarations ----------

fn convert_tools(tools: &[Tool], use_parameters: bool) -> Value {
    if tools.is_empty() {
        return json!(null);
    }
    let param_key = if use_parameters {
        "parameters"
    } else {
        "parametersJsonSchema"
    };
    json!([{
        "functionDeclarations": tools.iter().map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                param_key: tool.parameters
            })
        }).collect::<Vec<Value>>()
    }])
}

// ---------- Build request parameters ----------

fn build_params(model: &Model, context: &Context, options: &GoogleOptions) -> Value {
    let contents = convert_messages(model, context);

    let mut generation_config: Value = json!({});
    if let Some(temp) = options.base.temperature {
        generation_config["temperature"] = json!(temp);
    }
    if let Some(max_tokens) = options.base.max_tokens {
        generation_config["maxOutputTokens"] = json!(max_tokens);
    }

    // Build the config object (used as top-level fields in the REST API request)
    let mut config: Value = json!({});

    // Only add generationConfig if it has any fields
    if generation_config.as_object().is_some_and(|o| !o.is_empty()) {
        // Merge generation config fields directly into the top level
        if let Some(obj) = generation_config.as_object() {
            for (k, v) in obj {
                config[k] = v.clone();
            }
        }
    }

    // System instruction
    if let Some(sp) = &context.system_prompt {
        config["systemInstruction"] = json!({
            "parts": [{ "text": sanitize_surrogates(sp) }]
        });
    }

    // Tools
    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            config["tools"] = convert_tools(tools, false);

            // Tool config
            if let Some(choice) = &options.tool_choice {
                config["toolConfig"] = json!({
                    "functionCallingConfig": {
                        "mode": map_tool_choice(choice)
                    }
                });
            }
        }
    }

    // Thinking config
    if options.thinking_enabled && model.reasoning {
        let mut thinking_config = json!({ "includeThoughts": true });
        if let Some(level) = &options.thinking_level {
            thinking_config["thinkingLevel"] = json!(level);
        } else if let Some(budget) = options.thinking_budget_tokens {
            thinking_config["thinkingBudget"] = json!(budget);
        }
        config["thinkingConfig"] = thinking_config;
    }

    // Build final request body for Google's REST streaming API
    // POST /v1beta/models/{model}:streamGenerateContent?alt=sse
    let mut params = json!({
        "contents": contents
    });

    // Merge config fields into params
    if let Some(obj) = config.as_object() {
        for (k, v) in obj {
            params[k] = v.clone();
        }
    }

    params
}

// ---------- Build HTTP headers ----------

fn build_headers(
    model: &Model,
    api_key: &str,
    extra_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());

    // API key is passed as a query parameter for Google APIs, but we also
    // support Authorization header for custom base URLs
    if !api_key.is_empty() && model.base_url != "https://generativelanguage.googleapis.com/v1beta" {
        headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
    }

    // Merge model headers (skip sensitive auth headers)
    if let Some(model_headers) = &model.headers {
        crate::header_utils::merge_headers_safe(&mut headers, model_headers);
    }

    // Merge extra headers (skip sensitive auth headers)
    if let Some(extra) = extra_headers {
        crate::header_utils::merge_headers_safe(&mut headers, extra);
    }

    headers
}

// ---------- Build the streaming URL ----------

fn build_stream_url(model: &Model, api_key: &str) -> String {
    let base = &model.base_url;

    // If the base URL is the default Google Generative AI endpoint,
    // use the key=xxx query param pattern
    if base.contains("generativelanguage.googleapis.com") {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            base.trim_end_matches('/'),
            model.id,
            api_key
        )
    } else {
        // Custom base URL: use the model in the URL path but no key param
        // (auth handled via headers)
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            base.trim_end_matches('/'),
            model.id
        )
    }
}

// ---------- Gemini model helpers ----------

fn is_gemini3_pro_model(model: &Model) -> bool {
    model.id.contains("3-pro")
}

fn is_gemini3_flash_model(model: &Model) -> bool {
    model.id.contains("3-flash")
}

/// Map our thinking level to Gemini 3 thinking level string.
fn get_gemini3_thinking_level(effort: &ThinkingLevel, model: &Model) -> GoogleThinkingLevel {
    if is_gemini3_pro_model(model) {
        return match effort {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "LOW".to_string(),
            ThinkingLevel::Medium | ThinkingLevel::High => "HIGH".to_string(),
            ThinkingLevel::Xhigh => "HIGH".to_string(),
        };
    }
    match effort {
        ThinkingLevel::Minimal => "MINIMAL".to_string(),
        ThinkingLevel::Low => "LOW".to_string(),
        ThinkingLevel::Medium => "MEDIUM".to_string(),
        ThinkingLevel::High | ThinkingLevel::Xhigh => "HIGH".to_string(),
    }
}

/// Get the Google thinking budget for older (non-Gemini 3) models.
fn get_google_budget(
    model: &Model,
    effort: &ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
    let clamped = clamp_reasoning(effort);

    // Check custom budgets first
    if let Some(budgets) = custom_budgets {
        let custom = match clamped {
            ThinkingLevel::Minimal => budgets.minimal,
            ThinkingLevel::Low => budgets.low,
            ThinkingLevel::Medium => budgets.medium,
            ThinkingLevel::High => budgets.high,
            ThinkingLevel::Xhigh => budgets.high, // already clamped
        };
        if let Some(v) = custom {
            return v as i64;
        }
    }

    if model.id.contains("2.5-pro") {
        return match clamped {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::Xhigh => 32768,
        };
    }

    if model.id.contains("2.5-flash") {
        return match clamped {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::Xhigh => 24576,
        };
    }

    // Default: dynamic
    -1
}

// ---------- Stream functions ----------

/// Stream from Google Generative AI REST API using raw HTTP + SSE parsing.
///
/// Google's streaming endpoint returns SSE where each `data:` line contains
/// a complete JSON object with the structure:
/// ```json
/// {
///   "candidates": [{ "content": { "parts": [...] }, "finishReason": "..." }],
///   "usageMetadata": { "promptTokenCount": N, ... }
/// }
/// ```
pub fn stream_google(
    model: &Model,
    context: &Context,
    options: &GoogleOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&model);

        let api_key = options
            .base
            .api_key
            .clone()
            .or_else(|| get_env_api_key(&model.provider))
            .unwrap_or_default();

        let headers = build_headers(&model, &api_key, options.base.headers.as_ref());

        let params = build_params(&model, &context, &options);
        let url = build_stream_url(&model, &api_key);

        let client = reqwest::Client::new();
        let mut request = client.post(&url);
        for (k, v) in &headers {
            request = request.header(k.as_str(), v.as_str());
        }

        let response = match request.json(&params).send().await {
            Ok(resp) => resp,
            Err(e) => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(format!("HTTP error: {e}"));
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            output.stop_reason = StopReason::Error;
            output.error_message = Some(format!("{status} {body}"));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        stream_clone.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        // Track current block state (Google streams parts sequentially)
        enum CurrentBlock {
            Text {
                text: String,
                text_signature: Option<String>,
            },
            Thinking {
                thinking: String,
                thinking_signature: Option<String>,
            },
        }

        let mut current_block: Option<CurrentBlock> = None;
        let block_index = |output: &AssistantMessage| -> usize {
            if output.content.is_empty() {
                0
            } else {
                output.content.len() - 1
            }
        };

        // Parse SSE stream
        let mut sse_buffer = String::new();
        let mut byte_stream = response.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            if cancel.is_cancelled() {
                output.stop_reason = StopReason::Aborted;
                output.error_message = Some("Request was aborted".to_string());
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error: output,
                });
                return;
            }

            let chunk = match chunk_result {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(e) => {
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(format!("Stream error: {e}"));
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    });
                    return;
                }
            };

            // Google's SSE format: "data: {json}\n\n"
            // We parse line by line, extracting JSON from data: lines
            sse_buffer.push_str(&chunk);

            // Process complete lines
            while let Some(newline_pos) = sse_buffer.find('\n') {
                let line = sse_buffer[..newline_pos].trim_end_matches('\r').to_string();
                sse_buffer = sse_buffer[newline_pos + 1..].to_string();

                // Skip empty lines and comments
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                // Extract data from "data: {json}" lines
                let json_str = if let Some(data) = line.strip_prefix("data:") {
                    data.trim()
                } else {
                    // Skip non-data lines (event:, id:, retry:)
                    continue;
                };

                if json_str.is_empty() || json_str == "[DONE]" {
                    continue;
                }

                let data: Value = match serde_json::from_str(json_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Process candidates
                if let Some(candidates) = data.get("candidates").and_then(|c| c.as_array()) {
                    if let Some(candidate) = candidates.first() {
                        if let Some(parts) = candidate
                            .get("content")
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                        {
                            for part in parts {
                                // Handle text parts (both regular text and thinking)
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    let is_thinking = is_thinking_part(part);
                                    let thought_sig =
                                        part.get("thoughtSignature").and_then(|s| s.as_str());

                                    // Check if we need to start a new block
                                    let needs_new_block = match &current_block {
                                        None => true,
                                        Some(CurrentBlock::Text { .. }) if is_thinking => true,
                                        Some(CurrentBlock::Thinking { .. }) if !is_thinking => true,
                                        _ => false,
                                    };

                                    if needs_new_block {
                                        // End current block
                                        if let Some(ref cb) = current_block {
                                            let ci = block_index(&output);
                                            match cb {
                                                CurrentBlock::Text { text, .. } => {
                                                    stream_clone.push(
                                                        AssistantMessageEvent::TextEnd {
                                                            content_index: ci,
                                                            content: text.clone(),
                                                            partial: output.clone(),
                                                        },
                                                    );
                                                }
                                                CurrentBlock::Thinking { thinking, .. } => {
                                                    stream_clone.push(
                                                        AssistantMessageEvent::ThinkingEnd {
                                                            content_index: ci,
                                                            content: thinking.clone(),
                                                            partial: output.clone(),
                                                        },
                                                    );
                                                }
                                            }
                                        }

                                        // Start new block
                                        if is_thinking {
                                            current_block = Some(CurrentBlock::Thinking {
                                                thinking: String::new(),
                                                thinking_signature: None,
                                            });
                                            output.content.push(ContentBlock::Thinking(
                                                ThinkingContent {
                                                    thinking: String::new(),
                                                    thinking_signature: None,
                                                },
                                            ));
                                            let ci = block_index(&output);
                                            stream_clone.push(
                                                AssistantMessageEvent::ThinkingStart {
                                                    content_index: ci,
                                                    partial: output.clone(),
                                                },
                                            );
                                        } else {
                                            current_block = Some(CurrentBlock::Text {
                                                text: String::new(),
                                                text_signature: None,
                                            });
                                            output.content.push(ContentBlock::Text(TextContent {
                                                text: String::new(),
                                                text_signature: None,
                                            }));
                                            let ci = block_index(&output);
                                            stream_clone.push(AssistantMessageEvent::TextStart {
                                                content_index: ci,
                                                partial: output.clone(),
                                            });
                                        }
                                    }

                                    // Append to current block
                                    let ci = block_index(&output);
                                    match &mut current_block {
                                        Some(CurrentBlock::Thinking {
                                            thinking,
                                            thinking_signature,
                                        }) => {
                                            thinking.push_str(text);
                                            *thinking_signature = retain_thought_signature(
                                                thinking_signature.as_deref(),
                                                thought_sig,
                                            );
                                            // Update output content
                                            if let Some(ContentBlock::Thinking(t)) =
                                                output.content.get_mut(ci)
                                            {
                                                t.thinking.push_str(text);
                                                t.thinking_signature = thinking_signature.clone();
                                            }
                                            stream_clone.push(
                                                AssistantMessageEvent::ThinkingDelta {
                                                    content_index: ci,
                                                    delta: text.to_string(),
                                                    partial: output.clone(),
                                                },
                                            );
                                        }
                                        Some(CurrentBlock::Text {
                                            text: block_text,
                                            text_signature,
                                        }) => {
                                            block_text.push_str(text);
                                            *text_signature = retain_thought_signature(
                                                text_signature.as_deref(),
                                                thought_sig,
                                            );
                                            // Update output content
                                            if let Some(ContentBlock::Text(t)) =
                                                output.content.get_mut(ci)
                                            {
                                                t.text.push_str(text);
                                                t.text_signature = text_signature.clone();
                                            }
                                            stream_clone.push(AssistantMessageEvent::TextDelta {
                                                content_index: ci,
                                                delta: text.to_string(),
                                                partial: output.clone(),
                                            });
                                        }
                                        _ => {}
                                    }
                                }

                                // Handle function call parts
                                if let Some(fc) = part.get("functionCall") {
                                    // End current text/thinking block if active
                                    if let Some(ref cb) = current_block {
                                        let ci = block_index(&output);
                                        match cb {
                                            CurrentBlock::Text { text, .. } => {
                                                stream_clone.push(AssistantMessageEvent::TextEnd {
                                                    content_index: ci,
                                                    content: text.clone(),
                                                    partial: output.clone(),
                                                });
                                            }
                                            CurrentBlock::Thinking { thinking, .. } => {
                                                stream_clone.push(
                                                    AssistantMessageEvent::ThinkingEnd {
                                                        content_index: ci,
                                                        content: thinking.clone(),
                                                        partial: output.clone(),
                                                    },
                                                );
                                            }
                                        }
                                        current_block = None;
                                    }

                                    let fc_name = fc
                                        .get("name")
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let fc_args = fc.get("args").cloned().unwrap_or(json!({}));
                                    let fc_id = fc
                                        .get("id")
                                        .and_then(|i| i.as_str())
                                        .map(|s| s.to_string());

                                    let thought_sig = part
                                        .get("thoughtSignature")
                                        .and_then(|s| s.as_str())
                                        .map(|s| s.to_string());

                                    // Generate unique ID if not provided or if it's a duplicate
                                    let provided_id = fc_id;
                                    let needs_new_id = match &provided_id {
                                        None => true,
                                        Some(id) => output.content.iter().any(|b| {
                                            if let ContentBlock::ToolCall(tc) = b {
                                                tc.id == *id
                                            } else {
                                                false
                                            }
                                        }),
                                    };

                                    let tool_call_id = if needs_new_id {
                                        let counter =
                                            TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
                                        let now = chrono::Utc::now().timestamp_millis();
                                        format!("{fc_name}_{now}_{counter}")
                                    } else {
                                        provided_id.unwrap()
                                    };

                                    let tool_call = ToolCall {
                                        id: tool_call_id,
                                        name: fc_name,
                                        arguments: fc_args.clone(),
                                        thought_signature: thought_sig,
                                    };

                                    output
                                        .content
                                        .push(ContentBlock::ToolCall(tool_call.clone()));
                                    let ci = block_index(&output);

                                    stream_clone.push(AssistantMessageEvent::ToolCallStart {
                                        content_index: ci,
                                        partial: output.clone(),
                                    });
                                    stream_clone.push(AssistantMessageEvent::ToolCallDelta {
                                        content_index: ci,
                                        delta: serde_json::to_string(&fc_args).unwrap_or_default(),
                                        partial: output.clone(),
                                    });
                                    stream_clone.push(AssistantMessageEvent::ToolCallEnd {
                                        content_index: ci,
                                        tool_call,
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }

                        // Handle finish reason
                        if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str())
                        {
                            output.stop_reason = map_stop_reason(reason);
                            // Override to toolUse if we have any tool calls
                            if output
                                .content
                                .iter()
                                .any(|b| matches!(b, ContentBlock::ToolCall(_)))
                            {
                                output.stop_reason = StopReason::ToolUse;
                            }
                        }
                    }
                }

                // Handle usage metadata
                if let Some(usage) = data.get("usageMetadata") {
                    let prompt_tokens = usage
                        .get("promptTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let candidates_tokens = usage
                        .get("candidatesTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let thoughts_tokens = usage
                        .get("thoughtsTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let cached_tokens = usage
                        .get("cachedContentTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let total_tokens = usage
                        .get("totalTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);

                    output.usage = Usage {
                        input: prompt_tokens,
                        output: candidates_tokens + thoughts_tokens,
                        cache_read: cached_tokens,
                        cache_write: 0,
                        total_tokens,
                        cost: UsageCost::default(),
                    };
                    calculate_cost(&model, &mut output.usage);
                }
            }
        }

        // End final block
        if let Some(ref cb) = current_block {
            let ci = block_index(&output);
            match cb {
                CurrentBlock::Text { text, .. } => {
                    stream_clone.push(AssistantMessageEvent::TextEnd {
                        content_index: ci,
                        content: text.clone(),
                        partial: output.clone(),
                    });
                }
                CurrentBlock::Thinking { thinking, .. } => {
                    stream_clone.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: ci,
                        content: thinking.clone(),
                        partial: output.clone(),
                    });
                }
            }
        }

        if cancel.is_cancelled() {
            output.stop_reason = StopReason::Aborted;
            output.error_message = Some("Request was aborted".to_string());
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: output,
            });
            return;
        }

        if output.stop_reason == StopReason::Error || output.stop_reason == StopReason::Aborted {
            stream_clone.push(AssistantMessageEvent::Error {
                reason: output.stop_reason.clone(),
                error: output,
            });
        } else {
            stream_clone.push(AssistantMessageEvent::Done {
                reason: output.stop_reason.clone(),
                message: output,
            });
        }
    });

    stream
}

/// Simplified streaming: handles reasoning level and API key resolution.
pub fn stream_simple_google(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let api_key = options
        .base
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();

    if api_key.is_empty() {
        let stream = create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some(format!("No API key for provider: {}", model.provider));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg,
        });
        return stream;
    }

    let base = build_base_options(model, options, &api_key);

    if options.reasoning.is_none() {
        let google_opts = GoogleOptions {
            base,
            tool_choice: None,
            thinking_enabled: false,
            thinking_budget_tokens: None,
            thinking_level: None,
        };
        return stream_google(model, context, &google_opts, cancel);
    }

    let reasoning = options.reasoning.as_ref().unwrap();
    let effort = clamp_reasoning(reasoning);

    // Gemini 3 models use thinking levels
    if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) {
        let level = get_gemini3_thinking_level(&effort, model);
        let google_opts = GoogleOptions {
            base,
            tool_choice: None,
            thinking_enabled: true,
            thinking_budget_tokens: None,
            thinking_level: Some(level),
        };
        return stream_google(model, context, &google_opts, cancel);
    }

    // Older models use budget tokens
    let budget = get_google_budget(model, &effort, options.thinking_budgets.as_ref());
    let google_opts = GoogleOptions {
        base,
        tool_choice: None,
        thinking_enabled: true,
        thinking_budget_tokens: Some(budget),
        thinking_level: None,
    };
    stream_google(model, context, &google_opts, cancel)
}

// ---------- ApiProvider implementation ----------

pub struct GoogleProvider;

impl ApiProvider for GoogleProvider {
    fn api(&self) -> &str {
        "google-generative-ai"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let google_opts = GoogleOptions {
            base: options.clone(),
            tool_choice: None,
            thinking_enabled: false,
            thinking_budget_tokens: None,
            thinking_level: None,
        };
        stream_google(model, context, &google_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_google(model, context, options, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_google_model() -> Model {
        Model {
            id: "gemini-2.5-flash".to_string(),
            name: "Gemini 2.5 Flash".to_string(),
            api: "google-generative-ai".to_string(),
            provider: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            reasoning: true,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost {
                input: 0.3,
                output: 2.5,
                cache_read: 0.075,
                cache_write: 0.0,
            },
            context_window: 1048576,
            max_tokens: 65536,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_map_stop_reason() {
        assert_eq!(map_stop_reason("STOP"), StopReason::Stop);
        assert_eq!(map_stop_reason("MAX_TOKENS"), StopReason::Length);
        assert_eq!(map_stop_reason("SAFETY"), StopReason::Error);
        assert_eq!(
            map_stop_reason("MALFORMED_FUNCTION_CALL"),
            StopReason::Error
        );
        assert_eq!(map_stop_reason("UNKNOWN"), StopReason::Error);
    }

    #[test]
    fn test_map_tool_choice() {
        assert_eq!(map_tool_choice("auto"), "AUTO");
        assert_eq!(map_tool_choice("none"), "NONE");
        assert_eq!(map_tool_choice("any"), "ANY");
        assert_eq!(map_tool_choice("unknown"), "AUTO");
    }

    #[test]
    fn test_is_thinking_part() {
        assert!(is_thinking_part(
            &json!({ "thought": true, "text": "thinking..." })
        ));
        assert!(!is_thinking_part(&json!({ "text": "regular text" })));
        assert!(!is_thinking_part(
            &json!({ "thought": false, "text": "not thinking" })
        ));
        // thoughtSignature alone does NOT make a part thinking
        assert!(!is_thinking_part(
            &json!({ "text": "text", "thoughtSignature": "abc=" })
        ));
    }

    #[test]
    fn test_retain_thought_signature() {
        assert_eq!(
            retain_thought_signature(None, Some("abc=")),
            Some("abc=".to_string())
        );
        assert_eq!(
            retain_thought_signature(Some("old="), None),
            Some("old=".to_string())
        );
        assert_eq!(
            retain_thought_signature(Some("old="), Some("new=")),
            Some("new=".to_string())
        );
        assert_eq!(
            retain_thought_signature(Some("old="), Some("")),
            Some("old=".to_string())
        );
        assert_eq!(retain_thought_signature(None, None), None);
    }

    #[test]
    fn test_is_valid_thought_signature() {
        assert!(is_valid_thought_signature("AAAA"));
        assert!(is_valid_thought_signature("AAAAAAA=")); // 1 pad char
        assert!(is_valid_thought_signature("AAAAAA==")); // 2 pad chars
        assert!(is_valid_thought_signature("aGVsbG8="));
        assert!(!is_valid_thought_signature(""));
        assert!(!is_valid_thought_signature("abc")); // not multiple of 4
        assert!(!is_valid_thought_signature("ab!d")); // invalid char
        assert!(!is_valid_thought_signature("AAAA====")); // >2 pad chars (invalid base64)
    }

    #[test]
    fn test_requires_tool_call_id() {
        assert!(requires_tool_call_id("claude-sonnet-4"));
        assert!(requires_tool_call_id("gpt-oss-4o"));
        assert!(!requires_tool_call_id("gemini-2.5-pro"));
        assert!(!requires_tool_call_id("gemini-2.5-flash"));
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![Tool {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        }];
        let result = convert_tools(&tools, false);
        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let decls = arr[0]
            .get("functionDeclarations")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "search");
        assert!(decls[0].get("parametersJsonSchema").is_some());
    }

    #[test]
    fn test_convert_tools_use_parameters() {
        let tools = vec![Tool {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: json!({"type": "object"}),
        }];
        let result = convert_tools(&tools, true);
        let decls = result[0]["functionDeclarations"].as_array().unwrap();
        assert!(decls[0].get("parameters").is_some());
        assert!(decls[0].get("parametersJsonSchema").is_none());
    }

    #[test]
    fn test_convert_tools_empty() {
        let result = convert_tools(&[], false);
        assert!(result.is_null());
    }

    #[test]
    fn test_convert_messages_user_text() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Hello".to_string()),
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn test_convert_messages_assistant_text() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Hello world".to_string(),
                    text_signature: None,
                })],
                api: "google-generative-ai".to_string(),
                provider: "google".to_string(),
                model: "gemini-2.5-flash".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "model");
        assert_eq!(result[0]["parts"][0]["text"], "Hello world");
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "Found results".to_string(),
                    text_signature: None,
                })],
                details: None,
                is_error: false,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        let fn_resp = &result[0]["parts"][0]["functionResponse"];
        assert_eq!(fn_resp["name"], "search");
        assert_eq!(fn_resp["response"]["output"], "Found results");
    }

    #[test]
    fn test_convert_messages_tool_result_error() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_1".to_string(),
                tool_name: "bash".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "command not found".to_string(),
                    text_signature: None,
                })],
                details: None,
                is_error: true,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        let fn_resp = &result[0]["parts"][0]["functionResponse"];
        assert_eq!(fn_resp["response"]["error"], "command not found");
    }

    #[test]
    fn test_build_stream_url_default() {
        let model = test_google_model();
        let url = build_stream_url(&model, "test-key");
        assert!(url.contains("generativelanguage.googleapis.com"));
        assert!(url.contains("gemini-2.5-flash:streamGenerateContent"));
        assert!(url.contains("key=test-key"));
        assert!(url.contains("alt=sse"));
    }

    #[test]
    fn test_build_stream_url_custom_base() {
        let mut model = test_google_model();
        model.base_url = "https://custom-proxy.example.com/v1".to_string();
        let url = build_stream_url(&model, "test-key");
        assert!(!url.contains("key="));
        assert!(url.contains("custom-proxy.example.com"));
        assert!(url.contains("alt=sse"));
    }

    #[test]
    fn test_build_params_basic() {
        let model = test_google_model();
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Hello".to_string()),
                timestamp: 0,
            })],
            tools: None,
        };
        let options = GoogleOptions {
            base: StreamOptions {
                temperature: Some(0.7),
                max_tokens: Some(1024),
                ..Default::default()
            },
            tool_choice: None,
            thinking_enabled: false,
            thinking_budget_tokens: None,
            thinking_level: None,
        };
        let params = build_params(&model, &context, &options);

        assert!(params.get("contents").is_some());
        assert_eq!(params["temperature"], 0.7);
        assert_eq!(params["maxOutputTokens"], 1024);
        assert!(params.get("systemInstruction").is_some());
        assert_eq!(
            params["systemInstruction"]["parts"][0]["text"],
            "You are helpful."
        );
    }

    #[test]
    fn test_build_params_with_thinking() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Think hard".to_string()),
                timestamp: 0,
            })],
            tools: None,
        };
        let options = GoogleOptions {
            base: StreamOptions::default(),
            tool_choice: None,
            thinking_enabled: true,
            thinking_budget_tokens: Some(8192),
            thinking_level: None,
        };
        let params = build_params(&model, &context, &options);
        assert_eq!(params["thinkingConfig"]["includeThoughts"], true);
        assert_eq!(params["thinkingConfig"]["thinkingBudget"], 8192);
    }

    #[test]
    fn test_build_params_with_thinking_level() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Think".to_string()),
                timestamp: 0,
            })],
            tools: None,
        };
        let options = GoogleOptions {
            base: StreamOptions::default(),
            tool_choice: None,
            thinking_enabled: true,
            thinking_budget_tokens: None,
            thinking_level: Some("HIGH".to_string()),
        };
        let params = build_params(&model, &context, &options);
        assert_eq!(params["thinkingConfig"]["includeThoughts"], true);
        assert_eq!(params["thinkingConfig"]["thinkingLevel"], "HIGH");
    }

    #[test]
    fn test_build_params_with_tools() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("search for rust".to_string()),
                timestamp: 0,
            })],
            tools: Some(vec![Tool {
                name: "search".to_string(),
                description: "Search the web".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"]
                }),
            }]),
        };
        let options = GoogleOptions {
            base: StreamOptions::default(),
            tool_choice: Some("auto".to_string()),
            thinking_enabled: false,
            thinking_budget_tokens: None,
            thinking_level: None,
        };
        let params = build_params(&model, &context, &options);
        assert!(params.get("tools").is_some());
        assert_eq!(
            params["toolConfig"]["functionCallingConfig"]["mode"],
            "AUTO"
        );
    }

    #[test]
    fn test_get_google_budget_25_pro() {
        let mut model = test_google_model();
        model.id = "gemini-2.5-pro".to_string();
        assert_eq!(
            get_google_budget(&model, &ThinkingLevel::Minimal, None),
            128
        );
        assert_eq!(get_google_budget(&model, &ThinkingLevel::Low, None), 2048);
        assert_eq!(
            get_google_budget(&model, &ThinkingLevel::Medium, None),
            8192
        );
        assert_eq!(get_google_budget(&model, &ThinkingLevel::High, None), 32768);
    }

    #[test]
    fn test_get_google_budget_25_flash() {
        let mut model = test_google_model();
        model.id = "gemini-2.5-flash".to_string();
        assert_eq!(
            get_google_budget(&model, &ThinkingLevel::Minimal, None),
            128
        );
        assert_eq!(get_google_budget(&model, &ThinkingLevel::Low, None), 2048);
        assert_eq!(
            get_google_budget(&model, &ThinkingLevel::Medium, None),
            8192
        );
        assert_eq!(get_google_budget(&model, &ThinkingLevel::High, None), 24576);
    }

    #[test]
    fn test_get_google_budget_custom() {
        let model = test_google_model();
        let budgets = ThinkingBudgets {
            minimal: Some(256),
            low: Some(1024),
            medium: Some(4096),
            high: Some(16384),
        };
        assert_eq!(
            get_google_budget(&model, &ThinkingLevel::Medium, Some(&budgets)),
            4096
        );
    }

    #[test]
    fn test_get_google_budget_unknown_model() {
        let mut model = test_google_model();
        model.id = "some-other-model".to_string();
        assert_eq!(get_google_budget(&model, &ThinkingLevel::Medium, None), -1);
    }

    #[test]
    fn test_gemini3_thinking_level_pro() {
        let mut model = test_google_model();
        model.id = "gemini-3-pro".to_string();
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Minimal, &model),
            "LOW"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Low, &model),
            "LOW"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Medium, &model),
            "HIGH"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::High, &model),
            "HIGH"
        );
    }

    #[test]
    fn test_gemini3_thinking_level_flash() {
        let mut model = test_google_model();
        model.id = "gemini-3-flash".to_string();
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Minimal, &model),
            "MINIMAL"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Low, &model),
            "LOW"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::Medium, &model),
            "MEDIUM"
        );
        assert_eq!(
            get_gemini3_thinking_level(&ThinkingLevel::High, &model),
            "HIGH"
        );
    }

    #[test]
    fn test_convert_messages_thinking_same_model() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(ThinkingContent {
                        thinking: "Let me think...".to_string(),
                        thinking_signature: Some("AAAA".to_string()),
                    }),
                    ContentBlock::Text(TextContent {
                        text: "Here is the answer".to_string(),
                        text_signature: None,
                    }),
                ],
                api: "google-generative-ai".to_string(),
                provider: "google".to_string(),
                model: "gemini-2.5-flash".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        assert_eq!(result.len(), 1);
        let parts = result[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        // First part should be thinking
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["text"], "Let me think...");
        assert_eq!(parts[0]["thoughtSignature"], "AAAA");
        // Second part should be text
        assert_eq!(parts[1]["text"], "Here is the answer");
    }

    #[test]
    fn test_convert_messages_thinking_cross_model() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(ThinkingContent {
                        thinking: "Let me think...".to_string(),
                        thinking_signature: Some("sig".to_string()),
                    }),
                    ContentBlock::Text(TextContent {
                        text: "Answer".to_string(),
                        text_signature: None,
                    }),
                ],
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        let parts = result[0]["parts"].as_array().unwrap();
        // Thinking should be converted to plain text for cross-model
        assert!(parts[0].get("thought").is_none());
        assert_eq!(parts[0]["text"], "Let me think...");
    }

    #[test]
    fn test_convert_messages_function_call() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    arguments: json!({"query": "rust"}),
                    thought_signature: None,
                })],
                api: "google-generative-ai".to_string(),
                provider: "google".to_string(),
                model: "gemini-2.5-flash".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 0,
            })],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        let parts = result[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "search");
        assert_eq!(parts[0]["functionCall"]["args"]["query"], "rust");
        // Regular gemini models don't include id
        assert!(
            parts[0]["functionCall"].get("id").is_none()
                || parts[0]["functionCall"]["id"].is_null()
        );
    }

    #[test]
    fn test_convert_messages_consecutive_tool_results_merged() {
        let model = test_google_model();
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "search".to_string(),
                    content: vec![ContentBlock::Text(TextContent {
                        text: "Result 1".to_string(),
                        text_signature: None,
                    })],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_2".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ContentBlock::Text(TextContent {
                        text: "Result 2".to_string(),
                        text_signature: None,
                    })],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }),
            ],
            tools: None,
        };
        let result = convert_messages(&model, &context);
        // Both tool results should be merged into one user message
        assert_eq!(result.len(), 1);
        let parts = result[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionResponse"]["name"], "search");
        assert_eq!(parts[1]["functionResponse"]["name"], "read");
    }

    #[test]
    fn test_google_provider_api() {
        let provider = GoogleProvider;
        assert_eq!(provider.api(), "google-generative-ai");
    }

    #[test]
    fn test_normalize_tool_call_id() {
        assert_eq!(normalize_tool_call_id("call_123"), "call_123");
        assert_eq!(normalize_tool_call_id("call.123/foo"), "call_123_foo");
        // Test truncation to 64 chars
        let long_id = "a".repeat(100);
        assert_eq!(normalize_tool_call_id(&long_id).len(), 64);
    }

    #[test]
    fn test_resolve_thought_signature() {
        // Same provider, valid sig
        assert_eq!(
            resolve_thought_signature(true, Some("AAAA")),
            Some("AAAA".to_string())
        );
        // Same provider, invalid sig
        assert_eq!(resolve_thought_signature(true, Some("abc")), None);
        // Different provider
        assert_eq!(resolve_thought_signature(false, Some("AAAA")), None);
        // No signature
        assert_eq!(resolve_thought_signature(true, None), None);
    }
}
