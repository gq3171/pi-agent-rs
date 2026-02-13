use std::collections::HashMap;

use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use pi_agent_core::json_parse::parse_streaming_json;
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use super::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::env_keys::get_env_api_key;
use crate::models::calculate_cost;
use crate::registry::ApiProvider;
use crate::simple_options::{adjust_max_tokens_for_thinking, build_base_options};
use crate::sse::SseParser;

// ---------- Claude Code Stealth Mode ----------

const CLAUDE_CODE_VERSION: &str = "2.1.2";

const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

fn to_claude_code_name(name: &str) -> String {
    let lower = name.to_lowercase();
    for &tool in CLAUDE_CODE_TOOLS {
        if tool.to_lowercase() == lower {
            return tool.to_string();
        }
    }
    name.to_string()
}

fn from_claude_code_name(name: &str, tools: Option<&[Tool]>) -> String {
    if let Some(tools) = tools {
        let lower = name.to_lowercase();
        for tool in tools {
            if tool.name.to_lowercase() == lower {
                return tool.name.clone();
            }
        }
    }
    name.to_string()
}

// ---------- AnthropicOptions ----------

#[derive(Debug, Clone)]
pub struct AnthropicOptions {
    pub base: StreamOptions,
    pub thinking_enabled: bool,
    pub thinking_budget_tokens: Option<u64>,
    pub effort: Option<String>, // "low" | "medium" | "high" | "max"
    pub interleaved_thinking: bool,
    pub tool_choice: Option<Value>,
}

// ---------- Helper functions ----------

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn resolve_cache_retention(retention: Option<&CacheRetention>) -> CacheRetention {
    if let Some(r) = retention {
        return r.clone();
    }
    // Check environment variable fallback (matches TS PI_CACHE_RETENTION)
    if let Ok(val) = std::env::var("PI_CACHE_RETENTION") {
        match val.to_lowercase().as_str() {
            "none" => return CacheRetention::None,
            "short" => return CacheRetention::Short,
            "long" => return CacheRetention::Long,
            _ => {}
        }
    }
    CacheRetention::Short
}

fn get_cache_control(
    base_url: &str,
    cache_retention: Option<&CacheRetention>,
) -> (CacheRetention, Option<Value>) {
    let retention = resolve_cache_retention(cache_retention);
    if retention == CacheRetention::None {
        return (retention, None);
    }
    let ttl = if retention == CacheRetention::Long && base_url.contains("api.anthropic.com") {
        Some("1h")
    } else {
        None
    };
    let mut cache_control = json!({"type": "ephemeral"});
    if let Some(ttl) = ttl {
        cache_control["ttl"] = json!(ttl);
    }
    (retention, Some(cache_control))
}

fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6") || model_id.contains("opus-4.6")
}

fn map_thinking_level_to_effort(level: &ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal => "low",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "max",
    }
}

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

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "refusal" => StopReason::Error,
        "pause_turn" => StopReason::Stop,
        "stop_sequence" => StopReason::Stop,
        "sensitive" => StopReason::Error,
        unknown => {
            tracing::warn!("Unknown Anthropic stop reason: {}", unknown);
            StopReason::Error
        }
    }
}

// ---------- Convert content blocks to Anthropic format ----------

fn convert_content_blocks(content: &[ContentBlock]) -> Value {
    let has_images = content.iter().any(|c| c.as_image().is_some());

    if !has_images {
        // Text only: concatenate into a single string
        let text: String = content
            .iter()
            .filter_map(|c| c.as_text().map(|t| sanitize_surrogates(&t.text)))
            .collect::<Vec<_>>()
            .join("\n");
        return json!(text);
    }

    // Has images: convert to content block array
    let mut blocks: Vec<Value> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(t) => Some(json!({
                "type": "text",
                "text": sanitize_surrogates(&t.text)
            })),
            ContentBlock::Image(i) => Some(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": i.mime_type,
                    "data": i.data
                }
            })),
            _ => None,
        })
        .collect();

    // If only images, add placeholder text
    let has_text = blocks
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"));
    if !has_text {
        blocks.insert(0, json!({"type": "text", "text": "(see attached image)"}));
    }

    json!(blocks)
}

// ---------- Convert messages ----------

fn convert_messages(
    messages: &[Message],
    model: &Model,
    is_oauth: bool,
    cache_control: &Option<Value>,
    _tools: Option<&[Tool]>,
) -> Vec<Value> {
    let transformed =
        transform_messages(messages, model, Some(&normalize_tool_call_id_for_transform));

    let mut params: Vec<Value> = Vec::new();

    let mut i = 0;
    while i < transformed.len() {
        let msg = &transformed[i];

        match msg {
            Message::User(user_msg) => match &user_msg.content {
                UserContent::Text(text) => {
                    if !text.trim().is_empty() {
                        params.push(json!({
                            "role": "user",
                            "content": sanitize_surrogates(text)
                        }));
                    }
                }
                UserContent::Blocks(blocks) => {
                    let filtered: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(t) => {
                                if t.text.trim().is_empty() {
                                    None
                                } else {
                                    Some(json!({
                                        "type": "text",
                                        "text": sanitize_surrogates(&t.text)
                                    }))
                                }
                            }
                            ContentBlock::Image(img) => {
                                if model.input.contains(&"image".to_string()) {
                                    Some(json!({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": img.mime_type,
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

                    if !filtered.is_empty() {
                        params.push(json!({
                            "role": "user",
                            "content": filtered
                        }));
                    }
                }
            },
            Message::Assistant(assistant_msg) => {
                let mut blocks: Vec<Value> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            if !t.text.trim().is_empty() {
                                blocks.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&t.text)
                                }));
                            }
                        }
                        ContentBlock::Thinking(t) => {
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            if t.thinking_signature
                                .as_ref()
                                .is_some_and(|s| !s.trim().is_empty())
                            {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&t.thinking),
                                    "signature": t.thinking_signature
                                }));
                            } else {
                                // No signature: convert to text
                                blocks.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&t.thinking)
                                }));
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            let name = if is_oauth {
                                to_claude_code_name(&tc.name)
                            } else {
                                tc.name.clone()
                            };
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": name,
                                "input": tc.arguments
                            }));
                        }
                        _ => {}
                    }
                }

                if !blocks.is_empty() {
                    params.push(json!({
                        "role": "assistant",
                        "content": blocks
                    }));
                }
            }
            Message::ToolResult(tr) => {
                // Collect consecutive toolResult messages
                let mut tool_results: Vec<Value> = Vec::new();

                tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": convert_content_blocks(&tr.content),
                    "is_error": tr.is_error
                }));

                // Look ahead for consecutive toolResult messages
                let mut j = i + 1;
                while j < transformed.len() {
                    if let Message::ToolResult(next_tr) = &transformed[j] {
                        tool_results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": next_tr.tool_call_id,
                            "content": convert_content_blocks(&next_tr.content),
                            "is_error": next_tr.is_error
                        }));
                        j += 1;
                    } else {
                        break;
                    }
                }

                i = j - 1; // Will be incremented at end of loop

                params.push(json!({
                    "role": "user",
                    "content": tool_results
                }));
            }
        }

        i += 1;
    }

    // Add cache_control to the last user message
    if let Some(cc) = cache_control {
        if let Some(last) = params.last_mut() {
            if last.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(content) = last.get_mut("content") {
                    if let Some(arr) = content.as_array_mut() {
                        if let Some(last_block) = arr.last_mut() {
                            last_block["cache_control"] = cc.clone();
                        }
                    } else if content.is_string() {
                        let text = content.as_str().unwrap_or("").to_string();
                        *content = json!([{
                            "type": "text",
                            "text": text,
                            "cache_control": cc
                        }]);
                    }
                }
            }
        }
    }

    params
}

fn convert_tools(tools: &[Tool], is_oauth: bool) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let name = if is_oauth {
                to_claude_code_name(&tool.name)
            } else {
                tool.name.clone()
            };
            let schema = &tool.parameters;
            json!({
                "name": name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": schema.get("properties").cloned().unwrap_or(json!({})),
                    "required": schema.get("required").cloned().unwrap_or(json!([]))
                }
            })
        })
        .collect()
}

// ---------- Build request params ----------

fn build_params(
    model: &Model,
    context: &Context,
    is_oauth: bool,
    options: &AnthropicOptions,
) -> Value {
    let (_, cache_control) =
        get_cache_control(&model.base_url, options.base.cache_retention.as_ref());

    let tools_slice: Option<&[Tool]> = context.tools.as_deref();
    let messages = convert_messages(
        &context.messages,
        model,
        is_oauth,
        &cache_control,
        tools_slice,
    );

    let max_tokens = options
        .base
        .max_tokens
        .unwrap_or((model.max_tokens / 3).max(1));

    let mut params = json!({
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": true
    });

    // System prompt
    if is_oauth {
        let mut system_blocks = vec![json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude."
        })];
        if let Some(cc) = &cache_control {
            system_blocks[0]["cache_control"] = cc.clone();
        }
        if let Some(sp) = &context.system_prompt {
            let mut block = json!({
                "type": "text",
                "text": sanitize_surrogates(sp)
            });
            if let Some(cc) = &cache_control {
                block["cache_control"] = cc.clone();
            }
            system_blocks.push(block);
        }
        params["system"] = json!(system_blocks);
    } else if let Some(sp) = &context.system_prompt {
        let mut block = json!({
            "type": "text",
            "text": sanitize_surrogates(sp)
        });
        if let Some(cc) = &cache_control {
            block["cache_control"] = cc.clone();
        }
        params["system"] = json!([block]);
    }

    if let Some(temp) = options.base.temperature {
        params["temperature"] = json!(temp);
    }

    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            params["tools"] = json!(convert_tools(tools, is_oauth));
        }
    }

    // Thinking mode
    if options.thinking_enabled && model.reasoning {
        if supports_adaptive_thinking(&model.id) {
            params["thinking"] = json!({"type": "adaptive"});
            if let Some(effort) = &options.effort {
                params["output_config"] = json!({"effort": effort});
            }
        } else {
            params["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": options.thinking_budget_tokens.unwrap_or(1024)
            });
        }
    }

    if let Some(tc) = &options.tool_choice {
        params["tool_choice"] = tc.clone();
    }

    params
}

// ---------- Build HTTP headers ----------

fn build_headers(
    model: &Model,
    api_key: &str,
    is_oauth: bool,
    interleaved_thinking: bool,
    dynamic_headers: Option<&HashMap<String, String>>,
    extra_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("accept".to_string(), "application/json".to_string());
    headers.insert(
        "anthropic-dangerous-direct-browser-access".to_string(),
        "true".to_string(),
    );

    // Copilot Claude: Bearer auth + selective betas (no fine-grained-tool-streaming).
    if model.provider == "github-copilot" {
        let mut beta_features = Vec::new();
        if interleaved_thinking {
            beta_features.push("interleaved-thinking-2025-05-14".to_string());
        }
        if !beta_features.is_empty() {
            headers.insert("anthropic-beta".to_string(), beta_features.join(","));
        }
        headers.insert("authorization".to_string(), format!("Bearer {api_key}"));

        if let Some(model_headers) = &model.headers {
            crate::header_utils::merge_headers_safe(&mut headers, model_headers);
        }
        if let Some(dynamic) = dynamic_headers {
            crate::header_utils::merge_headers_safe(&mut headers, dynamic);
        }
        if let Some(extra) = extra_headers {
            crate::header_utils::merge_headers_safe(&mut headers, extra);
        }
        return headers;
    }

    let mut beta_features = vec!["fine-grained-tool-streaming-2025-05-14".to_string()];
    if interleaved_thinking {
        beta_features.push("interleaved-thinking-2025-05-14".to_string());
    }

    if is_oauth {
        headers.insert(
            "anthropic-beta".to_string(),
            format!(
                "claude-code-20250219,oauth-2025-04-20,{}",
                beta_features.join(",")
            ),
        );
        headers.insert(
            "user-agent".to_string(),
            format!("claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"),
        );
        headers.insert("x-app".to_string(), "cli".to_string());
        headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
    } else {
        headers.insert("anthropic-beta".to_string(), beta_features.join(","));
        headers.insert("x-api-key".to_string(), api_key.to_string());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
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

// ---------- Stream functions ----------

/// Stream from Anthropic Messages API using raw HTTP + SSE parsing.
pub fn stream_anthropic(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
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

        let is_oauth = is_oauth_token(&api_key);
        let interleaved = options.interleaved_thinking;
        let copilot_dynamic_headers = if model.provider == "github-copilot" {
            let has_images = has_copilot_vision_input(&context.messages);
            Some(build_copilot_dynamic_headers(&context.messages, has_images))
        } else {
            None
        };
        let headers = build_headers(
            &model,
            &api_key,
            is_oauth,
            interleaved,
            copilot_dynamic_headers.as_ref(),
            options.base.headers.as_ref(),
        );

        let params = build_params(&model, &context, is_oauth, &options);

        let url = format!("{}/v1/messages", model.base_url);
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

        // Track block indices (Anthropic uses event.index to identify blocks)
        let mut block_indices: Vec<usize> = Vec::new(); // Maps Anthropic index → our content index
        let mut partial_json_map: HashMap<usize, String> = HashMap::new();

        let mut sse_parser = SseParser::new();
        let tools_slice = context.tools.as_deref();

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

            let events = match sse_parser.feed(&chunk) {
                Ok(events) => events,
                Err(e) => {
                    tracing::error!("SSE parse error: {e}");
                    break;
                }
            };
            for sse_event in events {
                let data: Value = match serde_json::from_str(&sse_event.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match sse_event.event_type.as_str() {
                    "message_start" => {
                        if let Some(message) = data.get("message") {
                            if let Some(usage) = message.get("usage") {
                                output.usage.input = usage
                                    .get("input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                output.usage.output = usage
                                    .get("output_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                output.usage.cache_read = usage
                                    .get("cache_read_input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                output.usage.cache_write = usage
                                    .get("cache_creation_input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                output.usage.total_tokens = output.usage.input
                                    + output.usage.output
                                    + output.usage.cache_read
                                    + output.usage.cache_write;
                                calculate_cost(&model, &mut output.usage);
                            }
                        }
                    }
                    "content_block_start" => {
                        let index =
                            data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let content_block = data.get("content_block").unwrap_or(&Value::Null);
                        let block_type = content_block
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        match block_type {
                            "text" => {
                                output.content.push(ContentBlock::Text(TextContent {
                                    text: String::new(),
                                    text_signature: None,
                                }));
                                let ci = output.content.len() - 1;
                                // Map Anthropic index to our content index
                                while block_indices.len() <= index {
                                    block_indices.push(0);
                                }
                                block_indices[index] = ci;
                                stream_clone.push(AssistantMessageEvent::TextStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                });
                            }
                            "thinking" => {
                                output.content.push(ContentBlock::Thinking(ThinkingContent {
                                    thinking: String::new(),
                                    thinking_signature: None,
                                }));
                                let ci = output.content.len() - 1;
                                while block_indices.len() <= index {
                                    block_indices.push(0);
                                }
                                block_indices[index] = ci;
                                stream_clone.push(AssistantMessageEvent::ThinkingStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                });
                            }
                            "tool_use" => {
                                let id = content_block
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = content_block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = if is_oauth {
                                    from_claude_code_name(&name, tools_slice)
                                } else {
                                    name
                                };
                                output.content.push(ContentBlock::ToolCall(ToolCall {
                                    id,
                                    name,
                                    arguments: Value::Object(Default::default()),
                                    thought_signature: None,
                                }));
                                let ci = output.content.len() - 1;
                                while block_indices.len() <= index {
                                    block_indices.push(0);
                                }
                                block_indices[index] = ci;
                                partial_json_map.insert(ci, String::new());
                                stream_clone.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                });
                            }
                            _ => {}
                        }
                    }
                    "content_block_delta" => {
                        let index =
                            data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let ci = block_indices.get(index).copied().unwrap_or(0);
                        let delta = data.get("delta").unwrap_or(&Value::Null);
                        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

                        match delta_type {
                            "text_delta" => {
                                let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                if let Some(ContentBlock::Text(t)) = output.content.get_mut(ci) {
                                    t.text.push_str(text);
                                }
                                stream_clone.push(AssistantMessageEvent::TextDelta {
                                    content_index: ci,
                                    delta: text.to_string(),
                                    partial: output.clone(),
                                });
                            }
                            "thinking_delta" => {
                                let thinking =
                                    delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci)
                                {
                                    t.thinking.push_str(thinking);
                                }
                                stream_clone.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: ci,
                                    delta: thinking.to_string(),
                                    partial: output.clone(),
                                });
                            }
                            "input_json_delta" => {
                                let partial_json_text = delta
                                    .get("partial_json")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if let Some(pj) = partial_json_map.get_mut(&ci) {
                                    pj.push_str(partial_json_text);
                                    if let Some(ContentBlock::ToolCall(tc)) =
                                        output.content.get_mut(ci)
                                    {
                                        tc.arguments = parse_streaming_json(pj);
                                    }
                                }
                                stream_clone.push(AssistantMessageEvent::ToolCallDelta {
                                    content_index: ci,
                                    delta: partial_json_text.to_string(),
                                    partial: output.clone(),
                                });
                            }
                            "signature_delta" => {
                                let sig = delta
                                    .get("signature")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci)
                                {
                                    let current =
                                        t.thinking_signature.get_or_insert_with(String::new);
                                    current.push_str(sig);
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let index =
                            data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let ci = block_indices.get(index).copied().unwrap_or(0);

                        // Determine block type and extract needed data without holding borrow
                        enum BlockAction {
                            Text(String),
                            Thinking(String),
                            ToolCall,
                            None,
                        }
                        let action = match output.content.get(ci) {
                            Some(ContentBlock::Text(t)) => BlockAction::Text(t.text.clone()),
                            Some(ContentBlock::Thinking(t)) => {
                                BlockAction::Thinking(t.thinking.clone())
                            }
                            Some(ContentBlock::ToolCall(_)) => BlockAction::ToolCall,
                            _ => BlockAction::None,
                        };

                        match action {
                            BlockAction::Text(text) => {
                                stream_clone.push(AssistantMessageEvent::TextEnd {
                                    content_index: ci,
                                    content: text,
                                    partial: output.clone(),
                                });
                            }
                            BlockAction::Thinking(thinking) => {
                                stream_clone.push(AssistantMessageEvent::ThinkingEnd {
                                    content_index: ci,
                                    content: thinking,
                                    partial: output.clone(),
                                });
                            }
                            BlockAction::ToolCall => {
                                // Final parse of accumulated JSON
                                if let Some(pj) = partial_json_map.remove(&ci) {
                                    if let Some(ContentBlock::ToolCall(tc)) =
                                        output.content.get_mut(ci)
                                    {
                                        tc.arguments = parse_streaming_json(&pj);
                                    }
                                }
                                let final_tc = if let Some(ContentBlock::ToolCall(tc)) =
                                    output.content.get(ci)
                                {
                                    tc.clone()
                                } else {
                                    // Should not happen, but provide a fallback
                                    ToolCall {
                                        id: String::new(),
                                        name: String::new(),
                                        arguments: Value::Object(Default::default()),
                                        thought_signature: None,
                                    }
                                };
                                stream_clone.push(AssistantMessageEvent::ToolCallEnd {
                                    content_index: ci,
                                    tool_call: final_tc,
                                    partial: output.clone(),
                                });
                            }
                            BlockAction::None => {}
                        }
                    }
                    "message_delta" => {
                        if let Some(delta) = data.get("delta") {
                            if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
                            {
                                output.stop_reason = map_stop_reason(reason);
                            }
                        }
                        if let Some(usage) = data.get("usage") {
                            if let Some(v) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                output.usage.input = v;
                            }
                            if let Some(v) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                                output.usage.output = v;
                            }
                            if let Some(v) = usage
                                .get("cache_read_input_tokens")
                                .and_then(|v| v.as_u64())
                            {
                                output.usage.cache_read = v;
                            }
                            if let Some(v) = usage
                                .get("cache_creation_input_tokens")
                                .and_then(|v| v.as_u64())
                            {
                                output.usage.cache_write = v;
                            }
                            output.usage.total_tokens = output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write;
                            calculate_cost(&model, &mut output.usage);
                        }
                    }
                    "message_stop" => {
                        // Stream complete
                    }
                    "error" => {
                        let error_msg = data
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        output.stop_reason = StopReason::Error;
                        output.error_message = Some(error_msg);
                        stream_clone.push(AssistantMessageEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        });
                        return;
                    }
                    "ping" => {
                        // Heartbeat, ignore
                    }
                    _ => {}
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
pub fn stream_simple_anthropic(
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
        let anthropic_opts = AnthropicOptions {
            base,
            thinking_enabled: false,
            thinking_budget_tokens: None,
            effort: None,
            interleaved_thinking: true,
            tool_choice: None,
        };
        return stream_anthropic(model, context, &anthropic_opts, cancel);
    }

    let reasoning = options.reasoning.as_ref().unwrap();

    if supports_adaptive_thinking(&model.id) {
        let effort = map_thinking_level_to_effort(reasoning).to_string();
        let anthropic_opts = AnthropicOptions {
            base,
            thinking_enabled: true,
            thinking_budget_tokens: None,
            effort: Some(effort),
            interleaved_thinking: true,
            tool_choice: None,
        };
        return stream_anthropic(model, context, &anthropic_opts, cancel);
    }

    let (max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
        base.max_tokens.unwrap_or(0),
        model.max_tokens,
        reasoning,
        options.thinking_budgets.as_ref(),
    );

    let mut adjusted_base = base;
    adjusted_base.max_tokens = Some(max_tokens);

    let anthropic_opts = AnthropicOptions {
        base: adjusted_base,
        thinking_enabled: true,
        thinking_budget_tokens: Some(thinking_budget),
        effort: None,
        interleaved_thinking: true,
        tool_choice: None,
    };

    stream_anthropic(model, context, &anthropic_opts, cancel)
}

// ---------- ApiProvider implementation ----------

pub struct AnthropicProvider;

impl ApiProvider for AnthropicProvider {
    fn api(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let anthropic_opts = AnthropicOptions {
            base: options.clone(),
            thinking_enabled: false,
            thinking_budget_tokens: None,
            effort: None,
            interleaved_thinking: true,
            tool_choice: None,
        };
        stream_anthropic(model, context, &anthropic_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_anthropic(model, context, options, cancel)
    }
}
