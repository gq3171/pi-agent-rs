use std::collections::HashMap;

use futures::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use pi_agent_core::json_parse::parse_streaming_json;
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::{calculate_cost, supports_xhigh};
use crate::registry::ApiProvider;
use crate::simple_options::{build_base_options, clamp_reasoning};
use crate::sse::SseParser;

// ---------- Resolved compat (all fields required) ----------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ResolvedCompat {
    supports_store: bool,
    supports_developer_role: bool,
    supports_reasoning_effort: bool,
    supports_usage_in_streaming: bool,
    max_tokens_field: String, // "max_tokens" | "max_completion_tokens"
    requires_tool_result_name: bool,
    requires_assistant_after_tool_result: bool,
    requires_thinking_as_text: bool,
    requires_mistral_tool_ids: bool,
    thinking_format: String, // "openai" | "zai" | "qwen"
    open_router_routing: Value,
    vercel_gateway_routing: Value,
    supports_strict_mode: bool,
}

// ---------- OpenAICompletionsOptions ----------

#[derive(Debug, Clone)]
pub struct OpenAICompletionsOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<String>, // "minimal" | "low" | "medium" | "high" | "xhigh"
}

// ---------- Mistral tool ID normalization ----------

/// Normalize tool call ID for Mistral.
/// Mistral requires tool IDs to be exactly 9 alphanumeric characters (a-z, A-Z, 0-9).
fn normalize_mistral_tool_id(id: &str) -> String {
    let normalized: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    const PADDING: &str = "ABCDEFGHI";
    if normalized.len() < 9 {
        let pad_len = 9 - normalized.len();
        format!("{}{}", normalized, &PADDING[..pad_len])
    } else if normalized.len() > 9 {
        normalized[..9].to_string()
    } else {
        normalized
    }
}

// ---------- Normalize tool call ID ----------

fn normalize_tool_call_id(id: &str, model: &Model, compat: &ResolvedCompat) -> String {
    if compat.requires_mistral_tool_ids {
        return normalize_mistral_tool_id(id);
    }

    // Handle pipe-separated IDs from OpenAI Responses API
    // Format: {call_id}|{id} where {id} can be 400+ chars with special chars (+, /, =)
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or(id);
        let sanitized: String = call_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        return if sanitized.len() > 40 {
            sanitized[..40].to_string()
        } else {
            sanitized
        };
    }

    if model.provider == "openai" {
        return if id.len() > 40 { id[..40].to_string() } else { id.to_string() };
    }

    // Copilot Claude models route to Claude backend which requires Anthropic ID format
    if model.provider == "github-copilot" && model.id.to_lowercase().contains("claude") {
        let sanitized: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        return if sanitized.len() > 64 {
            sanitized[..64].to_string()
        } else {
            sanitized
        };
    }

    id.to_string()
}

/// Wrapper to match the `transform_messages` normalize signature.
fn normalize_tool_call_id_for_transform(
    id: &str,
    model: &Model,
    _assistant_msg: &AssistantMessage,
    compat: &ResolvedCompat,
) -> String {
    normalize_tool_call_id(id, model, compat)
}

// ---------- Check if conversation has tool history ----------

fn has_tool_history(messages: &[Message]) -> bool {
    for msg in messages {
        match msg {
            Message::ToolResult(_) => return true,
            Message::Assistant(a) => {
                if a.content.iter().any(|b| b.as_tool_call().is_some()) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

// ---------- Detect compat from provider/baseUrl ----------

fn detect_compat(model: &Model) -> ResolvedCompat {
    let provider = &model.provider;
    let base_url = &model.base_url;

    let is_zai = provider == "zai" || base_url.contains("api.z.ai");

    let is_non_standard = provider == "cerebras"
        || base_url.contains("cerebras.ai")
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || provider == "mistral"
        || base_url.contains("mistral.ai")
        || base_url.contains("chutes.ai")
        || base_url.contains("deepseek.com")
        || is_zai
        || provider == "opencode"
        || base_url.contains("opencode.ai");

    let use_max_tokens =
        provider == "mistral" || base_url.contains("mistral.ai") || base_url.contains("chutes.ai");

    let is_grok = provider == "xai" || base_url.contains("api.x.ai");

    let is_mistral = provider == "mistral" || base_url.contains("mistral.ai");

    ResolvedCompat {
        supports_store: !is_non_standard,
        supports_developer_role: !is_non_standard,
        supports_reasoning_effort: !is_grok && !is_zai,
        supports_usage_in_streaming: true,
        max_tokens_field: if use_max_tokens {
            "max_tokens".to_string()
        } else {
            "max_completion_tokens".to_string()
        },
        requires_tool_result_name: is_mistral,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: is_mistral,
        requires_mistral_tool_ids: is_mistral,
        thinking_format: if is_zai {
            "zai".to_string()
        } else {
            "openai".to_string()
        },
        open_router_routing: json!({}),
        vercel_gateway_routing: json!({}),
        supports_strict_mode: true,
    }
}

// ---------- Get resolved compat (merge model.compat with detected) ----------

fn get_compat(model: &Model) -> ResolvedCompat {
    let detected = detect_compat(model);

    let compat_val = match &model.compat {
        Some(v) => v,
        None => return detected,
    };

    // Try to deserialize the Value into OpenAICompletionsCompat
    let parsed: OpenAICompletionsCompat = match serde_json::from_value(compat_val.clone()) {
        Ok(c) => c,
        Err(_) => return detected,
    };

    ResolvedCompat {
        supports_store: parsed.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: parsed
            .supports_developer_role
            .unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: parsed
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: parsed
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        max_tokens_field: parsed
            .max_tokens_field
            .unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: parsed
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: parsed
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: parsed
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_mistral_tool_ids: parsed
            .requires_mistral_tool_ids
            .unwrap_or(detected.requires_mistral_tool_ids),
        thinking_format: parsed
            .thinking_format
            .unwrap_or(detected.thinking_format),
        open_router_routing: parsed
            .open_router_routing
            .map(|r| serde_json::to_value(r).unwrap_or(json!({})))
            .unwrap_or(json!({})),
        vercel_gateway_routing: parsed
            .vercel_gateway_routing
            .map(|r| serde_json::to_value(r).unwrap_or(json!({})))
            .unwrap_or(detected.vercel_gateway_routing),
        supports_strict_mode: parsed
            .supports_strict_mode
            .unwrap_or(detected.supports_strict_mode),
    }
}

// ---------- Map OpenAI stop reason ----------

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "function_call" | "tool_calls" => StopReason::ToolUse,
        "content_filter" => StopReason::Error,
        unknown => {
            tracing::warn!("Unknown OpenAI stop reason: {}", unknown);
            StopReason::Error
        }
    }
}

// ---------- Convert messages to OpenAI chat completion format ----------

fn convert_messages(
    model: &Model,
    context: &Context,
    compat: &ResolvedCompat,
) -> Vec<Value> {
    // Build a closure that captures compat for use inside transform_messages
    let compat_clone = compat.clone();
    let normalize_fn = move |id: &str, m: &Model, _a: &AssistantMessage| -> String {
        normalize_tool_call_id_for_transform(id, m, _a, &compat_clone)
    };

    let transformed = transform_messages(&context.messages, model, Some(&normalize_fn));

    let mut params: Vec<Value> = Vec::new();

    // System prompt
    if let Some(sp) = &context.system_prompt {
        let use_developer_role = model.reasoning && compat.supports_developer_role;
        let role = if use_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(json!({
            "role": role,
            "content": sanitize_surrogates(sp)
        }));
    }

    let mut last_role: Option<&str> = None;
    let mut i = 0;

    while i < transformed.len() {
        let msg = &transformed[i];

        // Some providers (e.g. Mistral/Devstral) don't allow user messages directly after tool results
        // Insert a synthetic assistant message to bridge the gap
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && msg.role() == "user"
        {
            params.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results."
            }));
        }

        match msg {
            Message::User(user_msg) => {
                match &user_msg.content {
                    UserContent::Text(text) => {
                        params.push(json!({
                            "role": "user",
                            "content": sanitize_surrogates(text)
                        }));
                    }
                    UserContent::Blocks(blocks) => {
                        let content: Vec<Value> = blocks
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(t) => Some(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&t.text)
                                })),
                                ContentBlock::Image(img) => {
                                    if model.input.contains(&"image".to_string()) {
                                        Some(json!({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": format!("data:{};base64,{}", img.mime_type, img.data)
                                            }
                                        }))
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            })
                            .collect();

                        // Filter out image_url blocks if model doesn't support images
                        let filtered: Vec<Value> = if !model.input.contains(&"image".to_string()) {
                            content
                                .into_iter()
                                .filter(|c| {
                                    c.get("type").and_then(|t| t.as_str()) != Some("image_url")
                                })
                                .collect()
                        } else {
                            content
                        };

                        if !filtered.is_empty() {
                            params.push(json!({
                                "role": "user",
                                "content": filtered
                            }));
                        }
                    }
                }
                last_role = Some("user");
            }
            Message::Assistant(assistant_msg) => {
                // Some providers (e.g. Mistral) don't accept null content, use empty string instead
                let mut assistant_obj = json!({
                    "role": "assistant"
                });

                if compat.requires_assistant_after_tool_result {
                    assistant_obj["content"] = json!("");
                } else {
                    assistant_obj["content"] = Value::Null;
                }

                // Collect text blocks
                let text_blocks: Vec<&TextContent> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .filter(|t| !t.text.trim().is_empty())
                    .collect();

                if !text_blocks.is_empty() {
                    if model.provider == "github-copilot" {
                        // GitHub Copilot requires assistant content as a string, not an array
                        let concatenated: String = text_blocks
                            .iter()
                            .map(|b| sanitize_surrogates(&b.text))
                            .collect::<Vec<_>>()
                            .join("");
                        assistant_obj["content"] = json!(concatenated);
                    } else {
                        let content_parts: Vec<Value> = text_blocks
                            .iter()
                            .map(|b| {
                                json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&b.text)
                                })
                            })
                            .collect();
                        assistant_obj["content"] = json!(content_parts);
                    }
                }

                // Handle thinking blocks
                let thinking_blocks: Vec<&ThinkingContent> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| b.as_thinking())
                    .filter(|t| !t.thinking.trim().is_empty())
                    .collect();

                if !thinking_blocks.is_empty() {
                    if compat.requires_thinking_as_text {
                        // Convert thinking blocks to plain text
                        let thinking_text: String = thinking_blocks
                            .iter()
                            .map(|b| b.thinking.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");

                        if let Some(arr) = assistant_obj["content"].as_array_mut() {
                            arr.insert(
                                0,
                                json!({
                                    "type": "text",
                                    "text": thinking_text
                                }),
                            );
                        } else {
                            assistant_obj["content"] = json!([{
                                "type": "text",
                                "text": thinking_text
                            }]);
                        }
                    } else {
                        // Use the signature from the first thinking block if available
                        // (for llama.cpp server + gpt-oss)
                        if let Some(sig) = thinking_blocks[0]
                            .thinking_signature
                            .as_ref()
                            .filter(|s| !s.is_empty())
                        {
                            let joined: String = thinking_blocks
                                .iter()
                                .map(|b| b.thinking.as_str())
                                .collect::<Vec<_>>()
                                .join("\n");
                            assistant_obj[sig] = json!(joined);
                        }
                    }
                }

                // Handle tool calls
                let tool_calls: Vec<&ToolCall> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| b.as_tool_call())
                    .collect();

                if !tool_calls.is_empty() {
                    let tc_arr: Vec<Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default()
                                }
                            })
                        })
                        .collect();
                    assistant_obj["tool_calls"] = json!(tc_arr);

                    // reasoning_details for encrypted reasoning
                    let reasoning_details: Vec<Value> = tool_calls
                        .iter()
                        .filter(|tc| tc.thought_signature.is_some())
                        .filter_map(|tc| {
                            tc.thought_signature
                                .as_ref()
                                .and_then(|s| serde_json::from_str(s).ok())
                        })
                        .collect();
                    if !reasoning_details.is_empty() {
                        assistant_obj["reasoning_details"] = json!(reasoning_details);
                    }
                }

                // Skip assistant messages that have no content and no tool calls
                let content = &assistant_obj["content"];
                let has_content = !content.is_null()
                    && match content {
                        Value::String(s) => !s.is_empty(),
                        Value::Array(a) => !a.is_empty(),
                        _ => false,
                    };
                let has_tool_calls = assistant_obj.get("tool_calls").is_some();

                if has_content || has_tool_calls {
                    params.push(assistant_obj);
                }

                last_role = Some("assistant");
            }
            Message::ToolResult(_) => {
                // Collect consecutive toolResult messages
                let mut image_blocks: Vec<Value> = Vec::new();
                let mut j = i;

                while j < transformed.len() {
                    if let Message::ToolResult(tool_msg) = &transformed[j] {
                        // Extract text content
                        let text_result: String = tool_msg
                            .content
                            .iter()
                            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let has_images = tool_msg.content.iter().any(|c| c.as_image().is_some());
                        let has_text = !text_result.is_empty();

                        let mut tool_result_msg = json!({
                            "role": "tool",
                            "content": sanitize_surrogates(if has_text { &text_result } else { "(see attached image)" }),
                            "tool_call_id": tool_msg.tool_call_id
                        });

                        if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                            tool_result_msg["name"] = json!(tool_msg.tool_name);
                        }

                        params.push(tool_result_msg);

                        if has_images && model.input.contains(&"image".to_string()) {
                            for block in &tool_msg.content {
                                if let ContentBlock::Image(img) = block {
                                    image_blocks.push(json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": format!("data:{};base64,{}", img.mime_type, img.data)
                                        }
                                    }));
                                }
                            }
                        }

                        j += 1;
                    } else {
                        break;
                    }
                }

                i = j - 1; // Will be incremented at end of loop

                if !image_blocks.is_empty() {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({
                            "role": "assistant",
                            "content": "I have processed the tool results."
                        }));
                    }

                    let mut user_content = vec![json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:"
                    })];
                    user_content.extend(image_blocks);

                    params.push(json!({
                        "role": "user",
                        "content": user_content
                    }));
                    last_role = Some("user");
                } else {
                    last_role = Some("toolResult");
                }

                i += 1;
                continue;
            }
        }

        i += 1;
    }

    params
}

// ---------- Convert tools ----------

fn convert_tools(tools: &[Tool], compat: &ResolvedCompat) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut func = json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            });
            if compat.supports_strict_mode {
                func["strict"] = json!(false);
            }
            json!({
                "type": "function",
                "function": func
            })
        })
        .collect()
}

// ---------- OpenRouter Anthropic cache control ----------

fn maybe_add_openrouter_anthropic_cache_control(model: &Model, messages: &mut [Value]) {
    if model.provider != "openrouter" || !model.id.starts_with("anthropic/") {
        return;
    }

    // Add cache_control on the last user/assistant message's last text part
    for msg in messages.iter_mut().rev() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "user" && role != "assistant" {
            continue;
        }

        let content = match msg.get_mut("content") {
            Some(c) => c,
            None => continue,
        };

        if content.is_string() {
            let text = content.as_str().unwrap_or("").to_string();
            *content = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
            return;
        }

        if let Some(arr) = content.as_array_mut() {
            for part in arr.iter_mut().rev() {
                if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                    part["cache_control"] = json!({"type": "ephemeral"});
                    return;
                }
            }
        }
    }
}

// ---------- Build request params ----------

fn build_params(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
) -> Value {
    let compat = get_compat(model);
    let mut messages = convert_messages(model, context, &compat);
    maybe_add_openrouter_anthropic_cache_control(model, &mut messages);

    let mut params = json!({
        "model": model.id,
        "messages": messages,
        "stream": true
    });

    if compat.supports_usage_in_streaming {
        params["stream_options"] = json!({"include_usage": true});
    }

    if compat.supports_store {
        params["store"] = json!(false);
    }

    if let Some(max_tokens) = options.base.max_tokens {
        if compat.max_tokens_field == "max_tokens" {
            params["max_tokens"] = json!(max_tokens);
        } else {
            params["max_completion_tokens"] = json!(max_tokens);
        }
    }

    if let Some(temp) = options.base.temperature {
        params["temperature"] = json!(temp);
    }

    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            params["tools"] = json!(convert_tools(tools, &compat));
        }
    } else if has_tool_history(&context.messages) {
        // Anthropic (via LiteLLM/proxy) requires tools param when conversation has tool_calls/tool_results
        params["tools"] = json!([]);
    }

    if let Some(tc) = &options.tool_choice {
        params["tool_choice"] = tc.clone();
    }

    // Thinking format handling
    if compat.thinking_format == "zai" && model.reasoning {
        // Z.ai uses binary thinking: { type: "enabled" | "disabled" }
        let thinking_type = if options.reasoning_effort.is_some() {
            "enabled"
        } else {
            "disabled"
        };
        params["thinking"] = json!({"type": thinking_type});
    } else if compat.thinking_format == "qwen" && model.reasoning {
        // Qwen uses enable_thinking: boolean
        params["enable_thinking"] = json!(options.reasoning_effort.is_some());
    } else if options.reasoning_effort.is_some() && model.reasoning && compat.supports_reasoning_effort
    {
        // OpenAI-style reasoning_effort
        params["reasoning_effort"] = json!(options.reasoning_effort.as_deref().unwrap_or(""));
    }

    // OpenRouter provider routing preferences
    if model.base_url.contains("openrouter.ai") {
        if let Some(compat_val) = &model.compat {
            if let Some(routing) = compat_val.get("openRouterRouting") {
                if routing.is_object() && !routing.as_object().unwrap().is_empty() {
                    params["provider"] = routing.clone();
                }
            }
        }
    }

    // Vercel AI Gateway provider routing preferences
    if model.base_url.contains("ai-gateway.vercel.sh") {
        if let Some(compat_val) = &model.compat {
            if let Some(routing) = compat_val.get("vercelGatewayRouting") {
                let only = routing.get("only");
                let order = routing.get("order");
                if only.is_some() || order.is_some() {
                    let mut gateway_options = json!({});
                    if let Some(o) = only {
                        gateway_options["only"] = o.clone();
                    }
                    if let Some(o) = order {
                        gateway_options["order"] = o.clone();
                    }
                    params["providerOptions"] = json!({"gateway": gateway_options});
                }
            }
        }
    }

    params
}

// ---------- Build HTTP headers ----------

fn build_headers(
    model: &Model,
    context: &Context,
    api_key: &str,
    extra_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("authorization".to_string(), format!("Bearer {api_key}"));

    // Copy model headers (skip sensitive auth headers)
    if let Some(model_headers) = &model.headers {
        crate::header_utils::merge_headers_safe(&mut headers, model_headers);
    }

    // GitHub Copilot specific headers
    if model.provider == "github-copilot" {
        let messages = &context.messages;
        let last_msg = messages.last();
        let is_agent_call = last_msg
            .map(|m| m.role() != "user")
            .unwrap_or(false);
        headers.insert(
            "X-Initiator".to_string(),
            if is_agent_call { "agent" } else { "user" }.to_string(),
        );
        headers.insert("Openai-Intent".to_string(), "conversation-edits".to_string());

        // Copilot requires this header when sending images
        let has_images = messages.iter().any(|msg| match msg {
            Message::User(u) => {
                if let UserContent::Blocks(blocks) = &u.content {
                    blocks.iter().any(|c| c.as_image().is_some())
                } else {
                    false
                }
            }
            Message::ToolResult(tr) => tr.content.iter().any(|c| c.as_image().is_some()),
            _ => false,
        });
        if has_images {
            headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
        }
    }

    // Merge extra headers (skip sensitive auth headers)
    if let Some(extra) = extra_headers {
        crate::header_utils::merge_headers_safe(&mut headers, extra);
    }

    headers
}

// ---------- Stream OpenAI Completions ----------

/// Stream from OpenAI Chat Completions API using raw HTTP + SSE parsing.
pub fn stream_openai_completions(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
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

        if api_key.is_empty() {
            output.stop_reason = StopReason::Error;
            output.error_message = Some(format!(
                "OpenAI API key is required. Set the appropriate environment variable or pass it as an argument. Provider: {}",
                model.provider
            ));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        let headers = build_headers(
            &model,
            &context,
            &api_key,
            options.base.headers.as_ref(),
        );

        let params = build_params(&model, &context, &options);

        // Determine the URL: OpenAI completions endpoint
        let url = if model.base_url.ends_with("/chat/completions") {
            model.base_url.clone()
        } else {
            let base = model.base_url.trim_end_matches('/');
            if base.ends_with("/v1") {
                format!("{base}/chat/completions")
            } else {
                format!("{base}/v1/chat/completions")
            }
        };

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

        // State for streaming: tracks current block being built
        enum CurrentBlock {
            Text(usize),     // content_index
            Thinking(usize), // content_index
            ToolCall(usize, String), // content_index, partial_args
        }

        let mut current_block: Option<CurrentBlock> = None;
        let mut sse_parser = SseParser::new();

        // Helper: finalize the current block
        macro_rules! finish_current_block {
            () => {
                if let Some(ref block) = current_block {
                    match block {
                        CurrentBlock::Text(ci) => {
                            let content = output
                                .content
                                .get(*ci)
                                .and_then(|b| b.as_text())
                                .map(|t| t.text.clone())
                                .unwrap_or_default();
                            stream_clone.push(AssistantMessageEvent::TextEnd {
                                content_index: *ci,
                                content,
                                partial: output.clone(),
                            });
                        }
                        CurrentBlock::Thinking(ci) => {
                            let content = output
                                .content
                                .get(*ci)
                                .and_then(|b| b.as_thinking())
                                .map(|t| t.thinking.clone())
                                .unwrap_or_default();
                            stream_clone.push(AssistantMessageEvent::ThinkingEnd {
                                content_index: *ci,
                                content,
                                partial: output.clone(),
                            });
                        }
                        CurrentBlock::ToolCall(ci, partial_args) => {
                            // Final parse of accumulated JSON args
                            let parsed_args = parse_streaming_json(partial_args);
                            if let Some(ContentBlock::ToolCall(tc)) = output.content.get_mut(*ci) {
                                tc.arguments = parsed_args;
                            }
                            let final_tc = output
                                .content
                                .get(*ci)
                                .and_then(|b| b.as_tool_call())
                                .cloned()
                                .unwrap_or(ToolCall {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: json!({}),
                                    thought_signature: None,
                                });
                            stream_clone.push(AssistantMessageEvent::ToolCallEnd {
                                content_index: *ci,
                                tool_call: final_tc,
                                partial: output.clone(),
                            });
                        }
                    }
                    #[allow(unused_assignments)]
                    { current_block = None; }
                }
            };
        }

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
                // OpenAI uses "message" as default event type (no explicit event: line)
                // The SSE parser normalizes to "message" when no event type specified
                let data_str = sse_event.data.trim();

                // Handle [DONE] signal
                if data_str == "[DONE]" {
                    continue;
                }

                let data: Value = match serde_json::from_str(data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Handle error responses in the SSE stream
                if let Some(err) = data.get("error") {
                    let error_msg = err
                        .get("message")
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

                // Handle usage data (sent in final chunk with stream_options.include_usage)
                if let Some(usage) = data.get("usage") {
                    let cached_tokens = usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let reasoning_tokens = usage
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    let completion_tokens =
                        usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);

                    // OpenAI includes cached tokens in prompt_tokens, so subtract to get non-cached input
                    let input = prompt_tokens.saturating_sub(cached_tokens);
                    let output_tokens = completion_tokens + reasoning_tokens;

                    output.usage = Usage {
                        input,
                        output: output_tokens,
                        cache_read: cached_tokens,
                        cache_write: 0,
                        total_tokens: input + output_tokens + cached_tokens,
                        cost: UsageCost::default(),
                    };
                    calculate_cost(&model, &mut output.usage);
                }

                // Process choices
                let choices = match data.get("choices").and_then(|c| c.as_array()) {
                    Some(c) => c,
                    None => continue,
                };
                let choice = match choices.first() {
                    Some(c) => c,
                    None => continue,
                };

                // Handle finish_reason
                if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    output.stop_reason = map_stop_reason(reason);
                }

                let delta = match choice.get("delta") {
                    Some(d) => d,
                    None => continue,
                };

                // --- Handle text content ---
                if let Some(content_str) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content_str.is_empty() {
                        // If current block is not text, finish it and start a new text block
                        let is_text = matches!(&current_block, Some(CurrentBlock::Text(_)));
                        if !is_text {
                            finish_current_block!();
                            output.content.push(ContentBlock::Text(TextContent {
                                text: String::new(),
                                text_signature: None,
                            }));
                            let ci = output.content.len() - 1;
                            current_block = Some(CurrentBlock::Text(ci));
                            stream_clone.push(AssistantMessageEvent::TextStart {
                                content_index: ci,
                                partial: output.clone(),
                            });
                        }

                        if let Some(CurrentBlock::Text(ci)) = &current_block {
                            let ci = *ci;
                            if let Some(ContentBlock::Text(t)) = output.content.get_mut(ci) {
                                t.text.push_str(content_str);
                            }
                            stream_clone.push(AssistantMessageEvent::TextDelta {
                                content_index: ci,
                                delta: content_str.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }

                // --- Handle reasoning/thinking content ---
                // Some endpoints return reasoning in reasoning_content (llama.cpp),
                // or reasoning (other openai compatible endpoints), or reasoning_text
                let reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"];
                let mut found_reasoning_field: Option<&str> = None;
                for field in &reasoning_fields {
                    if let Some(val) = delta.get(*field).and_then(|v| v.as_str()) {
                        if !val.is_empty() {
                            found_reasoning_field = Some(field);
                            break;
                        }
                    }
                }

                if let Some(field_name) = found_reasoning_field {
                    let reasoning_delta = delta
                        .get(field_name)
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let is_thinking = matches!(&current_block, Some(CurrentBlock::Thinking(_)));
                    if !is_thinking {
                        finish_current_block!();
                        output.content.push(ContentBlock::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: Some(field_name.to_string()),
                        }));
                        let ci = output.content.len() - 1;
                        current_block = Some(CurrentBlock::Thinking(ci));
                        stream_clone.push(AssistantMessageEvent::ThinkingStart {
                            content_index: ci,
                            partial: output.clone(),
                        });
                    }

                    if let Some(CurrentBlock::Thinking(ci)) = &current_block {
                        let ci = *ci;
                        if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci) {
                            t.thinking.push_str(reasoning_delta);
                        }
                        stream_clone.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: ci,
                            delta: reasoning_delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }

                // --- Handle tool calls ---
                if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tool_call in tool_calls {
                        let tc_id = tool_call.get("id").and_then(|v| v.as_str());
                        let tc_fn_name = tool_call
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str());
                        let tc_fn_args = tool_call
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str());

                        // Check if we need a new tool call block:
                        // - no current block, or current block is not a tool call
                        // - or a new tool call id is present and differs from current
                        let needs_new_block = match &current_block {
                            Some(CurrentBlock::ToolCall(ci, _)) => {
                                if let Some(new_id) = tc_id {
                                    // Check if the current tool call has a different id
                                    let current_id = output
                                        .content
                                        .get(*ci)
                                        .and_then(|b| b.as_tool_call())
                                        .map(|tc| tc.id.as_str())
                                        .unwrap_or("");
                                    current_id != new_id
                                } else {
                                    false
                                }
                            }
                            _ => true,
                        };

                        if needs_new_block {
                            finish_current_block!();
                            output.content.push(ContentBlock::ToolCall(ToolCall {
                                id: tc_id.unwrap_or("").to_string(),
                                name: tc_fn_name.unwrap_or("").to_string(),
                                arguments: json!({}),
                                thought_signature: None,
                            }));
                            let ci = output.content.len() - 1;
                            current_block =
                                Some(CurrentBlock::ToolCall(ci, String::new()));
                            stream_clone.push(AssistantMessageEvent::ToolCallStart {
                                content_index: ci,
                                partial: output.clone(),
                            });
                        }

                        if let Some(CurrentBlock::ToolCall(ci, partial_args)) =
                            &mut current_block
                        {
                            let ci = *ci;
                            // Update id if provided
                            if let Some(new_id) = tc_id {
                                if let Some(ContentBlock::ToolCall(tc)) =
                                    output.content.get_mut(ci)
                                {
                                    tc.id = new_id.to_string();
                                }
                            }
                            // Update name if provided
                            if let Some(new_name) = tc_fn_name {
                                if let Some(ContentBlock::ToolCall(tc)) =
                                    output.content.get_mut(ci)
                                {
                                    tc.name = new_name.to_string();
                                }
                            }
                            // Accumulate arguments delta
                            let mut args_delta = String::new();
                            if let Some(args) = tc_fn_args {
                                args_delta = args.to_string();
                                partial_args.push_str(args);
                                if let Some(ContentBlock::ToolCall(tc)) =
                                    output.content.get_mut(ci)
                                {
                                    tc.arguments = parse_streaming_json(partial_args);
                                }
                            }
                            stream_clone.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: ci,
                                delta: args_delta,
                                partial: output.clone(),
                            });
                        }
                    }
                }

                // --- Handle reasoning_details (encrypted reasoning for tool calls) ---
                if let Some(reasoning_details) =
                    delta.get("reasoning_details").and_then(|v| v.as_array())
                {
                    for detail in reasoning_details {
                        let detail_type = detail.get("type").and_then(|v| v.as_str());
                        let detail_id = detail.get("id").and_then(|v| v.as_str());
                        let detail_data = detail.get("data");

                        if detail_type == Some("reasoning.encrypted")
                            && detail_id.is_some()
                            && detail_data.is_some()
                        {
                            let target_id = detail_id.unwrap();
                            // Find matching tool call in output content
                            for block in &mut output.content {
                                if let ContentBlock::ToolCall(tc) = block {
                                    if tc.id == target_id {
                                        tc.thought_signature =
                                            Some(serde_json::to_string(detail).unwrap_or_default());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Finalize any remaining open block
        finish_current_block!();

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

// ---------- Stream Simple OpenAI Completions ----------

/// Simplified streaming: handles reasoning level and API key resolution.
pub fn stream_simple_openai_completions(
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

    let reasoning_effort = options.reasoning.as_ref().map(|level| {
        if supports_xhigh(model) {
            level.to_string()
        } else {
            clamp_reasoning(level).to_string()
        }
    });

    let openai_opts = OpenAICompletionsOptions {
        base,
        reasoning_effort,
        tool_choice: None,
    };

    stream_openai_completions(model, context, &openai_opts, cancel)
}

// ---------- ApiProvider implementation ----------

pub struct OpenAIProvider;

impl ApiProvider for OpenAIProvider {
    fn api(&self) -> &str {
        "openai-completions"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let openai_opts = OpenAICompletionsOptions {
            base: options.clone(),
            tool_choice: None,
            reasoning_effort: None,
        };
        stream_openai_completions(model, context, &openai_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_openai_completions(model, context, options, cancel)
    }
}
