use std::collections::HashSet;

use regex::Regex;
use serde_json::{json, Value};

use pi_agent_core::json_parse::parse_streaming_json;
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use crate::models::calculate_cost;

// =============================================================================
// Utilities
// =============================================================================

/// Fast deterministic hash to shorten long strings.
/// Port of the TypeScript cyrb53 variant used in openai-responses-shared.ts.
pub fn short_hash(s: &str) -> String {
    let mut h1: u32 = 0xdeadbeef;
    let mut h2: u32 = 0x41c6ce57;
    for ch in s.encode_utf16() {
        let c = ch as u32;
        h1 = (h1 ^ c).wrapping_mul(2654435761);
        h2 = (h2 ^ c).wrapping_mul(1597334677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2246822507) ^ (h2 ^ (h2 >> 13)).wrapping_mul(3266489909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2246822507) ^ (h1 ^ (h1 >> 13)).wrapping_mul(3266489909);

    // JS `>>> 0` is a no-op for u32 (already unsigned), then `.toString(36)`
    format!("{}{}", to_base36(h2), to_base36(h1))
}

fn to_base36(n: u32) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut val = n;
    let mut digits = Vec::new();
    while val > 0 {
        let d = (val % 36) as u8;
        digits.push(if d < 10 { b'0' + d } else { b'a' + d - 10 });
        val /= 36;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

// =============================================================================
// Options types
// =============================================================================

/// Options for controlling service-tier pricing adjustments.
pub struct OpenAIResponsesStreamOptions {
    pub service_tier: Option<String>,
    pub apply_service_tier_pricing: Option<Box<dyn Fn(&mut Usage, Option<&str>) + Send>>,
}

/// Options for convert_responses_messages.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesMessagesOptions {
    pub include_system_prompt: Option<bool>,
}

/// Options for convert_responses_tools.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesToolsOptions {
    pub strict: Option<bool>,
}

// =============================================================================
// Message conversion
// =============================================================================

/// Normalize a tool-call ID that uses the `{call_id}|{item_id}` format used by
/// the Responses API.  This mirrors the TS `normalizeToolCallId` closure inside
/// `convertResponsesMessages`.
fn normalize_tool_call_id(
    id: &str,
    model: &Model,
    allowed_tool_call_providers: &HashSet<&str>,
) -> String {
    if !allowed_tool_call_providers.contains(model.provider.as_str()) {
        return id.to_string();
    }
    if !id.contains('|') {
        return id.to_string();
    }
    let parts: Vec<&str> = id.splitn(2, '|').collect();
    let call_id = parts[0];
    let item_id = parts.get(1).unwrap_or(&"");

    let re = Regex::new(r"[^a-zA-Z0-9_\-]").unwrap();
    let sanitized_call_id = re.replace_all(call_id, "_").to_string();
    let mut sanitized_item_id = re.replace_all(item_id, "_").to_string();

    // OpenAI Responses API requires item id to start with "fc"
    if !sanitized_item_id.starts_with("fc") {
        sanitized_item_id = format!("fc_{sanitized_item_id}");
    }

    // Truncate to 64 chars
    let mut normalized_call_id = if sanitized_call_id.len() > 64 {
        sanitized_call_id[..64].to_string()
    } else {
        sanitized_call_id
    };
    let mut normalized_item_id = if sanitized_item_id.len() > 64 {
        sanitized_item_id[..64].to_string()
    } else {
        sanitized_item_id
    };

    // Strip trailing underscores (OpenAI Codex rejects them)
    normalized_call_id = normalized_call_id.trim_end_matches('_').to_string();
    normalized_item_id = normalized_item_id.trim_end_matches('_').to_string();

    format!("{normalized_call_id}|{normalized_item_id}")
}

/// Convert internal messages to OpenAI Responses API input format.
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &HashSet<&str>,
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();

    // Build the normalize closure for transform_messages
    let providers = allowed_tool_call_providers.clone();
    let normalize_fn = move |id: &str, m: &Model, _a: &AssistantMessage| -> String {
        normalize_tool_call_id(id, m, &providers)
    };

    let transformed = transform_messages(&context.messages, model, Some(&normalize_fn));

    let include_system_prompt = options
        .and_then(|o| o.include_system_prompt)
        .unwrap_or(true);

    if include_system_prompt {
        if let Some(sp) = &context.system_prompt {
            let role = if model.reasoning { "developer" } else { "system" };
            messages.push(json!({
                "role": role,
                "content": sanitize_surrogates(sp)
            }));
        }
    }

    let mut msg_index: usize = 0;
    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                match &user_msg.content {
                    UserContent::Text(text) => {
                        messages.push(json!({
                            "role": "user",
                            "content": [{"type": "input_text", "text": sanitize_surrogates(text)}]
                        }));
                    }
                    UserContent::Blocks(blocks) => {
                        let content: Vec<Value> = blocks
                            .iter()
                            .map(|item| match item {
                                ContentBlock::Text(t) => {
                                    json!({"type": "input_text", "text": sanitize_surrogates(&t.text)})
                                }
                                ContentBlock::Image(img) => {
                                    json!({
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": format!("data:{};base64,{}", img.mime_type, img.data)
                                    })
                                }
                                _ => json!(null),
                            })
                            .filter(|v| !v.is_null())
                            .collect();

                        let filtered: Vec<Value> = if !model.input.contains(&"image".to_string()) {
                            content
                                .into_iter()
                                .filter(|c| {
                                    c.get("type").and_then(|t| t.as_str()) != Some("input_image")
                                })
                                .collect()
                        } else {
                            content
                        };

                        if filtered.is_empty() {
                            msg_index += 1;
                            continue;
                        }

                        messages.push(json!({
                            "role": "user",
                            "content": filtered
                        }));
                    }
                }
            }
            Message::Assistant(assistant_msg) => {
                let mut output: Vec<Value> = Vec::new();
                let is_different_model = assistant_msg.model != model.id
                    && assistant_msg.provider == model.provider
                    && assistant_msg.api == model.api;

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Thinking(t) => {
                            if let Some(sig) = &t.thinking_signature {
                                if let Ok(reasoning_item) = serde_json::from_str::<Value>(sig) {
                                    output.push(reasoning_item);
                                }
                            }
                        }
                        ContentBlock::Text(t) => {
                            let mut msg_id = t.text_signature.clone();
                            if msg_id.is_none() {
                                msg_id = Some(format!("msg_{msg_index}"));
                            } else if msg_id.as_ref().is_some_and(|id| id.len() > 64) {
                                msg_id = Some(format!("msg_{}", short_hash(msg_id.as_ref().unwrap())));
                            }
                            output.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": sanitize_surrogates(&t.text), "annotations": []}],
                                "status": "completed",
                                "id": msg_id.unwrap_or_default()
                            }));
                        }
                        ContentBlock::ToolCall(tc) => {
                            let parts: Vec<&str> = tc.id.splitn(2, '|').collect();
                            let call_id = parts[0].to_string();
                            let item_id_raw = parts.get(1).map(|s| s.to_string());

                            // For different-model messages, set id to null to avoid pairing validation.
                            let item_id = if is_different_model
                                && item_id_raw
                                    .as_ref()
                                    .is_some_and(|id| id.starts_with("fc_"))
                            {
                                Value::Null
                            } else {
                                item_id_raw
                                    .map(|s| Value::String(s))
                                    .unwrap_or(Value::Null)
                            };

                            output.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default()
                            }));
                        }
                        _ => {}
                    }
                }

                if output.is_empty() {
                    msg_index += 1;
                    continue;
                }
                messages.extend(output);
            }
            Message::ToolResult(tr) => {
                // Extract text and image content
                let text_result: String = tr
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = tr.content.iter().any(|c| c.as_image().is_some());
                let has_text = !text_result.is_empty();

                let call_id = tr.tool_call_id.split('|').next().unwrap_or(&tr.tool_call_id);
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": sanitize_surrogates(if has_text { &text_result } else { "(see attached image)" })
                }));

                // If there are images and model supports them, send a follow-up user message
                if has_images && model.input.contains(&"image".to_string()) {
                    let mut content_parts: Vec<Value> = vec![json!({
                        "type": "input_text",
                        "text": "Attached image(s) from tool result:"
                    })];
                    for block in &tr.content {
                        if let ContentBlock::Image(img) = block {
                            content_parts.push(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", img.mime_type, img.data)
                            }));
                        }
                    }
                    messages.push(json!({
                        "role": "user",
                        "content": content_parts
                    }));
                }
            }
        }
        msg_index += 1;
    }

    messages
}

// =============================================================================
// Tool conversion
// =============================================================================

/// Convert internal Tool definitions to OpenAI Responses API tool format.
pub fn convert_responses_tools(
    tools: &[Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Vec<Value> {
    let strict = options.and_then(|o| o.strict);
    tools
        .iter()
        .map(|tool| {
            let mut t = json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            });
            // Only add strict if explicitly set (not None)
            match strict {
                Some(val) => { t["strict"] = json!(val); }
                None => { t["strict"] = json!(false); }
            }
            t
        })
        .collect()
}

// =============================================================================
// Stream processing
// =============================================================================

/// State tracking the current item being streamed from the Responses API.
pub struct CurrentItem {
    pub item_type: String,          // "reasoning" | "message" | "function_call"
    pub item: Value,                // the full item JSON
}

/// State tracking the current content block being built.
pub enum CurrentBlock {
    Thinking {
        thinking: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
        partial_json: String,
    },
}

/// Process a stream of OpenAI Responses API events, updating the output message
/// and pushing events to the event stream.
///
/// This is the core shared processing logic used by all three Responses providers.
pub fn process_responses_events(
    events: &[Value],
    output: &mut AssistantMessage,
    stream: &pi_agent_core::event_stream::AssistantMessageEventStream,
    model: &Model,
    current_item: &mut Option<CurrentItem>,
    current_block: &mut Option<CurrentBlock>,
    options: Option<&OpenAIResponsesStreamOptions>,
) -> Result<(), String> {
    for event in events {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "response.output_item.added" => {
                let item = event.get("item").cloned().unwrap_or(json!(null));
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let block_index = output.content.len();

                match item_type {
                    "reasoning" => {
                        *current_item = Some(CurrentItem {
                            item_type: "reasoning".to_string(),
                            item: item.clone(),
                        });
                        *current_block = Some(CurrentBlock::Thinking {
                            thinking: String::new(),
                        });
                        output.content.push(ContentBlock::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                        }));
                        stream.push(AssistantMessageEvent::ThinkingStart {
                            content_index: block_index,
                            partial: output.clone(),
                        });
                    }
                    "message" => {
                        *current_item = Some(CurrentItem {
                            item_type: "message".to_string(),
                            item: item.clone(),
                        });
                        *current_block = Some(CurrentBlock::Text {
                            text: String::new(),
                        });
                        output.content.push(ContentBlock::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }));
                        stream.push(AssistantMessageEvent::TextStart {
                            content_index: block_index,
                            partial: output.clone(),
                        });
                    }
                    "function_call" => {
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let args_str = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        *current_item = Some(CurrentItem {
                            item_type: "function_call".to_string(),
                            item: item.clone(),
                        });
                        *current_block = Some(CurrentBlock::ToolCall {
                            id: format!("{call_id}|{item_id}"),
                            name: name.clone(),
                            arguments: json!({}),
                            partial_json: args_str,
                        });
                        output.content.push(ContentBlock::ToolCall(ToolCall {
                            id: format!("{call_id}|{item_id}"),
                            name,
                            arguments: json!({}),
                            thought_signature: None,
                        }));
                        stream.push(AssistantMessageEvent::ToolCallStart {
                            content_index: block_index,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "response.reasoning_summary_part.added" => {
                if let Some(ci) = current_item.as_mut() {
                    if ci.item_type == "reasoning" {
                        let summary = ci.item.get("summary").cloned();
                        let mut summary_arr = match summary {
                            Some(Value::Array(arr)) => arr,
                            _ => Vec::new(),
                        };
                        if let Some(part) = event.get("part") {
                            summary_arr.push(part.clone());
                        }
                        ci.item["summary"] = json!(summary_arr);
                    }
                }
            }
            "response.reasoning_summary_text.delta" => {
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "reasoning" {
                        if let CurrentBlock::Thinking { thinking } = cb {
                            // Update the summary array's last part
                            if let Some(summary_arr) = ci.item.get_mut("summary").and_then(|v| v.as_array_mut()) {
                                if let Some(last_part) = summary_arr.last_mut() {
                                    if let Some(text) = last_part.get_mut("text") {
                                        let current = text.as_str().unwrap_or("");
                                        *text = json!(format!("{current}{delta}"));
                                    }
                                }
                            }

                            thinking.push_str(delta);
                            let block_index = output.content.len().saturating_sub(1);
                            if let Some(ContentBlock::Thinking(t)) = output.content.last_mut() {
                                t.thinking.push_str(delta);
                            }
                            stream.push(AssistantMessageEvent::ThinkingDelta {
                                content_index: block_index,
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "reasoning" {
                        if let CurrentBlock::Thinking { thinking } = cb {
                            // Update the summary array's last part
                            if let Some(summary_arr) = ci.item.get_mut("summary").and_then(|v| v.as_array_mut()) {
                                if let Some(last_part) = summary_arr.last_mut() {
                                    if let Some(text) = last_part.get_mut("text") {
                                        let current = text.as_str().unwrap_or("");
                                        *text = json!(format!("{current}\n\n"));
                                    }
                                }
                            }

                            thinking.push_str("\n\n");
                            let block_index = output.content.len().saturating_sub(1);
                            if let Some(ContentBlock::Thinking(t)) = output.content.last_mut() {
                                t.thinking.push_str("\n\n");
                            }
                            stream.push(AssistantMessageEvent::ThinkingDelta {
                                content_index: block_index,
                                delta: "\n\n".to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
            "response.content_part.added" => {
                if let Some(ci) = current_item.as_mut() {
                    if ci.item_type == "message" {
                        let content = ci.item.get("content").cloned();
                        let mut content_arr = match content {
                            Some(Value::Array(arr)) => arr,
                            _ => Vec::new(),
                        };
                        if let Some(part) = event.get("part") {
                            let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if part_type == "output_text" || part_type == "refusal" {
                                content_arr.push(part.clone());
                            }
                        }
                        ci.item["content"] = json!(content_arr);
                    }
                }
            }
            "response.output_text.delta" => {
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "message" {
                        if let CurrentBlock::Text { text } = cb {
                            let content_arr = ci.item.get("content").and_then(|v| v.as_array());
                            if content_arr.map(|a| a.is_empty()).unwrap_or(true) {
                                continue;
                            }
                            let last_type = content_arr
                                .and_then(|a| a.last())
                                .and_then(|p| p.get("type"))
                                .and_then(|t| t.as_str())
                                .unwrap_or("");
                            if last_type == "output_text" {
                                // Update the content array's last part
                                if let Some(arr) = ci.item.get_mut("content").and_then(|v| v.as_array_mut()) {
                                    if let Some(last) = arr.last_mut() {
                                        if let Some(t) = last.get_mut("text") {
                                            let current = t.as_str().unwrap_or("");
                                            *t = json!(format!("{current}{delta}"));
                                        }
                                    }
                                }
                                text.push_str(delta);
                                let block_index = output.content.len().saturating_sub(1);
                                if let Some(ContentBlock::Text(t)) = output.content.last_mut() {
                                    t.text.push_str(delta);
                                }
                                stream.push(AssistantMessageEvent::TextDelta {
                                    content_index: block_index,
                                    delta: delta.to_string(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }
            }
            "response.refusal.delta" => {
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "message" {
                        if let CurrentBlock::Text { text } = cb {
                            let content_arr = ci.item.get("content").and_then(|v| v.as_array());
                            if content_arr.map(|a| a.is_empty()).unwrap_or(true) {
                                continue;
                            }
                            let last_type = content_arr
                                .and_then(|a| a.last())
                                .and_then(|p| p.get("type"))
                                .and_then(|t| t.as_str())
                                .unwrap_or("");
                            if last_type == "refusal" {
                                // Update the content array's last part
                                if let Some(arr) = ci.item.get_mut("content").and_then(|v| v.as_array_mut()) {
                                    if let Some(last) = arr.last_mut() {
                                        if let Some(r) = last.get_mut("refusal") {
                                            let current = r.as_str().unwrap_or("");
                                            *r = json!(format!("{current}{delta}"));
                                        }
                                    }
                                }
                                text.push_str(delta);
                                let block_index = output.content.len().saturating_sub(1);
                                if let Some(ContentBlock::Text(t)) = output.content.last_mut() {
                                    t.text.push_str(delta);
                                }
                                stream.push(AssistantMessageEvent::TextDelta {
                                    content_index: block_index,
                                    delta: delta.to_string(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "function_call" {
                        if let CurrentBlock::ToolCall { partial_json, arguments, .. } = cb {
                            partial_json.push_str(delta);
                            *arguments = parse_streaming_json(partial_json);
                            let block_index = output.content.len().saturating_sub(1);
                            if let Some(ContentBlock::ToolCall(tc)) = output.content.last_mut() {
                                tc.arguments = arguments.clone();
                            }
                            stream.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: block_index,
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let args_str = event.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(ci), Some(cb)) = (current_item.as_mut(), current_block.as_mut()) {
                    if ci.item_type == "function_call" {
                        if let CurrentBlock::ToolCall { partial_json, arguments, .. } = cb {
                            *partial_json = args_str.to_string();
                            *arguments = parse_streaming_json(partial_json);
                            if let Some(ContentBlock::ToolCall(tc)) = output.content.last_mut() {
                                tc.arguments = arguments.clone();
                            }
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let item = event.get("item").cloned().unwrap_or(json!(null));
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let block_index = output.content.len().saturating_sub(1);

                match item_type {
                    "reasoning" => {
                        if let Some(cb) = current_block.as_mut() {
                            if let CurrentBlock::Thinking { .. } = cb {
                                // Update thinking from full summary
                                let summary_text = item
                                    .get("summary")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|s| s.get("text").and_then(|t| t.as_str()))
                                            .collect::<Vec<_>>()
                                            .join("\n\n")
                                    })
                                    .unwrap_or_default();

                                if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(block_index) {
                                    t.thinking = summary_text.clone();
                                    t.thinking_signature = Some(serde_json::to_string(&item).unwrap_or_default());
                                }
                                stream.push(AssistantMessageEvent::ThinkingEnd {
                                    content_index: block_index,
                                    content: summary_text,
                                    partial: output.clone(),
                                });
                                *current_block = None;
                            }
                        }
                    }
                    "message" => {
                        if let Some(cb) = current_block.as_mut() {
                            if let CurrentBlock::Text { .. } = cb {
                                let final_text = item
                                    .get("content")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|c| {
                                                let ct = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                match ct {
                                                    "output_text" => c.get("text").and_then(|t| t.as_str()),
                                                    "refusal" => c.get("refusal").and_then(|t| t.as_str()),
                                                    _ => None,
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                            .join("")
                                    })
                                    .unwrap_or_default();
                                let item_id = item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());

                                if let Some(ContentBlock::Text(t)) = output.content.get_mut(block_index) {
                                    t.text = final_text.clone();
                                    t.text_signature = item_id;
                                }
                                stream.push(AssistantMessageEvent::TextEnd {
                                    content_index: block_index,
                                    content: final_text,
                                    partial: output.clone(),
                                });
                                *current_block = None;
                            }
                        }
                    }
                    "function_call" => {
                        // Parse final arguments
                        let args = if let Some(CurrentBlock::ToolCall { partial_json, .. }) = current_block.as_ref() {
                            if !partial_json.is_empty() {
                                serde_json::from_str(partial_json).unwrap_or_else(|_| {
                                    serde_json::from_str(
                                        item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}")
                                    ).unwrap_or(json!({}))
                                })
                            } else {
                                serde_json::from_str(
                                    item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}")
                                ).unwrap_or(json!({}))
                            }
                        } else {
                            serde_json::from_str(
                                item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}")
                            ).unwrap_or(json!({}))
                        };

                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        let tool_call = ToolCall {
                            id: format!("{call_id}|{item_id}"),
                            name,
                            arguments: args,
                            thought_signature: None,
                        };

                        // Update the output content block
                        if let Some(ContentBlock::ToolCall(tc)) = output.content.get_mut(block_index) {
                            tc.id = tool_call.id.clone();
                            tc.name = tool_call.name.clone();
                            tc.arguments = tool_call.arguments.clone();
                        }

                        *current_block = None;
                        stream.push(AssistantMessageEvent::ToolCallEnd {
                            content_index: block_index,
                            tool_call,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "response.completed" => {
                let response = event.get("response");
                if let Some(usage) = response.and_then(|r| r.get("usage")) {
                    let cached_tokens = usage
                        .get("input_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    let output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    let total_tokens = usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0);

                    // OpenAI includes cached tokens in input_tokens, so subtract to get non-cached input
                    output.usage = Usage {
                        input: input_tokens.saturating_sub(cached_tokens),
                        output: output_tokens,
                        cache_read: cached_tokens,
                        cache_write: 0,
                        total_tokens,
                        cost: UsageCost::default(),
                    };
                }
                calculate_cost(model, &mut output.usage);

                if let Some(opts) = options {
                    if let Some(apply_fn) = &opts.apply_service_tier_pricing {
                        let service_tier = response
                            .and_then(|r| r.get("service_tier"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| opts.service_tier.clone());
                        apply_fn(&mut output.usage, service_tier.as_deref());
                    }
                }

                // Map status to stop reason
                let status = response
                    .and_then(|r| r.get("status"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("completed");
                output.stop_reason = map_stop_reason(status);

                // If there are tool calls and stop reason is "stop", upgrade to "toolUse"
                let has_tool_calls = output.content.iter().any(|b| b.as_tool_call().is_some());
                if has_tool_calls && output.stop_reason == StopReason::Stop {
                    output.stop_reason = StopReason::ToolUse;
                }
            }
            "error" => {
                let code = event.get("code").and_then(|v| v.as_str()).unwrap_or("");
                let message = event.get("message").and_then(|v| v.as_str()).unwrap_or("Unknown error");
                return Err(format!("Error Code {code}: {message}"));
            }
            "response.failed" => {
                return Err("Unknown error".to_string());
            }
            _ => {
                // Ignore unhandled events
            }
        }
    }

    Ok(())
}

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map an OpenAI Responses API status string to our StopReason enum.
pub fn map_stop_reason(status: &str) -> StopReason {
    match status {
        "completed" => StopReason::Stop,
        "incomplete" => StopReason::Length,
        "failed" | "cancelled" => StopReason::Error,
        "in_progress" | "queued" => StopReason::Stop,
        "" => StopReason::Stop,
        unknown => {
            tracing::warn!("Unknown OpenAI Responses stop reason: {}", unknown);
            StopReason::Stop
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_hash_deterministic() {
        let h1 = short_hash("hello world");
        let h2 = short_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_short_hash_different_inputs() {
        let h1 = short_hash("hello");
        let h2 = short_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_short_hash_empty() {
        let h = short_hash("");
        assert!(!h.is_empty());
    }

    #[test]
    fn test_to_base36() {
        assert_eq!(to_base36(0), "0");
        assert_eq!(to_base36(35), "z");
        assert_eq!(to_base36(36), "10");
        assert_eq!(to_base36(100), "2s");
    }

    #[test]
    fn test_map_stop_reason_completed() {
        assert_eq!(map_stop_reason("completed"), StopReason::Stop);
    }

    #[test]
    fn test_map_stop_reason_incomplete() {
        assert_eq!(map_stop_reason("incomplete"), StopReason::Length);
    }

    #[test]
    fn test_map_stop_reason_failed() {
        assert_eq!(map_stop_reason("failed"), StopReason::Error);
    }

    #[test]
    fn test_map_stop_reason_cancelled() {
        assert_eq!(map_stop_reason("cancelled"), StopReason::Error);
    }

    #[test]
    fn test_map_stop_reason_in_progress() {
        assert_eq!(map_stop_reason("in_progress"), StopReason::Stop);
    }

    #[test]
    fn test_map_stop_reason_empty() {
        assert_eq!(map_stop_reason(""), StopReason::Stop);
    }

    #[test]
    fn test_map_stop_reason_unknown() {
        // Unknown status should still return Stop (with a warning log)
        assert_eq!(map_stop_reason("some_unknown_status"), StopReason::Stop);
    }

    #[test]
    fn test_convert_responses_tools_basic() {
        let tools = vec![Tool {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        }];
        let result = convert_responses_tools(&tools, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "function");
        assert_eq!(result[0]["name"], "search");
        assert_eq!(result[0]["strict"], false);
    }

    #[test]
    fn test_convert_responses_tools_with_strict() {
        let tools = vec![Tool {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({"type": "object"}),
        }];
        let opts = ConvertResponsesToolsOptions { strict: Some(true) };
        let result = convert_responses_tools(&tools, Some(&opts));
        assert_eq!(result[0]["strict"], true);
    }

    #[test]
    fn test_normalize_tool_call_id_no_pipe() {
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let model = Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let result = normalize_tool_call_id("call_123", &model, &providers);
        assert_eq!(result, "call_123");
    }

    #[test]
    fn test_normalize_tool_call_id_with_pipe() {
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let model = Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let result = normalize_tool_call_id("call_123|item_456", &model, &providers);
        assert!(result.contains('|'));
        assert!(result.contains("call_123"));
        // Item should get fc_ prefix
        assert!(result.contains("fc_item_456"));
    }

    #[test]
    fn test_normalize_tool_call_id_strips_trailing_underscores() {
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let model = Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let result = normalize_tool_call_id("call_123_|fc_item___", &model, &providers);
        assert!(!result.ends_with('_'));
    }

    #[test]
    fn test_normalize_tool_call_id_non_allowed_provider() {
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let model = Model {
            id: "claude-sonnet-4".to_string(),
            name: "Claude Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 200000,
            max_tokens: 8192,
            headers: None,
            compat: None,
        };
        // Non-allowed provider should return the ID unchanged
        let result = normalize_tool_call_id("call_123|item_456", &model, &providers);
        assert_eq!(result, "call_123|item_456");
    }

    #[test]
    fn test_convert_responses_messages_basic() {
        let model = Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Hello".to_string()),
                timestamp: 0,
            })],
            tools: None,
        };
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let result = convert_responses_messages(&model, &context, &providers, None);
        assert!(result.len() >= 2); // system + user
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[1]["role"], "user");
    }

    #[test]
    fn test_convert_responses_messages_reasoning_model_developer_role() {
        let model = Model {
            id: "o3-mini".to_string(),
            name: "O3 Mini".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: true,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![],
            tools: None,
        };
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let result = convert_responses_messages(&model, &context, &providers, None);
        assert_eq!(result[0]["role"], "developer");
    }

    #[test]
    fn test_convert_responses_messages_no_system_prompt() {
        let model = Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![],
            tools: None,
        };
        let providers: HashSet<&str> = ["openai"].into_iter().collect();
        let opts = ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
        };
        let result = convert_responses_messages(&model, &context, &providers, Some(&opts));
        assert!(result.is_empty());
    }
}
