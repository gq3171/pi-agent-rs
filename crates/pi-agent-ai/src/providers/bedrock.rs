use std::collections::HashMap;

use base64::Engine;
use futures::StreamExt;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use pi_agent_core::json_parse::parse_streaming_json;
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::calculate_cost;
use crate::registry::ApiProvider;
use crate::simple_options::{adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning};

// ---------- BedrockOptions ----------

#[derive(Debug, Clone)]
pub struct BedrockOptions {
    pub base: StreamOptions,
    pub region: Option<String>,
    pub profile: Option<String>,
    pub tool_choice: Option<BedrockToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub interleaved_thinking: bool,
}

#[derive(Debug, Clone)]
pub enum BedrockToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

// ---------- AWS SigV4 Signing ----------

type HmacSha256 = Hmac<Sha256>;

fn sha256_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key)
        .expect("HMAC can take key of any size");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

fn get_signature_key(key: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{key}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Resolve the AWS region from options, environment, or default.
fn resolve_region(options_region: Option<&str>) -> String {
    if let Some(r) = options_region {
        if !r.is_empty() {
            return r.to_string();
        }
    }
    if let Ok(r) = std::env::var("AWS_REGION") {
        if !r.is_empty() {
            return r;
        }
    }
    if let Ok(r) = std::env::var("AWS_DEFAULT_REGION") {
        if !r.is_empty() {
            return r;
        }
    }
    "us-east-1".to_string()
}

/// Resolve AWS credentials from environment variables.
/// Returns (access_key_id, secret_access_key, session_token).
fn resolve_aws_credentials() -> Option<(String, String, Option<String>)> {
    // Check for skip-auth mode (proxy that doesn't need real creds)
    if std::env::var("AWS_BEDROCK_SKIP_AUTH").ok().as_deref() == Some("1") {
        return Some((
            "dummy-access-key".to_string(),
            "dummy-secret-key".to_string(),
            None,
        ));
    }

    let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
    let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
    Some((access_key, secret_key, session_token))
}

/// Resolve the AWS bearer token for Bedrock.
fn resolve_bearer_token() -> Option<String> {
    std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok()
}

/// Build a SigV4 signed request for Bedrock ConverseStream.
fn sign_request(
    method: &str,
    url_str: &str,
    body: &[u8],
    region: &str,
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
) -> Result<HashMap<String, String>, String> {
    let now = chrono::Utc::now();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date_stamp = now.format("%Y%m%d").to_string();

    let parsed = url::Url::parse(url_str).map_err(|e| format!("Invalid URL: {e}"))?;
    let host = parsed.host_str().unwrap_or("").to_string();
    let canonical_uri = parsed.path().to_string();
    let canonical_querystring = parsed.query().unwrap_or("").to_string();

    let service = "bedrock";
    let content_type = "application/json";
    let payload_hash = sha256_hash(body);

    // Build canonical headers
    let mut signed_header_names = vec!["content-type", "host", "x-amz-content-sha256", "x-amz-date"];
    if session_token.is_some() {
        signed_header_names.push("x-amz-security-token");
    }
    signed_header_names.sort();

    let mut canonical_headers = String::new();
    for name in &signed_header_names {
        let value = match *name {
            "content-type" => content_type.to_string(),
            "host" => host.clone(),
            "x-amz-content-sha256" => payload_hash.clone(),
            "x-amz-date" => amz_date.clone(),
            "x-amz-security-token" => session_token.unwrap_or("").to_string(),
            _ => String::new(),
        };
        canonical_headers.push_str(&format!("{name}:{value}\n"));
    }

    let signed_headers = signed_header_names.join(";");

    let canonical_request = format!(
        "{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );

    let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        sha256_hash(canonical_request.as_bytes())
    );

    let signing_key = get_signature_key(secret_key, &date_stamp, region, service);
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), content_type.to_string());
    headers.insert("x-amz-date".to_string(), amz_date);
    headers.insert("x-amz-content-sha256".to_string(), payload_hash);
    headers.insert("authorization".to_string(), authorization);

    if let Some(token) = session_token {
        headers.insert("x-amz-security-token".to_string(), token.to_string());
    }

    Ok(headers)
}

// ---------- Helper functions ----------

fn resolve_cache_retention(cache_retention: Option<&CacheRetention>) -> CacheRetention {
    if let Some(r) = cache_retention {
        return r.clone();
    }
    if let Ok(val) = std::env::var("PI_CACHE_RETENTION") {
        if val.to_lowercase() == "long" {
            return CacheRetention::Long;
        }
    }
    CacheRetention::Short
}

/// Check if the model supports adaptive thinking (Opus 4.6+).
fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6") || model_id.contains("opus-4.6")
}

fn map_thinking_level_to_effort(level: &ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "max",
    }
}

/// Check if the model supports prompt caching.
/// Supported: Claude 3.5 Haiku, Claude 3.7 Sonnet, Claude 4.x models
fn supports_prompt_caching(model: &Model) -> bool {
    if model.cost.cache_read > 0.0 || model.cost.cache_write > 0.0 {
        return true;
    }
    let id = model.id.to_lowercase();
    // Claude 4.x models
    if id.contains("claude") && (id.contains("-4-") || id.contains("-4.")) {
        return true;
    }
    // Claude 3.7 Sonnet
    if id.contains("claude-3-7-sonnet") {
        return true;
    }
    // Claude 3.5 Haiku
    if id.contains("claude-3-5-haiku") {
        return true;
    }
    false
}

/// Check if the model supports thinking signatures in reasoningContent.
/// Only Anthropic Claude models support the signature field.
fn supports_thinking_signature(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    id.contains("anthropic.claude") || id.contains("anthropic/claude")
}

fn normalize_tool_call_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if sanitized.len() > 64 {
        sanitized[..64].to_string()
    } else {
        sanitized
    }
}

fn normalize_tool_call_id_for_transform(id: &str, _model: &Model, _source: &AssistantMessage) -> String {
    normalize_tool_call_id(id)
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        unknown => {
            tracing::warn!("Unknown Bedrock stop reason: {}", unknown);
            StopReason::Error
        }
    }
}

fn map_image_format(mime_type: &str) -> Result<&'static str, String> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Ok("jpeg"),
        "image/png" => Ok("png"),
        "image/gif" => Ok("gif"),
        "image/webp" => Ok("webp"),
        other => Err(format!("Unknown image type: {other}")),
    }
}

fn create_image_block(mime_type: &str, data: &str) -> Value {
    let format = map_image_format(mime_type).unwrap_or("png");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .unwrap_or_default();
    // Re-encode to base64 for JSON (Bedrock expects bytes as base64 in JSON)
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    json!({
        "source": { "bytes": b64 },
        "format": format
    })
}

// ---------- Build system prompt ----------

fn build_system_prompt(
    system_prompt: Option<&str>,
    model: &Model,
    cache_retention: &CacheRetention,
) -> Option<Value> {
    let sp = system_prompt?;
    if sp.is_empty() {
        return None;
    }

    let mut blocks: Vec<Value> = vec![json!({"text": sanitize_surrogates(sp)})];

    // Add cache point for supported Claude models when caching is enabled
    if *cache_retention != CacheRetention::None && supports_prompt_caching(model) {
        let mut cache_point = json!({"type": "default"});
        if *cache_retention == CacheRetention::Long {
            cache_point["ttl"] = json!("ONE_HOUR");
        }
        blocks.push(json!({"cachePoint": cache_point}));
    }

    Some(json!(blocks))
}

// ---------- Convert messages ----------

fn convert_messages(
    messages: &[Message],
    model: &Model,
    cache_retention: &CacheRetention,
) -> Vec<Value> {
    let transformed = transform_messages(messages, model, Some(&normalize_tool_call_id_for_transform));
    let mut result: Vec<Value> = Vec::new();

    let mut i = 0;
    while i < transformed.len() {
        let msg = &transformed[i];

        match msg {
            Message::User(user_msg) => {
                let content = match &user_msg.content {
                    UserContent::Text(text) => {
                        json!([{"text": sanitize_surrogates(text)}])
                    }
                    UserContent::Blocks(blocks) => {
                        let items: Vec<Value> = blocks
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(t) => {
                                    Some(json!({"text": sanitize_surrogates(&t.text)}))
                                }
                                ContentBlock::Image(img) => {
                                    Some(json!({"image": create_image_block(&img.mime_type, &img.data)}))
                                }
                                _ => None,
                            })
                            .collect();
                        json!(items)
                    }
                };
                result.push(json!({
                    "role": "user",
                    "content": content
                }));
            }
            Message::Assistant(assistant_msg) => {
                // Skip assistant messages with empty content
                if assistant_msg.content.is_empty() {
                    i += 1;
                    continue;
                }

                let mut content_blocks: Vec<Value> = Vec::new();
                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            // Skip empty text blocks
                            if t.text.trim().is_empty() {
                                continue;
                            }
                            content_blocks.push(json!({"text": sanitize_surrogates(&t.text)}));
                        }
                        ContentBlock::ToolCall(tc) => {
                            content_blocks.push(json!({
                                "toolUse": {
                                    "toolUseId": tc.id,
                                    "name": tc.name,
                                    "input": tc.arguments
                                }
                            }));
                        }
                        ContentBlock::Thinking(t) => {
                            // Skip empty thinking blocks
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            if supports_thinking_signature(model) {
                                content_blocks.push(json!({
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": sanitize_surrogates(&t.thinking),
                                            "signature": t.thinking_signature.as_deref().unwrap_or("")
                                        }
                                    }
                                }));
                            } else {
                                content_blocks.push(json!({
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": sanitize_surrogates(&t.thinking)
                                        }
                                    }
                                }));
                            }
                        }
                        _ => {}
                    }
                }

                // Skip if all content blocks were filtered out
                if content_blocks.is_empty() {
                    i += 1;
                    continue;
                }

                result.push(json!({
                    "role": "assistant",
                    "content": content_blocks
                }));
            }
            Message::ToolResult(tr) => {
                // Collect all consecutive toolResult messages into a single user message
                let mut tool_results: Vec<Value> = Vec::new();

                tool_results.push(build_tool_result_block(tr));

                // Look ahead for consecutive toolResult messages
                let mut j = i + 1;
                while j < transformed.len() {
                    if let Message::ToolResult(next_tr) = &transformed[j] {
                        tool_results.push(build_tool_result_block(next_tr));
                        j += 1;
                    } else {
                        break;
                    }
                }

                i = j - 1; // Will be incremented at end of loop

                result.push(json!({
                    "role": "user",
                    "content": tool_results
                }));
            }
        }

        i += 1;
    }

    // Add cache point to the last user message for supported Claude models when caching is enabled
    if *cache_retention != CacheRetention::None && supports_prompt_caching(model) && !result.is_empty() {
        if let Some(last_msg) = result.last_mut() {
            if last_msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(content) = last_msg.get_mut("content") {
                    if let Some(arr) = content.as_array_mut() {
                        let mut cache_point = json!({"type": "default"});
                        if *cache_retention == CacheRetention::Long {
                            cache_point["ttl"] = json!("ONE_HOUR");
                        }
                        arr.push(json!({"cachePoint": cache_point}));
                    }
                }
            }
        }
    }

    result
}

fn build_tool_result_block(tr: &ToolResultMessage) -> Value {
    let content: Vec<Value> = tr
        .content
        .iter()
        .map(|c| match c {
            ContentBlock::Image(img) => {
                json!({"image": create_image_block(&img.mime_type, &img.data)})
            }
            ContentBlock::Text(t) => json!({"text": sanitize_surrogates(&t.text)}),
            _ => json!({"text": ""}),
        })
        .collect();

    json!({
        "toolResult": {
            "toolUseId": tr.tool_call_id,
            "content": content,
            "status": if tr.is_error { "error" } else { "success" }
        }
    })
}

// ---------- Convert tool config ----------

fn convert_tool_config(tools: Option<&[Tool]>, tool_choice: &Option<BedrockToolChoice>) -> Option<Value> {
    let tools = match tools {
        Some(t) if !t.is_empty() => t,
        _ => return None,
    };

    if matches!(tool_choice, Some(BedrockToolChoice::None)) {
        return None;
    }

    let bedrock_tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": { "json": tool.parameters }
                }
            })
        })
        .collect();

    let bedrock_tool_choice = match tool_choice {
        Some(BedrockToolChoice::Auto) => Some(json!({"auto": {}})),
        Some(BedrockToolChoice::Any) => Some(json!({"any": {}})),
        Some(BedrockToolChoice::Tool { name }) => Some(json!({"tool": {"name": name}})),
        _ => None,
    };

    let mut config = json!({"tools": bedrock_tools});
    if let Some(tc) = bedrock_tool_choice {
        config["toolChoice"] = tc;
    }

    Some(config)
}

// ---------- Build additional model request fields ----------

fn build_additional_model_request_fields(
    model: &Model,
    options: &BedrockOptions,
) -> Option<Value> {
    let reasoning = match &options.reasoning {
        Some(r) => r,
        None => return None,
    };

    if !model.reasoning {
        return None;
    }

    if model.id.contains("anthropic.claude") {
        let mut result = if supports_adaptive_thinking(&model.id) {
            json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": map_thinking_level_to_effort(reasoning)}
            })
        } else {
            let default_budgets: HashMap<&str, u64> = [
                ("minimal", 1024),
                ("low", 2048),
                ("medium", 8192),
                ("high", 16384),
                ("xhigh", 16384),
            ].into();

            let level_str = match reasoning {
                ThinkingLevel::Xhigh => "high",
                ThinkingLevel::Minimal => "minimal",
                ThinkingLevel::Low => "low",
                ThinkingLevel::Medium => "medium",
                ThinkingLevel::High => "high",
            };

            let budget = options
                .thinking_budgets
                .as_ref()
                .and_then(|budgets| match level_str {
                    "minimal" => budgets.minimal,
                    "low" => budgets.low,
                    "medium" => budgets.medium,
                    "high" => budgets.high,
                    _ => None,
                })
                .unwrap_or_else(|| *default_budgets.get(&reasoning.to_string().as_str()).unwrap_or(&16384));

            json!({
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget
                }
            })
        };

        if options.interleaved_thinking && !supports_adaptive_thinking(&model.id) {
            result["anthropic_beta"] = json!(["interleaved-thinking-2025-05-14"]);
        }

        return Some(result);
    }

    None
}

// ---------- AWS Event Stream Binary Decoder ----------

/// Decode an AWS Event Stream binary message.
/// AWS event stream messages have:
///   4 bytes: total byte length (big-endian)
///   4 bytes: headers byte length (big-endian)
///   4 bytes: prelude CRC
///   N bytes: headers
///   M bytes: payload
///   4 bytes: message CRC
///
/// Returns a list of (headers, payload) tuples parsed from the buffer,
/// and the number of bytes consumed.
fn decode_event_stream_messages(buf: &[u8]) -> (Vec<(HashMap<String, String>, Vec<u8>)>, usize) {
    let mut results = Vec::new();
    let mut offset = 0;

    while offset + 12 <= buf.len() {
        let total_len = u32::from_be_bytes([
            buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
        ]) as usize;

        if total_len < 16 || offset + total_len > buf.len() {
            break;
        }

        let headers_len = u32::from_be_bytes([
            buf[offset + 4], buf[offset + 5], buf[offset + 6], buf[offset + 7],
        ]) as usize;

        // Skip prelude CRC (4 bytes at offset+8)
        let headers_start = offset + 12;
        let headers_end = headers_start + headers_len;
        let payload_start = headers_end;
        let payload_end = offset + total_len - 4; // Subtract message CRC

        if headers_end > buf.len() || payload_end > buf.len() || payload_end < payload_start {
            break;
        }

        let headers = decode_event_headers(&buf[headers_start..headers_end]);
        let payload = buf[payload_start..payload_end].to_vec();

        results.push((headers, payload));
        offset += total_len;
    }

    (results, offset)
}

/// Decode event stream headers.
/// Each header:
///   1 byte: header name length
///   N bytes: header name (UTF-8)
///   1 byte: header value type (7 = string)
///   2 bytes: header value length (big-endian)
///   M bytes: header value
fn decode_event_headers(data: &[u8]) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    let mut pos = 0;

    while pos < data.len() {
        if pos >= data.len() {
            break;
        }
        let name_len = data[pos] as usize;
        pos += 1;

        if pos + name_len > data.len() {
            break;
        }
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        if pos >= data.len() {
            break;
        }
        let value_type = data[pos];
        pos += 1;

        match value_type {
            7 => {
                // String type
                if pos + 2 > data.len() {
                    break;
                }
                let value_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                if pos + value_len > data.len() {
                    break;
                }
                let value = String::from_utf8_lossy(&data[pos..pos + value_len]).to_string();
                pos += value_len;
                headers.insert(name, value);
            }
            _ => {
                // For non-string types, skip based on known sizes
                // Type 0 (bool_true), type 1 (bool_false): 0 bytes
                // Type 2 (byte): 1 byte
                // Type 3 (short): 2 bytes
                // Type 4 (int): 4 bytes
                // Type 5 (long): 8 bytes
                // Type 6 (bytes): 2-byte length + data
                // Type 8 (timestamp): 8 bytes
                // Type 9 (uuid): 16 bytes
                match value_type {
                    0 | 1 => {}
                    2 => { pos += 1; }
                    3 => { pos += 2; }
                    4 => { pos += 4; }
                    5 | 8 => { pos += 8; }
                    6 => {
                        if pos + 2 > data.len() { break; }
                        let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                        pos += 2 + len;
                    }
                    9 => { pos += 16; }
                    _ => { break; }
                }
            }
        }
    }

    headers
}

// ---------- Stream event handling ----------

/// Internal block type with tracking fields
struct StreamBlock {
    content_block_index: Option<u64>,
    partial_json: Option<String>,
}

fn handle_content_block_start(
    event: &Value,
    blocks: &mut Vec<StreamBlock>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let index = event.get("contentBlockIndex").and_then(|v| v.as_u64());
    let start = event.get("start");

    if let Some(start) = start {
        if let Some(tool_use) = start.get("toolUse") {
            let id = tool_use.get("toolUseId").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let name = tool_use.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();

            let block = ContentBlock::ToolCall(ToolCall {
                id,
                name,
                arguments: Value::Object(Default::default()),
                thought_signature: None,
            });
            output.content.push(block);
            blocks.push(StreamBlock {
                content_block_index: index,
                partial_json: Some(String::new()),
            });
            let ci = output.content.len() - 1;
            stream.push(AssistantMessageEvent::ToolCallStart {
                content_index: ci,
                partial: output.clone(),
            });
        }
    }
}

fn handle_content_block_delta(
    event: &Value,
    blocks: &mut Vec<StreamBlock>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let content_block_index = event.get("contentBlockIndex").and_then(|v| v.as_u64());
    let delta = event.get("delta");

    let block_pos = blocks.iter().position(|b| b.content_block_index == content_block_index);

    if let Some(delta) = delta {
        // Text delta
        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
            let (ci, is_new) = if let Some(pos) = block_pos {
                // Find the corresponding content index
                let ci = find_content_index(blocks, pos, output);
                (ci, false)
            } else {
                // No text block exists yet, create one
                let block = ContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                });
                output.content.push(block);
                blocks.push(StreamBlock {
                    content_block_index,
                    partial_json: None,
                });
                (output.content.len() - 1, true)
            };

            if is_new {
                stream.push(AssistantMessageEvent::TextStart {
                    content_index: ci,
                    partial: output.clone(),
                });
            }

            if let Some(ContentBlock::Text(t)) = output.content.get_mut(ci) {
                t.text.push_str(text);
            }
            stream.push(AssistantMessageEvent::TextDelta {
                content_index: ci,
                delta: text.to_string(),
                partial: output.clone(),
            });
        }
        // Tool use delta
        else if let Some(tool_use) = delta.get("toolUse") {
            if let Some(pos) = block_pos {
                let ci = find_content_index(blocks, pos, output);
                let input_str = tool_use.get("input").and_then(|v| v.as_str()).unwrap_or("");

                if let Some(pj) = &mut blocks[pos].partial_json {
                    pj.push_str(input_str);
                    let parsed = parse_streaming_json(pj);
                    if let Some(ContentBlock::ToolCall(tc)) = output.content.get_mut(ci) {
                        tc.arguments = parsed;
                    }
                }

                stream.push(AssistantMessageEvent::ToolCallDelta {
                    content_index: ci,
                    delta: input_str.to_string(),
                    partial: output.clone(),
                });
            }
        }
        // Reasoning content delta
        else if let Some(reasoning) = delta.get("reasoningContent") {
            let (ci, is_new) = if let Some(pos) = block_pos {
                (find_content_index(blocks, pos, output), false)
            } else {
                let block = ContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                });
                output.content.push(block);
                blocks.push(StreamBlock {
                    content_block_index,
                    partial_json: None,
                });
                (output.content.len() - 1, true)
            };

            if is_new {
                stream.push(AssistantMessageEvent::ThinkingStart {
                    content_index: ci,
                    partial: output.clone(),
                });
            }

            let reasoning_text = reasoning.get("text").and_then(|v| v.as_str()).map(|s| s.to_string());
            let reasoning_sig = reasoning.get("signature").and_then(|v| v.as_str()).map(|s| s.to_string());

            if let Some(text) = &reasoning_text {
                if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci) {
                    t.thinking.push_str(text);
                }
            }

            if let Some(sig) = &reasoning_sig {
                if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci) {
                    let current = t.thinking_signature.get_or_insert_with(String::new);
                    current.push_str(sig);
                }
            }

            if let Some(text) = reasoning_text {
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: ci,
                    delta: text,
                    partial: output.clone(),
                });
            }
        }
    }
}

fn handle_content_block_stop(
    event: &Value,
    blocks: &mut Vec<StreamBlock>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let content_block_index = event.get("contentBlockIndex").and_then(|v| v.as_u64());
    let pos = blocks.iter().position(|b| b.content_block_index == content_block_index);

    if let Some(pos) = pos {
        let ci = find_content_index(blocks, pos, output);

        // Clear the content_block_index
        blocks[pos].content_block_index = None;

        // Determine block type and emit appropriate end event
        match output.content.get(ci) {
            Some(ContentBlock::Text(t)) => {
                stream.push(AssistantMessageEvent::TextEnd {
                    content_index: ci,
                    content: t.text.clone(),
                    partial: output.clone(),
                });
            }
            Some(ContentBlock::Thinking(t)) => {
                stream.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: ci,
                    content: t.thinking.clone(),
                    partial: output.clone(),
                });
            }
            Some(ContentBlock::ToolCall(_)) => {
                // Final parse of accumulated JSON
                if let Some(pj) = blocks[pos].partial_json.take() {
                    let parsed = parse_streaming_json(&pj);
                    if let Some(ContentBlock::ToolCall(tc)) = output.content.get_mut(ci) {
                        tc.arguments = parsed;
                    }
                }
                let final_tc = if let Some(ContentBlock::ToolCall(tc)) = output.content.get(ci) {
                    tc.clone()
                } else {
                    ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: Value::Object(Default::default()),
                        thought_signature: None,
                    }
                };
                stream.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: ci,
                    tool_call: final_tc,
                    partial: output.clone(),
                });
            }
            _ => {}
        }
    }
}

fn handle_metadata(event: &Value, model: &Model, output: &mut AssistantMessage) {
    if let Some(usage) = event.get("usage") {
        output.usage.input = usage.get("inputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        output.usage.output = usage.get("outputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        output.usage.cache_read = usage.get("cacheReadInputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        output.usage.cache_write = usage.get("cacheWriteInputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        output.usage.total_tokens = usage
            .get("totalTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(output.usage.input + output.usage.output);
        calculate_cost(model, &mut output.usage);
    }
}

/// Find the content index for a StreamBlock at position `pos` within output.content.
/// Since we maintain blocks in insertion order parallel to output.content, the position
/// in blocks corresponds to position in output.content when blocks are 1:1.
fn find_content_index(_blocks: &[StreamBlock], pos: usize, output: &AssistantMessage) -> usize {
    // The blocks vector parallels output.content (each StreamBlock maps to one ContentBlock).
    // But blocks might not be in 1:1 correspondence since we add to both. Use position directly.
    // This is a simplification: the block at position `pos` was the `pos`-th block added.
    // Since output.content grows in the same order as blocks, pos is the content index.
    pos.min(output.content.len().saturating_sub(1))
}

// ---------- Build the Bedrock API URL ----------

fn build_bedrock_url(region: &str, model_id: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{}/converse-stream",
        urlencoded(model_id)
    )
}

fn urlencoded(s: &str) -> String {
    // URL-encode the model ID (mainly for the colon in model IDs like "amazon.nova-2-lite-v1:0")
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

// ---------- Stream functions ----------

/// Stream from Bedrock ConverseStream API using raw HTTP + AWS SigV4 signing.
pub fn stream_bedrock(
    model: &Model,
    context: &Context,
    options: &BedrockOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&model);

        let region = resolve_region(options.region.as_deref());
        let cache_retention = resolve_cache_retention(options.base.cache_retention.as_ref());

        // Build request body
        let messages = convert_messages(&context.messages, &model, &cache_retention);
        let system = build_system_prompt(context.system_prompt.as_deref(), &model, &cache_retention);
        let tool_config = convert_tool_config(
            context.tools.as_deref(),
            &options.tool_choice,
        );
        let additional_fields = build_additional_model_request_fields(&model, &options);

        let mut body = json!({
            "modelId": model.id,
            "messages": messages
        });

        if let Some(sys) = system {
            body["system"] = sys;
        }

        let mut inference_config = json!({});
        if let Some(max_tokens) = options.base.max_tokens {
            inference_config["maxTokens"] = json!(max_tokens);
        }
        if let Some(temp) = options.base.temperature {
            inference_config["temperature"] = json!(temp);
        }
        if inference_config.as_object().map_or(false, |o| !o.is_empty()) {
            body["inferenceConfig"] = inference_config;
        }

        if let Some(tc) = tool_config {
            body["toolConfig"] = tc;
        }

        if let Some(additional) = additional_fields {
            body["additionalModelRequestFields"] = additional;
        }

        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        let url = build_bedrock_url(&region, &model.id);

        // Build request with authentication
        let client = reqwest::Client::new();
        let mut request = client.post(&url);

        // Determine auth method and set headers
        if let Some(bearer) = resolve_bearer_token() {
            request = request
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {bearer}"));
        } else if let Some((access_key, secret_key, session_token)) = resolve_aws_credentials() {
            let headers = match sign_request(
                "POST",
                &url,
                &body_bytes,
                &region,
                &access_key,
                &secret_key,
                session_token.as_deref(),
            ) {
                Ok(h) => h,
                Err(e) => {
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(format!("SigV4 signing error: {e}"));
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    });
                    return;
                }
            };
            for (k, v) in &headers {
                request = request.header(k.as_str(), v.as_str());
            }
        } else {
            // No credentials available
            output.stop_reason = StopReason::Error;
            output.error_message = Some("No AWS credentials available. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_BEARER_TOKEN_BEDROCK, or AWS_PROFILE.".to_string());
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        let response = match request.body(body_bytes).send().await {
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

        // Parse the AWS event stream binary format
        let mut blocks: Vec<StreamBlock> = Vec::new();
        let mut event_buf: Vec<u8> = Vec::new();

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
                Ok(bytes) => bytes,
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

            event_buf.extend_from_slice(&chunk);

            let (messages, consumed) = decode_event_stream_messages(&event_buf);
            if consumed > 0 {
                event_buf.drain(..consumed);
            }

            for (headers, payload) in messages {
                let event_type = headers.get(":event-type").cloned()
                    .or_else(|| headers.get(":exception-type").cloned());
                let message_type = headers.get(":message-type").cloned();

                // Handle exceptions
                if message_type.as_deref() == Some("exception") {
                    let error_msg = String::from_utf8_lossy(&payload);
                    let exception_type = event_type.as_deref().unwrap_or("unknown");
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(format!("{exception_type}: {error_msg}"));
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    });
                    return;
                }

                // Parse the JSON payload
                let data: Value = match serde_json::from_slice(&payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match event_type.as_deref() {
                    Some("messageStart") => {
                        let role = data.get("role").and_then(|v| v.as_str()).unwrap_or("");
                        if role != "assistant" {
                            output.stop_reason = StopReason::Error;
                            output.error_message = Some(
                                "Unexpected assistant message start but got user message start instead".to_string()
                            );
                            stream_clone.push(AssistantMessageEvent::Error {
                                reason: StopReason::Error,
                                error: output,
                            });
                            return;
                        }
                    }
                    Some("contentBlockStart") => {
                        handle_content_block_start(&data, &mut blocks, &mut output, &stream_clone);
                    }
                    Some("contentBlockDelta") => {
                        handle_content_block_delta(&data, &mut blocks, &mut output, &stream_clone);
                    }
                    Some("contentBlockStop") => {
                        handle_content_block_stop(&data, &mut blocks, &mut output, &stream_clone);
                    }
                    Some("messageStop") => {
                        let stop_reason = data.get("stopReason").and_then(|v| v.as_str()).unwrap_or("end_turn");
                        output.stop_reason = map_stop_reason(stop_reason);
                    }
                    Some("metadata") => {
                        handle_metadata(&data, &model, &mut output);
                    }
                    Some("internalServerException") | Some("modelStreamErrorException")
                    | Some("validationException") | Some("throttlingException")
                    | Some("serviceUnavailableException") => {
                        let error_msg = data.get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error");
                        output.stop_reason = StopReason::Error;
                        output.error_message = Some(format!("{}: {error_msg}", event_type.as_deref().unwrap_or("error")));
                        stream_clone.push(AssistantMessageEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        });
                        return;
                    }
                    _ => {
                        // Unknown event type, skip
                    }
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

/// Simplified streaming: handles reasoning level and credential resolution.
pub fn stream_simple_bedrock(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    // Check for AWS credentials
    let has_creds = get_env_api_key(&model.provider);
    if has_creds.is_none() {
        let stream = create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some("No AWS credentials available for Bedrock".to_string());
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg,
        });
        return stream;
    }

    let base = build_base_options(model, options, "<authenticated>");

    if options.reasoning.is_none() {
        let bedrock_opts = BedrockOptions {
            base,
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: None,
            thinking_budgets: None,
            interleaved_thinking: false,
        };
        return stream_bedrock(model, context, &bedrock_opts, cancel);
    }

    let reasoning = options.reasoning.as_ref().unwrap();

    // Check if it's an Anthropic Claude model on Bedrock
    if model.id.contains("anthropic.claude") || model.id.contains("anthropic/claude") {
        if supports_adaptive_thinking(&model.id) {
            let bedrock_opts = BedrockOptions {
                base,
                region: None,
                profile: None,
                tool_choice: None,
                reasoning: Some(reasoning.clone()),
                thinking_budgets: options.thinking_budgets.clone(),
                interleaved_thinking: false,
            };
            return stream_bedrock(model, context, &bedrock_opts, cancel);
        }

        let (max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
            base.max_tokens.unwrap_or(0),
            model.max_tokens,
            reasoning,
            options.thinking_budgets.as_ref(),
        );

        let clamped = clamp_reasoning(reasoning);
        let mut adjusted_budgets = options.thinking_budgets.clone().unwrap_or_default();
        match clamped {
            ThinkingLevel::Minimal => adjusted_budgets.minimal = Some(thinking_budget),
            ThinkingLevel::Low => adjusted_budgets.low = Some(thinking_budget),
            ThinkingLevel::Medium => adjusted_budgets.medium = Some(thinking_budget),
            ThinkingLevel::High => adjusted_budgets.high = Some(thinking_budget),
            ThinkingLevel::Xhigh => { /* unreachable after clamp */ }
        }

        let mut adjusted_base = base;
        adjusted_base.max_tokens = Some(max_tokens);

        let bedrock_opts = BedrockOptions {
            base: adjusted_base,
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: Some(reasoning.clone()),
            thinking_budgets: Some(adjusted_budgets),
            interleaved_thinking: false,
        };
        return stream_bedrock(model, context, &bedrock_opts, cancel);
    }

    // Non-Claude models: pass through reasoning options as-is
    let bedrock_opts = BedrockOptions {
        base,
        region: None,
        profile: None,
        tool_choice: None,
        reasoning: Some(reasoning.clone()),
        thinking_budgets: options.thinking_budgets.clone(),
        interleaved_thinking: false,
    };
    stream_bedrock(model, context, &bedrock_opts, cancel)
}

// ---------- ApiProvider implementation ----------

pub struct BedrockProvider;

impl ApiProvider for BedrockProvider {
    fn api(&self) -> &str {
        "bedrock-converse-stream"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let bedrock_opts = BedrockOptions {
            base: options.clone(),
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: None,
            thinking_budgets: None,
            interleaved_thinking: false,
        };
        stream_bedrock(model, context, &bedrock_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_bedrock(model, context, options, cancel)
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> Model {
        Model {
            id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            api: "bedrock-converse-stream".to_string(),
            provider: "amazon-bedrock".to_string(),
            base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
            reasoning: true,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            context_window: 200000,
            max_tokens: 8192,
            headers: None,
            compat: None,
        }
    }

    fn test_non_claude_model() -> Model {
        Model {
            id: "amazon.nova-2-lite-v1:0".to_string(),
            name: "Nova 2 Lite".to_string(),
            api: "bedrock-converse-stream".to_string(),
            provider: "amazon-bedrock".to_string(),
            base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost {
                input: 0.33,
                output: 2.75,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        }
    }

    fn test_opus46_model() -> Model {
        Model {
            id: "anthropic.claude-opus-4-6-20250610-v1:0".to_string(),
            name: "Claude Opus 4.6".to_string(),
            api: "bedrock-converse-stream".to_string(),
            provider: "amazon-bedrock".to_string(),
            base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
            reasoning: true,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost {
                input: 15.0,
                output: 75.0,
                cache_read: 1.5,
                cache_write: 18.75,
            },
            context_window: 200000,
            max_tokens: 32000,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_supports_adaptive_thinking() {
        assert!(supports_adaptive_thinking("anthropic.claude-opus-4-6-20250610-v1:0"));
        assert!(supports_adaptive_thinking("some-model-opus-4.6-thing"));
        assert!(!supports_adaptive_thinking("anthropic.claude-3-5-sonnet-20241022-v2:0"));
        assert!(!supports_adaptive_thinking("amazon.nova-2-lite-v1:0"));
    }

    #[test]
    fn test_supports_prompt_caching() {
        let claude_model = test_model();
        assert!(supports_prompt_caching(&claude_model));

        let non_claude = test_non_claude_model();
        assert!(!supports_prompt_caching(&non_claude));

        let opus46 = test_opus46_model();
        assert!(supports_prompt_caching(&opus46));
    }

    #[test]
    fn test_supports_thinking_signature() {
        let claude_model = test_model();
        assert!(supports_thinking_signature(&claude_model));

        let non_claude = test_non_claude_model();
        assert!(!supports_thinking_signature(&non_claude));
    }

    #[test]
    fn test_map_stop_reason() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::Stop);
        assert_eq!(map_stop_reason("stop_sequence"), StopReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_stop_reason("model_context_window_exceeded"), StopReason::Length);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("unknown_thing"), StopReason::Error);
    }

    #[test]
    fn test_normalize_tool_call_id() {
        assert_eq!(normalize_tool_call_id("abc-123_def"), "abc-123_def");
        assert_eq!(normalize_tool_call_id("abc.def:ghi"), "abc_def_ghi");
        assert_eq!(normalize_tool_call_id("a@b#c$d"), "a_b_c_d");

        // Test truncation to 64 chars
        let long_id = "a".repeat(100);
        assert_eq!(normalize_tool_call_id(&long_id).len(), 64);
    }

    #[test]
    fn test_map_image_format() {
        assert_eq!(map_image_format("image/jpeg").unwrap(), "jpeg");
        assert_eq!(map_image_format("image/jpg").unwrap(), "jpeg");
        assert_eq!(map_image_format("image/png").unwrap(), "png");
        assert_eq!(map_image_format("image/gif").unwrap(), "gif");
        assert_eq!(map_image_format("image/webp").unwrap(), "webp");
        assert!(map_image_format("image/bmp").is_err());
    }

    #[test]
    fn test_convert_tool_config() {
        let tools = vec![Tool {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        }];

        // With auto tool choice
        let config = convert_tool_config(Some(&tools), &Some(BedrockToolChoice::Auto));
        assert!(config.is_some());
        let config = config.unwrap();
        assert!(config.get("tools").is_some());
        assert!(config.get("toolChoice").is_some());
        assert!(config["toolChoice"]["auto"].is_object());

        // With none tool choice
        let config = convert_tool_config(Some(&tools), &Some(BedrockToolChoice::None));
        assert!(config.is_none());

        // With no tools
        let config = convert_tool_config(None, &Some(BedrockToolChoice::Auto));
        assert!(config.is_none());

        // With tool choice pointing to a specific tool
        let config = convert_tool_config(Some(&tools), &Some(BedrockToolChoice::Tool { name: "search".to_string() }));
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config["toolChoice"]["tool"]["name"], "search");
    }

    #[test]
    fn test_build_system_prompt_with_caching() {
        let model = test_model();
        let result = build_system_prompt(Some("You are helpful"), &model, &CacheRetention::Short);
        assert!(result.is_some());
        let blocks = result.unwrap();
        let arr = blocks.as_array().unwrap();
        assert_eq!(arr.len(), 2); // text + cache point
        assert_eq!(arr[0]["text"], "You are helpful");
        assert!(arr[1].get("cachePoint").is_some());
    }

    #[test]
    fn test_build_system_prompt_no_caching() {
        let model = test_non_claude_model();
        let result = build_system_prompt(Some("You are helpful"), &model, &CacheRetention::None);
        assert!(result.is_some());
        let blocks = result.unwrap();
        let arr = blocks.as_array().unwrap();
        assert_eq!(arr.len(), 1); // Just the text block, no cache point
    }

    #[test]
    fn test_build_system_prompt_none() {
        let model = test_model();
        assert!(build_system_prompt(None, &model, &CacheRetention::Short).is_none());
        assert!(build_system_prompt(Some(""), &model, &CacheRetention::Short).is_none());
    }

    #[test]
    fn test_convert_messages_basic() {
        let model = test_model();
        let messages = vec![
            Message::User(UserMessage {
                content: UserContent::Text("Hello".to_string()),
                timestamp: 0,
            }),
        ];
        let result = convert_messages(&messages, &model, &CacheRetention::Short);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        let content = result[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["text"], "Hello");
        // Should have cache point appended (last user message of cacheable model)
        assert!(content.last().unwrap().get("cachePoint").is_some());
    }

    #[test]
    fn test_convert_messages_tool_results_aggregated() {
        let model = test_model();
        let messages = vec![
            Message::User(UserMessage {
                content: UserContent::Text("Search for rust".to_string()),
                timestamp: 0,
            }),
            Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::ToolCall(ToolCall {
                        id: "call_1".to_string(),
                        name: "search".to_string(),
                        arguments: json!({"query": "rust"}),
                        thought_signature: None,
                    }),
                    ContentBlock::ToolCall(ToolCall {
                        id: "call_2".to_string(),
                        name: "read".to_string(),
                        arguments: json!({"file": "main.rs"}),
                        thought_signature: None,
                    }),
                ],
                api: "bedrock-converse-stream".to_string(),
                provider: "amazon-bedrock".to_string(),
                model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 0,
            }),
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
        ];

        let result = convert_messages(&messages, &model, &CacheRetention::None);
        // Should be: user, assistant, user (aggregated tool results)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[1]["role"], "assistant");
        assert_eq!(result[2]["role"], "user");

        // The aggregated user message should contain both tool results
        let tool_results = result[2]["content"].as_array().unwrap();
        assert_eq!(tool_results.len(), 2);
        assert!(tool_results[0].get("toolResult").is_some());
        assert!(tool_results[1].get("toolResult").is_some());
    }

    #[test]
    fn test_convert_messages_thinking_with_signature_for_claude() {
        let model = test_model();
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(ThinkingContent {
                        thinking: "Let me think...".to_string(),
                        thinking_signature: Some("sig123".to_string()),
                    }),
                    ContentBlock::Text(TextContent {
                        text: "Here is my answer".to_string(),
                        text_signature: None,
                    }),
                ],
                api: "bedrock-converse-stream".to_string(),
                provider: "amazon-bedrock".to_string(),
                model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
        ];

        let result = convert_messages(&messages, &model, &CacheRetention::None);
        assert_eq!(result.len(), 1);
        let content = result[0]["content"].as_array().unwrap();
        // Should have reasoningContent block with signature
        let reasoning = &content[0]["reasoningContent"]["reasoningText"];
        assert_eq!(reasoning["text"], "Let me think...");
        assert_eq!(reasoning["signature"], "sig123");
    }

    #[test]
    fn test_build_additional_model_request_fields_adaptive() {
        let model = test_opus46_model();
        let options = BedrockOptions {
            base: StreamOptions::default(),
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: Some(ThinkingLevel::High),
            thinking_budgets: None,
            interleaved_thinking: false,
        };
        let result = build_additional_model_request_fields(&model, &options);
        assert!(result.is_some());
        let fields = result.unwrap();
        assert_eq!(fields["thinking"]["type"], "adaptive");
        assert_eq!(fields["output_config"]["effort"], "high");
    }

    #[test]
    fn test_build_additional_model_request_fields_extended() {
        let model = test_model();
        let options = BedrockOptions {
            base: StreamOptions::default(),
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: Some(ThinkingLevel::Medium),
            thinking_budgets: None,
            interleaved_thinking: false,
        };
        let result = build_additional_model_request_fields(&model, &options);
        assert!(result.is_some());
        let fields = result.unwrap();
        assert_eq!(fields["thinking"]["type"], "enabled");
        assert_eq!(fields["thinking"]["budget_tokens"], 8192);
    }

    #[test]
    fn test_build_additional_model_request_fields_none_for_non_reasoning() {
        let model = test_non_claude_model();
        let options = BedrockOptions {
            base: StreamOptions::default(),
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: Some(ThinkingLevel::High),
            thinking_budgets: None,
            interleaved_thinking: false,
        };
        let result = build_additional_model_request_fields(&model, &options);
        // Non-reasoning model should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_build_bedrock_url() {
        let url = build_bedrock_url("us-east-1", "anthropic.claude-3-5-sonnet-20241022-v2:0");
        assert!(url.starts_with("https://bedrock-runtime.us-east-1.amazonaws.com/model/"));
        assert!(url.ends_with("/converse-stream"));
        assert!(url.contains("anthropic.claude-3-5-sonnet-20241022-v2"));
    }

    #[test]
    fn test_sigv4_signing_produces_authorization() {
        let headers = sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse-stream",
            b"{}",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        )
        .unwrap();
        assert!(headers.contains_key("authorization"));
        assert!(headers["authorization"].starts_with("AWS4-HMAC-SHA256"));
        assert!(headers.contains_key("x-amz-date"));
        assert!(headers.contains_key("x-amz-content-sha256"));
    }

    #[test]
    fn test_sigv4_signing_with_session_token() {
        let headers = sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse-stream",
            b"{}",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            Some("session-token-value"),
        )
        .unwrap();
        assert!(headers.contains_key("x-amz-security-token"));
        assert_eq!(headers["x-amz-security-token"], "session-token-value");
        // Authorization should include the security token header in signed headers
        assert!(headers["authorization"].contains("x-amz-security-token"));
    }

    #[test]
    fn test_decode_event_stream_messages_empty() {
        let (messages, consumed) = decode_event_stream_messages(&[]);
        assert!(messages.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_decode_event_stream_messages_partial() {
        // Less than 12 bytes (minimum prelude)
        let (messages, consumed) = decode_event_stream_messages(&[0, 0, 0, 20]);
        assert!(messages.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_decode_event_headers_string_type() {
        // Build a header: name_len=4, name="test", type=7 (string), value_len=5, value="hello"
        let header_bytes: Vec<u8> = vec![
            4, // name length
            b't', b'e', b's', b't', // name
            7, // string type
            0, 5, // value length (big-endian)
            b'h', b'e', b'l', b'l', b'o', // value
        ];
        let headers = decode_event_headers(&header_bytes);
        assert_eq!(headers.get("test").unwrap(), "hello");
    }

    #[test]
    fn test_map_thinking_level_to_effort() {
        assert_eq!(map_thinking_level_to_effort(&ThinkingLevel::Minimal), "low");
        assert_eq!(map_thinking_level_to_effort(&ThinkingLevel::Low), "low");
        assert_eq!(map_thinking_level_to_effort(&ThinkingLevel::Medium), "medium");
        assert_eq!(map_thinking_level_to_effort(&ThinkingLevel::High), "high");
        assert_eq!(map_thinking_level_to_effort(&ThinkingLevel::Xhigh), "max");
    }

    #[test]
    fn test_create_image_block() {
        // Simple base64-encoded single pixel PNG (just test the structure)
        let data = base64::engine::general_purpose::STANDARD.encode(b"fake-png-data");
        let block = create_image_block("image/png", &data);
        assert_eq!(block["format"], "png");
        assert!(block["source"]["bytes"].is_string());
    }

    #[test]
    fn test_bedrock_provider_api_name() {
        let provider = BedrockProvider;
        assert_eq!(provider.api(), "bedrock-converse-stream");
    }

    #[test]
    fn test_resolve_cache_retention_default() {
        // Without env var, should default to Short
        let result = resolve_cache_retention(None);
        assert_eq!(result, CacheRetention::Short);
    }

    #[test]
    fn test_resolve_cache_retention_explicit() {
        assert_eq!(resolve_cache_retention(Some(&CacheRetention::None)), CacheRetention::None);
        assert_eq!(resolve_cache_retention(Some(&CacheRetention::Long)), CacheRetention::Long);
    }

    #[test]
    fn test_build_tool_result_block() {
        let tr = ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "search".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "Found results".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        };
        let block = build_tool_result_block(&tr);
        assert_eq!(block["toolResult"]["toolUseId"], "call_1");
        assert_eq!(block["toolResult"]["status"], "success");
        let content = block["toolResult"]["content"].as_array().unwrap();
        assert_eq!(content[0]["text"], "Found results");
    }

    #[test]
    fn test_build_tool_result_block_error() {
        let tr = ToolResultMessage {
            tool_call_id: "call_2".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "File not found".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: true,
            timestamp: 0,
        };
        let block = build_tool_result_block(&tr);
        assert_eq!(block["toolResult"]["status"], "error");
    }
}
