use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::supports_xhigh;
use crate::registry::ApiProvider;
use crate::simple_options::{build_base_options, clamp_reasoning};
use crate::sse::SseParser;

use super::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools,
    process_responses_events, ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions,
};

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const MAX_RETRIES: usize = 3;
const BASE_DELAY_MS: u64 = 1000;

fn codex_tool_call_providers() -> HashSet<&'static str> {
    ["openai", "openai-codex", "opencode"].into_iter().collect()
}

const CODEX_RESPONSE_STATUSES: &[&str] = &[
    "completed",
    "incomplete",
    "failed",
    "cancelled",
    "queued",
    "in_progress",
];

// =============================================================================
// Options
// =============================================================================

/// OpenAI Codex Responses-specific stream options.
#[derive(Debug, Clone)]
pub struct OpenAICodexResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<String>,   // "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
    pub reasoning_summary: Option<String>,  // "auto" | "concise" | "detailed" | "off" | "on" | null
    pub text_verbosity: Option<String>,     // "low" | "medium" | "high"
}

// =============================================================================
// Retry helpers
// =============================================================================

/// Check if an HTTP error status or message text indicates a retryable error.
fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 || status == 500 || status == 502 || status == 503 || status == 504 {
        return true;
    }
    let re = regex::Regex::new(r"(?i)rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused").unwrap();
    re.is_match(error_text)
}

/// Sleep for a given duration, respecting cancellation.
async fn sleep_with_cancel(ms: u64, cancel: &CancellationToken) -> Result<(), String> {
    tokio::select! {
        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
        _ = cancel.cancelled() => Err("Request was aborted".to_string()),
    }
}

// =============================================================================
// JWT / Auth helpers
// =============================================================================

/// Extract the account ID from a JWT token.
/// The token is expected to have the claim at `https://api.openai.com/auth`.
pub fn extract_account_id(token: &str) -> Result<String, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("Failed to extract accountId from token".to_string());
    }

    // Decode the payload (base64url)
    let payload_b64 = parts[1];
    // Add padding if needed
    let padded = match payload_b64.len() % 4 {
        2 => format!("{payload_b64}=="),
        3 => format!("{payload_b64}="),
        _ => payload_b64.to_string(),
    };
    // Replace URL-safe chars
    let standard_b64 = padded.replace('-', "+").replace('_', "/");

    let decoded_bytes = base64_decode(&standard_b64)
        .map_err(|_| "Failed to extract accountId from token".to_string())?;

    let payload_str = String::from_utf8(decoded_bytes)
        .map_err(|_| "Failed to extract accountId from token".to_string())?;

    let payload: Value = serde_json::from_str(&payload_str)
        .map_err(|_| "Failed to extract accountId from token".to_string())?;

    let account_id = payload
        .get(JWT_CLAIM_PATH)
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Failed to extract accountId from token".to_string())?;

    Ok(account_id.to_string())
}

/// Simple base64 decode without external crate dependency.
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in input.as_bytes() {
        if byte == b'=' {
            break;
        }
        let val = CHARS.iter().position(|&c| c == byte);
        let val = match val {
            Some(v) => v as u32,
            None => continue, // skip whitespace or invalid chars
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((buf >> bits) & 0xFF) as u8);
        }
    }

    Ok(output)
}

// =============================================================================
// URL resolution
// =============================================================================

/// Resolve the Codex API endpoint URL.
fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = match base_url {
        Some(url) if !url.trim().is_empty() => url.to_string(),
        _ => DEFAULT_CODEX_BASE_URL.to_string(),
    };
    let normalized = raw.trim_end_matches('/').to_string();
    if normalized.ends_with("/codex/responses") {
        normalized
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

// =============================================================================
// Reasoning effort clamping
// =============================================================================

/// Clamp reasoning effort for specific Codex model IDs.
fn clamp_reasoning_effort(model_id: &str, effort: &str) -> String {
    let id = if model_id.contains('/') {
        model_id.rsplit('/').next().unwrap_or(model_id)
    } else {
        model_id
    };

    if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3")) && effort == "minimal" {
        return "low".to_string();
    }
    if id == "gpt-5.1" && effort == "xhigh" {
        return "high".to_string();
    }
    if id == "gpt-5.1-codex-mini" {
        return if effort == "high" || effort == "xhigh" {
            "high".to_string()
        } else {
            "medium".to_string()
        };
    }
    effort.to_string()
}

// =============================================================================
// Codex SSE event mapping
// =============================================================================

/// Normalize a Codex response status string.
fn normalize_codex_status(status: &str) -> Option<String> {
    if CODEX_RESPONSE_STATUSES.contains(&status) {
        Some(status.to_string())
    } else {
        None
    }
}

/// Map raw Codex SSE events to standard Responses API events.
/// Handles `response.done` -> `response.completed` remapping,
/// error events, and `response.failed`.
fn map_codex_events(events: Vec<Value>) -> Result<Vec<Value>, String> {
    let mut mapped = Vec::new();

    for event in events {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "error" => {
                let code = event.get("code").and_then(|v| v.as_str()).unwrap_or("");
                let message = event.get("message").and_then(|v| v.as_str()).unwrap_or("");
                let error_detail = if !message.is_empty() {
                    message.to_string()
                } else if !code.is_empty() {
                    code.to_string()
                } else {
                    serde_json::to_string(&event).unwrap_or_default()
                };
                return Err(format!("Codex error: {error_detail}"));
            }
            "response.failed" => {
                let msg = event
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Codex response failed");
                return Err(msg.to_string());
            }
            "response.done" | "response.completed" => {
                let mut new_event = event.clone();
                new_event["type"] = json!("response.completed");

                // Normalize the response status
                if let Some(response) = new_event.get_mut("response") {
                    if let Some(status) = response.get("status").and_then(|v| v.as_str()) {
                        if let Some(normalized) = normalize_codex_status(status) {
                            response["status"] = json!(normalized);
                        } else {
                            // Remove invalid status
                            if let Some(obj) = response.as_object_mut() {
                                obj.remove("status");
                            }
                        }
                    }
                }

                mapped.push(new_event);
            }
            "" => {
                // Skip events without a type
            }
            _ => {
                mapped.push(event);
            }
        }
    }

    Ok(mapped)
}

// =============================================================================
// Error parsing
// =============================================================================

/// Parse an error response body for a user-friendly message.
fn parse_error_response(status: u16, body: &str) -> (String, Option<String>) {
    let mut message = if body.is_empty() {
        "Request failed".to_string()
    } else {
        body.to_string()
    };
    let mut friendly_message: Option<String> = None;

    if let Ok(parsed) = serde_json::from_str::<Value>(body) {
        if let Some(err) = parsed.get("error") {
            let code = err
                .get("code")
                .or_else(|| err.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let is_usage_limit = regex::Regex::new(
                r"(?i)usage_limit_reached|usage_not_included|rate_limit_exceeded",
            )
            .map(|re| re.is_match(code))
            .unwrap_or(false)
                || status == 429;

            if is_usage_limit {
                let plan = err
                    .get("plan_type")
                    .and_then(|v| v.as_str())
                    .map(|p| format!(" ({} plan)", p.to_lowercase()))
                    .unwrap_or_default();

                let resets_at = err.get("resets_at").and_then(|v| v.as_f64());
                let when = resets_at
                    .map(|r| {
                        let now_ms = chrono::Utc::now().timestamp_millis() as f64;
                        let mins = ((r * 1000.0 - now_ms) / 60000.0).round().max(0.0) as i64;
                        format!(" Try again in ~{mins} min.")
                    })
                    .unwrap_or_default();

                friendly_message =
                    Some(format!("You have hit your ChatGPT usage limit{plan}.{when}").trim().to_string());
            }

            if let Some(err_msg) = err.get("message").and_then(|v| v.as_str()) {
                message = err_msg.to_string();
            } else if let Some(fm) = &friendly_message {
                message = fm.clone();
            }
        }
    }

    (message, friendly_message)
}

// =============================================================================
// HTTP request building
// =============================================================================

fn build_request_body(
    model: &Model,
    context: &Context,
    options: Option<&OpenAICodexResponsesOptions>,
) -> Value {
    let providers = codex_tool_call_providers();
    let msg_opts = ConvertResponsesMessagesOptions {
        include_system_prompt: Some(false),
    };
    let messages = convert_responses_messages(model, context, &providers, Some(&msg_opts));

    let text_verbosity = options
        .and_then(|o| o.text_verbosity.as_deref())
        .unwrap_or("medium");

    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context.system_prompt,
        "input": messages,
        "text": {"verbosity": text_verbosity},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(opts) = options {
        if let Some(session_id) = &opts.base.session_id {
            body["prompt_cache_key"] = json!(session_id);
        }
        if let Some(temp) = opts.base.temperature {
            body["temperature"] = json!(temp);
        }
    }

    if let Some(tools) = &context.tools {
        let tool_opts = ConvertResponsesToolsOptions { strict: None };
        body["tools"] = json!(convert_responses_tools(tools, Some(&tool_opts)));
    }

    if let Some(opts) = options {
        if let Some(effort) = &opts.reasoning_effort {
            let clamped = clamp_reasoning_effort(&model.id, effort);
            let summary = opts.reasoning_summary.as_deref().unwrap_or("auto");
            body["reasoning"] = json!({
                "effort": clamped,
                "summary": summary,
            });
        }
    }

    body
}

fn build_codex_headers(
    init_headers: Option<&HashMap<String, String>>,
    additional_headers: Option<&HashMap<String, String>>,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();

    // Copy init headers (model.headers)
    if let Some(init) = init_headers {
        for (k, v) in init {
            headers.insert(k.clone(), v.clone());
        }
    }

    headers.insert("authorization".to_string(), format!("Bearer {token}"));
    headers.insert("chatgpt-account-id".to_string(), account_id.to_string());
    headers.insert("openai-beta".to_string(), "responses=experimental".to_string());
    headers.insert("originator".to_string(), "pi".to_string());

    // Build user agent (use std::env::consts for platform info)
    let user_agent = format!(
        "pi ({} {}; {})",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::consts::ARCH
    );
    headers.insert("user-agent".to_string(), user_agent);
    headers.insert("accept".to_string(), "text/event-stream".to_string());
    headers.insert("content-type".to_string(), "application/json".to_string());

    // Additional headers override
    if let Some(extra) = additional_headers {
        for (k, v) in extra {
            headers.insert(k.clone(), v.clone());
        }
    }

    if let Some(sid) = session_id {
        headers.insert("session_id".to_string(), sid.to_string());
    }

    headers
}

// =============================================================================
// Stream functions
// =============================================================================

/// Stream from OpenAI Codex Responses API using raw HTTP + SSE parsing with retry logic.
pub fn stream_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: &OpenAICodexResponsesOptions,
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
            output.error_message = Some(format!("No API key for provider: {}", model.provider));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        // Extract account ID from JWT
        let account_id = match extract_account_id(&api_key) {
            Ok(id) => id,
            Err(e) => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(e);
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        };

        let body = build_request_body(&model, &context, Some(&options));
        let body_json = serde_json::to_string(&body).unwrap_or_default();
        let headers = build_codex_headers(
            model.headers.as_ref(),
            options.base.headers.as_ref(),
            &account_id,
            &api_key,
            options.base.session_id.as_deref(),
        );
        let url = resolve_codex_url(Some(&model.base_url));

        // Retry logic
        let client = reqwest::Client::new();
        let mut response: Option<reqwest::Response> = None;
        let mut last_error: Option<String> = None;

        for attempt in 0..=MAX_RETRIES {
            if cancel.is_cancelled() {
                output.stop_reason = StopReason::Aborted;
                output.error_message = Some("Request was aborted".to_string());
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error: output,
                });
                return;
            }

            let mut request = client.post(&url);
            for (k, v) in &headers {
                request = request.header(k.as_str(), v.as_str());
            }
            request = request.body(body_json.clone());

            match request.send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        response = Some(resp);
                        break;
                    }

                    let status = resp.status().as_u16();
                    let error_text = resp.text().await.unwrap_or_default();

                    if attempt < MAX_RETRIES && is_retryable_error(status, &error_text) {
                        let delay_ms = BASE_DELAY_MS * 2u64.pow(attempt as u32);
                        if sleep_with_cancel(delay_ms, &cancel).await.is_err() {
                            output.stop_reason = StopReason::Aborted;
                            output.error_message = Some("Request was aborted".to_string());
                            stream_clone.push(AssistantMessageEvent::Error {
                                reason: StopReason::Aborted,
                                error: output,
                            });
                            return;
                        }
                        continue;
                    }

                    let (msg, friendly) = parse_error_response(status, &error_text);
                    last_error = Some(friendly.unwrap_or(msg));
                    break;
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("aborted") || err_msg.contains("Request was aborted") {
                        output.stop_reason = StopReason::Aborted;
                        output.error_message = Some("Request was aborted".to_string());
                        stream_clone.push(AssistantMessageEvent::Error {
                            reason: StopReason::Aborted,
                            error: output,
                        });
                        return;
                    }
                    last_error = Some(err_msg.clone());
                    // Network errors are retryable
                    if attempt < MAX_RETRIES && !err_msg.contains("usage limit") {
                        let delay_ms = BASE_DELAY_MS * 2u64.pow(attempt as u32);
                        if sleep_with_cancel(delay_ms, &cancel).await.is_err() {
                            output.stop_reason = StopReason::Aborted;
                            output.error_message = Some("Request was aborted".to_string());
                            stream_clone.push(AssistantMessageEvent::Error {
                                reason: StopReason::Aborted,
                                error: output,
                            });
                            return;
                        }
                        continue;
                    }
                    break;
                }
            }
        }

        let response = match response {
            Some(r) => r,
            None => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(last_error.unwrap_or_else(|| "Failed after retries".to_string()));
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        };

        stream_clone.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        // Process SSE stream
        let mut current_item: Option<super::openai_responses_shared::CurrentItem> = None;
        let mut current_block: Option<super::openai_responses_shared::CurrentBlock> = None;
        let mut sse_parser = SseParser::new();

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

            let events = sse_parser.feed(&chunk);
            let mut raw_events: Vec<Value> = Vec::new();
            for sse_event in events {
                let data_str = sse_event.data.trim();
                if data_str == "[DONE]" {
                    continue;
                }
                if let Ok(data) = serde_json::from_str::<Value>(data_str) {
                    raw_events.push(data);
                }
            }

            // Map Codex events to standard Responses events
            let mapped_events = match map_codex_events(raw_events) {
                Ok(events) => events,
                Err(err) => {
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(err);
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    });
                    return;
                }
            };

            if let Err(err) = process_responses_events(
                &mapped_events,
                &mut output,
                &stream_clone,
                &model,
                &mut current_item,
                &mut current_block,
                None,
            ) {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(err);
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
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
pub fn stream_simple_openai_codex_responses(
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

    let codex_opts = OpenAICodexResponsesOptions {
        base,
        reasoning_effort,
        reasoning_summary: None,
        text_verbosity: None,
    };

    stream_openai_codex_responses(model, context, &codex_opts, cancel)
}

// =============================================================================
// ApiProvider implementation
// =============================================================================

pub struct OpenAICodexResponsesProvider;

impl ApiProvider for OpenAICodexResponsesProvider {
    fn api(&self) -> &str {
        "openai-codex-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let codex_opts = OpenAICodexResponsesOptions {
            base: options.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            text_verbosity: None,
        };
        stream_openai_codex_responses(model, context, &codex_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_openai_codex_responses(model, context, options, cancel)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> Model {
        Model {
            id: "gpt-5.2-codex".to_string(),
            name: "GPT-5.2 Codex".to_string(),
            api: "openai-codex-responses".to_string(),
            provider: "openai-codex".to_string(),
            base_url: "https://chatgpt.com/backend-api".to_string(),
            reasoning: true,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 200000,
            max_tokens: 16384,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_resolve_codex_url_default() {
        assert_eq!(
            resolve_codex_url(None),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_resolve_codex_url_empty() {
        assert_eq!(
            resolve_codex_url(Some("")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_resolve_codex_url_custom() {
        assert_eq!(
            resolve_codex_url(Some("https://custom.api.com")),
            "https://custom.api.com/codex/responses"
        );
    }

    #[test]
    fn test_resolve_codex_url_already_codex() {
        assert_eq!(
            resolve_codex_url(Some("https://custom.api.com/codex")),
            "https://custom.api.com/codex/responses"
        );
    }

    #[test]
    fn test_resolve_codex_url_full_path() {
        assert_eq!(
            resolve_codex_url(Some("https://custom.api.com/codex/responses")),
            "https://custom.api.com/codex/responses"
        );
    }

    #[test]
    fn test_resolve_codex_url_trailing_slash() {
        assert_eq!(
            resolve_codex_url(Some("https://custom.api.com/")),
            "https://custom.api.com/codex/responses"
        );
    }

    #[test]
    fn test_clamp_reasoning_effort_gpt52_minimal() {
        assert_eq!(clamp_reasoning_effort("gpt-5.2", "minimal"), "low");
    }

    #[test]
    fn test_clamp_reasoning_effort_gpt53_minimal() {
        assert_eq!(clamp_reasoning_effort("gpt-5.3-codex", "minimal"), "low");
    }

    #[test]
    fn test_clamp_reasoning_effort_gpt51_xhigh() {
        assert_eq!(clamp_reasoning_effort("gpt-5.1", "xhigh"), "high");
    }

    #[test]
    fn test_clamp_reasoning_effort_gpt51_codex_mini_high() {
        assert_eq!(clamp_reasoning_effort("gpt-5.1-codex-mini", "high"), "high");
    }

    #[test]
    fn test_clamp_reasoning_effort_gpt51_codex_mini_low() {
        assert_eq!(clamp_reasoning_effort("gpt-5.1-codex-mini", "low"), "medium");
    }

    #[test]
    fn test_clamp_reasoning_effort_passthrough() {
        assert_eq!(clamp_reasoning_effort("gpt-4o", "high"), "high");
        assert_eq!(clamp_reasoning_effort("gpt-4o", "medium"), "medium");
    }

    #[test]
    fn test_clamp_reasoning_effort_with_org_prefix() {
        assert_eq!(clamp_reasoning_effort("org/gpt-5.2-codex", "minimal"), "low");
    }

    #[test]
    fn test_is_retryable_error_429() {
        assert!(is_retryable_error(429, ""));
    }

    #[test]
    fn test_is_retryable_error_500() {
        assert!(is_retryable_error(500, ""));
    }

    #[test]
    fn test_is_retryable_error_502() {
        assert!(is_retryable_error(502, ""));
    }

    #[test]
    fn test_is_retryable_error_text_match() {
        assert!(is_retryable_error(400, "rate limit exceeded"));
        assert!(is_retryable_error(400, "Service Unavailable"));
    }

    #[test]
    fn test_is_retryable_error_non_retryable() {
        assert!(!is_retryable_error(400, "bad request"));
        assert!(!is_retryable_error(401, "unauthorized"));
    }

    #[test]
    fn test_extract_account_id_invalid_token() {
        assert!(extract_account_id("not-a-jwt").is_err());
        assert!(extract_account_id("a.b").is_err());
    }

    #[test]
    fn test_extract_account_id_valid_token() {
        // Build a simple JWT with the expected claim
        let header = base64_encode(b"{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        let payload = base64_encode(
            br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acc_12345"}}"#,
        );
        let token = format!("{header}.{payload}.signature");
        let result = extract_account_id(&token);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "acc_12345");
    }

    #[test]
    fn test_extract_account_id_missing_claim() {
        let header = base64_encode(b"{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        let payload = base64_encode(b"{\"some_other_claim\":\"value\"}");
        let token = format!("{header}.{payload}.signature");
        let result = extract_account_id(&token);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_codex_status() {
        assert_eq!(normalize_codex_status("completed"), Some("completed".to_string()));
        assert_eq!(normalize_codex_status("in_progress"), Some("in_progress".to_string()));
        assert_eq!(normalize_codex_status("unknown"), None);
    }

    #[test]
    fn test_map_codex_events_done_to_completed() {
        let events = vec![json!({"type": "response.done", "response": {"status": "completed", "usage": {}}})];
        let mapped = map_codex_events(events).unwrap();
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0]["type"], "response.completed");
    }

    #[test]
    fn test_map_codex_events_error() {
        let events = vec![json!({"type": "error", "code": "500", "message": "Server error"})];
        let result = map_codex_events(events);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Server error"));
    }

    #[test]
    fn test_map_codex_events_failed() {
        let events = vec![json!({"type": "response.failed", "response": {"error": {"message": "Out of tokens"}}})];
        let result = map_codex_events(events);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Out of tokens"));
    }

    #[test]
    fn test_map_codex_events_passthrough() {
        let events = vec![
            json!({"type": "response.output_item.added", "item": {"type": "message"}}),
            json!({"type": "response.output_text.delta", "delta": "hello"}),
        ];
        let mapped = map_codex_events(events).unwrap();
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0]["type"], "response.output_item.added");
    }

    #[test]
    fn test_parse_error_response_plain_text() {
        let (msg, friendly) = parse_error_response(400, "Bad request");
        assert_eq!(msg, "Bad request");
        assert!(friendly.is_none());
    }

    #[test]
    fn test_parse_error_response_rate_limit() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","message":"Rate limit exceeded","plan_type":"Plus"}}"#;
        let (msg, friendly) = parse_error_response(429, body);
        assert_eq!(msg, "Rate limit exceeded");
        assert!(friendly.is_some());
        assert!(friendly.unwrap().contains("usage limit"));
    }

    #[test]
    fn test_parse_error_response_empty() {
        let (msg, friendly) = parse_error_response(500, "");
        assert_eq!(msg, "Request failed");
        assert!(friendly.is_none());
    }

    #[test]
    fn test_base64_decode_simple() {
        let result = base64_decode("SGVsbG8gV29ybGQ=").unwrap();
        assert_eq!(String::from_utf8(result).unwrap(), "Hello World");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        let result = base64_decode("SGVsbG8").unwrap();
        assert_eq!(String::from_utf8(result).unwrap(), "Hello");
    }

    #[test]
    fn test_build_request_body() {
        let model = test_model();
        let context = Context {
            system_prompt: Some("You are a coding assistant.".to_string()),
            messages: vec![],
            tools: None,
        };
        let opts = OpenAICodexResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: Some("high".to_string()),
            reasoning_summary: Some("auto".to_string()),
            text_verbosity: Some("medium".to_string()),
        };
        let body = build_request_body(&model, &context, Some(&opts));
        assert_eq!(body["model"], "gpt-5.2-codex");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["text"]["verbosity"], "medium");
        assert!(body.get("reasoning").is_some());
    }

    /// Simple base64 encode for test fixtures.
    fn base64_encode(input: &[u8]) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut result = String::new();
        let mut i = 0;
        while i < input.len() {
            let b0 = input[i] as u32;
            let b1 = if i + 1 < input.len() { input[i + 1] as u32 } else { 0 };
            let b2 = if i + 2 < input.len() { input[i + 2] as u32 } else { 0 };
            let triple = (b0 << 16) | (b1 << 8) | b2;

            result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
            result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

            if i + 1 < input.len() {
                result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }

            if i + 2 < input.len() {
                result.push(CHARS[(triple & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }

            i += 3;
        }
        result
    }
}
