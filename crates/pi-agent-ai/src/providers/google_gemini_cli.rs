/// Google Gemini CLI / Antigravity provider.
///
/// Shared implementation for both `google-gemini-cli` and `google-antigravity` providers.
/// Uses the Cloud Code Assist API endpoint to access Gemini and Claude models.
///
/// Key features:
/// - Dual-endpoint support (production CloudCode API + Antigravity sandbox)
/// - OAuth credential handling (JSON-encoded apiKey with token + projectId)
/// - Retry logic with endpoint fallback and exponential backoff
/// - Empty stream retry handling
/// - Advanced retry delay extraction from headers and error messages
/// - Claude model special handling (thinking models get beta header)
/// - Antigravity system instruction injection
/// - Manual SSE stream parsing
/// - Tool call ID generation with uniqueness checking
use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::StreamExt;
use regex::Regex;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use pi_agent_core::sanitize::sanitize_surrogates;
use pi_agent_core::transform::transform_messages;
use pi_agent_core::types::*;

use crate::models::calculate_cost;
use crate::registry::ApiProvider;
use crate::simple_options::{adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning};

// ---------- Constants ----------

const DEFAULT_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const ANTIGRAVITY_DAILY_ENDPOINT: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";

/// Headers for Gemini CLI (prod endpoint)
fn gemini_cli_headers() -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert(
        "User-Agent".to_string(),
        "google-cloud-sdk vscode_cloudshelleditor/0.1".to_string(),
    );
    headers.insert(
        "X-Goog-Api-Client".to_string(),
        "gl-node/22.17.0".to_string(),
    );
    headers.insert(
        "Client-Metadata".to_string(),
        json!({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        })
        .to_string(),
    );
    headers
}

const DEFAULT_ANTIGRAVITY_VERSION: &str = "1.15.8";

/// Headers for Antigravity (sandbox endpoint)
fn antigravity_headers() -> HashMap<String, String> {
    let version = std::env::var("PI_AI_ANTIGRAVITY_VERSION")
        .unwrap_or_else(|_| DEFAULT_ANTIGRAVITY_VERSION.to_string());
    let mut headers = HashMap::new();
    headers.insert(
        "User-Agent".to_string(),
        format!("antigravity/{version} darwin/arm64"),
    );
    headers.insert(
        "X-Goog-Api-Client".to_string(),
        "google-cloud-sdk vscode_cloudshelleditor/0.1".to_string(),
    );
    headers.insert(
        "Client-Metadata".to_string(),
        json!({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        })
        .to_string(),
    );
    headers
}

/// Antigravity system instruction (compact version from CLIProxyAPI).
const ANTIGRAVITY_SYSTEM_INSTRUCTION: &str = concat!(
    "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.",
    "You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.",
    "**Absolute paths only**",
    "**Proactiveness**"
);

// Counter for generating unique tool call IDs
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

// Retry configuration
const MAX_RETRIES: u32 = 3;
const BASE_DELAY_MS: u64 = 1000;
const MAX_EMPTY_STREAM_RETRIES: u32 = 2;
const EMPTY_STREAM_BASE_DELAY_MS: u64 = 500;
const CLAUDE_THINKING_BETA_HEADER: &str = "interleaved-thinking-2025-05-14";

// ---------- GoogleGeminiCliOptions ----------

#[derive(Debug, Clone)]
pub struct GoogleGeminiCliOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<String>,
    pub thinking_enabled: bool,
    pub thinking_budget_tokens: Option<i64>,
    pub thinking_level: Option<String>,
    pub project_id: Option<String>,
}

// ---------- Shared helpers (from google-shared, private in google.rs) ----------

fn is_thinking_part(part: &Value) -> bool {
    part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false)
}

fn retain_thought_signature(existing: Option<&str>, incoming: Option<&str>) -> Option<String> {
    if let Some(s) = incoming {
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    existing.map(|s| s.to_string())
}

fn base64_signature_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9+/]+={0,2}$").unwrap())
}

fn is_valid_thought_signature(signature: &str) -> bool {
    if signature.is_empty() || signature.len() % 4 != 0 {
        return false;
    }
    base64_signature_regex().is_match(signature)
}

fn resolve_thought_signature(is_same: bool, signature: Option<&str>) -> Option<String> {
    match signature {
        Some(sig) if is_same && is_valid_thought_signature(sig) => Some(sig.to_string()),
        _ => None,
    }
}

fn requires_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

fn map_stop_reason_string(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

fn map_tool_choice(choice: &str) -> &'static str {
    match choice {
        "auto" => "AUTO",
        "none" => "NONE",
        "any" => "ANY",
        _ => "AUTO",
    }
}

fn normalize_tool_call_id(id: &str) -> String {
    let normalized: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if normalized.len() > 64 { normalized[..64].to_string() } else { normalized }
}

fn normalize_tool_call_id_for_transform(id: &str, _model: &Model, _source: &AssistantMessage) -> String {
    normalize_tool_call_id(id)
}

// ---------- Convert messages ----------

fn convert_messages(model: &Model, context: &Context) -> Vec<Value> {
    let transformed = transform_messages(&context.messages, model, Some(&normalize_tool_call_id_for_transform));
    let mut contents: Vec<Value> = Vec::new();

    for msg in &transformed {
        match msg {
            Message::User(user_msg) => match &user_msg.content {
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
                            ContentBlock::Text(t) => Some(json!({ "text": sanitize_surrogates(&t.text) })),
                            ContentBlock::Image(img) if model.input.contains(&"image".to_string()) => {
                                Some(json!({ "inlineData": { "mimeType": img.mime_type, "data": img.data } }))
                            }
                            _ => None,
                        })
                        .collect();
                    let filtered = if !model.input.contains(&"image".to_string()) {
                        parts.into_iter().filter(|p| p.get("text").is_some()).collect()
                    } else {
                        parts
                    };
                    if !filtered.is_empty() {
                        contents.push(json!({ "role": "user", "parts": filtered }));
                    }
                }
            },
            Message::Assistant(a) => {
                let mut parts: Vec<Value> = Vec::new();
                let same = a.provider == model.provider && a.model == model.id;
                for block in &a.content {
                    match block {
                        ContentBlock::Text(t) => {
                            if t.text.trim().is_empty() { continue; }
                            let sig = resolve_thought_signature(same, t.text_signature.as_deref());
                            let mut part = json!({ "text": sanitize_surrogates(&t.text) });
                            if let Some(s) = sig { part["thoughtSignature"] = json!(s); }
                            parts.push(part);
                        }
                        ContentBlock::Thinking(t) => {
                            if t.thinking.trim().is_empty() { continue; }
                            if same {
                                let sig = resolve_thought_signature(same, t.thinking_signature.as_deref());
                                let mut part = json!({ "thought": true, "text": sanitize_surrogates(&t.thinking) });
                                if let Some(s) = sig { part["thoughtSignature"] = json!(s); }
                                parts.push(part);
                            } else {
                                parts.push(json!({ "text": sanitize_surrogates(&t.thinking) }));
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            let sig = resolve_thought_signature(same, tc.thought_signature.as_deref());
                            let is_gemini3 = model.id.to_lowercase().contains("gemini-3");
                            if is_gemini3 && sig.is_none() {
                                let args_str = serde_json::to_string_pretty(&tc.arguments).unwrap_or_else(|_| "{}".to_string());
                                parts.push(json!({
                                    "text": format!(
                                        "[Historical context: a different model called tool \"{}\" with arguments: {}. Do not mimic this format - use proper function calling.]",
                                        tc.name, args_str
                                    )
                                }));
                            } else {
                                let mut fc = json!({ "name": tc.name, "args": tc.arguments });
                                if requires_tool_call_id(&model.id) { fc["id"] = json!(tc.id); }
                                let mut part = json!({ "functionCall": fc });
                                if let Some(s) = sig { part["thoughtSignature"] = json!(s); }
                                parts.push(part);
                            }
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({ "role": "model", "parts": parts }));
                }
            }
            Message::ToolResult(tr) => {
                let text_result: String = tr.content.iter()
                    .filter_map(|c| c.as_text())
                    .map(|c| c.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let image_content: Vec<&ImageContent> = if model.input.contains(&"image".to_string()) {
                    tr.content.iter().filter_map(|c| c.as_image()).collect()
                } else {
                    Vec::new()
                };
                let has_text = !text_result.is_empty();
                let has_images = !image_content.is_empty();
                let supports_multimodal = model.id.contains("gemini-3");
                let response_value = if has_text {
                    sanitize_surrogates(&text_result)
                } else if has_images {
                    "(see attached image)".to_string()
                } else {
                    String::new()
                };
                let image_parts: Vec<Value> = image_content.iter()
                    .map(|img| json!({ "inlineData": { "mimeType": img.mime_type, "data": img.data } }))
                    .collect();
                let response_obj = if tr.is_error { json!({ "error": response_value }) } else { json!({ "output": response_value }) };
                let mut fn_resp = json!({ "name": tr.tool_name, "response": response_obj });
                if has_images && supports_multimodal { fn_resp["parts"] = json!(image_parts); }
                if requires_tool_call_id(&model.id) { fn_resp["id"] = json!(tr.tool_call_id); }
                let fn_resp_part = json!({ "functionResponse": fn_resp });

                let should_merge = contents.last()
                    .map(|last| {
                        last.get("role").and_then(|r| r.as_str()) == Some("user")
                            && last.get("parts").and_then(|p| p.as_array())
                                .map(|arr| arr.iter().any(|p| p.get("functionResponse").is_some()))
                                .unwrap_or(false)
                    })
                    .unwrap_or(false);

                if should_merge {
                    if let Some(last) = contents.last_mut() {
                        if let Some(parts) = last.get_mut("parts").and_then(|p| p.as_array_mut()) {
                            parts.push(fn_resp_part);
                        }
                    }
                } else {
                    contents.push(json!({ "role": "user", "parts": [fn_resp_part] }));
                }

                if has_images && !supports_multimodal {
                    let mut img_parts = vec![json!({ "text": "Tool result image:" })];
                    img_parts.extend(image_parts);
                    contents.push(json!({ "role": "user", "parts": img_parts }));
                }
            }
        }
    }
    contents
}

fn convert_tools(tools: &[Tool], use_parameters: bool) -> Value {
    if tools.is_empty() { return json!(null); }
    let key = if use_parameters { "parameters" } else { "parametersJsonSchema" };
    json!([{
        "functionDeclarations": tools.iter().map(|tool| {
            json!({ "name": tool.name, "description": tool.description, key: tool.parameters })
        }).collect::<Vec<Value>>()
    }])
}

// ---------- Retry helpers ----------

/// Extract retry delay from Gemini error response (in milliseconds).
pub fn extract_retry_delay(error_text: &str, headers: Option<&reqwest::header::HeaderMap>) -> Option<u64> {
    let normalize_delay = |ms: f64| -> Option<u64> {
        if ms > 0.0 { Some((ms + 1000.0).ceil() as u64) } else { None }
    };

    if let Some(hdrs) = headers {
        if let Some(retry_after) = hdrs.get("retry-after").and_then(|v| v.to_str().ok()) {
            if let Ok(secs) = retry_after.parse::<f64>() {
                if secs.is_finite() {
                    if let Some(d) = normalize_delay(secs * 1000.0) { return Some(d); }
                }
            }
            if let Ok(date) = chrono::DateTime::parse_from_rfc2822(retry_after) {
                let diff = (date.signed_duration_since(chrono::Utc::now())).num_milliseconds() as f64;
                if let Some(d) = normalize_delay(diff) { return Some(d); }
            }
        }
        if let Some(reset) = hdrs.get("x-ratelimit-reset").and_then(|v| v.to_str().ok()) {
            if let Ok(reset_secs) = reset.parse::<i64>() {
                let diff = (reset_secs * 1000 - chrono::Utc::now().timestamp_millis()) as f64;
                if let Some(d) = normalize_delay(diff) { return Some(d); }
            }
        }
        if let Some(ra) = hdrs.get("x-ratelimit-reset-after").and_then(|v| v.to_str().ok()) {
            if let Ok(secs) = ra.parse::<f64>() {
                if secs.is_finite() {
                    if let Some(d) = normalize_delay(secs * 1000.0) { return Some(d); }
                }
            }
        }
    }

    // Pattern 1: "reset after [Xh][Xm]Xs"
    fn re1() -> &'static Regex {
        use std::sync::OnceLock;
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s").unwrap())
    }
    if let Some(c) = re1().captures(error_text) {
        let h: f64 = c.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let m: f64 = c.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        if let Some(sm) = c.get(3) {
            if let Ok(s) = sm.as_str().parse::<f64>() {
                if !s.is_nan() {
                    if let Some(d) = normalize_delay(((h * 60.0 + m) * 60.0 + s) * 1000.0) { return Some(d); }
                }
            }
        }
    }

    // Pattern 2: "Please retry in X[ms|s]"
    fn re2() -> &'static Regex {
        use std::sync::OnceLock;
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)Please retry in ([0-9.]+)(ms|s)").unwrap())
    }
    if let Some(c) = re2().captures(error_text) {
        if let (Some(v), Some(u)) = (c.get(1), c.get(2)) {
            if let Ok(val) = v.as_str().parse::<f64>() {
                if !val.is_nan() && val > 0.0 {
                    let ms = if u.as_str().eq_ignore_ascii_case("ms") { val } else { val * 1000.0 };
                    if let Some(d) = normalize_delay(ms) { return Some(d); }
                }
            }
        }
    }

    // Pattern 3: "retryDelay": "Xs"
    fn re3() -> &'static Regex {
        use std::sync::OnceLock;
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r#"(?i)"retryDelay":\s*"([0-9.]+)(ms|s)""#).unwrap())
    }
    if let Some(c) = re3().captures(error_text) {
        if let (Some(v), Some(u)) = (c.get(1), c.get(2)) {
            if let Ok(val) = v.as_str().parse::<f64>() {
                if !val.is_nan() && val > 0.0 {
                    let ms = if u.as_str().eq_ignore_ascii_case("ms") { val } else { val * 1000.0 };
                    if let Some(d) = normalize_delay(ms) { return Some(d); }
                }
            }
        }
    }

    None
}

fn is_claude_thinking_model(model_id: &str) -> bool {
    let n = model_id.to_lowercase();
    n.contains("claude") && n.contains("thinking")
}

fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if matches!(status, 429 | 500 | 502 | 503 | 504) { return true; }
    fn re() -> &'static Regex {
        use std::sync::OnceLock;
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed").unwrap()
        })
    }
    re().is_match(error_text)
}

fn extract_error_message(error_text: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<Value>(error_text) {
        if let Some(msg) = parsed.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
            return msg.to_string();
        }
    }
    error_text.to_string()
}

async fn cancellable_sleep(ms: u64, cancel: &CancellationToken) -> Result<(), String> {
    tokio::select! {
        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
        _ = cancel.cancelled() => Err("Request was aborted".to_string()),
    }
}

// ---------- Build request ----------

pub fn build_request(
    model: &Model,
    context: &Context,
    project_id: &str,
    options: &GoogleGeminiCliOptions,
    is_antigravity: bool,
) -> Value {
    let contents = convert_messages(model, context);

    let mut generation_config = json!({});
    if let Some(temp) = options.base.temperature { generation_config["temperature"] = json!(temp); }
    if let Some(mt) = options.base.max_tokens { generation_config["maxOutputTokens"] = json!(mt); }

    if options.thinking_enabled && model.reasoning {
        let mut tc = json!({ "includeThoughts": true });
        if let Some(level) = &options.thinking_level { tc["thinkingLevel"] = json!(level); }
        else if let Some(budget) = options.thinking_budget_tokens { tc["thinkingBudget"] = json!(budget); }
        generation_config["thinkingConfig"] = tc;
    }

    let mut request = json!({ "contents": contents });
    if let Some(sid) = &options.base.session_id { request["sessionId"] = json!(sid); }
    if let Some(sp) = &context.system_prompt {
        request["systemInstruction"] = json!({ "parts": [{ "text": sanitize_surrogates(sp) }] });
    }
    if generation_config.as_object().map_or(false, |o| !o.is_empty()) {
        request["generationConfig"] = generation_config;
    }

    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            let use_params = model.id.starts_with("claude-");
            let tv = convert_tools(tools, use_params);
            if !tv.is_null() { request["tools"] = tv; }
            if let Some(choice) = &options.tool_choice {
                request["toolConfig"] = json!({ "functionCallingConfig": { "mode": map_tool_choice(choice) } });
            }
        }
    }

    if is_antigravity {
        let existing = request.get("systemInstruction")
            .and_then(|si| si.get("parts"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();
        let mut parts = vec![
            json!({ "text": ANTIGRAVITY_SYSTEM_INSTRUCTION }),
            json!({ "text": format!("Please ignore following [ignore]{}[/ignore]", ANTIGRAVITY_SYSTEM_INSTRUCTION) }),
        ];
        parts.extend(existing);
        request["systemInstruction"] = json!({ "role": "user", "parts": parts });
    }

    let now = chrono::Utc::now().timestamp_millis();
    let counter = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let request_id = format!(
        "{}-{}-{:x}",
        if is_antigravity { "agent" } else { "pi" },
        now,
        now as u64 ^ counter,
    );

    json!({
        "project": project_id,
        "model": model.id,
        "request": request,
        "requestType": if is_antigravity { "agent" } else { "" },
        "userAgent": if is_antigravity { "antigravity" } else { "pi-coding-agent" },
        "requestId": request_id,
    })
}

// ---------- Thinking level helpers ----------

fn get_gemini_cli_thinking_level(effort: &ThinkingLevel, model_id: &str) -> String {
    if model_id.contains("3-pro") {
        return match effort {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "LOW",
            ThinkingLevel::Medium | ThinkingLevel::High | ThinkingLevel::Xhigh => "HIGH",
        }.to_string();
    }
    match effort {
        ThinkingLevel::Minimal => "MINIMAL",
        ThinkingLevel::Low => "LOW",
        ThinkingLevel::Medium => "MEDIUM",
        ThinkingLevel::High | ThinkingLevel::Xhigh => "HIGH",
    }.to_string()
}

// ---------- Stream response processing ----------

/// Process an SSE stream response (takes ownership). Returns Ok(true) if content was received.
async fn stream_response_owned(
    response: reqwest::Response,
    model: &Model,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    cancel: &CancellationToken,
) -> Result<bool, String> {
    let mut has_content = false;
    let mut started = false;

    enum CurrentBlock {
        Text { text: String, text_signature: Option<String> },
        Thinking { thinking: String, thinking_signature: Option<String> },
    }

    let mut current_block: Option<CurrentBlock> = None;
    let block_index = |output: &AssistantMessage| -> usize {
        if output.content.is_empty() { 0 } else { output.content.len() - 1 }
    };

    let mut sse_buffer = String::new();
    let mut byte_stream = response.bytes_stream();

    loop {
        let chunk_result = tokio::select! {
            chunk = byte_stream.next() => {
                match chunk {
                    Some(r) => r,
                    None => break, // stream ended
                }
            },
            _ = cancel.cancelled() => {
                return Err("Request was aborted".to_string());
            }
        };

        let chunk = match chunk_result {
            Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
            Err(e) => return Err(format!("Stream error: {e}")),
        };

        sse_buffer.push_str(&chunk);
        let lines: Vec<String> = {
            let parts: Vec<&str> = sse_buffer.split('\n').collect();
            let remainder = parts.last().cloned().unwrap_or("");
            let complete: Vec<String> = parts[..parts.len().saturating_sub(1)]
                .iter().map(|s| s.to_string()).collect();
            sse_buffer = remainder.to_string();
            complete
        };

        for line in &lines {
            if !line.starts_with("data:") { continue; }
            let json_str = line[5..].trim();
            if json_str.is_empty() { continue; }

            let chunk_data: Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Cloud Code Assist wraps in { "response": { ... } }
            let response_data = match chunk_data.get("response") {
                Some(r) => r,
                None => continue,
            };

            if let Some(candidate) = response_data.get("candidates")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
            {
                if let Some(parts) = candidate.get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|p| p.as_array())
                {
                    for part in parts {
                        // Handle text parts
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            has_content = true;
                            let is_thinking = is_thinking_part(part);
                            let thought_sig = part.get("thoughtSignature").and_then(|s| s.as_str());

                            let needs_new = match &current_block {
                                None => true,
                                Some(CurrentBlock::Text { .. }) if is_thinking => true,
                                Some(CurrentBlock::Thinking { .. }) if !is_thinking => true,
                                _ => false,
                            };

                            if needs_new {
                                // End current block
                                if let Some(ref cb) = current_block {
                                    let ci = block_index(output);
                                    match cb {
                                        CurrentBlock::Text { text, .. } => {
                                            stream.push(AssistantMessageEvent::TextEnd {
                                                content_index: ci, content: text.clone(), partial: output.clone(),
                                            });
                                        }
                                        CurrentBlock::Thinking { thinking, .. } => {
                                            stream.push(AssistantMessageEvent::ThinkingEnd {
                                                content_index: ci, content: thinking.clone(), partial: output.clone(),
                                            });
                                        }
                                    }
                                }
                                // Start new block
                                if is_thinking {
                                    current_block = Some(CurrentBlock::Thinking { thinking: String::new(), thinking_signature: None });
                                    output.content.push(ContentBlock::Thinking(ThinkingContent { thinking: String::new(), thinking_signature: None }));
                                    if !started { stream.push(AssistantMessageEvent::Start { partial: output.clone() }); started = true; }
                                    stream.push(AssistantMessageEvent::ThinkingStart { content_index: block_index(output), partial: output.clone() });
                                } else {
                                    current_block = Some(CurrentBlock::Text { text: String::new(), text_signature: None });
                                    output.content.push(ContentBlock::Text(TextContent { text: String::new(), text_signature: None }));
                                    if !started { stream.push(AssistantMessageEvent::Start { partial: output.clone() }); started = true; }
                                    stream.push(AssistantMessageEvent::TextStart { content_index: block_index(output), partial: output.clone() });
                                }
                            }

                            let ci = block_index(output);
                            match &mut current_block {
                                Some(CurrentBlock::Thinking { thinking, thinking_signature }) => {
                                    thinking.push_str(text);
                                    *thinking_signature = retain_thought_signature(thinking_signature.as_deref(), thought_sig);
                                    if let Some(ContentBlock::Thinking(t)) = output.content.get_mut(ci) {
                                        t.thinking.push_str(text);
                                        t.thinking_signature = thinking_signature.clone();
                                    }
                                    stream.push(AssistantMessageEvent::ThinkingDelta { content_index: ci, delta: text.to_string(), partial: output.clone() });
                                }
                                Some(CurrentBlock::Text { text: bt, text_signature }) => {
                                    bt.push_str(text);
                                    *text_signature = retain_thought_signature(text_signature.as_deref(), thought_sig);
                                    if let Some(ContentBlock::Text(t)) = output.content.get_mut(ci) {
                                        t.text.push_str(text);
                                        t.text_signature = text_signature.clone();
                                    }
                                    stream.push(AssistantMessageEvent::TextDelta { content_index: ci, delta: text.to_string(), partial: output.clone() });
                                }
                                _ => {}
                            }
                        }

                        // Handle function calls
                        if let Some(fc) = part.get("functionCall") {
                            has_content = true;
                            if let Some(ref cb) = current_block {
                                let ci = block_index(output);
                                match cb {
                                    CurrentBlock::Text { text, .. } => {
                                        stream.push(AssistantMessageEvent::TextEnd { content_index: ci, content: text.clone(), partial: output.clone() });
                                    }
                                    CurrentBlock::Thinking { thinking, .. } => {
                                        stream.push(AssistantMessageEvent::ThinkingEnd { content_index: ci, content: thinking.clone(), partial: output.clone() });
                                    }
                                }
                                current_block = None;
                            }

                            let fc_name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                            let fc_args = fc.get("args").cloned().unwrap_or(json!({}));
                            let fc_id = fc.get("id").and_then(|i| i.as_str()).map(|s| s.to_string());
                            let sig = part.get("thoughtSignature").and_then(|s| s.as_str()).map(|s| s.to_string());

                            let needs_new_id = match &fc_id {
                                None => true,
                                Some(id) => output.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(tc) if tc.id == *id)),
                            };
                            let tool_call_id = if needs_new_id {
                                let ctr = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
                                format!("{}_{}_{}",fc_name, chrono::Utc::now().timestamp_millis(), ctr)
                            } else {
                                fc_id.unwrap()
                            };

                            let tool_call = ToolCall { id: tool_call_id, name: fc_name, arguments: fc_args.clone(), thought_signature: sig };
                            output.content.push(ContentBlock::ToolCall(tool_call.clone()));
                            if !started { stream.push(AssistantMessageEvent::Start { partial: output.clone() }); started = true; }
                            let ci = block_index(output);
                            stream.push(AssistantMessageEvent::ToolCallStart { content_index: ci, partial: output.clone() });
                            stream.push(AssistantMessageEvent::ToolCallDelta { content_index: ci, delta: serde_json::to_string(&fc_args).unwrap_or_default(), partial: output.clone() });
                            stream.push(AssistantMessageEvent::ToolCallEnd { content_index: ci, tool_call, partial: output.clone() });
                        }
                    }
                }

                if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str()) {
                    output.stop_reason = map_stop_reason_string(reason);
                    if output.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) {
                        output.stop_reason = StopReason::ToolUse;
                    }
                }
            }

            // Usage metadata
            if let Some(usage) = response_data.get("usageMetadata") {
                let prompt = usage.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                let candidates = usage.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                let thoughts = usage.get("thoughtsTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                let cached = usage.get("cachedContentTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                let total = usage.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                output.usage = Usage {
                    input: prompt.saturating_sub(cached),
                    output: candidates + thoughts,
                    cache_read: cached,
                    cache_write: 0,
                    total_tokens: total,
                    cost: UsageCost::default(),
                };
                calculate_cost(model, &mut output.usage);
            }
        }
    }

    // End final block
    if let Some(ref cb) = current_block {
        let ci = block_index(output);
        match cb {
            CurrentBlock::Text { text, .. } => {
                stream.push(AssistantMessageEvent::TextEnd { content_index: ci, content: text.clone(), partial: output.clone() });
            }
            CurrentBlock::Thinking { thinking, .. } => {
                stream.push(AssistantMessageEvent::ThinkingEnd { content_index: ci, content: thinking.clone(), partial: output.clone() });
            }
        }
    }

    Ok(has_content)
}

// ---------- Main stream function ----------

/// Stream from Google Cloud Code Assist API (Gemini CLI / Antigravity).
pub fn stream_google_gemini_cli(
    model: &Model,
    context: &Context,
    options: &GoogleGeminiCliOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();
    let stream_clone = stream.clone();

    tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&model);
        match run_stream(&model, &context, &options, &cancel, &stream_clone, &mut output).await {
            Ok(()) => { /* done event already pushed */ }
            Err(msg) => {
                output.stop_reason = if cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error };
                output.error_message = Some(msg);
                stream_clone.push(AssistantMessageEvent::Error { reason: output.stop_reason.clone(), error: output });
            }
        }
    });

    stream
}

/// Core streaming logic with retry handling.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &GoogleGeminiCliOptions,
    cancel: &CancellationToken,
    stream: &AssistantMessageEventStream,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    let api_key_raw = options.base.api_key.as_deref()
        .ok_or("Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.")?;

    #[derive(serde::Deserialize)]
    struct Creds { token: String, #[serde(rename = "projectId")] project_id: String }

    let creds: Creds = serde_json::from_str(api_key_raw)
        .map_err(|_| "Invalid Google Cloud Code Assist credentials. Use /login to re-authenticate.".to_string())?;
    if creds.token.is_empty() || creds.project_id.is_empty() {
        return Err("Missing token or projectId in Google Cloud credentials. Use /login to re-authenticate.".to_string());
    }

    let is_antigravity = model.provider == known_provider::GOOGLE_ANTIGRAVITY;
    let base_url = model.base_url.trim();
    let endpoints: Vec<String> = if !base_url.is_empty() && base_url != DEFAULT_ENDPOINT && base_url != ANTIGRAVITY_DAILY_ENDPOINT {
        vec![base_url.to_string()]
    } else if is_antigravity {
        vec![ANTIGRAVITY_DAILY_ENDPOINT.to_string(), DEFAULT_ENDPOINT.to_string()]
    } else {
        vec![DEFAULT_ENDPOINT.to_string()]
    };

    let request_body = build_request(model, context, &creds.project_id, options, is_antigravity);
    let provider_hdrs = if is_antigravity { antigravity_headers() } else { gemini_cli_headers() };

    let mut req_headers: HashMap<String, String> = HashMap::new();
    req_headers.insert("Authorization".to_string(), format!("Bearer {}", creds.token));
    req_headers.insert("Content-Type".to_string(), "application/json".to_string());
    req_headers.insert("Accept".to_string(), "text/event-stream".to_string());
    for (k, v) in &provider_hdrs { req_headers.insert(k.clone(), v.clone()); }
    if is_claude_thinking_model(&model.id) {
        req_headers.insert("anthropic-beta".to_string(), CLAUDE_THINKING_BETA_HEADER.to_string());
    }
    if let Some(extra) = &options.base.headers {
        crate::header_utils::merge_headers_safe(&mut req_headers, extra);
    }

    let body_json = serde_json::to_string(&request_body).map_err(|e| format!("Serialize error: {e}"))?;
    let client = reqwest::Client::new();

    // --- Initial fetch with retries ---
    let mut ok_response: Option<reqwest::Response> = None;
    let mut last_err: Option<String> = None;
    let mut req_url: Option<String> = None;

    for attempt in 0..=MAX_RETRIES {
        if cancel.is_cancelled() { return Err("Request was aborted".to_string()); }

        let idx = (attempt as usize).min(endpoints.len() - 1);
        let url = format!("{}/v1internal:streamGenerateContent?alt=sse", endpoints[idx]);
        req_url = Some(url.clone());

        let mut req = client.post(&url);
        for (k, v) in &req_headers { req = req.header(k.as_str(), v.as_str()); }
        req = req.body(body_json.clone());

        match req.send().await {
            Ok(resp) => {
                if resp.status().is_success() { ok_response = Some(resp); break; }
                let status = resp.status().as_u16();
                let hdrs = resp.headers().clone();
                let et = resp.text().await.unwrap_or_default();
                if attempt < MAX_RETRIES && is_retryable_error(status, &et) {
                    let sd = extract_retry_delay(&et, Some(&hdrs));
                    let delay = sd.unwrap_or(BASE_DELAY_MS * 2u64.pow(attempt));
                    let max_d = options.base.max_retry_delay_ms.unwrap_or(60000);
                    if max_d > 0 { if let Some(s) = sd { if s > max_d {
                        return Err(format!("Server requested {}s retry delay (max: {}s). {}",
                            (s as f64 / 1000.0).ceil() as u64, (max_d as f64 / 1000.0).ceil() as u64, extract_error_message(&et)));
                    }}}
                    cancellable_sleep(delay, cancel).await?;
                    continue;
                }
                return Err(format!("Cloud Code Assist API error ({}): {}", status, extract_error_message(&et)));
            }
            Err(e) => {
                last_err = Some(if let Some(src) = e.source() { format!("Network error: {src}") } else { format!("Network error: {e}") });
                if attempt < MAX_RETRIES { cancellable_sleep(BASE_DELAY_MS * 2u64.pow(attempt), cancel).await?; continue; }
                return Err(last_err.unwrap());
            }
        }
    }

    let initial_resp = ok_response.ok_or_else(|| last_err.unwrap_or("Failed after retries".to_string()))?;

    // --- Empty stream retry loop ---
    // We make requests and stream them one at a time. The first attempt uses initial_resp.
    let make_request = |client: &reqwest::Client, url: &str, headers: &HashMap<String,String>, body: &str| {
        let mut r = client.post(url);
        for (k, v) in headers { r = r.header(k.as_str(), v.as_str()); }
        r.body(body.to_string())
    };

    let url_ref = req_url.as_deref().unwrap_or("");
    let mut received = false;

    // Attempt 0: use initial_resp
    {
        let streamed = stream_response_owned(initial_resp, model, output, stream, cancel).await?;
        if streamed { received = true; }
    }

    if !received {
        for retry in 1..=MAX_EMPTY_STREAM_RETRIES {
            if cancel.is_cancelled() { return Err("Request was aborted".to_string()); }

            let backoff = EMPTY_STREAM_BASE_DELAY_MS * 2u64.pow(retry - 1);
            cancellable_sleep(backoff, cancel).await?;

            // Reset output
            output.content.clear();
            output.usage = Usage::default();
            output.stop_reason = StopReason::Stop;
            output.error_message = None;
            output.timestamp = chrono::Utc::now().timestamp_millis();

            let retry_resp = make_request(&client, url_ref, &req_headers, &body_json)
                .send().await
                .map_err(|e| format!("Retry failed: {e}"))?;

            if !retry_resp.status().is_success() {
                let s = retry_resp.status().as_u16();
                let t = retry_resp.text().await.unwrap_or_default();
                return Err(format!("Cloud Code Assist API error ({}): {}", s, t));
            }

            let streamed = stream_response_owned(retry_resp, model, output, stream, cancel).await?;
            if streamed { received = true; break; }
        }
    }

    if !received {
        return Err("Cloud Code Assist API returned an empty response".to_string());
    }

    if cancel.is_cancelled() { return Err("Request was aborted".to_string()); }
    if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
        return Err("An unknown error occurred".to_string());
    }

    stream.push(AssistantMessageEvent::Done { reason: output.stop_reason.clone(), message: output.clone() });
    Ok(())
}

// ---------- Simple stream ----------

pub fn stream_simple_google_gemini_cli(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let api_key = options.base.api_key.clone().unwrap_or_default();
    if api_key.is_empty() {
        let s = create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some("Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.".to_string());
        s.push(AssistantMessageEvent::Error { reason: StopReason::Error, error: msg });
        return s;
    }

    let base = build_base_options(model, options, &api_key);
    if options.reasoning.is_none() {
        return stream_google_gemini_cli(model, context, &GoogleGeminiCliOptions {
            base, tool_choice: None, thinking_enabled: false, thinking_budget_tokens: None, thinking_level: None, project_id: None,
        }, cancel);
    }

    let effort = clamp_reasoning(options.reasoning.as_ref().unwrap());
    if model.id.contains("3-pro") || model.id.contains("3-flash") {
        return stream_google_gemini_cli(model, context, &GoogleGeminiCliOptions {
            base, tool_choice: None, thinking_enabled: true, thinking_budget_tokens: None,
            thinking_level: Some(get_gemini_cli_thinking_level(&effort, &model.id)), project_id: None,
        }, cancel);
    }

    let base_max = base.max_tokens.unwrap_or(0);
    let (max_tokens, budget) = adjust_max_tokens_for_thinking(base_max, model.max_tokens, &effort, options.thinking_budgets.as_ref());
    stream_google_gemini_cli(model, context, &GoogleGeminiCliOptions {
        base: StreamOptions { max_tokens: Some(max_tokens), ..base },
        tool_choice: None, thinking_enabled: true, thinking_budget_tokens: Some(budget as i64), thinking_level: None, project_id: None,
    }, cancel)
}

// ---------- ApiProvider ----------

pub struct GoogleGeminiCliProvider;

impl ApiProvider for GoogleGeminiCliProvider {
    fn api(&self) -> &str { "google-gemini-cli" }

    fn stream(&self, model: &Model, context: &Context, options: &StreamOptions, cancel: CancellationToken) -> AssistantMessageEventStream {
        stream_google_gemini_cli(model, context, &GoogleGeminiCliOptions {
            base: options.clone(), tool_choice: None, thinking_enabled: false, thinking_budget_tokens: None, thinking_level: None, project_id: None,
        }, cancel)
    }

    fn stream_simple(&self, model: &Model, context: &Context, options: &SimpleStreamOptions, cancel: CancellationToken) -> AssistantMessageEventStream {
        stream_simple_google_gemini_cli(model, context, options, cancel)
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gemini_cli_model() -> Model {
        Model {
            id: "gemini-2.5-flash".to_string(),
            name: "Gemini 2.5 Flash".to_string(),
            api: "google-gemini-cli".to_string(),
            provider: "google-gemini-cli".to_string(),
            base_url: DEFAULT_ENDPOINT.to_string(),
            reasoning: true,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost { input: 0.15, output: 0.6, cache_read: 0.0375, cache_write: 0.0 },
            context_window: 1048576,
            max_tokens: 65536,
            headers: None,
            compat: None,
        }
    }

    fn test_antigravity_model() -> Model {
        Model {
            id: "gemini-2.5-pro".to_string(),
            name: "Gemini 2.5 Pro".to_string(),
            api: "google-gemini-cli".to_string(),
            provider: "google-antigravity".to_string(),
            base_url: ANTIGRAVITY_DAILY_ENDPOINT.to_string(),
            reasoning: true,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost::default(),
            context_window: 1048576,
            max_tokens: 65536,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_extract_retry_delay_from_text() {
        assert_eq!(extract_retry_delay("Your quota will reset after 39s", None), Some(40000));
        assert_eq!(extract_retry_delay("Your quota will reset after 1h2m3s", None), Some(3724000));
        assert_eq!(extract_retry_delay("Please retry in 5s", None), Some(6000));
        assert_eq!(extract_retry_delay("Please retry in 500ms", None), Some(1500));
        assert_eq!(extract_retry_delay(r#""retryDelay": "10.5s""#, None), Some(11500));
        assert_eq!(extract_retry_delay("Some random error", None), None);
    }

    #[test]
    fn test_extract_retry_delay_from_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "5".parse().unwrap());
        assert_eq!(extract_retry_delay("", Some(&headers)), Some(6000));
    }

    #[test]
    fn test_is_retryable_error() {
        assert!(is_retryable_error(429, ""));
        assert!(is_retryable_error(500, ""));
        assert!(is_retryable_error(503, ""));
        assert!(!is_retryable_error(400, ""));
        assert!(!is_retryable_error(401, ""));
        assert!(is_retryable_error(200, "RESOURCE_EXHAUSTED"));
        assert!(is_retryable_error(200, "rate limit exceeded"));
        assert!(!is_retryable_error(200, "invalid argument"));
    }

    #[test]
    fn test_extract_error_message() {
        assert_eq!(extract_error_message(r#"{"error":{"message":"Quota exceeded"}}"#), "Quota exceeded");
        assert_eq!(extract_error_message("plain error"), "plain error");
        assert_eq!(extract_error_message(r#"{"status":"error"}"#), r#"{"status":"error"}"#);
    }

    #[test]
    fn test_is_claude_thinking_model() {
        assert!(is_claude_thinking_model("claude-3.5-thinking-20250501"));
        assert!(is_claude_thinking_model("Claude-Thinking-v2"));
        assert!(!is_claude_thinking_model("claude-sonnet-4"));
        assert!(!is_claude_thinking_model("gemini-2.5-pro"));
    }

    #[test]
    fn test_build_request_basic() {
        let model = test_gemini_cli_model();
        let ctx = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User(UserMessage { content: UserContent::Text("Hello".to_string()), timestamp: 0 })],
            tools: None,
        };
        let opts = GoogleGeminiCliOptions {
            base: StreamOptions { temperature: Some(0.7), max_tokens: Some(1024), ..Default::default() },
            tool_choice: None, thinking_enabled: false, thinking_budget_tokens: None, thinking_level: None, project_id: None,
        };
        let req = build_request(&model, &ctx, "test-project", &opts, false);
        assert_eq!(req["project"], "test-project");
        assert_eq!(req["model"], "gemini-2.5-flash");
        assert_eq!(req["userAgent"], "pi-coding-agent");
        assert_eq!(req["request"]["systemInstruction"]["parts"][0]["text"], "You are helpful.");
        assert_eq!(req["request"]["generationConfig"]["temperature"], 0.7);
        assert_eq!(req["request"]["generationConfig"]["maxOutputTokens"], 1024);
    }

    #[test]
    fn test_build_request_antigravity() {
        let model = test_antigravity_model();
        let ctx = Context {
            system_prompt: Some("Original.".to_string()),
            messages: vec![Message::User(UserMessage { content: UserContent::Text("Hi".to_string()), timestamp: 0 })],
            tools: None,
        };
        let opts = GoogleGeminiCliOptions {
            base: StreamOptions::default(), tool_choice: None, thinking_enabled: false,
            thinking_budget_tokens: None, thinking_level: None, project_id: None,
        };
        let req = build_request(&model, &ctx, "proj", &opts, true);
        assert_eq!(req["requestType"], "agent");
        assert_eq!(req["userAgent"], "antigravity");
        let si = &req["request"]["systemInstruction"];
        assert_eq!(si["role"], "user");
        let parts = si["parts"].as_array().unwrap();
        assert!(parts.len() >= 3);
        assert!(parts[0]["text"].as_str().unwrap().contains("Antigravity"));
        assert_eq!(parts.last().unwrap()["text"], "Original.");
    }

    #[test]
    fn test_build_request_with_thinking() {
        let model = test_gemini_cli_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage { content: UserContent::Text("Think".to_string()), timestamp: 0 })],
            tools: None,
        };
        let opts = GoogleGeminiCliOptions {
            base: StreamOptions::default(), tool_choice: None, thinking_enabled: true,
            thinking_budget_tokens: Some(8192), thinking_level: None, project_id: None,
        };
        let req = build_request(&model, &ctx, "p", &opts, false);
        let gc = &req["request"]["generationConfig"];
        assert_eq!(gc["thinkingConfig"]["includeThoughts"], true);
        assert_eq!(gc["thinkingConfig"]["thinkingBudget"], 8192);
    }

    #[test]
    fn test_build_request_with_thinking_level() {
        let model = test_gemini_cli_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage { content: UserContent::Text("x".to_string()), timestamp: 0 })],
            tools: None,
        };
        let opts = GoogleGeminiCliOptions {
            base: StreamOptions::default(), tool_choice: None, thinking_enabled: true,
            thinking_budget_tokens: None, thinking_level: Some("HIGH".to_string()), project_id: None,
        };
        let req = build_request(&model, &ctx, "p", &opts, false);
        assert_eq!(req["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH");
    }

    #[test]
    fn test_build_request_tools_claude_uses_parameters() {
        let mut model = test_gemini_cli_model();
        model.id = "claude-sonnet-4".to_string();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage { content: UserContent::Text("search".to_string()), timestamp: 0 })],
            tools: Some(vec![Tool { name: "s".to_string(), description: "d".to_string(), parameters: json!({"type":"object"}) }]),
        };
        let opts = GoogleGeminiCliOptions {
            base: StreamOptions::default(), tool_choice: Some("auto".to_string()),
            thinking_enabled: false, thinking_budget_tokens: None, thinking_level: None, project_id: None,
        };
        let req = build_request(&model, &ctx, "p", &opts, false);
        let decls = &req["request"]["tools"][0]["functionDeclarations"];
        assert!(decls[0].get("parameters").is_some());
        assert!(decls[0].get("parametersJsonSchema").is_none());
        assert_eq!(req["request"]["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
    }

    #[test]
    fn test_get_gemini_cli_thinking_level() {
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Minimal, "gemini-3-pro"), "LOW");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Low, "gemini-3-pro"), "LOW");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Medium, "gemini-3-pro"), "HIGH");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::High, "gemini-3-pro"), "HIGH");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Minimal, "gemini-3-flash"), "MINIMAL");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Low, "gemini-3-flash"), "LOW");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Medium, "gemini-3-flash"), "MEDIUM");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::High, "gemini-3-flash"), "HIGH");
        assert_eq!(get_gemini_cli_thinking_level(&ThinkingLevel::Low, "gemini-2.5-flash"), "LOW");
    }

    #[test]
    fn test_provider_api() {
        assert_eq!(GoogleGeminiCliProvider.api(), "google-gemini-cli");
    }

    #[test]
    fn test_convert_tools_parameters_vs_json_schema() {
        let tools = vec![Tool { name: "t".to_string(), description: "d".to_string(), parameters: json!({"type":"object"}) }];
        let r1 = convert_tools(&tools, false);
        assert!(r1[0]["functionDeclarations"][0].get("parametersJsonSchema").is_some());
        let r2 = convert_tools(&tools, true);
        assert!(r2[0]["functionDeclarations"][0].get("parameters").is_some());
        assert!(convert_tools(&[], false).is_null());
    }

    #[test]
    fn test_map_stop_reason_string() {
        assert_eq!(map_stop_reason_string("STOP"), StopReason::Stop);
        assert_eq!(map_stop_reason_string("MAX_TOKENS"), StopReason::Length);
        assert_eq!(map_stop_reason_string("SAFETY"), StopReason::Error);
        assert_eq!(map_stop_reason_string("UNKNOWN"), StopReason::Error);
    }

    #[test]
    fn test_headers_structure() {
        let cli = gemini_cli_headers();
        assert!(cli["User-Agent"].contains("google-cloud-sdk"));
        assert!(cli.contains_key("X-Goog-Api-Client"));
        assert!(cli.contains_key("Client-Metadata"));
        let ag = antigravity_headers();
        assert!(ag["User-Agent"].contains("antigravity"));
    }
}
