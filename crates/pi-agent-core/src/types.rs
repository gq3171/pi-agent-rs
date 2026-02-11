use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------- Api & Provider ----------

pub type Api = String;
pub type Provider = String;

pub mod known_api {
    pub const OPENAI_COMPLETIONS: &str = "openai-completions";
    pub const OPENAI_RESPONSES: &str = "openai-responses";
    pub const AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
    pub const OPENAI_CODEX_RESPONSES: &str = "openai-codex-responses";
    pub const ANTHROPIC_MESSAGES: &str = "anthropic-messages";
    pub const BEDROCK_CONVERSE_STREAM: &str = "bedrock-converse-stream";
    pub const GOOGLE_GENERATIVE_AI: &str = "google-generative-ai";
    pub const GOOGLE_GEMINI_CLI: &str = "google-gemini-cli";
    pub const GOOGLE_VERTEX: &str = "google-vertex";
}

pub mod known_provider {
    pub const AMAZON_BEDROCK: &str = "amazon-bedrock";
    pub const ANTHROPIC: &str = "anthropic";
    pub const GOOGLE: &str = "google";
    pub const GOOGLE_GEMINI_CLI: &str = "google-gemini-cli";
    pub const GOOGLE_ANTIGRAVITY: &str = "google-antigravity";
    pub const GOOGLE_VERTEX: &str = "google-vertex";
    pub const OPENAI: &str = "openai";
    pub const AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
    pub const OPENAI_CODEX: &str = "openai-codex";
    pub const GITHUB_COPILOT: &str = "github-copilot";
    pub const XAI: &str = "xai";
    pub const GROQ: &str = "groq";
    pub const CEREBRAS: &str = "cerebras";
    pub const OPENROUTER: &str = "openrouter";
    pub const VERCEL_AI_GATEWAY: &str = "vercel-ai-gateway";
    pub const ZAI: &str = "zai";
    pub const MISTRAL: &str = "mistral";
    pub const MINIMAX: &str = "minimax";
    pub const MINIMAX_CN: &str = "minimax-cn";
    pub const HUGGINGFACE: &str = "huggingface";
    pub const OPENCODE: &str = "opencode";
    pub const KIMI_CODING: &str = "kimi-coding";
}

// ---------- ThinkingLevel ----------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThinkingLevel::Minimal => write!(f, "minimal"),
            ThinkingLevel::Low => write!(f, "low"),
            ThinkingLevel::Medium => write!(f, "medium"),
            ThinkingLevel::High => write!(f, "high"),
            ThinkingLevel::Xhigh => write!(f, "xhigh"),
        }
    }
}

// ---------- ThinkingBudgets ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    pub minimal: Option<u64>,
    pub low: Option<u64>,
    pub medium: Option<u64>,
    pub high: Option<u64>,
}

// ---------- CacheRetention ----------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

// ---------- StreamOptions ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub api_key: Option<String>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub max_retry_delay_ms: Option<u64>,
}

// ---------- SimpleStreamOptions ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleStreamOptions {
    #[serde(flatten)]
    pub base: StreamOptions,
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
}

// ---------- Content Blocks ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

// ---------- ContentBlock ----------

#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    Image(ImageContent),
    ToolCall(ToolCall),
}

impl ContentBlock {
    pub fn content_type(&self) -> &'static str {
        match self {
            ContentBlock::Text(_) => "text",
            ContentBlock::Thinking(_) => "thinking",
            ContentBlock::Image(_) => "image",
            ContentBlock::ToolCall(_) => "toolCall",
        }
    }

    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            ContentBlock::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_thinking(&self) -> Option<&ThinkingContent> {
        match self {
            ContentBlock::Thinking(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageContent> {
        match self {
            ContentBlock::Image(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            ContentBlock::ToolCall(tc) => Some(tc),
            _ => None,
        }
    }
}

impl Serialize for ContentBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ContentBlock::Text(t) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", &t.text)?;
                if let Some(sig) = &t.text_signature {
                    map.serialize_entry("textSignature", sig)?;
                }
                map.end()
            }
            ContentBlock::Thinking(t) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "thinking")?;
                map.serialize_entry("thinking", &t.thinking)?;
                if let Some(sig) = &t.thinking_signature {
                    map.serialize_entry("thinkingSignature", sig)?;
                }
                map.end()
            }
            ContentBlock::Image(i) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "image")?;
                map.serialize_entry("data", &i.data)?;
                map.serialize_entry("mimeType", &i.mime_type)?;
                map.end()
            }
            ContentBlock::ToolCall(tc) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "toolCall")?;
                map.serialize_entry("id", &tc.id)?;
                map.serialize_entry("name", &tc.name)?;
                map.serialize_entry("arguments", &tc.arguments)?;
                if let Some(sig) = &tc.thought_signature {
                    map.serialize_entry("thoughtSignature", sig)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let obj = value.as_object().ok_or_else(|| serde::de::Error::custom("expected object"))?;
        let type_str = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| serde::de::Error::custom("missing 'type' field"))?;

        match type_str {
            "text" => {
                let text = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text_signature = obj.get("textSignature").and_then(|v| v.as_str()).map(String::from);
                Ok(ContentBlock::Text(TextContent { text, text_signature }))
            }
            "thinking" => {
                let thinking = obj
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let thinking_signature = obj
                    .get("thinkingSignature")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                Ok(ContentBlock::Thinking(ThinkingContent {
                    thinking,
                    thinking_signature,
                }))
            }
            "image" => {
                let data = obj
                    .get("data")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let mime_type = obj
                    .get("mimeType")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(ContentBlock::Image(ImageContent { data, mime_type }))
            }
            "toolCall" => {
                let id = obj.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let arguments = obj.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));
                let thought_signature = obj
                    .get("thoughtSignature")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                Ok(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments,
                    thought_signature,
                }))
            }
            other => Err(serde::de::Error::custom(format!("unknown content type: {other}"))),
        }
    }
}

// ---------- Usage ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

// ---------- StopReason ----------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Stop => write!(f, "stop"),
            StopReason::Length => write!(f, "length"),
            StopReason::ToolUse => write!(f, "toolUse"),
            StopReason::Error => write!(f, "error"),
            StopReason::Aborted => write!(f, "aborted"),
        }
    }
}

// ---------- UserContent ----------

#[derive(Debug, Clone)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl Serialize for UserContent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            UserContent::Text(s) => serializer.serialize_str(s),
            UserContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for UserContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => Ok(UserContent::Text(s)),
            Value::Array(_) => {
                let blocks: Vec<ContentBlock> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(UserContent::Blocks(blocks))
            }
            _ => Err(serde::de::Error::custom("expected string or array")),
        }
    }
}

// ---------- Messages ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub api: Api,
    pub provider: Provider,
    pub model: String,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub timestamp: i64,
}

impl AssistantMessage {
    pub fn empty(model_info: &Model) -> Self {
        Self {
            content: Vec::new(),
            api: model_info.api.clone(),
            provider: model_info.provider.clone(),
            model: model_info.id.clone(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub is_error: bool,
    pub timestamp: i64,
}

// ---------- Message (tagged union by "role") ----------

#[derive(Debug, Clone)]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        }
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            Message::ToolResult(m) => Some(m),
            _ => None,
        }
    }

    pub fn timestamp(&self) -> i64 {
        match self {
            Message::User(m) => m.timestamp,
            Message::Assistant(m) => m.timestamp,
            Message::ToolResult(m) => m.timestamp,
        }
    }
}

impl Serialize for Message {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            Message::User(msg) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("role", "user")?;
                map.serialize_entry("content", &msg.content)?;
                map.serialize_entry("timestamp", &msg.timestamp)?;
                map.end()
            }
            Message::Assistant(msg) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("role", "assistant")?;
                map.serialize_entry("content", &msg.content)?;
                map.serialize_entry("api", &msg.api)?;
                map.serialize_entry("provider", &msg.provider)?;
                map.serialize_entry("model", &msg.model)?;
                map.serialize_entry("usage", &msg.usage)?;
                map.serialize_entry("stopReason", &msg.stop_reason)?;
                if let Some(err) = &msg.error_message {
                    map.serialize_entry("errorMessage", err)?;
                }
                map.serialize_entry("timestamp", &msg.timestamp)?;
                map.end()
            }
            Message::ToolResult(msg) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("role", "toolResult")?;
                map.serialize_entry("toolCallId", &msg.tool_call_id)?;
                map.serialize_entry("toolName", &msg.tool_name)?;
                map.serialize_entry("content", &msg.content)?;
                if let Some(details) = &msg.details {
                    map.serialize_entry("details", details)?;
                }
                map.serialize_entry("isError", &msg.is_error)?;
                map.serialize_entry("timestamp", &msg.timestamp)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let obj = value.as_object().ok_or_else(|| serde::de::Error::custom("expected object"))?;
        let role = obj
            .get("role")
            .and_then(|v| v.as_str())
            .ok_or_else(|| serde::de::Error::custom("missing 'role' field"))?;

        match role {
            "user" => {
                let content: UserContent =
                    serde_json::from_value(obj.get("content").cloned().unwrap_or(Value::String(String::new())))
                        .map_err(serde::de::Error::custom)?;
                let timestamp = obj.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(Message::User(UserMessage { content, timestamp }))
            }
            "assistant" => {
                let content: Vec<ContentBlock> =
                    serde_json::from_value(obj.get("content").cloned().unwrap_or(Value::Array(vec![])))
                        .map_err(serde::de::Error::custom)?;
                let api = obj.get("api").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let provider = obj.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let model = obj.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let usage: Usage =
                    serde_json::from_value(obj.get("usage").cloned().unwrap_or_default())
                        .unwrap_or_default();
                let stop_reason: StopReason =
                    serde_json::from_value(obj.get("stopReason").cloned().unwrap_or(Value::String("stop".into())))
                        .unwrap_or(StopReason::Stop);
                let error_message = obj.get("errorMessage").and_then(|v| v.as_str()).map(String::from);
                let timestamp = obj.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(Message::Assistant(AssistantMessage {
                    content,
                    api,
                    provider,
                    model,
                    usage,
                    stop_reason,
                    error_message,
                    timestamp,
                }))
            }
            "toolResult" => {
                let tool_call_id = obj
                    .get("toolCallId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tool_name = obj
                    .get("toolName")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let content: Vec<ContentBlock> =
                    serde_json::from_value(obj.get("content").cloned().unwrap_or(Value::Array(vec![])))
                        .map_err(serde::de::Error::custom)?;
                let details = obj.get("details").cloned();
                let is_error = obj.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
                let timestamp = obj.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(Message::ToolResult(ToolResultMessage {
                    tool_call_id,
                    tool_name,
                    content,
                    details,
                    is_error,
                    timestamp,
                }))
            }
            other => Err(serde::de::Error::custom(format!("unknown role: {other}"))),
        }
    }
}

// ---------- Tool ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ---------- Context ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
}

// ---------- AssistantMessageEvent ----------

#[derive(Debug, Clone)]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolCallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolCallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolCallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. })
    }
}

// ---------- Model ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Default for ModelCost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: Provider,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

// ---------- OpenAICompletionsCompat ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_usage_in_streaming: Option<bool>,
    pub max_tokens_field: Option<String>,
    pub requires_tool_result_name: Option<bool>,
    pub requires_assistant_after_tool_result: Option<bool>,
    pub requires_thinking_as_text: Option<bool>,
    pub requires_mistral_tool_ids: Option<bool>,
    pub thinking_format: Option<String>,
    pub open_router_routing: Option<OpenRouterRouting>,
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    pub supports_strict_mode: Option<bool>,
}

// ---------- OpenAIResponsesCompat ----------

/// Compatibility settings for OpenAI Responses APIs (reserved for future use).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAIResponsesCompat {}

// ---------- OpenRouterRouting ----------

/// OpenRouter provider routing preferences.
/// Controls which upstream providers OpenRouter routes requests to.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenRouterRouting {
    /// List of provider slugs to exclusively use for this request.
    pub only: Option<Vec<String>>,
    /// List of provider slugs to try in order.
    pub order: Option<Vec<String>>,
}

// ---------- VercelGatewayRouting ----------

/// Vercel AI Gateway routing preferences.
/// Controls which upstream providers the gateway routes requests to.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VercelGatewayRouting {
    /// List of provider slugs to exclusively use for this request.
    pub only: Option<Vec<String>>,
    /// List of provider slugs to try in order.
    pub order: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_round_trip() {
        let assistant_msg = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "Let me think...".to_string(),
                    thinking_signature: Some("sig123".to_string()),
                }),
                ContentBlock::Text(TextContent {
                    text: "Hello, world!".to_string(),
                    text_signature: None,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query": "rust"}),
                    thought_signature: None,
                }),
            ],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            usage: Usage {
                input: 100,
                output: 50,
                cache_read: 10,
                cache_write: 5,
                total_tokens: 165,
                cost: UsageCost {
                    input: 0.3,
                    output: 0.75,
                    cache_read: 0.03,
                    cache_write: 0.0375,
                    total: 1.1175,
                },
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 1700000000000,
        });

        let json = serde_json::to_string_pretty(&assistant_msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        // Verify round-trip
        let json2 = serde_json::to_string_pretty(&deserialized).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn test_user_message_text() {
        let msg = Message::User(UserMessage {
            content: UserContent::Text("Hello".to_string()),
            timestamp: 1700000000000,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role(), "user");
    }

    #[test]
    fn test_user_message_blocks() {
        let msg = Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                ContentBlock::Text(TextContent {
                    text: "Look at this".to_string(),
                    text_signature: None,
                }),
                ContentBlock::Image(ImageContent {
                    data: "base64data".to_string(),
                    mime_type: "image/png".to_string(),
                }),
            ]),
            timestamp: 1700000000000,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role(), "user");
    }

    #[test]
    fn test_tool_result_message() {
        let msg = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "search".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "Found 3 results".to_string(),
                text_signature: None,
            })],
            details: Some(serde_json::json!({"count": 3})),
            is_error: false,
            timestamp: 1700000000000,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role(), "toolResult");
    }

    #[test]
    fn test_stop_reason_serde() {
        assert_eq!(
            serde_json::to_string(&StopReason::ToolUse).unwrap(),
            "\"toolUse\""
        );
        let deserialized: StopReason = serde_json::from_str("\"toolUse\"").unwrap();
        assert_eq!(deserialized, StopReason::ToolUse);
    }
}
