use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::event_stream::AssistantMessageEventStream;
use crate::types::*;

// ---------- AgentThinkingLevel (includes "off") ----------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl AgentThinkingLevel {
    /// Convert to AI ThinkingLevel, returning None for "off"
    pub fn to_thinking_level(&self) -> Option<ThinkingLevel> {
        match self {
            AgentThinkingLevel::Off => None,
            AgentThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
            AgentThinkingLevel::Low => Some(ThinkingLevel::Low),
            AgentThinkingLevel::Medium => Some(ThinkingLevel::Medium),
            AgentThinkingLevel::High => Some(ThinkingLevel::High),
            AgentThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
        }
    }
}

impl fmt::Display for AgentThinkingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentThinkingLevel::Off => write!(f, "off"),
            AgentThinkingLevel::Minimal => write!(f, "minimal"),
            AgentThinkingLevel::Low => write!(f, "low"),
            AgentThinkingLevel::Medium => write!(f, "medium"),
            AgentThinkingLevel::High => write!(f, "high"),
            AgentThinkingLevel::Xhigh => write!(f, "xhigh"),
        }
    }
}

// ---------- AgentMessage ----------

#[derive(Debug, Clone)]
pub enum AgentMessage {
    Llm(Message),
    Custom(Value),
}

impl AgentMessage {
    pub fn role(&self) -> Option<&str> {
        match self {
            AgentMessage::Llm(msg) => Some(msg.role()),
            AgentMessage::Custom(_) => None,
        }
    }

    pub fn as_message(&self) -> Option<&Message> {
        match self {
            AgentMessage::Llm(msg) => Some(msg),
            AgentMessage::Custom(_) => None,
        }
    }

    pub fn into_message(self) -> Option<Message> {
        match self {
            AgentMessage::Llm(msg) => Some(msg),
            AgentMessage::Custom(_) => None,
        }
    }

    /// Convenience: create a user AgentMessage from text
    pub fn user(text: impl Into<String>) -> Self {
        AgentMessage::Llm(Message::User(UserMessage {
            content: UserContent::Text(text.into()),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }))
    }
}

// Allow implicit conversion from Message
impl From<Message> for AgentMessage {
    fn from(msg: Message) -> Self {
        AgentMessage::Llm(msg)
    }
}

// Allow implicit conversion from AssistantMessage
impl From<AssistantMessage> for AgentMessage {
    fn from(msg: AssistantMessage) -> Self {
        AgentMessage::Llm(Message::Assistant(msg))
    }
}

// Allow implicit conversion from ToolResultMessage
impl From<ToolResultMessage> for AgentMessage {
    fn from(msg: ToolResultMessage) -> Self {
        AgentMessage::Llm(Message::ToolResult(msg))
    }
}

// ---------- AgentToolResult ----------

#[derive(Debug, Clone)]
pub struct AgentToolResult {
    pub content: Vec<ContentBlock>,
    pub details: Option<Value>,
}

// ---------- AgentTool ----------

#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn definition(&self) -> &Tool;

    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        cancel: CancellationToken,
        on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>>;
}

// ---------- AgentState ----------

pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: AgentThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub stream_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error: Option<String>,
}

// ---------- AgentContext ----------

pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

pub type ConvertToLlmFuture = Pin<Box<dyn Future<Output = Vec<Message>> + Send>>;
pub type ConvertToLlmFn = dyn Fn(&[AgentMessage]) -> ConvertToLlmFuture + Send + Sync;
pub type TransformContextFuture = Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>;
pub type TransformContextFn =
    dyn Fn(Vec<AgentMessage>, CancellationToken) -> TransformContextFuture + Send + Sync;
pub type GetApiKeyFuture = Pin<Box<dyn Future<Output = Option<String>> + Send>>;
pub type GetApiKeyFn = dyn Fn(&str) -> GetApiKeyFuture + Send + Sync;
pub type MessageQueueFuture = Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>;
pub type MessageQueueFn = dyn Fn() -> MessageQueueFuture + Send + Sync;

// ---------- AgentEvent ----------

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

impl AgentEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        }
    }
}

pub type AgentEventListener = dyn Fn(&AgentEvent) + Send + Sync;

// ---------- QueueMode ----------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueMode {
    All,
    OneAtATime,
}

// ---------- AgentLoopConfig ----------

pub struct AgentLoopConfig {
    pub model: Model,
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub api_key: Option<String>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
    pub max_retry_delay_ms: Option<u64>,
    pub convert_to_llm: Arc<ConvertToLlmFn>,
    pub transform_context: Option<Arc<TransformContextFn>>,
    pub get_api_key: Option<Arc<GetApiKeyFn>>,
    pub get_steering_messages: Option<Arc<MessageQueueFn>>,
    pub get_follow_up_messages: Option<Arc<MessageQueueFn>>,
}

// ---------- StreamFn type ----------

pub type StreamFnBox = Arc<
    dyn Fn(&Model, &Context, &SimpleStreamOptions) -> AssistantMessageEventStream + Send + Sync,
>;
