use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A bash execution message (custom message type for tool execution context).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub tool_call_id: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub duration_ms: Option<u64>,
    pub was_cancelled: bool,
}

/// A custom message for storing compaction summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub summary: String,
    pub summarized_count: usize,
    pub token_estimate_before: u64,
    pub token_estimate_after: u64,
    pub timestamp: i64,
}

/// Wrapper for custom messages that can be serialized/deserialized
/// as AgentMessage::Custom.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CustomMessage {
    #[serde(rename = "bashExecution")]
    BashExecution(BashExecutionMessage),

    #[serde(rename = "compactionSummary")]
    CompactionSummary(CompactionSummaryMessage),

    /// Fallback for unknown custom message types.
    #[serde(other)]
    Unknown,
}

impl CustomMessage {
    /// Convert to a serde_json::Value for use as AgentMessage::Custom.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}
