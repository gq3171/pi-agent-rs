use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CURRENT_SESSION_VERSION: u32 = 1;

/// Session file header — the first line of a .jsonl session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    pub version: u32,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_entry_id: Option<String>,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// A session entry — one line in the JSONL file (after the header).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionEntry {
    /// User prompt.
    #[serde(rename = "user")]
    User {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        content: String,
    },

    /// Assistant response.
    #[serde(rename = "assistant")]
    Assistant {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        message: Value,
    },

    /// Tool use (from assistant).
    #[serde(rename = "toolUse")]
    ToolUse {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        tool_call_id: String,
        tool_name: String,
        arguments: Value,
    },

    /// Tool result.
    #[serde(rename = "toolResult")]
    ToolResult {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        tool_call_id: String,
        tool_name: String,
        content: Value,
        is_error: bool,
    },

    /// Summary (from compaction).
    #[serde(rename = "summary")]
    Summary {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        summary: String,
        /// IDs of entries that were summarized.
        summarized_ids: Vec<String>,
    },

    /// Model switch event.
    #[serde(rename = "modelSwitch")]
    ModelSwitch {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        from_model: String,
        to_model: String,
    },

    /// Fork point marker.
    #[serde(rename = "fork")]
    Fork {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        source_session_id: String,
        source_entry_id: String,
    },

    /// Custom metadata/event.
    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        data: Value,
    },

    /// System info entry.
    #[serde(rename = "system")]
    System {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: i64,
        message: String,
    },
}

impl SessionEntry {
    /// Get the entry ID.
    pub fn id(&self) -> &str {
        match self {
            SessionEntry::User { id, .. }
            | SessionEntry::Assistant { id, .. }
            | SessionEntry::ToolUse { id, .. }
            | SessionEntry::ToolResult { id, .. }
            | SessionEntry::Summary { id, .. }
            | SessionEntry::ModelSwitch { id, .. }
            | SessionEntry::Fork { id, .. }
            | SessionEntry::Custom { id, .. }
            | SessionEntry::System { id, .. } => id,
        }
    }

    /// Get the parent entry ID.
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            SessionEntry::User { parent_id, .. }
            | SessionEntry::Assistant { parent_id, .. }
            | SessionEntry::ToolUse { parent_id, .. }
            | SessionEntry::ToolResult { parent_id, .. }
            | SessionEntry::Summary { parent_id, .. }
            | SessionEntry::ModelSwitch { parent_id, .. }
            | SessionEntry::Fork { parent_id, .. }
            | SessionEntry::Custom { parent_id, .. }
            | SessionEntry::System { parent_id, .. } => parent_id.as_deref(),
        }
    }

    /// Get the timestamp.
    pub fn timestamp(&self) -> i64 {
        match self {
            SessionEntry::User { timestamp, .. }
            | SessionEntry::Assistant { timestamp, .. }
            | SessionEntry::ToolUse { timestamp, .. }
            | SessionEntry::ToolResult { timestamp, .. }
            | SessionEntry::Summary { timestamp, .. }
            | SessionEntry::ModelSwitch { timestamp, .. }
            | SessionEntry::Fork { timestamp, .. }
            | SessionEntry::Custom { timestamp, .. }
            | SessionEntry::System { timestamp, .. } => *timestamp,
        }
    }

    /// Get the entry type as string.
    pub fn entry_type(&self) -> &'static str {
        match self {
            SessionEntry::User { .. } => "user",
            SessionEntry::Assistant { .. } => "assistant",
            SessionEntry::ToolUse { .. } => "toolUse",
            SessionEntry::ToolResult { .. } => "toolResult",
            SessionEntry::Summary { .. } => "summary",
            SessionEntry::ModelSwitch { .. } => "modelSwitch",
            SessionEntry::Fork { .. } => "fork",
            SessionEntry::Custom { .. } => "custom",
            SessionEntry::System { .. } => "system",
        }
    }

    /// Generate a new unique entry ID.
    pub fn new_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

/// Session metadata for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub session_id: String,
    pub title: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub entry_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
}
