use chrono::{SecondsFormat, TimeZone, Utc};
use pi_agent_core::types::Message;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

pub const CURRENT_SESSION_VERSION: u32 = 3;

fn session_entry_type() -> String {
    "session".to_string()
}

pub fn now_iso_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn millis_to_iso(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(Utc::now)
        .to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn iso_to_millis(value: &str) -> Option<i64> {
    if let Ok(v) = value.parse::<i64>() {
        return Some(v);
    }

    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

fn deserialize_timestamp_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(s) => Ok(s),
        Value::Number(n) => {
            let ms = n
                .as_i64()
                .ok_or_else(|| serde::de::Error::custom("invalid numeric timestamp"))?;
            Ok(millis_to_iso(ms))
        }
        Value::Null => Ok(now_iso_timestamp()),
        _ => Err(serde::de::Error::custom(
            "timestamp must be string or number",
        )),
    }
}

fn serialize_timestamp_string<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(value)
}

/// Session file header — the first line of a .jsonl session file.
///
/// Aligned with pi-mono v3 format:
/// {"type":"session","version":3,"id":"...","timestamp":"...","cwd":"..."}
///
/// Also supports reading older Rust header fields via serde aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type", default = "session_entry_type")]
    pub entry_type: String,

    #[serde(default)]
    pub version: Option<u32>,

    #[serde(alias = "session_id")]
    pub id: String,

    #[serde(
        default = "now_iso_timestamp",
        deserialize_with = "deserialize_timestamp_string",
        serialize_with = "serialize_timestamp_string",
        alias = "created_at"
    )]
    pub timestamp: String,

    #[serde(default)]
    pub cwd: String,

    #[serde(
        rename = "parentSession",
        skip_serializing_if = "Option::is_none",
        alias = "parent_session_id"
    )]
    pub parent_session: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl SessionHeader {
    pub fn version_or_current(&self) -> u32 {
        self.version.unwrap_or(CURRENT_SESSION_VERSION)
    }

    pub fn timestamp_ms(&self) -> i64 {
        iso_to_millis(&self.timestamp).unwrap_or_else(|| Utc::now().timestamp_millis())
    }
}

/// A session entry line in the JSONL file.
///
/// Includes modern pi-mono v3 variants and legacy Rust variants for read-compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    // ---------- Modern v3 variants ----------
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        message: Message,
    },

    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
    },

    #[serde(rename = "model_change")]
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },

    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        summary: String,
        #[serde(rename = "firstKeptEntryId", skip_serializing_if = "Option::is_none")]
        first_kept_entry_id: Option<String>,
        #[serde(rename = "tokensBefore", default)]
        tokens_before: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },

    #[serde(rename = "branch_summary")]
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        #[serde(rename = "fromId")]
        from_id: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },

    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },

    #[serde(rename = "custom_message")]
    CustomMessage {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        content: Value,
        display: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },

    #[serde(rename = "label")]
    Label {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: String,
        label: Option<String>,
    },

    #[serde(rename = "session_info")]
    SessionInfo {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        #[serde(
            default = "now_iso_timestamp",
            deserialize_with = "deserialize_timestamp_string",
            serialize_with = "serialize_timestamp_string"
        )]
        timestamp: String,
        name: Option<String>,
    },

    // ---------- Legacy Rust variants (read compatibility) ----------
    #[serde(rename = "user")]
    LegacyUser {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        content: String,
    },

    #[serde(rename = "assistant")]
    LegacyAssistant {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        message: Value,
    },

    #[serde(rename = "toolUse")]
    LegacyToolUse {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        arguments: Value,
    },

    #[serde(rename = "toolResult")]
    LegacyToolResult {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        content: Value,
        #[serde(rename = "isError")]
        is_error: bool,
    },

    #[serde(rename = "summary")]
    LegacySummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        summary: String,
        #[serde(rename = "summarizedIds", default)]
        summarized_ids: Vec<String>,
    },

    #[serde(rename = "modelSwitch")]
    LegacyModelSwitch {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        #[serde(rename = "fromModel")]
        from_model: String,
        #[serde(rename = "toModel")]
        to_model: String,
    },

    #[serde(rename = "fork")]
    LegacyFork {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        #[serde(rename = "sourceSessionId")]
        source_session_id: String,
        #[serde(rename = "sourceEntryId")]
        source_entry_id: String,
    },

    #[serde(rename = "system")]
    LegacySystem {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: i64,
        message: String,
    },
}

impl SessionEntry {
    /// Get the entry ID.
    pub fn id(&self) -> &str {
        match self {
            SessionEntry::Message { id, .. }
            | SessionEntry::ThinkingLevelChange { id, .. }
            | SessionEntry::ModelChange { id, .. }
            | SessionEntry::Compaction { id, .. }
            | SessionEntry::BranchSummary { id, .. }
            | SessionEntry::Custom { id, .. }
            | SessionEntry::CustomMessage { id, .. }
            | SessionEntry::Label { id, .. }
            | SessionEntry::SessionInfo { id, .. }
            | SessionEntry::LegacyUser { id, .. }
            | SessionEntry::LegacyAssistant { id, .. }
            | SessionEntry::LegacyToolUse { id, .. }
            | SessionEntry::LegacyToolResult { id, .. }
            | SessionEntry::LegacySummary { id, .. }
            | SessionEntry::LegacyModelSwitch { id, .. }
            | SessionEntry::LegacyFork { id, .. }
            | SessionEntry::LegacySystem { id, .. } => id,
        }
    }

    /// Get the parent entry ID.
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            SessionEntry::Message { parent_id, .. }
            | SessionEntry::ThinkingLevelChange { parent_id, .. }
            | SessionEntry::ModelChange { parent_id, .. }
            | SessionEntry::Compaction { parent_id, .. }
            | SessionEntry::BranchSummary { parent_id, .. }
            | SessionEntry::Custom { parent_id, .. }
            | SessionEntry::CustomMessage { parent_id, .. }
            | SessionEntry::Label { parent_id, .. }
            | SessionEntry::SessionInfo { parent_id, .. }
            | SessionEntry::LegacyUser { parent_id, .. }
            | SessionEntry::LegacyAssistant { parent_id, .. }
            | SessionEntry::LegacyToolUse { parent_id, .. }
            | SessionEntry::LegacyToolResult { parent_id, .. }
            | SessionEntry::LegacySummary { parent_id, .. }
            | SessionEntry::LegacyModelSwitch { parent_id, .. }
            | SessionEntry::LegacyFork { parent_id, .. }
            | SessionEntry::LegacySystem { parent_id, .. } => parent_id.as_deref(),
        }
    }

    /// Get timestamp in milliseconds.
    pub fn timestamp(&self) -> i64 {
        match self {
            SessionEntry::Message { timestamp, .. }
            | SessionEntry::ThinkingLevelChange { timestamp, .. }
            | SessionEntry::ModelChange { timestamp, .. }
            | SessionEntry::Compaction { timestamp, .. }
            | SessionEntry::BranchSummary { timestamp, .. }
            | SessionEntry::Custom { timestamp, .. }
            | SessionEntry::CustomMessage { timestamp, .. }
            | SessionEntry::Label { timestamp, .. }
            | SessionEntry::SessionInfo { timestamp, .. } => {
                iso_to_millis(timestamp).unwrap_or_else(|| Utc::now().timestamp_millis())
            }
            SessionEntry::LegacyUser { timestamp, .. }
            | SessionEntry::LegacyAssistant { timestamp, .. }
            | SessionEntry::LegacyToolUse { timestamp, .. }
            | SessionEntry::LegacyToolResult { timestamp, .. }
            | SessionEntry::LegacySummary { timestamp, .. }
            | SessionEntry::LegacyModelSwitch { timestamp, .. }
            | SessionEntry::LegacyFork { timestamp, .. }
            | SessionEntry::LegacySystem { timestamp, .. } => *timestamp,
        }
    }

    /// Get the entry type as string.
    pub fn entry_type(&self) -> &'static str {
        match self {
            SessionEntry::Message { .. } => "message",
            SessionEntry::ThinkingLevelChange { .. } => "thinking_level_change",
            SessionEntry::ModelChange { .. } => "model_change",
            SessionEntry::Compaction { .. } => "compaction",
            SessionEntry::BranchSummary { .. } => "branch_summary",
            SessionEntry::Custom { .. } => "custom",
            SessionEntry::CustomMessage { .. } => "custom_message",
            SessionEntry::Label { .. } => "label",
            SessionEntry::SessionInfo { .. } => "session_info",
            SessionEntry::LegacyUser { .. } => "user",
            SessionEntry::LegacyAssistant { .. } => "assistant",
            SessionEntry::LegacyToolUse { .. } => "toolUse",
            SessionEntry::LegacyToolResult { .. } => "toolResult",
            SessionEntry::LegacySummary { .. } => "summary",
            SessionEntry::LegacyModelSwitch { .. } => "modelSwitch",
            SessionEntry::LegacyFork { .. } => "fork",
            SessionEntry::LegacySystem { .. } => "system",
        }
    }

    /// Generate a new short unique entry ID (8 hex chars, matching TS behavior).
    pub fn new_id() -> String {
        uuid::Uuid::new_v4()
            .simple()
            .to_string()
            .chars()
            .take(8)
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_deserialize_from_legacy_fields() {
        let raw = serde_json::json!({
            "session_id": "abc",
            "created_at": 1710000000000_i64
        });

        let header: SessionHeader = serde_json::from_value(raw).unwrap();
        assert_eq!(header.id, "abc");
        assert!(header.timestamp.contains('T'));
    }

    #[test]
    fn test_legacy_entry_round_trip() {
        let entry = SessionEntry::LegacyToolUse {
            id: "t1".to_string(),
            parent_id: Some("e2".to_string()),
            timestamp: 2000,
            tool_call_id: "call_123".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls -la"}),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let loaded: SessionEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.id(), "t1");
        assert_eq!(loaded.entry_type(), "toolUse");
    }

    #[test]
    fn test_new_id_is_short() {
        let id = SessionEntry::new_id();
        assert_eq!(id.len(), 8);
    }
}
