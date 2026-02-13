use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;
use serde_json::Value;

use crate::session::types::SessionEntry;

/// Build agent context messages from session entries.
///
/// Supports both modern pi-mono v3 entries (`message`, `compaction`, ...)
/// and legacy Rust entries (`user`, `assistant`, `toolResult`, ...).
pub fn build_session_context(entries: &[SessionEntry]) -> Vec<AgentMessage> {
    let mut messages = Vec::new();

    for entry in entries {
        match entry {
            SessionEntry::Message { message, .. } => {
                messages.push(AgentMessage::Llm(message.clone()));
            }

            SessionEntry::LegacyUser {
                content, timestamp, ..
            } => {
                messages.push(AgentMessage::Llm(Message::User(UserMessage {
                    content: UserContent::Text(content.clone()),
                    timestamp: *timestamp,
                })));
            }

            SessionEntry::LegacyAssistant {
                message, timestamp, ..
            } => {
                // 1) Try parsing as full Message
                if let Ok(msg) = serde_json::from_value::<Message>(message.clone()) {
                    messages.push(AgentMessage::Llm(msg));
                    continue;
                }

                // 2) Fallback: parse as assistant payload object
                if let Some(assistant_msg) = parse_assistant_message(message, *timestamp) {
                    messages.push(AgentMessage::Llm(Message::Assistant(assistant_msg)));
                } else {
                    messages.push(AgentMessage::Custom(message.clone()));
                }
            }

            SessionEntry::LegacyToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
                timestamp,
                ..
            } => {
                let content_blocks = parse_content_blocks(content);
                messages.push(AgentMessage::Llm(Message::ToolResult(ToolResultMessage {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    content: content_blocks,
                    details: None,
                    is_error: *is_error,
                    timestamp: *timestamp,
                })));
            }

            SessionEntry::Compaction {
                summary,
                tokens_before,
                timestamp,
                ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "compactionSummary",
                    "summary": summary,
                    "tokensBefore": tokens_before,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::BranchSummary {
                from_id,
                summary,
                timestamp,
                ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "branchSummary",
                    "fromId": from_id,
                    "summary": summary,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::CustomMessage {
                custom_type,
                content,
                display,
                details,
                timestamp,
                ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "custom",
                    "customType": custom_type,
                    "content": content,
                    "display": display,
                    "details": details,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::LegacySummary {
                summary, timestamp, ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "summary",
                    "summary": summary,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::LegacyToolUse {
                tool_call_id,
                tool_name,
                arguments,
                timestamp,
                ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "toolUse",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "arguments": arguments,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::LegacyModelSwitch {
                from_model,
                to_model,
                timestamp,
                ..
            } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "modelSwitch",
                    "fromModel": from_model,
                    "toModel": to_model,
                    "timestamp": timestamp,
                })));
            }

            // Metadata entries not currently injected into LLM context.
            SessionEntry::ThinkingLevelChange { .. }
            | SessionEntry::ModelChange { .. }
            | SessionEntry::Custom { .. }
            | SessionEntry::Label { .. }
            | SessionEntry::SessionInfo { .. }
            | SessionEntry::LegacyFork { .. }
            | SessionEntry::LegacySystem { .. } => {}
        }
    }

    messages
}

/// Try to parse a JSON value as an AssistantMessage.
fn parse_assistant_message(value: &Value, default_timestamp: i64) -> Option<AssistantMessage> {
    let obj = value.as_object()?;

    let content_val = obj.get("content")?;
    let content: Vec<ContentBlock> = serde_json::from_value(content_val.clone()).ok()?;

    let api = obj
        .get("api")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let provider = obj
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = obj
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let usage: Usage = obj
        .get("usage")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let stop_reason: StopReason = obj
        .get("stopReason")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(StopReason::Stop);
    let error_message = obj
        .get("errorMessage")
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = obj
        .get("timestamp")
        .and_then(|v| v.as_i64())
        .unwrap_or(default_timestamp);

    Some(AssistantMessage {
        content,
        api,
        provider,
        model,
        usage,
        stop_reason,
        error_message,
        timestamp,
    })
}

/// Parse a JSON value as content blocks.
fn parse_content_blocks(value: &Value) -> Vec<ContentBlock> {
    match value {
        Value::Array(_) => serde_json::from_value(value.clone()).unwrap_or_default(),
        Value::String(s) => vec![ContentBlock::Text(TextContent {
            text: s.clone(),
            text_signature: None,
        })],
        _ => vec![ContentBlock::Text(TextContent {
            text: value.to_string(),
            text_signature: None,
        })],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_context_from_modern_message_entries() {
        let entries = vec![
            SessionEntry::Message {
                id: "e1".to_string(),
                parent_id: None,
                timestamp: crate::session::types::now_iso_timestamp(),
                message: Message::User(UserMessage {
                    content: UserContent::Text("Hello".to_string()),
                    timestamp: 1000,
                }),
            },
            SessionEntry::Message {
                id: "e2".to_string(),
                parent_id: Some("e1".to_string()),
                timestamp: crate::session::types::now_iso_timestamp(),
                message: Message::Assistant(AssistantMessage {
                    content: vec![ContentBlock::Text(TextContent {
                        text: "Hi there!".to_string(),
                        text_signature: None,
                    })],
                    api: "anthropic-messages".to_string(),
                    provider: "anthropic".to_string(),
                    model: "claude-sonnet-4".to_string(),
                    usage: Usage::default(),
                    stop_reason: StopReason::Stop,
                    error_message: None,
                    timestamp: 1001,
                }),
            },
        ];

        let messages = build_session_context(&entries);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role(), Some("user"));
        assert_eq!(messages[1].role(), Some("assistant"));
    }

    #[test]
    fn test_build_context_with_legacy_tool_result() {
        let entries = vec![SessionEntry::LegacyToolResult {
            id: "tr1".to_string(),
            parent_id: None,
            timestamp: 2000,
            tool_call_id: "call_1".to_string(),
            tool_name: "bash".to_string(),
            content: serde_json::json!([{"type": "text", "text": "file1.rs\nfile2.rs"}]),
            is_error: false,
        }];

        let messages = build_session_context(&entries);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Some("toolResult"));
    }
}
