use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;
use serde_json::Value;

use crate::session::types::SessionEntry;

/// Build agent context messages from session entries.
///
/// Converts a list of `SessionEntry` into `Vec<AgentMessage>` suitable for
/// feeding into the agent loop.
pub fn build_session_context(entries: &[SessionEntry]) -> Vec<AgentMessage> {
    let mut messages = Vec::new();

    for entry in entries {
        match entry {
            SessionEntry::User { content, timestamp, .. } => {
                messages.push(AgentMessage::Llm(Message::User(UserMessage {
                    content: UserContent::Text(content.clone()),
                    timestamp: *timestamp,
                })));
            }

            SessionEntry::Assistant { message, timestamp, .. } => {
                // Try to parse the stored message as an AssistantMessage
                if let Some(assistant_msg) = parse_assistant_message(message, *timestamp) {
                    messages.push(AgentMessage::Llm(Message::Assistant(assistant_msg)));
                } else {
                    // Store as custom if it can't be parsed
                    messages.push(AgentMessage::Custom(message.clone()));
                }
            }

            SessionEntry::ToolResult {
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

            SessionEntry::Summary { summary, timestamp, .. } => {
                // Summaries become custom messages (they're metadata)
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "summary",
                    "summary": summary,
                    "timestamp": timestamp,
                })));
            }

            // ToolUse entries are typically part of assistant messages
            // and handled through the assistant message content blocks.
            // We still keep them as custom for context.
            SessionEntry::ToolUse {
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

            // Non-message entries (ModelSwitch, Fork, System, Custom) are kept as custom.
            SessionEntry::ModelSwitch { from_model, to_model, timestamp, .. } => {
                messages.push(AgentMessage::Custom(serde_json::json!({
                    "type": "modelSwitch",
                    "fromModel": from_model,
                    "toModel": to_model,
                    "timestamp": timestamp,
                })));
            }

            SessionEntry::Fork { .. } | SessionEntry::Custom { .. } | SessionEntry::System { .. } => {
                // These are metadata entries, skip or keep as custom
            }
        }
    }

    messages
}

/// Try to parse a JSON value as an AssistantMessage.
fn parse_assistant_message(value: &Value, default_timestamp: i64) -> Option<AssistantMessage> {
    // If it has "content" as an array, try to parse it as AssistantMessage
    let obj = value.as_object()?;

    let content_val = obj.get("content")?;
    let content: Vec<ContentBlock> = serde_json::from_value(content_val.clone()).ok()?;

    let api = obj.get("api").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let provider = obj.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let model = obj.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let usage: Usage = obj.get("usage").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let stop_reason: StopReason = obj
        .get("stopReason")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(StopReason::Stop);
    let error_message = obj.get("errorMessage").and_then(|v| v.as_str()).map(String::from);
    let timestamp = obj.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(default_timestamp);

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
    fn test_build_context_from_entries() {
        let entries = vec![
            SessionEntry::User {
                id: "e1".to_string(),
                parent_id: None,
                timestamp: 1000,
                content: "Hello".to_string(),
            },
            SessionEntry::Assistant {
                id: "e2".to_string(),
                parent_id: Some("e1".to_string()),
                timestamp: 1001,
                message: serde_json::json!({
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "api": "anthropic-messages",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15, "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}},
                    "stopReason": "stop",
                    "timestamp": 1001
                }),
            },
        ];

        let messages = build_session_context(&entries);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role(), Some("user"));
        assert_eq!(messages[1].role(), Some("assistant"));
    }

    #[test]
    fn test_build_context_with_tool_result() {
        let entries = vec![SessionEntry::ToolResult {
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
