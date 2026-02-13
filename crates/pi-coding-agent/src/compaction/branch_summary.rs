use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;

// ============================================================================
// Summarization prompt constants — aligned with pi-mono compaction.ts / utils.ts
// ============================================================================

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your job is to create structured summaries of coding conversations that preserve all critical technical details needed for an AI coding assistant to continue working effectively.";

pub const SUMMARIZATION_PROMPT: &str = r#"Analyze the following conversation between a user and an AI coding assistant, and produce a structured summary with the following sections:

## Goal
What is the user trying to accomplish? Be specific about the desired outcome.

## Progress
What has been done so far? List:
- Files that were created, modified, or deleted
- Commands that were run and their outcomes
- Key code changes made (be specific about what changed)
- Tests that were run and their results

## Key Decisions
What important technical decisions were made during the conversation?
- Architecture choices
- Library/tool selections
- Trade-offs that were discussed
- Constraints that were identified

## Current State
What is the current state of the work?
- What works?
- What's broken or incomplete?
- What are the next steps?
- Any blockers or open questions?

## Important Context
Any other context that would be essential for continuing this work:
- Environment details
- Configuration specifics
- Error messages that are still relevant
- File paths that are important

Be concise but thorough. Prioritize technical accuracy over brevity. Include specific file paths, function names, error messages, and command outputs where relevant."#;

pub const UPDATE_SUMMARIZATION_PROMPT: &str = r#"Update the existing structured summary below with information from the new conversation segment. Maintain the same structure but:

1. Update the Progress section with new actions taken
2. Add any new Key Decisions
3. Update the Current State to reflect the latest state
4. Update Important Context with any new relevant details
5. Keep the Goal section unless the user's goal has changed

Merge information rather than replacing it — the updated summary should contain all important context from both the old summary and the new conversation."#;

pub const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "The following is a PREFIX of a turn that was too large to include in full. Only the beginning of the content is shown, and the rest was truncated. Summarize what you can see:";

// ============================================================================
// Tool argument formatting helper
// ============================================================================

/// Format tool call arguments as `key=value, ...` pairs.
///
/// For JSON objects, formats each key-value pair. For other values,
/// returns the JSON representation directly.
fn format_tool_args(args: &serde_json::Value) -> String {
    match args {
        serde_json::Value::Object(map) => {
            let pairs: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let val_str = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    format!("{k}={val_str}")
                })
                .collect();
            pairs.join(", ")
        }
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ============================================================================
// Conversation serialization — aligned with pi-mono serializeConversation()
// ============================================================================

/// Serialize a conversation for summarization, aligned with pi-mono's
/// `serializeConversation()` (utils.ts:93-146).
///
/// Output format:
/// - `[User]: text`
/// - `[Assistant]: text` (text blocks only)
/// - `[Assistant thinking]: thinking` (thinking blocks)
/// - `[Assistant tool calls]: name(key=value, ...); ...` (tool call blocks)
/// - `[Tool result]: text`
pub fn serialize_conversation(messages: &[AgentMessage]) -> String {
    let mut parts = Vec::new();

    for msg in messages {
        match msg {
            AgentMessage::Llm(Message::User(m)) => {
                let text = match &m.content {
                    UserContent::Text(t) => t.clone(),
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|b| b.as_text().map(|t| t.text.clone()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                parts.push(format!("[User]: {text}"));
            }
            AgentMessage::Llm(Message::Assistant(m)) => {
                // Separate thinking, text, and tool call blocks
                let mut thinking_parts: Vec<String> = Vec::new();
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<String> = Vec::new();

                for block in &m.content {
                    match block {
                        ContentBlock::Thinking(t) => {
                            thinking_parts.push(t.thinking.clone());
                        }
                        ContentBlock::Text(t) => {
                            text_parts.push(t.text.clone());
                        }
                        ContentBlock::ToolCall(tc) => {
                            let args_str = format_tool_args(&tc.arguments);
                            tool_calls.push(format!("{}({})", tc.name, args_str));
                        }
                        _ => {}
                    }
                }

                // Emit thinking blocks
                for thinking in &thinking_parts {
                    parts.push(format!("[Assistant thinking]: {thinking}"));
                }

                // Emit text blocks
                if !text_parts.is_empty() {
                    parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
                }

                // Emit tool calls as a single line
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            AgentMessage::Llm(Message::ToolResult(m)) => {
                let content: String = m
                    .content
                    .iter()
                    .filter_map(|b| b.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let truncated = if content.len() > 500 {
                    let end = content
                        .char_indices()
                        .take_while(|&(i, _)| i <= 500)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(0);
                    format!("{}...[truncated]", &content[..end])
                } else {
                    content
                };
                parts.push(format!("[Tool result]: {truncated}"));
            }
            AgentMessage::Custom(_) => {
                // Skip custom messages in summary context
            }
        }
    }

    parts.join("\n\n")
}

/// Deprecated alias for `serialize_conversation()`.
#[deprecated(note = "Use serialize_conversation() instead")]
pub fn generate_summary_context(messages: &[AgentMessage]) -> String {
    serialize_conversation(messages)
}

// ============================================================================
// Summary prompt generation — aligned with pi-mono compaction.ts
// ============================================================================

/// Generate a prompt for the LLM to create a conversation summary.
///
/// - Without `previous_summary`: uses `SUMMARIZATION_PROMPT` for a fresh summary.
/// - With `previous_summary`: uses `UPDATE_SUMMARIZATION_PROMPT` to incrementally
///   update the existing summary with new conversation context.
pub fn generate_summary_prompt(conversation_text: &str, previous_summary: Option<&str>) -> String {
    match previous_summary {
        None => {
            format!(
                "{SUMMARIZATION_PROMPT}\n\n<conversation>\n{conversation_text}\n</conversation>"
            )
        }
        Some(prev) => {
            format!(
                "{UPDATE_SUMMARIZATION_PROMPT}\n\n\
                 <previous-summary>\n{prev}\n</previous-summary>\n\n\
                 <conversation>\n{conversation_text}\n</conversation>"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_conversation_user_format() {
        let messages = vec![AgentMessage::user("Fix the bug in main.rs")];
        let context = serialize_conversation(&messages);
        assert!(context.contains("[User]: Fix the bug in main.rs"));
    }

    #[test]
    fn test_serialize_conversation_assistant_text() {
        let messages = vec![AgentMessage::Llm(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "I'll look at main.rs.".to_string(),
                text_signature: None,
            })],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }))];
        let context = serialize_conversation(&messages);
        assert!(context.contains("[Assistant]: I'll look at main.rs."));
    }

    #[test]
    fn test_serialize_conversation_thinking() {
        let messages = vec![AgentMessage::Llm(Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "Let me analyze this...".to_string(),
                    thinking_signature: None,
                }),
                ContentBlock::Text(TextContent {
                    text: "Here's what I found.".to_string(),
                    text_signature: None,
                }),
            ],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }))];
        let context = serialize_conversation(&messages);
        assert!(context.contains("[Assistant thinking]: Let me analyze this..."));
        assert!(context.contains("[Assistant]: Here's what I found."));
    }

    #[test]
    fn test_serialize_conversation_tool_calls() {
        let messages = vec![AgentMessage::Llm(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "tc1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "test.rs"}),
                thought_signature: None,
            })],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }))];
        let context = serialize_conversation(&messages);
        assert!(context.contains("[Assistant tool calls]: read(path=test.rs)"));
    }

    #[test]
    fn test_serialize_conversation_tool_result() {
        let messages = vec![AgentMessage::Llm(Message::ToolResult(ToolResultMessage {
            tool_call_id: "tc1".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: "file contents here".into(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        }))];
        let context = serialize_conversation(&messages);
        assert!(context.contains("[Tool result]: file contents here"));
    }

    #[test]
    fn test_serialize_conversation_full() {
        let messages = vec![
            AgentMessage::user("Fix the bug in main.rs"),
            AgentMessage::Llm(Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "I'll look at main.rs.".to_string(),
                    text_signature: None,
                })],
                api: "test".to_string(),
                provider: "test".to_string(),
                model: "test".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })),
        ];

        let context = serialize_conversation(&messages);
        assert!(context.contains("[User]: Fix the bug in main.rs"));
        assert!(context.contains("[Assistant]: I'll look at main.rs."));
    }

    #[test]
    fn test_generate_summary_prompt_fresh() {
        let prompt = generate_summary_prompt("some conversation", None);
        assert!(prompt.contains("## Goal"));
        assert!(prompt.contains("## Progress"));
        assert!(prompt.contains("<conversation>"));
        assert!(prompt.contains("some conversation"));
    }

    #[test]
    fn test_generate_summary_prompt_incremental() {
        let prompt = generate_summary_prompt("new conversation", Some("previous summary text"));
        assert!(prompt.contains("Update the existing structured summary"));
        assert!(prompt.contains("<previous-summary>"));
        assert!(prompt.contains("previous summary text"));
        assert!(prompt.contains("<conversation>"));
        assert!(prompt.contains("new conversation"));
    }

    #[test]
    fn test_format_tool_args_object() {
        let args = serde_json::json!({"path": "test.rs", "content": "hello"});
        let result = format_tool_args(&args);
        assert!(result.contains("path=test.rs"));
        assert!(result.contains("content=hello"));
    }

    #[test]
    fn test_format_tool_args_null() {
        let result = format_tool_args(&serde_json::Value::Null);
        assert_eq!(result, "");
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_content = "a".repeat(600);
        let messages = vec![AgentMessage::Llm(Message::ToolResult(ToolResultMessage {
            tool_call_id: "tc1".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: long_content,
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        }))];
        let context = serialize_conversation(&messages);
        assert!(context.contains("...[truncated]"));
    }

    #[test]
    fn test_deprecated_alias() {
        #[allow(deprecated)]
        let result = generate_summary_context(&[AgentMessage::user("hello")]);
        assert!(result.contains("[User]: hello"));
    }
}
