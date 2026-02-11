use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;

/// Generate a text representation of the conversation for summarization.
///
/// This creates a concise representation of the conversation that can be
/// sent to an LLM for summarization.
pub fn generate_summary_context(messages: &[AgentMessage]) -> String {
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
                parts.push(format!("User: {text}"));
            }
            AgentMessage::Llm(Message::Assistant(m)) => {
                let text_parts: Vec<String> = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text(t) => Some(t.text.clone()),
                        ContentBlock::ToolCall(tc) => {
                            Some(format!("[Tool call: {} ({})]", tc.name, tc.id))
                        }
                        _ => None,
                    })
                    .collect();
                if !text_parts.is_empty() {
                    parts.push(format!("Assistant: {}", text_parts.join("\n")));
                }
            }
            AgentMessage::Llm(Message::ToolResult(m)) => {
                let content: String = m
                    .content
                    .iter()
                    .filter_map(|b| b.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let status = if m.is_error { "error" } else { "success" };
                let truncated = if content.len() > 500 {
                    format!("{}...[truncated]", &content[..500])
                } else {
                    content
                };
                parts.push(format!(
                    "Tool result ({}, {}): {}",
                    m.tool_name, status, truncated
                ));
            }
            AgentMessage::Custom(_) => {
                // Skip custom messages in summary context
            }
        }
    }

    parts.join("\n\n")
}

/// Generate a prompt for the LLM to create a conversation summary.
pub fn generate_summary_prompt(context: &str) -> String {
    format!(
        "Please provide a concise summary of the following conversation. \
        Focus on:\n\
        1. What was the user trying to accomplish\n\
        2. What actions were taken (files modified, commands run)\n\
        3. The current state and any pending tasks\n\
        4. Important context that should be preserved for continuing the conversation\n\n\
        Conversation:\n{context}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_summary_context() {
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

        let context = generate_summary_context(&messages);
        assert!(context.contains("User: Fix the bug in main.rs"));
        assert!(context.contains("Assistant: I'll look at main.rs."));
    }

    #[test]
    fn test_generate_summary_prompt() {
        let prompt = generate_summary_prompt("User: Hello\nAssistant: Hi!");
        assert!(prompt.contains("concise summary"));
        assert!(prompt.contains("User: Hello"));
    }
}
