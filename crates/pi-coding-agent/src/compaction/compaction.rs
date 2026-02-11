use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;

/// Estimate the number of tokens in a string.
///
/// Uses a simple heuristic: ~4 characters per token for English text.
/// This matches the TS implementation's `estimateTokens()`.
pub fn estimate_tokens(text: &str) -> u64 {
    // Rough approximation: 1 token ≈ 4 chars for English
    // This is intentionally simple — exact token counting requires a tokenizer.
    (text.len() as f64 / 4.0).ceil() as u64
}

/// Estimate total tokens for a list of agent messages.
pub fn estimate_messages_tokens(messages: &[AgentMessage]) -> u64 {
    let mut total = 0u64;
    for msg in messages {
        match msg {
            AgentMessage::Llm(Message::User(m)) => {
                let text = match &m.content {
                    UserContent::Text(t) => t.len(),
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text(t) => t.text.len(),
                            ContentBlock::Thinking(t) => t.thinking.len(),
                            ContentBlock::Image(_) => 1000, // rough estimate for images
                            ContentBlock::ToolCall(tc) => tc.arguments.to_string().len(),
                        })
                        .sum(),
                };
                total += (text as f64 / 4.0).ceil() as u64;
            }
            AgentMessage::Llm(Message::Assistant(m)) => {
                for block in &m.content {
                    match block {
                        ContentBlock::Text(t) => total += estimate_tokens(&t.text),
                        ContentBlock::Thinking(t) => total += estimate_tokens(&t.thinking),
                        ContentBlock::ToolCall(tc) => {
                            total += estimate_tokens(&tc.arguments.to_string())
                        }
                        ContentBlock::Image(_) => total += 250,
                    }
                }
            }
            AgentMessage::Llm(Message::ToolResult(m)) => {
                for block in &m.content {
                    match block {
                        ContentBlock::Text(t) => total += estimate_tokens(&t.text),
                        ContentBlock::Image(_) => total += 250,
                        _ => {}
                    }
                }
            }
            AgentMessage::Custom(v) => {
                total += estimate_tokens(&v.to_string());
            }
        }
    }
    total
}

/// Check if compaction should be triggered.
///
/// Returns true if estimated token usage exceeds the threshold fraction
/// of the model's context window.
pub fn should_compact(
    messages: &[AgentMessage],
    context_window: u64,
    threshold: f64,
) -> bool {
    let estimated = estimate_messages_tokens(messages);
    let limit = (context_window as f64 * threshold) as u64;
    estimated > limit
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary: String,
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// Compact messages by replacing older messages with a summary.
///
/// This function prepares the compaction request — the actual LLM call
/// to generate the summary should be done by the caller.
///
/// Returns the messages that should be summarized (for the caller to
/// pass to the LLM for summarization).
pub fn prepare_compaction(
    messages: &[AgentMessage],
    keep_recent: usize,
) -> (Vec<AgentMessage>, Vec<AgentMessage>) {
    if messages.len() <= keep_recent {
        return (Vec::new(), messages.to_vec());
    }

    let split_point = messages.len() - keep_recent;
    let to_summarize = messages[..split_point].to_vec();
    let to_keep = messages[split_point..].to_vec();

    (to_summarize, to_keep)
}

/// Apply a compaction summary — replace old messages with a summary message.
pub fn apply_compaction(
    summary: &str,
    kept_messages: Vec<AgentMessage>,
) -> Vec<AgentMessage> {
    let mut result = Vec::new();

    // Insert the summary as a user message
    result.push(AgentMessage::Llm(Message::User(UserMessage {
        content: UserContent::Text(format!(
            "[Previous conversation summary]\n\n{summary}"
        )),
        timestamp: chrono::Utc::now().timestamp_millis(),
    })));

    // Keep the recent messages
    result.extend(kept_messages);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars / 4 ≈ 2
        assert_eq!(estimate_tokens("a".repeat(100).as_str()), 25);
    }

    #[test]
    fn test_estimate_messages_tokens() {
        let messages = vec![
            AgentMessage::user("Hello, how are you?"),
            AgentMessage::Llm(Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "I'm doing well!".to_string(),
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

        let tokens = estimate_messages_tokens(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_should_compact() {
        let messages: Vec<AgentMessage> = (0..100)
            .map(|i| AgentMessage::user(format!("Message {i} with some content to fill tokens")))
            .collect();

        // With a very small context window, should trigger
        assert!(should_compact(&messages, 100, 0.8));

        // With a huge context window, should not
        assert!(!should_compact(&messages, 10_000_000, 0.8));
    }

    #[test]
    fn test_prepare_compaction() {
        let messages: Vec<AgentMessage> = (0..10)
            .map(|i| AgentMessage::user(format!("Message {i}")))
            .collect();

        let (to_summarize, to_keep) = prepare_compaction(&messages, 3);
        assert_eq!(to_summarize.len(), 7);
        assert_eq!(to_keep.len(), 3);
    }

    #[test]
    fn test_apply_compaction() {
        let kept = vec![
            AgentMessage::user("Recent message 1"),
            AgentMessage::user("Recent message 2"),
        ];

        let result = apply_compaction("Previous conversation was about Rust.", kept);
        assert_eq!(result.len(), 3); // summary + 2 kept
    }
}
