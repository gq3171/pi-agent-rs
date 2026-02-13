use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::*;

// ============================================================================
// CompactionSettings — mirrors pi-mono's CompactionSettings
// ============================================================================

/// Settings that control when and how compaction is performed.
///
/// Aligned with pi-mono's `CompactionSettings` interface.
#[derive(Debug, Clone)]
pub struct CompactionSettings {
    /// Whether automatic compaction is enabled.
    pub enabled: bool,
    /// Tokens to reserve for new messages (absolute value subtracted from context_window).
    pub reserve_tokens: u64,
    /// Approximate number of tokens of recent context to keep after compaction.
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        }
    }
}

// ============================================================================
// Token estimation
// ============================================================================

/// Estimate the number of tokens in a string.
///
/// Uses a simple heuristic: ~4 characters per token for English text.
/// This matches the TS implementation's `estimateTokens()`.
pub fn estimate_tokens(text: &str) -> u64 {
    (text.len() as f64 / 4.0).ceil() as u64
}

/// Estimate tokens for a single agent message.
///
/// Mirrors pi-mono's per-message `estimateTokens()` which handles each
/// message role separately.
pub fn estimate_message_tokens(msg: &AgentMessage) -> u64 {
    match msg {
        AgentMessage::Llm(Message::User(m)) => {
            let chars: usize = match &m.content {
                UserContent::Text(t) => t.len(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text(t) => t.text.len(),
                        ContentBlock::Thinking(t) => t.thinking.len(),
                        ContentBlock::Image(_) => 4800, // match pi-mono: images ≈ 1200 tokens
                        ContentBlock::ToolCall(tc) => {
                            tc.name.len() + tc.arguments.to_string().len()
                        }
                    })
                    .sum(),
            };
            (chars as f64 / 4.0).ceil() as u64
        }
        AgentMessage::Llm(Message::Assistant(m)) => {
            let mut chars = 0usize;
            for block in &m.content {
                match block {
                    ContentBlock::Text(t) => chars += t.text.len(),
                    ContentBlock::Thinking(t) => chars += t.thinking.len(),
                    ContentBlock::ToolCall(tc) => {
                        chars += tc.name.len() + tc.arguments.to_string().len()
                    }
                    ContentBlock::Image(_) => chars += 4800,
                }
            }
            (chars as f64 / 4.0).ceil() as u64
        }
        AgentMessage::Llm(Message::ToolResult(m)) => {
            let mut chars = 0usize;
            for block in &m.content {
                match block {
                    ContentBlock::Text(t) => chars += t.text.len(),
                    ContentBlock::Image(_) => chars += 4800,
                    _ => {}
                }
            }
            (chars as f64 / 4.0).ceil() as u64
        }
        AgentMessage::Custom(v) => estimate_tokens(&v.to_string()),
    }
}

/// Estimate total tokens for a list of agent messages.
pub fn estimate_messages_tokens(messages: &[AgentMessage]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}

// ============================================================================
// Compaction trigger check — aligned with pi-mono
// ============================================================================

/// Check if compaction should be triggered.
///
/// Aligned with pi-mono: triggers when `context_tokens > context_window - reserve_tokens`.
pub fn should_compact(
    messages: &[AgentMessage],
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    let estimated = estimate_messages_tokens(messages);
    estimated > context_window.saturating_sub(settings.reserve_tokens)
}

// ============================================================================
// Token-based cut point detection — aligned with pi-mono's findCutPoint
// ============================================================================

/// Check if a message is a valid cut point (not a tool result).
///
/// Mirrors pi-mono's `findValidCutPoints()`: user, assistant, and custom
/// messages are valid; tool results are not (they must follow their tool call).
fn is_valid_cut_point(msg: &AgentMessage) -> bool {
    match msg {
        AgentMessage::Llm(Message::User(_)) => true,
        AgentMessage::Llm(Message::Assistant(_)) => true,
        AgentMessage::Llm(Message::ToolResult(_)) => false,
        AgentMessage::Custom(_) => true,
    }
}

/// Find the token-based cut point in messages.
///
/// Walks backwards from the end, accumulating estimated token counts.
/// Stops when `keep_recent_tokens` worth of context has been accumulated.
/// Only cuts at valid cut points (never at tool results).
///
/// Returns the index of the first message to keep.
fn find_cut_point(messages: &[AgentMessage], keep_recent_tokens: u64) -> usize {
    if messages.is_empty() {
        return 0;
    }

    let mut accumulated: u64 = 0;
    let mut cut_index = 0; // default: keep everything

    for i in (0..messages.len()).rev() {
        accumulated += estimate_message_tokens(&messages[i]);

        if accumulated >= keep_recent_tokens {
            // Find the closest valid cut point at or after this index
            for j in i..messages.len() {
                if is_valid_cut_point(&messages[j]) {
                    cut_index = j;
                    break;
                }
            }
            break;
        }
    }

    cut_index
}

// ============================================================================
// Compaction summary format — aligned with pi-mono messages.ts:11-17
// ============================================================================

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

// ============================================================================
// Compaction preparation — token-based, aligned with pi-mono
// ============================================================================

/// Prepare messages for compaction using token-based cut point detection.
///
/// Aligned with pi-mono's `findCutPoint()` algorithm: walks backwards from
/// the end, accumulating tokens until `keep_recent_tokens` is reached,
/// then cuts at the nearest valid boundary.
///
/// Returns `(to_summarize, to_keep)`.
pub fn prepare_compaction(
    messages: &[AgentMessage],
    settings: &CompactionSettings,
) -> (Vec<AgentMessage>, Vec<AgentMessage>) {
    let cut_index = find_cut_point(messages, settings.keep_recent_tokens);

    if cut_index == 0 {
        // Nothing to summarize (or everything fits within budget)
        return (Vec::new(), messages.to_vec());
    }

    let to_summarize = messages[..cut_index].to_vec();
    let to_keep = messages[cut_index..].to_vec();

    (to_summarize, to_keep)
}

/// Apply a compaction summary — replace old messages with a summary message.
pub fn apply_compaction(summary: &str, kept_messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut result = Vec::new();

    // Insert the summary as a user message (aligned with pi-mono messages.ts)
    result.push(AgentMessage::Llm(Message::User(UserMessage {
        content: UserContent::Text(format!(
            "{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"
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
        assert_eq!(estimate_tokens(&"a".repeat(100)), 25);
    }

    #[test]
    fn test_estimate_message_tokens() {
        let msg = AgentMessage::user("Hello, how are you?");
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 0);
        assert_eq!(tokens, 5); // 19 chars / 4 ≈ 5
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
    fn test_should_compact_enabled() {
        let messages: Vec<AgentMessage> = (0..100)
            .map(|i| AgentMessage::user(format!("Message {i} with some content to fill tokens")))
            .collect();

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        };

        // With a very small context window, should trigger
        assert!(should_compact(&messages, 100, &settings));

        // With a huge context window, should not
        assert!(!should_compact(&messages, 10_000_000, &settings));
    }

    #[test]
    fn test_should_compact_disabled() {
        let messages: Vec<AgentMessage> = (0..100)
            .map(|i| AgentMessage::user(format!("Message {i}")))
            .collect();

        let settings = CompactionSettings {
            enabled: false,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        };

        // Should never trigger when disabled
        assert!(!should_compact(&messages, 100, &settings));
    }

    #[test]
    fn test_should_compact_reserve_tokens() {
        // Create messages totalling roughly 100 tokens
        let messages: Vec<AgentMessage> = (0..10)
            .map(|i| AgentMessage::user(format!("Message number {i} with some text")))
            .collect();

        let estimated = estimate_messages_tokens(&messages);

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 20000,
        };

        // context_window - reserve_tokens should be the threshold
        // If context_window = estimated + 50, then threshold = estimated + 50 - 100 = estimated - 50
        // Since estimated > estimated - 50, should trigger
        assert!(should_compact(&messages, estimated + 50, &settings));

        // If context_window = estimated + 200, threshold = estimated + 200 - 100 = estimated + 100
        // Since estimated < estimated + 100, should NOT trigger
        assert!(!should_compact(&messages, estimated + 200, &settings));
    }

    #[test]
    fn test_prepare_compaction_token_based() {
        // Create messages with known token sizes
        let messages: Vec<AgentMessage> = (0..10)
            .map(|i| AgentMessage::user(format!("Message {i} with some text padding here")))
            .collect();

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16384,
            // Keep only ~20 tokens worth (very small, should keep few messages)
            keep_recent_tokens: 20,
        };

        let (to_summarize, to_keep) = prepare_compaction(&messages, &settings);

        // Should summarize most messages and keep only the recent ones
        assert!(!to_summarize.is_empty());
        assert!(!to_keep.is_empty());
        assert_eq!(to_summarize.len() + to_keep.len(), messages.len());

        // Verify kept messages are roughly within the token budget
        let kept_tokens = estimate_messages_tokens(&to_keep);
        // The kept tokens should be around keep_recent_tokens (may exceed slightly
        // because we cut at the nearest valid boundary)
        assert!(kept_tokens >= settings.keep_recent_tokens);
    }

    #[test]
    fn test_prepare_compaction_all_fits() {
        let messages = vec![AgentMessage::user("Hello"), AgentMessage::user("World")];

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 100000, // very large budget
        };

        let (to_summarize, to_keep) = prepare_compaction(&messages, &settings);

        // Everything fits in budget, nothing to summarize
        assert!(to_summarize.is_empty());
        assert_eq!(to_keep.len(), 2);
    }

    #[test]
    fn test_prepare_compaction_skips_tool_results() {
        let messages = vec![
            AgentMessage::user("Do something"),
            AgentMessage::Llm(Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "tc1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "test.rs"}),
                    thought_signature: None,
                })],
                api: "test".into(),
                provider: "test".into(),
                model: "test".into(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })),
            AgentMessage::Llm(Message::ToolResult(ToolResultMessage {
                tool_call_id: "tc1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "file contents here".into(),
                    text_signature: None,
                })],
                details: None,
                is_error: false,
                timestamp: 0,
            })),
            AgentMessage::user("Thanks"),
        ];

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 5, // very small, should try to cut early
        };

        let (to_summarize, to_keep) = prepare_compaction(&messages, &settings);

        // Should not cut at tool result — the cut should be at user or assistant
        if !to_summarize.is_empty() {
            // First kept message should not be a tool result
            assert!(!matches!(
                to_keep.first(),
                Some(AgentMessage::Llm(Message::ToolResult(_)))
            ));
        }
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

    #[test]
    fn test_default_settings() {
        let settings = CompactionSettings::default();
        assert!(settings.enabled);
        assert_eq!(settings.reserve_tokens, 16384);
        assert_eq!(settings.keep_recent_tokens, 20000);
    }
}
