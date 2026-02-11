use once_cell::sync::Lazy;
use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

static OVERFLOW_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)prompt is too long").unwrap(),
        Regex::new(r"(?i)input is too long for requested model").unwrap(),
        Regex::new(r"(?i)exceeds the context window").unwrap(),
        Regex::new(r"(?i)input token count.*exceeds the maximum").unwrap(),
        Regex::new(r"(?i)maximum prompt length is \d+").unwrap(),
        Regex::new(r"(?i)reduce the length of the messages").unwrap(),
        Regex::new(r"(?i)maximum context length is \d+ tokens").unwrap(),
        Regex::new(r"(?i)exceeds the limit of \d+").unwrap(),
        Regex::new(r"(?i)exceeds the available context size").unwrap(),
        Regex::new(r"(?i)greater than the context length").unwrap(),
        Regex::new(r"(?i)context window exceeds limit").unwrap(),
        Regex::new(r"(?i)exceeded model token limit").unwrap(),
        Regex::new(r"(?i)context[_ ]length[_ ]exceeded").unwrap(),
        Regex::new(r"(?i)too many tokens").unwrap(),
        Regex::new(r"(?i)token limit exceeded").unwrap(),
    ]
});

static STATUS_CODE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^4(00|13)\s*(status code)?\s*\(no body\)").unwrap());

/// Check if an assistant message represents a context overflow error.
///
/// Handles two cases:
/// 1. Error-based overflow: Most providers return stopReason "error" with a specific error message.
/// 2. Silent overflow: Some providers accept overflow but usage exceeds context window.
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    // Case 1: Check error message patterns
    if message.stop_reason == StopReason::Error {
        if let Some(error_message) = &message.error_message {
            if OVERFLOW_PATTERNS.iter().any(|p| p.is_match(error_message)) {
                return true;
            }
            // Cerebras and Mistral: 400/413 with no body
            if STATUS_CODE_PATTERN.is_match(error_message) {
                return true;
            }
        }
    }

    // Case 2: Silent overflow (z.ai style)
    if let Some(cw) = context_window {
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens > cw {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_error_message(error: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some(error.to_string()),
            timestamp: 0,
        }
    }

    #[test]
    fn test_anthropic_overflow() {
        let msg = make_error_message("prompt is too long: 213462 tokens > 200000 maximum");
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_openai_overflow() {
        let msg = make_error_message("Your input exceeds the context window of this model");
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_google_overflow() {
        let msg = make_error_message(
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_status_code_overflow() {
        let msg = make_error_message("400 status code (no body)");
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_silent_overflow() {
        let msg = AssistantMessage {
            content: vec![],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage {
                input: 150000,
                cache_read: 60000,
                ..Usage::default()
            },
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        };
        assert!(is_context_overflow(&msg, Some(200000)));
    }

    #[test]
    fn test_not_overflow() {
        let msg = make_error_message("rate limit exceeded");
        assert!(!is_context_overflow(&msg, None));
    }
}
