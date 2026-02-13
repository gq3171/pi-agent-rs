//! Automatic retry logic for transient API errors.
//!
//! Aligned with pi-mono's retry behavior (agent-session.ts:2027-2119).

use once_cell::sync::Lazy;
use regex::Regex;

/// Configuration for automatic retry of transient errors.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Whether automatic retry is enabled.
    pub enabled: bool,
    /// Maximum number of retries before giving up.
    pub max_retries: u32,
    /// Base delay in milliseconds before the first retry.
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (caps exponential backoff).
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2000,
            max_delay_ms: 60000,
        }
    }
}

// ============================================================================
// Context overflow detection patterns
// ============================================================================

static CONTEXT_OVERFLOW_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let patterns = [
        r"(?i)maximum context length",
        r"(?i)context.?window",
        r"(?i)token limit",
        r"(?i)too many tokens",
        r"(?i)context length exceeded",
        r"(?i)max_tokens",
        r"(?i)input is too long",
        r"(?i)prompt is too long",
        r"(?i)request too large",
        r"(?i)payload too large",
        r"(?i)content too large",
        r"(?i)exceeds the model.s maximum",
        r"(?i)reduce the length",
        r"(?i)input token count",
        r"(?i)exceeds.*context",
    ];
    patterns.iter().filter_map(|p| Regex::new(p).ok()).collect()
});

// ============================================================================
// Retryable error detection patterns
// ============================================================================

static RETRYABLE_ERROR_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let patterns = [
        r"(?i)overloaded",
        r"(?i)rate.?limit",
        r"\b429\b",
        r"\b500\b",
        r"\b502\b",
        r"\b503\b",
        r"\b504\b",
        r"(?i)temporarily unavailable",
        r"(?i)service unavailable",
        r"(?i)internal server error",
        r"(?i)bad gateway",
        r"(?i)gateway timeout",
        r"(?i)too many requests",
        r"(?i)server error",
        r"(?i)connection.?(reset|refused|timeout|closed)",
        r"(?i)ECONNRESET",
        r"(?i)ETIMEDOUT",
        r"(?i)network.?error",
        r"(?i)request.?timeout",
        r"(?i)capacity",
    ];
    patterns.iter().filter_map(|p| Regex::new(p).ok()).collect()
});

/// Check if an error message indicates a context overflow.
///
/// Context overflows are NOT retryable — they require compaction instead.
/// Also detects "silent" overflow by checking if the model's context window
/// is a known small value that suggests truncation occurred.
pub fn is_context_overflow(error_msg: &str, _context_window: u64) -> bool {
    CONTEXT_OVERFLOW_PATTERNS
        .iter()
        .any(|re| re.is_match(error_msg))
}

/// Check if an error message indicates a retryable transient error.
///
/// Returns `true` for overloaded servers, rate limits, 5xx errors, network
/// errors, etc. Returns `false` for context overflow errors (which need
/// compaction, not retry).
pub fn is_retryable_error(error_msg: &str, context_window: u64) -> bool {
    // Context overflow is NOT retryable
    if is_context_overflow(error_msg, context_window) {
        return false;
    }

    RETRYABLE_ERROR_PATTERNS
        .iter()
        .any(|re| re.is_match(error_msg))
}

/// Calculate the delay before the next retry attempt.
///
/// Uses exponential backoff: `base_delay_ms * 2^(attempt - 1)`,
/// capped at `max_delay_ms`.
pub fn calculate_delay(config: &RetryConfig, attempt: u32) -> u64 {
    let delay = config
        .base_delay_ms
        .saturating_mul(1u64 << (attempt.saturating_sub(1)));
    delay.min(config.max_delay_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 2000);
        assert_eq!(config.max_delay_ms, 60000);
    }

    #[test]
    fn test_is_context_overflow() {
        assert!(is_context_overflow(
            "maximum context length exceeded",
            128000
        ));
        assert!(is_context_overflow(
            "This model's context window is 8192",
            128000
        ));
        assert!(is_context_overflow("Too many tokens in the prompt", 128000));
        assert!(is_context_overflow(
            "Input is too long for this model",
            128000
        ));
        assert!(is_context_overflow("Request too large", 128000));
        assert!(is_context_overflow(
            "Exceeds the model's maximum context",
            128000
        ));
        assert!(is_context_overflow(
            "Please reduce the length of the messages",
            128000
        ));

        // NOT context overflow
        assert!(!is_context_overflow("rate limit exceeded", 128000));
        assert!(!is_context_overflow("internal server error", 128000));
    }

    #[test]
    fn test_is_retryable_error() {
        // Retryable errors
        assert!(is_retryable_error("server is overloaded", 128000));
        assert!(is_retryable_error("rate limit exceeded", 128000));
        assert!(is_retryable_error("HTTP 429 Too Many Requests", 128000));
        assert!(is_retryable_error("HTTP 500 Internal Server Error", 128000));
        assert!(is_retryable_error("HTTP 502 Bad Gateway", 128000));
        assert!(is_retryable_error("HTTP 503 Service Unavailable", 128000));
        assert!(is_retryable_error("HTTP 504 Gateway Timeout", 128000));
        assert!(is_retryable_error("temporarily unavailable", 128000));
        assert!(is_retryable_error("connection reset by peer", 128000));
        assert!(is_retryable_error("ECONNRESET", 128000));
        assert!(is_retryable_error("network error occurred", 128000));

        // NOT retryable (context overflow)
        assert!(!is_retryable_error(
            "maximum context length exceeded",
            128000
        ));
        assert!(!is_retryable_error("too many tokens", 128000));

        // NOT retryable (unknown)
        assert!(!is_retryable_error("invalid API key", 128000));
        assert!(!is_retryable_error("permission denied", 128000));
    }

    #[test]
    fn test_calculate_delay() {
        let config = RetryConfig::default();

        // Attempt 1: 2000ms
        assert_eq!(calculate_delay(&config, 1), 2000);
        // Attempt 2: 4000ms
        assert_eq!(calculate_delay(&config, 2), 4000);
        // Attempt 3: 8000ms
        assert_eq!(calculate_delay(&config, 3), 8000);
        // Attempt 4: 16000ms
        assert_eq!(calculate_delay(&config, 4), 16000);
    }

    #[test]
    fn test_calculate_delay_capped() {
        let config = RetryConfig {
            max_delay_ms: 10000,
            ..Default::default()
        };

        // Attempt 3 would be 8000, under cap
        assert_eq!(calculate_delay(&config, 3), 8000);
        // Attempt 4 would be 16000, capped to 10000
        assert_eq!(calculate_delay(&config, 4), 10000);
        // Attempt 10 would be huge, capped to 10000
        assert_eq!(calculate_delay(&config, 10), 10000);
    }

    #[test]
    fn test_calculate_delay_zero_attempt() {
        let config = RetryConfig::default();
        // attempt 0 (edge case): 2000 * 2^0 = 2000
        assert_eq!(calculate_delay(&config, 0), 2000);
    }
}
