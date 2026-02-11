use pi_agent_core::types::*;

/// Build base stream options from simple options.
pub fn build_base_options(
    model: &Model,
    options: &SimpleStreamOptions,
    api_key: &str,
) -> StreamOptions {
    StreamOptions {
        temperature: options.base.temperature,
        max_tokens: Some(
            options
                .base
                .max_tokens
                .filter(|&v| v > 0)  // TS uses || which treats 0 as falsy
                .unwrap_or_else(|| model.max_tokens.min(32000)),
        ),
        api_key: if api_key.is_empty() {
            options.base.api_key.clone()
        } else {
            Some(api_key.to_string())
        },
        cache_retention: options.base.cache_retention.clone(),
        session_id: options.base.session_id.clone(),
        headers: options.base.headers.clone(),
        max_retry_delay_ms: options.base.max_retry_delay_ms,
    }
}

/// Clamp xhigh to high for providers that don't support xhigh.
pub fn clamp_reasoning(level: &ThinkingLevel) -> ThinkingLevel {
    match level {
        ThinkingLevel::Xhigh => ThinkingLevel::High,
        other => other.clone(),
    }
}

/// Adjust max_tokens to accommodate thinking budget.
/// Returns (max_tokens, thinking_budget).
pub fn adjust_max_tokens_for_thinking(
    base_max: u64,
    model_max: u64,
    level: &ThinkingLevel,
    budgets: Option<&ThinkingBudgets>,
) -> (u64, u64) {
    let default_budgets = ThinkingBudgets {
        minimal: Some(1024),
        low: Some(2048),
        medium: Some(8192),
        high: Some(16384),
    };

    let effective_budgets = budgets.unwrap_or(&default_budgets);
    let min_output_tokens = 1024u64;

    let clamped = clamp_reasoning(level);
    let budget = match clamped {
        ThinkingLevel::Minimal => effective_budgets.minimal.or(default_budgets.minimal).unwrap_or(1024),
        ThinkingLevel::Low => effective_budgets.low.or(default_budgets.low).unwrap_or(2048),
        ThinkingLevel::Medium => effective_budgets.medium.or(default_budgets.medium).unwrap_or(8192),
        ThinkingLevel::High => effective_budgets.high.or(default_budgets.high).unwrap_or(16384),
        ThinkingLevel::Xhigh => unreachable!("xhigh should be clamped to high"),
    };

    let max_tokens = (base_max + budget).min(model_max);
    let thinking_budget = if max_tokens <= budget {
        max_tokens.saturating_sub(min_output_tokens)
    } else {
        budget
    };

    (max_tokens, thinking_budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_reasoning() {
        assert_eq!(clamp_reasoning(&ThinkingLevel::Xhigh), ThinkingLevel::High);
        assert_eq!(clamp_reasoning(&ThinkingLevel::Low), ThinkingLevel::Low);
    }

    #[test]
    fn test_adjust_max_tokens_basic() {
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(8192, 200000, &ThinkingLevel::High, None);
        assert_eq!(max_tokens, 8192 + 16384);
        assert_eq!(budget, 16384);
    }

    #[test]
    fn test_adjust_max_tokens_capped_by_model() {
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(8192, 10000, &ThinkingLevel::High, None);
        assert_eq!(max_tokens, 10000);
        // budget should be capped: 10000 <= 16384, so budget = max(0, 10000 - 1024) = 8976
        assert_eq!(budget, 8976);
    }

    #[test]
    fn test_adjust_max_tokens_custom_budgets() {
        let custom = ThinkingBudgets {
            minimal: Some(512),
            low: Some(1024),
            medium: Some(4096),
            high: Some(8192),
        };
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(8192, 200000, &ThinkingLevel::Medium, Some(&custom));
        assert_eq!(max_tokens, 8192 + 4096);
        assert_eq!(budget, 4096);
    }
}
