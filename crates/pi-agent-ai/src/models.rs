use std::collections::HashMap;
use std::sync::OnceLock;

use pi_agent_core::types::*;

use crate::models_generated;

fn model_registry() -> &'static HashMap<String, HashMap<String, Model>> {
    static REGISTRY: OnceLock<HashMap<String, HashMap<String, Model>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry: HashMap<String, HashMap<String, Model>> = HashMap::new();

        for model in models_generated::all_models() {
            registry
                .entry(model.provider.clone())
                .or_default()
                .insert(model.id.clone(), model);
        }

        registry
    })
}

/// Get a model by provider and model ID.
pub fn get_model(provider: &str, model_id: &str) -> Option<Model> {
    model_registry()
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

/// Get all known provider names.
pub fn get_providers() -> Vec<String> {
    model_registry().keys().cloned().collect()
}

/// Get all models for a provider.
pub fn get_models(provider: &str) -> Vec<Model> {
    model_registry()
        .get(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

/// Get all models across all providers.
pub fn get_all_models() -> Vec<Model> {
    model_registry()
        .values()
        .flat_map(|models| models.values().cloned())
        .collect()
}

/// Calculate cost based on model pricing and usage.
/// Modifies `usage.cost` in-place.
pub fn calculate_cost(model: &Model, usage: &mut Usage) {
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// Check if a model supports xhigh thinking level.
pub fn supports_xhigh(model: &Model) -> bool {
    if model.id.contains("gpt-5.2") || model.id.contains("gpt-5.3") {
        return true;
    }
    if model.api == "anthropic-messages" {
        return model.id.contains("opus-4-6") || model.id.contains("opus-4.6");
    }
    false
}

/// Check if two models are equal by comparing both their id and provider.
pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model() {
        let model = get_model("anthropic", "claude-sonnet-4-20250514");
        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m.name, "Claude Sonnet 4");
        assert_eq!(m.api, "anthropic-messages");
    }

    #[test]
    fn test_get_model_not_found() {
        assert!(get_model("anthropic", "nonexistent").is_none());
        assert!(get_model("nonexistent", "anything").is_none());
    }

    #[test]
    fn test_get_providers() {
        let providers = get_providers();
        assert!(providers.contains(&"anthropic".to_string()));
        assert!(providers.contains(&"openai".to_string()));
        assert!(providers.contains(&"google".to_string()));
    }

    #[test]
    fn test_get_all_models_count() {
        let all = get_all_models();
        assert!(all.len() > 100, "Expected 700+ models, got {}", all.len());
    }

    #[test]
    fn test_calculate_cost() {
        let model = get_model("anthropic", "claude-sonnet-4-20250514").unwrap();
        let mut usage = Usage {
            input: 1000,
            output: 500,
            cache_read: 200,
            cache_write: 100,
            total_tokens: 1800,
            cost: UsageCost::default(),
        };
        calculate_cost(&model, &mut usage);

        assert!((usage.cost.input - 0.003).abs() < 1e-10);
        assert!((usage.cost.output - 0.0075).abs() < 1e-10);
        assert!((usage.cost.cache_read - 0.00006).abs() < 1e-10);
        assert!((usage.cost.cache_write - 0.000375).abs() < 1e-10);
    }

    #[test]
    fn test_supports_xhigh() {
        let opus46 = get_model("anthropic", "claude-opus-4-6").unwrap();
        assert!(supports_xhigh(&opus46));

        let sonnet = get_model("anthropic", "claude-sonnet-4-20250514").unwrap();
        assert!(!supports_xhigh(&sonnet));
    }

    #[test]
    fn test_models_are_equal() {
        let a = get_model("anthropic", "claude-sonnet-4-20250514");
        let b = get_model("anthropic", "claude-sonnet-4-20250514");
        assert!(models_are_equal(a.as_ref(), b.as_ref()));

        let c = get_model("openai", "gpt-4o");
        assert!(!models_are_equal(a.as_ref(), c.as_ref()));
        assert!(!models_are_equal(None, a.as_ref()));
    }
}
