use pi_agent_core::types::Model;

use crate::model::registry::ModelRegistry;

/// Resolve a model identifier to a Model.
///
/// Tries in order:
/// 1. Exact ID match
/// 2. Provider/ID format ("anthropic/claude-sonnet-4")
/// 3. Prefix match
/// 4. Glob pattern match
pub fn resolve_model<'a>(registry: &'a ModelRegistry, query: &str) -> Option<&'a Model> {
    registry.find(query)
}

/// Resolve a model with scope (provider hint).
pub fn resolve_model_scoped<'a>(
    registry: &'a ModelRegistry,
    query: &str,
    provider: Option<&str>,
) -> Option<&'a Model> {
    if let Some(provider) = provider {
        // Try provider-scoped first
        if let Some(model) = registry.find_by_provider(provider, query) {
            return Some(model);
        }
    }
    resolve_model(registry, query)
}

/// Cycle through available models for a given provider.
/// Returns the next model after `current_id`, wrapping around.
pub fn cycle_model<'a>(
    registry: &'a ModelRegistry,
    provider: &str,
    current_id: &str,
) -> Option<&'a Model> {
    let models = registry.models_for_provider(provider);
    if models.is_empty() {
        return None;
    }

    let current_idx = models.iter().position(|m| m.id == current_id);
    let next_idx = match current_idx {
        Some(idx) => (idx + 1) % models.len(),
        None => 0,
    };

    Some(models[next_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model() {
        let registry = ModelRegistry::new();
        let models = registry.all_models();
        if let Some(first) = models.first() {
            let resolved = resolve_model(&registry, &first.id);
            assert!(resolved.is_some());
        }
    }

    #[test]
    fn test_cycle_model() {
        let registry = ModelRegistry::new();
        let providers = registry.providers();
        if let Some(provider) = providers.first() {
            let models = registry.models_for_provider(provider);
            if models.len() >= 2 {
                let next = cycle_model(&registry, provider, &models[0].id);
                assert!(next.is_some());
                assert_eq!(next.unwrap().id, models[1].id);
            }
        }
    }
}
