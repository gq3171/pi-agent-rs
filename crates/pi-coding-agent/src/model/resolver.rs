use std::collections::HashMap;

use pi_agent_core::types::Model;

use crate::cli::args::is_valid_thinking_level;
use crate::model::registry::ModelRegistry;

fn is_alias(id: &str) -> bool {
    if id.ends_with("-latest") {
        return true;
    }
    !regex::Regex::new(r"-\d{8}$")
        .expect("valid date suffix regex")
        .is_match(id)
}

fn try_match_model<'a>(pattern: &str, available_models: &'a [Model]) -> Option<&'a Model> {
    if let Some(slash_index) = pattern.find('/') {
        let provider = &pattern[..slash_index];
        let model_id = &pattern[slash_index + 1..];
        if let Some(provider_match) = available_models.iter().find(|m| {
            m.provider.eq_ignore_ascii_case(provider) && m.id.eq_ignore_ascii_case(model_id)
        }) {
            return Some(provider_match);
        }
    }

    if let Some(exact_match) = available_models
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(pattern))
    {
        return Some(exact_match);
    }

    let lowered = pattern.to_lowercase();
    let mut matches: Vec<&Model> = available_models
        .iter()
        .filter(|m| {
            m.id.to_lowercase().contains(&lowered) || m.name.to_lowercase().contains(&lowered)
        })
        .collect();
    if matches.is_empty() {
        return None;
    }

    let mut aliases: Vec<&Model> = matches
        .iter()
        .copied()
        .filter(|m| is_alias(&m.id))
        .collect();
    if !aliases.is_empty() {
        aliases.sort_by(|a, b| b.id.cmp(&a.id));
        return aliases.first().copied();
    }

    matches.sort_by(|a, b| b.id.cmp(&a.id));
    matches.first().copied()
}

#[derive(Debug, Clone, Default)]
pub struct ParsedModelResult {
    pub model: Option<Model>,
    pub thinking_level: Option<String>,
    pub warning: Option<String>,
}

pub fn parse_model_pattern(
    pattern: &str,
    available_models: &[Model],
    allow_invalid_thinking_level_fallback: bool,
) -> ParsedModelResult {
    if let Some(exact_match) = try_match_model(pattern, available_models) {
        return ParsedModelResult {
            model: Some(exact_match.clone()),
            thinking_level: None,
            warning: None,
        };
    }

    let Some(last_colon_index) = pattern.rfind(':') else {
        return ParsedModelResult::default();
    };

    let prefix = &pattern[..last_colon_index];
    let suffix = &pattern[last_colon_index + 1..];
    if is_valid_thinking_level(suffix) {
        let result = parse_model_pattern(
            prefix,
            available_models,
            allow_invalid_thinking_level_fallback,
        );
        if result.model.is_some() {
            return ParsedModelResult {
                model: result.model,
                thinking_level: if result.warning.is_some() {
                    None
                } else {
                    Some(suffix.to_string())
                },
                warning: result.warning,
            };
        }
        return result;
    }

    if !allow_invalid_thinking_level_fallback {
        return ParsedModelResult::default();
    }

    let result = parse_model_pattern(
        prefix,
        available_models,
        allow_invalid_thinking_level_fallback,
    );
    if result.model.is_some() {
        return ParsedModelResult {
            model: result.model,
            thinking_level: None,
            warning: Some(format!(
                "Invalid thinking level \"{suffix}\" in pattern \"{pattern}\". Using default instead."
            )),
        };
    }
    result
}

#[derive(Debug, Clone, Default)]
pub struct ResolveCliModelResult {
    pub model: Option<Model>,
    pub thinking_level: Option<String>,
    pub warning: Option<String>,
    pub error: Option<String>,
}

pub fn resolve_cli_model(
    cli_provider: Option<&str>,
    cli_model: Option<&str>,
    model_registry: &ModelRegistry,
) -> ResolveCliModelResult {
    let Some(cli_model) = cli_model else {
        return ResolveCliModelResult::default();
    };

    let available_models = model_registry.all_models();
    if available_models.is_empty() {
        return ResolveCliModelResult {
            error: Some(
                "No models available. Check your installation or add models to models.json."
                    .to_string(),
            ),
            ..Default::default()
        };
    }

    let mut provider_map = HashMap::new();
    for m in available_models {
        provider_map.insert(m.provider.to_lowercase(), m.provider.clone());
    }

    let mut provider = cli_provider.and_then(|p| provider_map.get(&p.to_lowercase()).cloned());
    if cli_provider.is_some() && provider.is_none() {
        return ResolveCliModelResult {
            error: Some(format!(
                "Unknown provider \"{}\". Use --list-models to see available providers/models.",
                cli_provider.unwrap_or_default()
            )),
            ..Default::default()
        };
    }

    if provider.is_none() {
        let lower = cli_model.to_lowercase();
        if let Some(exact) = available_models.iter().find(|m| {
            m.id.to_lowercase() == lower
                || format!("{}/{}", m.provider, m.id).to_lowercase() == lower
        }) {
            return ResolveCliModelResult {
                model: Some(exact.clone()),
                ..Default::default()
            };
        }
    }

    let mut pattern = cli_model.to_string();
    if provider.is_none() {
        if let Some(slash_index) = cli_model.find('/') {
            let maybe_provider = &cli_model[..slash_index];
            if let Some(canonical) = provider_map.get(&maybe_provider.to_lowercase()) {
                provider = Some(canonical.clone());
                pattern = cli_model[slash_index + 1..].to_string();
            }
        }
    } else if let Some(ref provider_name) = provider {
        let prefix = format!("{provider_name}/");
        if cli_model.to_lowercase().starts_with(&prefix.to_lowercase()) {
            pattern = cli_model[prefix.len()..].to_string();
        }
    }

    let candidates: Vec<Model> = if let Some(ref provider_name) = provider {
        available_models
            .iter()
            .filter(|m| m.provider == *provider_name)
            .cloned()
            .collect()
    } else {
        available_models.to_vec()
    };

    let parsed = parse_model_pattern(&pattern, &candidates, false);
    if parsed.model.is_none() {
        let display = if let Some(ref provider_name) = provider {
            format!("{provider_name}/{pattern}")
        } else {
            cli_model.to_string()
        };
        return ResolveCliModelResult {
            warning: parsed.warning,
            error: Some(format!(
                "Model \"{display}\" not found. Use --list-models to see available models."
            )),
            ..Default::default()
        };
    }

    ResolveCliModelResult {
        model: parsed.model,
        thinking_level: parsed.thinking_level,
        warning: parsed.warning,
        error: None,
    }
}

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
                assert_eq!(next.expect("next model should exist").id, models[1].id);
            }
        }
    }

    #[test]
    fn test_resolve_cli_model_provider_prefixed() {
        let registry = ModelRegistry::new();
        let result = resolve_cli_model(None, Some("openai/gpt-4o"), &registry);
        assert!(result.error.is_none(), "{:?}", result.error);
        let model = result.model.expect("model should resolve");
        assert_eq!(model.provider, "openai");
        assert_eq!(model.id, "gpt-4o");
    }

    #[test]
    fn test_resolve_cli_model_thinking_suffix() {
        let registry = ModelRegistry::new();
        let result = resolve_cli_model(None, Some("sonnet:high"), &registry);
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(result.thinking_level.as_deref(), Some("high"));
        let model = result.model.expect("model should resolve");
        assert!(model.id.contains("sonnet"));
    }

    #[test]
    fn test_resolve_cli_model_prefers_exact_openrouter_style_id() {
        let registry = ModelRegistry::new();
        let result = resolve_cli_model(None, Some("openai/gpt-4o:extended"), &registry);
        assert!(result.error.is_none(), "{:?}", result.error);
        let model = result.model.expect("model should resolve");
        assert_eq!(model.provider, "openrouter");
        assert_eq!(model.id, "openai/gpt-4o:extended");
    }

    #[test]
    fn test_resolve_cli_model_invalid_suffix_strict_mode() {
        let registry = ModelRegistry::new();
        let result = resolve_cli_model(Some("openai"), Some("gpt-4o:extended"), &registry);
        assert!(result.model.is_none());
        assert!(result.error.is_some());
    }
}
