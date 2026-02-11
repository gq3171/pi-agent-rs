use std::collections::HashMap;
use std::path::Path;

use pi_agent_core::types::Model;

use crate::error::CodingAgentError;
use crate::settings::types::CustomModelConfig;

/// Registry for managing available models (built-in + custom).
pub struct ModelRegistry {
    /// Built-in models keyed by "provider/id".
    builtin: HashMap<String, Model>,
    /// Custom models from models.json and settings.
    custom: Vec<CustomModelConfig>,
    /// All models combined for lookup.
    all_models: Vec<Model>,
}

impl ModelRegistry {
    /// Create a new registry with built-in models.
    pub fn new() -> Self {
        let builtin_models = Self::load_builtin_models();
        let mut builtin = HashMap::new();
        for model in &builtin_models {
            let key = format!("{}/{}", model.provider, model.id);
            builtin.insert(key, model.clone());
        }

        Self {
            builtin,
            custom: Vec::new(),
            all_models: builtin_models,
        }
    }

    /// Load custom models from a models.json file.
    pub fn load_custom_models(&mut self, path: &Path) -> Result<(), CodingAgentError> {
        if !path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(path)?;
        let configs: Vec<CustomModelConfig> = serde_json::from_str(&content)?;
        self.add_custom_models(&configs);
        Ok(())
    }

    /// Add custom model configurations.
    pub fn add_custom_models(&mut self, configs: &[CustomModelConfig]) {
        for config in configs {
            self.custom.push(config.clone());

            // Convert to Model if enough info is provided
            if let Some(model) = custom_config_to_model(config) {
                self.all_models.push(model);
            }
        }
    }

    /// Find a model by ID (exact match or glob pattern).
    pub fn find(&self, query: &str) -> Option<&Model> {
        // Exact match by ID
        if let Some(model) = self.all_models.iter().find(|m| m.id == query) {
            return Some(model);
        }

        // Match by "provider/id"
        if let Some(model) = self.builtin.get(query) {
            return Some(model);
        }

        // Fuzzy match: check if query is a prefix
        if let Some(model) = self.all_models.iter().find(|m| m.id.starts_with(query)) {
            return Some(model);
        }

        // Glob match
        if let Ok(pattern) = glob::Pattern::new(query) {
            if let Some(model) = self.all_models.iter().find(|m| pattern.matches(&m.id)) {
                return Some(model);
            }
        }

        None
    }

    /// Find a model by provider and model ID.
    pub fn find_by_provider(&self, provider: &str, model_id: &str) -> Option<&Model> {
        self.all_models
            .iter()
            .find(|m| m.provider == provider && m.id == model_id)
    }

    /// Get all available models.
    pub fn all_models(&self) -> &[Model] {
        &self.all_models
    }

    /// Get models for a specific provider.
    pub fn models_for_provider(&self, provider: &str) -> Vec<&Model> {
        self.all_models
            .iter()
            .filter(|m| m.provider == provider)
            .collect()
    }

    /// List all available provider names.
    pub fn providers(&self) -> Vec<String> {
        let mut providers: Vec<String> = self
            .all_models
            .iter()
            .map(|m| m.provider.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        providers.sort();
        providers
    }

    /// Get the default model.
    pub fn default_model(&self) -> Option<&Model> {
        // Use the pi-agent-ai default model lookup
        self.all_models.first()
    }

    fn load_builtin_models() -> Vec<Model> {
        // Delegate to pi-agent-ai's model list
        pi_agent_ai::models::get_all_models()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a CustomModelConfig to a Model (if enough fields are provided).
fn custom_config_to_model(config: &CustomModelConfig) -> Option<Model> {
    let api = config.api.clone()?;
    let provider = config.provider.clone()?;

    Some(Model {
        id: config.id.clone(),
        name: config.name.clone().unwrap_or_else(|| config.id.clone()),
        api,
        provider,
        base_url: config.base_url.clone().unwrap_or_default(),
        reasoning: config.reasoning.unwrap_or(false),
        input: vec!["text".to_string()],
        cost: pi_agent_core::types::ModelCost::default(),
        context_window: config.context_window.unwrap_or(128_000),
        max_tokens: config.max_tokens.unwrap_or(4096),
        headers: config.headers.clone(),
        compat: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_builtin_models() {
        let registry = ModelRegistry::new();
        assert!(!registry.all_models().is_empty());
    }

    #[test]
    fn test_find_model_by_id() {
        let registry = ModelRegistry::new();
        // Should find some model that exists in built-in models
        let models = registry.all_models();
        if let Some(first) = models.first() {
            let found = registry.find(&first.id);
            assert!(found.is_some());
            assert_eq!(found.unwrap().id, first.id);
        }
    }

    #[test]
    fn test_add_custom_model() {
        let mut registry = ModelRegistry::new();
        let initial_count = registry.all_models().len();

        registry.add_custom_models(&[CustomModelConfig {
            id: "my-custom-model".to_string(),
            name: Some("My Custom Model".to_string()),
            api: Some("openai-completions".to_string()),
            provider: Some("custom".to_string()),
            base_url: Some("https://api.custom.com".to_string()),
            reasoning: None,
            context_window: Some(32_000),
            max_tokens: Some(4096),
            headers: None,
        }]);

        assert_eq!(registry.all_models().len(), initial_count + 1);

        let found = registry.find("my-custom-model");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "My Custom Model");
    }

    #[test]
    fn test_find_nonexistent_model() {
        let registry = ModelRegistry::new();
        assert!(registry.find("nonexistent-model-xyz").is_none());
    }
}
