use std::sync::Arc;

use crate::providers::anthropic::AnthropicProvider;
use crate::providers::azure_openai_responses::AzureOpenAIResponsesProvider;
use crate::providers::bedrock::BedrockProvider;
use crate::providers::google::GoogleProvider;
use crate::providers::google_gemini_cli::GoogleGeminiCliProvider;
use crate::providers::google_vertex::GoogleVertexProvider;
use crate::providers::openai::OpenAIProvider;
use crate::providers::openai_codex_responses::OpenAICodexResponsesProvider;
use crate::providers::openai_responses::OpenAIResponsesProvider;
use crate::registry::ApiRegistry;

/// Register all built-in API providers.
pub fn register_builtin_providers(registry: &mut ApiRegistry) {
    registry.register(Arc::new(AnthropicProvider));
    registry.register(Arc::new(OpenAIProvider));
    registry.register(Arc::new(OpenAIResponsesProvider));
    registry.register(Arc::new(AzureOpenAIResponsesProvider));
    registry.register(Arc::new(OpenAICodexResponsesProvider));
    registry.register(Arc::new(GoogleProvider));
    registry.register(Arc::new(GoogleGeminiCliProvider));
    registry.register(Arc::new(GoogleVertexProvider));
    registry.register(Arc::new(BedrockProvider));
}

/// Create a new registry with all built-in providers already registered.
pub fn create_default_registry() -> ApiRegistry {
    let mut registry = ApiRegistry::new();
    register_builtin_providers(&mut registry);
    registry
}
