use std::collections::HashMap;
use std::sync::Arc;

use pi_agent_core::event_stream::AssistantMessageEventStream;
use pi_agent_core::types::*;
use tokio_util::sync::CancellationToken;

/// Trait for API providers that can stream LLM responses.
pub trait ApiProvider: Send + Sync {
    fn api(&self) -> &str;

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream;

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream;
}

/// Internal entry tracking the provider and its source.
struct RegisteredApiProvider {
    provider: Arc<dyn ApiProvider>,
    source_id: Option<String>,
}

/// Registry of API providers.
pub struct ApiRegistry {
    providers: HashMap<String, RegisteredApiProvider>,
}

impl ApiRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider. If a provider with the same api name already exists, it is replaced.
    pub fn register(&mut self, provider: Arc<dyn ApiProvider>) {
        self.register_with_source(provider, None);
    }

    /// Register a provider with an optional sourceId for later batch unregistration.
    pub fn register_with_source(
        &mut self,
        provider: Arc<dyn ApiProvider>,
        source_id: Option<String>,
    ) {
        let api = provider.api().to_string();
        self.providers.insert(
            api,
            RegisteredApiProvider {
                provider,
                source_id,
            },
        );
    }

    /// Get a provider by API name.
    pub fn get(&self, api: &str) -> Option<&Arc<dyn ApiProvider>> {
        self.providers.get(api).map(|entry| &entry.provider)
    }

    /// Get all registered providers.
    pub fn providers(&self) -> Vec<&Arc<dyn ApiProvider>> {
        self.providers
            .values()
            .map(|entry| &entry.provider)
            .collect()
    }

    /// Unregister all providers that were registered with the given sourceId.
    pub fn unregister_by_source(&mut self, source_id: &str) {
        self.providers
            .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
    }

    /// Clear all registered providers.
    pub fn clear(&mut self) {
        self.providers.clear();
    }
}

impl Default for ApiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Wraps a provider's stream call with an API name validation check.
/// Returns an error event stream if the model's api doesn't match.
pub fn wrap_stream(
    provider: &dyn ApiProvider,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    if model.api != provider.api() {
        let stream = pi_agent_core::event_stream::create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some(format!(
            "Mismatched api: {} expected {}",
            model.api,
            provider.api()
        ));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg,
        });
        return stream;
    }
    provider.stream(model, context, options, cancel)
}

/// Wraps a provider's stream_simple call with an API name validation check.
pub fn wrap_stream_simple(
    provider: &dyn ApiProvider,
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    if model.api != provider.api() {
        let stream = pi_agent_core::event_stream::create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some(format!(
            "Mismatched api: {} expected {}",
            model.api,
            provider.api()
        ));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg,
        });
        return stream;
    }
    provider.stream_simple(model, context, options, cancel)
}
