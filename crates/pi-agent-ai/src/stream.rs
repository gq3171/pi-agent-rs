use pi_agent_core::event_stream::AssistantMessageEventStream;
use pi_agent_core::types::*;
use tokio_util::sync::CancellationToken;

use crate::registry::ApiRegistry;

fn resolve_provider<'a>(registry: &'a ApiRegistry, api: &str) -> Result<&'a std::sync::Arc<dyn crate::registry::ApiProvider>, String> {
    registry
        .get(api)
        .ok_or_else(|| format!("No API provider registered for api: {api}"))
}

/// Stream an LLM response with raw provider options.
pub fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    registry: &ApiRegistry,
    cancel: CancellationToken,
) -> Result<AssistantMessageEventStream, String> {
    let provider = resolve_provider(registry, &model.api)?;
    Ok(provider.stream(model, context, options, cancel))
}

/// Stream an LLM response with simplified options (handles reasoning level).
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    registry: &ApiRegistry,
    cancel: CancellationToken,
) -> Result<AssistantMessageEventStream, String> {
    let provider = resolve_provider(registry, &model.api)?;
    Ok(provider.stream_simple(model, context, options, cancel))
}

/// Complete (non-streaming): stream and collect the final result.
pub async fn complete(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    registry: &ApiRegistry,
    cancel: CancellationToken,
) -> Result<AssistantMessage, String> {
    let s = stream(model, context, options, registry, cancel)?;
    Ok(s.result().await)
}

/// Complete with simplified options.
pub async fn complete_simple(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    registry: &ApiRegistry,
    cancel: CancellationToken,
) -> Result<AssistantMessage, String> {
    let s = stream_simple(model, context, options, registry, cancel)?;
    Ok(s.result().await)
}
