use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::supports_xhigh;
use crate::registry::ApiProvider;
use crate::simple_options::{build_base_options, clamp_reasoning};
use crate::sse::SseParser;

use super::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools, process_responses_events,
};

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_AZURE_API_VERSION: &str = "v1";

fn azure_tool_call_providers() -> HashSet<&'static str> {
    [
        "openai",
        "openai-codex",
        "opencode",
        "azure-openai-responses",
    ]
    .into_iter()
    .collect()
}

// =============================================================================
// Options
// =============================================================================

/// Azure OpenAI Responses-specific stream options.
#[derive(Debug, Clone)]
pub struct AzureOpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<String>,
    pub reasoning_summary: Option<String>,
    pub azure_api_version: Option<String>,
    pub azure_resource_name: Option<String>,
    pub azure_base_url: Option<String>,
    pub azure_deployment_name: Option<String>,
}

// =============================================================================
// Deployment name mapping
// =============================================================================

/// Parse a comma-separated `model=deployment` mapping from an environment variable value.
fn parse_deployment_name_map(value: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let val = match value {
        Some(v) => v,
        None => return map,
    };
    for entry in val.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(2, '=').collect();
        if parts.len() != 2 || parts[0].trim().is_empty() || parts[1].trim().is_empty() {
            continue;
        }
        map.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
    }
    map
}

/// Resolve the Azure deployment name from options, env var map, or model id.
fn resolve_deployment_name(model: &Model, options: Option<&AzureOpenAIResponsesOptions>) -> String {
    if let Some(opts) = options {
        if let Some(name) = &opts.azure_deployment_name {
            return name.clone();
        }
    }

    let env_map = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP").ok();
    let deployment_map = parse_deployment_name_map(env_map.as_deref());
    if let Some(mapped) = deployment_map.get(&model.id) {
        return mapped.clone();
    }

    model.id.clone()
}

// =============================================================================
// Azure URL helpers
// =============================================================================

fn normalize_azure_base_url(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_string()
}

fn build_default_base_url(resource_name: &str) -> String {
    format!("https://{resource_name}.openai.azure.com/openai/v1")
}

/// Resolve Azure configuration from options, env vars, and model config.
fn resolve_azure_config(
    model: &Model,
    options: Option<&AzureOpenAIResponsesOptions>,
) -> Result<(String, String), String> {
    let api_version = options
        .and_then(|o| o.azure_api_version.as_deref())
        .map(|s| s.to_string())
        .or_else(|| std::env::var("AZURE_OPENAI_API_VERSION").ok())
        .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string());

    let base_url = options
        .and_then(|o| o.azure_base_url.as_deref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("AZURE_OPENAI_BASE_URL")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        });

    let resource_name = options
        .and_then(|o| o.azure_resource_name.as_deref())
        .map(|s| s.to_string())
        .or_else(|| std::env::var("AZURE_OPENAI_RESOURCE_NAME").ok());

    let mut resolved_url = base_url;

    if resolved_url.is_none() {
        if let Some(rn) = &resource_name {
            resolved_url = Some(build_default_base_url(rn));
        }
    }

    if resolved_url.is_none() && !model.base_url.is_empty() {
        resolved_url = Some(model.base_url.clone());
    }

    let final_url = resolved_url.ok_or_else(|| {
        "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.".to_string()
    })?;

    Ok((normalize_azure_base_url(&final_url), api_version))
}

fn resolve_azure_responses_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/responses") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/responses")
    } else {
        format!("{trimmed}/v1/responses")
    }
}

// =============================================================================
// HTTP helpers
// =============================================================================

fn build_headers(
    model: &Model,
    api_key: &str,
    extra_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("api-key".to_string(), api_key.to_string());

    // Copy model headers (skip sensitive auth headers)
    if let Some(model_headers) = &model.headers {
        crate::header_utils::merge_headers_safe(&mut headers, model_headers);
    }

    // Merge extra headers (skip sensitive auth headers)
    if let Some(extra) = extra_headers {
        crate::header_utils::merge_headers_safe(&mut headers, extra);
    }

    headers
}

fn build_params(
    model: &Model,
    context: &Context,
    options: Option<&AzureOpenAIResponsesOptions>,
    deployment_name: &str,
) -> Value {
    let providers = azure_tool_call_providers();
    let messages = convert_responses_messages(model, context, &providers, None);

    let mut params = json!({
        "model": deployment_name,
        "input": messages,
        "stream": true,
    });

    if let Some(opts) = options {
        if let Some(session_id) = &opts.base.session_id {
            params["prompt_cache_key"] = json!(session_id);
        }
        if let Some(max_tokens) = opts.base.max_tokens {
            params["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = opts.base.temperature {
            params["temperature"] = json!(temp);
        }
    }

    if let Some(tools) = &context.tools {
        params["tools"] = json!(convert_responses_tools(tools, None));
    }

    if model.reasoning {
        let has_reasoning_opts = options
            .map(|o| o.reasoning_effort.is_some() || o.reasoning_summary.is_some())
            .unwrap_or(false);

        if has_reasoning_opts {
            let effort = options
                .and_then(|o| o.reasoning_effort.as_deref())
                .unwrap_or("medium");
            let summary = options
                .and_then(|o| o.reasoning_summary.as_deref())
                .unwrap_or("auto");
            params["reasoning"] = json!({
                "effort": effort,
                "summary": summary,
            });
            params["include"] = json!(["reasoning.encrypted_content"]);
        } else if model.name.to_lowercase().starts_with("gpt-5") {
            // Workaround for GPT-5 models
            if let Some(arr) = params.get_mut("input").and_then(|v| v.as_array_mut()) {
                arr.push(json!({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "# Juice: 0 !important"}]
                }));
            }
        }
    }

    params
}

// =============================================================================
// Stream functions
// =============================================================================

/// Stream from Azure OpenAI Responses API using raw HTTP + SSE parsing.
pub fn stream_azure_openai_responses(
    model: &Model,
    context: &Context,
    options: &AzureOpenAIResponsesOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&model);
        let deployment_name = resolve_deployment_name(&model, Some(&options));

        let api_key = options
            .base
            .api_key
            .clone()
            .or_else(|| get_env_api_key(&model.provider))
            .unwrap_or_default();

        if api_key.is_empty() {
            output.stop_reason = StopReason::Error;
            output.error_message = Some(
                "Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass it as an argument.".to_string()
            );
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        let (base_url, _api_version) = match resolve_azure_config(&model, Some(&options)) {
            Ok(config) => config,
            Err(e) => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(e);
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        };

        let headers = build_headers(&model, &api_key, options.base.headers.as_ref());
        let params = build_params(&model, &context, Some(&options), &deployment_name);
        let url = resolve_azure_responses_url(&base_url);

        let client = reqwest::Client::new();
        let mut request = client.post(&url);
        for (k, v) in &headers {
            request = request.header(k.as_str(), v.as_str());
        }

        let response = match request.json(&params).send().await {
            Ok(resp) => resp,
            Err(e) => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(format!("HTTP error: {e}"));
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            output.stop_reason = StopReason::Error;
            output.error_message = Some(format!("{status} {body}"));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        stream_clone.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        // Process SSE stream
        let mut current_item: Option<super::openai_responses_shared::CurrentItem> = None;
        let mut current_block: Option<super::openai_responses_shared::CurrentBlock> = None;
        let mut sse_parser = SseParser::new();

        let mut byte_stream = response.bytes_stream();
        while let Some(chunk_result) = byte_stream.next().await {
            if cancel.is_cancelled() {
                output.stop_reason = StopReason::Aborted;
                output.error_message = Some("Request was aborted".to_string());
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error: output,
                });
                return;
            }

            let chunk = match chunk_result {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(e) => {
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(format!("Stream error: {e}"));
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    });
                    return;
                }
            };

            let events = match sse_parser.feed(&chunk) {
                Ok(events) => events,
                Err(e) => {
                    tracing::error!("SSE parse error: {e}");
                    break;
                }
            };
            let mut parsed_events: Vec<Value> = Vec::new();
            for sse_event in events {
                let data_str = sse_event.data.trim();
                if data_str == "[DONE]" {
                    continue;
                }
                if let Ok(data) = serde_json::from_str::<Value>(data_str) {
                    parsed_events.push(data);
                }
            }

            if let Err(err) = process_responses_events(
                &parsed_events,
                &mut output,
                &stream_clone,
                &model,
                &mut current_item,
                &mut current_block,
                None,
            ) {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(err);
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: output,
                });
                return;
            }
        }

        if cancel.is_cancelled() {
            output.stop_reason = StopReason::Aborted;
            output.error_message = Some("Request was aborted".to_string());
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: output,
            });
            return;
        }

        if output.stop_reason == StopReason::Error || output.stop_reason == StopReason::Aborted {
            stream_clone.push(AssistantMessageEvent::Error {
                reason: output.stop_reason.clone(),
                error: output,
            });
        } else {
            stream_clone.push(AssistantMessageEvent::Done {
                reason: output.stop_reason.clone(),
                message: output,
            });
        }
    });

    stream
}

/// Simplified streaming: handles reasoning level and API key resolution.
pub fn stream_simple_azure_openai_responses(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let api_key = options
        .base
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();

    if api_key.is_empty() {
        let stream = create_assistant_message_event_stream();
        let mut msg = AssistantMessage::empty(model);
        msg.stop_reason = StopReason::Error;
        msg.error_message = Some(format!("No API key for provider: {}", model.provider));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg,
        });
        return stream;
    }

    let base = build_base_options(model, options, &api_key);
    let reasoning_effort = options.reasoning.as_ref().map(|level| {
        if supports_xhigh(model) {
            level.to_string()
        } else {
            clamp_reasoning(level).to_string()
        }
    });

    let azure_opts = AzureOpenAIResponsesOptions {
        base,
        reasoning_effort,
        reasoning_summary: None,
        azure_api_version: None,
        azure_resource_name: None,
        azure_base_url: None,
        azure_deployment_name: None,
    };

    stream_azure_openai_responses(model, context, &azure_opts, cancel)
}

// =============================================================================
// ApiProvider implementation
// =============================================================================

pub struct AzureOpenAIResponsesProvider;

impl ApiProvider for AzureOpenAIResponsesProvider {
    fn api(&self) -> &str {
        "azure-openai-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let azure_opts = AzureOpenAIResponsesOptions {
            base: options.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: None,
            azure_resource_name: None,
            azure_base_url: None,
            azure_deployment_name: None,
        };
        stream_azure_openai_responses(model, context, &azure_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_azure_openai_responses(model, context, options, cancel)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> Model {
        Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            api: "azure-openai-responses".to_string(),
            provider: "azure-openai-responses".to_string(),
            base_url: "https://myresource.openai.azure.com/openai/v1".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_parse_deployment_name_map_basic() {
        let map = parse_deployment_name_map(Some("gpt-4o=my-gpt4o-deployment,o3-mini=my-o3"));
        assert_eq!(map.get("gpt-4o"), Some(&"my-gpt4o-deployment".to_string()));
        assert_eq!(map.get("o3-mini"), Some(&"my-o3".to_string()));
    }

    #[test]
    fn test_parse_deployment_name_map_empty() {
        let map = parse_deployment_name_map(None);
        assert!(map.is_empty());
        let map2 = parse_deployment_name_map(Some(""));
        assert!(map2.is_empty());
    }

    #[test]
    fn test_parse_deployment_name_map_whitespace() {
        let map = parse_deployment_name_map(Some(" model1 = deploy1 , model2 = deploy2 "));
        assert_eq!(map.get("model1"), Some(&"deploy1".to_string()));
        assert_eq!(map.get("model2"), Some(&"deploy2".to_string()));
    }

    #[test]
    fn test_parse_deployment_name_map_invalid_entries() {
        let map = parse_deployment_name_map(Some("valid=deploy,invalid,,=bad,also_bad="));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("valid"), Some(&"deploy".to_string()));
    }

    #[test]
    fn test_resolve_deployment_name_from_options() {
        let model = test_model();
        let opts = AzureOpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: None,
            azure_resource_name: None,
            azure_base_url: None,
            azure_deployment_name: Some("my-custom-deployment".to_string()),
        };
        assert_eq!(
            resolve_deployment_name(&model, Some(&opts)),
            "my-custom-deployment"
        );
    }

    #[test]
    fn test_resolve_deployment_name_fallback_to_model_id() {
        let model = test_model();
        let opts = AzureOpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: None,
            azure_resource_name: None,
            azure_base_url: None,
            azure_deployment_name: None,
        };
        assert_eq!(resolve_deployment_name(&model, Some(&opts)), "gpt-4o");
    }

    #[test]
    fn test_normalize_azure_base_url() {
        assert_eq!(
            normalize_azure_base_url("https://example.com/openai/v1/"),
            "https://example.com/openai/v1"
        );
        assert_eq!(
            normalize_azure_base_url("https://example.com/openai/v1"),
            "https://example.com/openai/v1"
        );
    }

    #[test]
    fn test_build_default_base_url() {
        assert_eq!(
            build_default_base_url("myresource"),
            "https://myresource.openai.azure.com/openai/v1"
        );
    }

    #[test]
    fn test_resolve_azure_config_with_base_url_option() {
        let model = test_model();
        let opts = AzureOpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: Some("2024-06-01".to_string()),
            azure_resource_name: None,
            azure_base_url: Some("https://custom.azure.com/openai".to_string()),
            azure_deployment_name: None,
        };
        let (url, version) = resolve_azure_config(&model, Some(&opts)).unwrap();
        assert_eq!(url, "https://custom.azure.com/openai");
        assert_eq!(version, "2024-06-01");
    }

    #[test]
    fn test_resolve_azure_config_fallback_to_model_base_url() {
        let model = test_model();
        let opts = AzureOpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: None,
            azure_resource_name: None,
            azure_base_url: None,
            azure_deployment_name: None,
        };
        let (url, version) = resolve_azure_config(&model, Some(&opts)).unwrap();
        assert_eq!(url, "https://myresource.openai.azure.com/openai/v1");
        assert_eq!(version, DEFAULT_AZURE_API_VERSION);
    }

    #[test]
    fn test_resolve_azure_config_no_url() {
        let mut model = test_model();
        model.base_url = String::new();
        let opts = AzureOpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            azure_api_version: None,
            azure_resource_name: None,
            azure_base_url: None,
            azure_deployment_name: None,
        };
        let result = resolve_azure_config(&model, Some(&opts));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_azure_responses_url() {
        assert_eq!(
            resolve_azure_responses_url("https://myresource.openai.azure.com/openai/v1"),
            "https://myresource.openai.azure.com/openai/v1/responses"
        );
    }

    #[test]
    fn test_build_params_basic() {
        let model = test_model();
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![],
            tools: None,
        };
        let params = build_params(&model, &context, None, "my-deployment");
        assert_eq!(params["model"], "my-deployment");
        assert_eq!(params["stream"], true);
    }
}
