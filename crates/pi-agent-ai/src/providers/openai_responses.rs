use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use pi_agent_core::types::*;

use crate::env_keys::get_env_api_key;
use crate::models::supports_xhigh;
use crate::registry::ApiProvider;
use crate::simple_options::{build_base_options, clamp_reasoning};
use crate::sse::SseParser;

use super::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools,
    process_responses_events, OpenAIResponsesStreamOptions,
};

// =============================================================================
// Constants
// =============================================================================

fn openai_tool_call_providers() -> HashSet<&'static str> {
    ["openai", "openai-codex", "opencode"].into_iter().collect()
}

// =============================================================================
// Options
// =============================================================================

/// OpenAI Responses-specific stream options.
#[derive(Debug, Clone)]
pub struct OpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<String>,   // "minimal" | "low" | "medium" | "high" | "xhigh"
    pub reasoning_summary: Option<String>,  // "auto" | "detailed" | "concise" | null
    pub service_tier: Option<String>,       // "flex" | "priority" | "auto" | etc.
}

// =============================================================================
// Cache retention helpers
// =============================================================================

/// Resolve cache retention preference.
/// Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
fn resolve_cache_retention(retention: Option<&CacheRetention>) -> CacheRetention {
    if let Some(r) = retention {
        return r.clone();
    }
    if let Ok(val) = std::env::var("PI_CACHE_RETENTION") {
        if val == "long" {
            return CacheRetention::Long;
        }
    }
    CacheRetention::Short
}

/// Get prompt cache retention based on cacheRetention and base URL.
/// Only applies to direct OpenAI API calls (api.openai.com).
fn get_prompt_cache_retention(base_url: &str, cache_retention: &CacheRetention) -> Option<String> {
    if *cache_retention != CacheRetention::Long {
        return None;
    }
    if base_url.contains("api.openai.com") {
        return Some("24h".to_string());
    }
    None
}

// =============================================================================
// Service tier pricing
// =============================================================================

fn get_service_tier_cost_multiplier(service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => 2.0,
        _ => 1.0,
    }
}

fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&str>) {
    let multiplier = get_service_tier_cost_multiplier(service_tier);
    if (multiplier - 1.0).abs() < f64::EPSILON {
        return;
    }
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

// =============================================================================
// HTTP helpers
// =============================================================================

fn build_headers(
    model: &Model,
    context: &Context,
    api_key: &str,
    extra_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("authorization".to_string(), format!("Bearer {api_key}"));

    // Copy model headers (skip sensitive auth headers)
    if let Some(model_headers) = &model.headers {
        crate::header_utils::merge_headers_safe(&mut headers, model_headers);
    }

    // GitHub Copilot specific headers
    if model.provider == "github-copilot" {
        let messages = &context.messages;
        let last_msg = messages.last();
        let is_agent_call = last_msg.map(|m| m.role() != "user").unwrap_or(false);
        headers.insert(
            "X-Initiator".to_string(),
            if is_agent_call { "agent" } else { "user" }.to_string(),
        );
        headers.insert("Openai-Intent".to_string(), "conversation-edits".to_string());

        // Copilot requires this header when sending images
        let has_images = messages.iter().any(|msg| match msg {
            Message::User(u) => {
                if let UserContent::Blocks(blocks) = &u.content {
                    blocks.iter().any(|c| c.as_image().is_some())
                } else {
                    false
                }
            }
            Message::ToolResult(tr) => tr.content.iter().any(|c| c.as_image().is_some()),
            _ => false,
        });
        if has_images {
            headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
        }
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
    options: Option<&OpenAIResponsesOptions>,
) -> Value {
    let providers = openai_tool_call_providers();
    let input_messages = convert_responses_messages(model, context, &providers, None);

    let cache_retention = resolve_cache_retention(
        options.and_then(|o| o.base.cache_retention.as_ref()),
    );

    let mut params = json!({
        "model": model.id,
        "input": input_messages,
        "stream": true,
        "store": false,
    });

    // Prompt cache settings
    let is_none_retention = cache_retention == CacheRetention::None;
    if !is_none_retention {
        if let Some(opts) = options {
            if let Some(session_id) = &opts.base.session_id {
                params["prompt_cache_key"] = json!(session_id);
            }
        }
    }
    if let Some(retention_str) = get_prompt_cache_retention(&model.base_url, &cache_retention) {
        params["prompt_cache_retention"] = json!(retention_str);
    }

    if let Some(opts) = options {
        if let Some(max_tokens) = opts.base.max_tokens {
            params["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = opts.base.temperature {
            params["temperature"] = json!(temp);
        }
        if let Some(tier) = &opts.service_tier {
            params["service_tier"] = json!(tier);
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
        } else if model.name.starts_with("gpt-5") {
            // Workaround for GPT-5 models that require explicit reasoning disable
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

fn resolve_url(base_url: &str) -> String {
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
// Stream functions
// =============================================================================

/// Stream from OpenAI Responses API using raw HTTP + SSE parsing.
pub fn stream_openai_responses(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    cancel: CancellationToken,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&model);

        let api_key = options
            .base
            .api_key
            .clone()
            .or_else(|| get_env_api_key(&model.provider))
            .unwrap_or_default();

        if api_key.is_empty() {
            output.stop_reason = StopReason::Error;
            output.error_message = Some(format!(
                "OpenAI API key is required. Set the appropriate environment variable or pass it as an argument. Provider: {}",
                model.provider
            ));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output,
            });
            return;
        }

        let headers = build_headers(&model, &context, &api_key, options.base.headers.as_ref());
        let params = build_params(&model, &context, Some(&options));
        let url = resolve_url(&model.base_url);

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
        let service_tier_clone = options.service_tier.clone();
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

            let stream_opts = OpenAIResponsesStreamOptions {
                service_tier: service_tier_clone.clone(),
                apply_service_tier_pricing: Some(Box::new(apply_service_tier_pricing)),
            };

            if let Err(err) = process_responses_events(
                &parsed_events,
                &mut output,
                &stream_clone,
                &model,
                &mut current_item,
                &mut current_block,
                Some(&stream_opts),
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
pub fn stream_simple_openai_responses(
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

    let openai_opts = OpenAIResponsesOptions {
        base,
        reasoning_effort,
        reasoning_summary: None,
        service_tier: None,
    };

    stream_openai_responses(model, context, &openai_opts, cancel)
}

// =============================================================================
// ApiProvider implementation
// =============================================================================

pub struct OpenAIResponsesProvider;

impl ApiProvider for OpenAIResponsesProvider {
    fn api(&self) -> &str {
        "openai-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        let openai_opts = OpenAIResponsesOptions {
            base: options.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            service_tier: None,
        };
        stream_openai_responses(model, context, &openai_opts, cancel)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
        cancel: CancellationToken,
    ) -> AssistantMessageEventStream {
        stream_simple_openai_responses(model, context, options, cancel)
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
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
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
    fn test_resolve_url_base() {
        assert_eq!(
            resolve_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn test_resolve_url_already_has_responses() {
        assert_eq!(
            resolve_url("https://api.openai.com/v1/responses"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn test_resolve_url_no_v1() {
        assert_eq!(
            resolve_url("https://custom.api.com"),
            "https://custom.api.com/v1/responses"
        );
    }

    #[test]
    fn test_resolve_url_trailing_slash() {
        assert_eq!(
            resolve_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn test_get_service_tier_cost_multiplier_flex() {
        assert!((get_service_tier_cost_multiplier(Some("flex")) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_service_tier_cost_multiplier_priority() {
        assert!((get_service_tier_cost_multiplier(Some("priority")) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_service_tier_cost_multiplier_default() {
        assert!((get_service_tier_cost_multiplier(None) - 1.0).abs() < f64::EPSILON);
        assert!((get_service_tier_cost_multiplier(Some("auto")) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_service_tier_pricing_flex() {
        let mut usage = Usage {
            input: 100,
            output: 50,
            cache_read: 10,
            cache_write: 0,
            total_tokens: 160,
            cost: UsageCost {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 0.0,
                total: 3.5,
            },
        };
        apply_service_tier_pricing(&mut usage, Some("flex"));
        assert!((usage.cost.input - 0.5).abs() < f64::EPSILON);
        assert!((usage.cost.output - 1.0).abs() < f64::EPSILON);
        assert!((usage.cost.cache_read - 0.25).abs() < f64::EPSILON);
        assert!((usage.cost.total - 1.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_service_tier_pricing_no_change() {
        let mut usage = Usage {
            input: 100,
            output: 50,
            cache_read: 10,
            cache_write: 0,
            total_tokens: 160,
            cost: UsageCost {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 0.0,
                total: 3.5,
            },
        };
        apply_service_tier_pricing(&mut usage, Some("auto"));
        assert!((usage.cost.input - 1.0).abs() < f64::EPSILON);
        assert!((usage.cost.total - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_prompt_cache_retention_long_openai() {
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", &CacheRetention::Long),
            Some("24h".to_string())
        );
    }

    #[test]
    fn test_get_prompt_cache_retention_long_non_openai() {
        assert_eq!(
            get_prompt_cache_retention("https://custom.api.com/v1", &CacheRetention::Long),
            None
        );
    }

    #[test]
    fn test_get_prompt_cache_retention_short() {
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", &CacheRetention::Short),
            None
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
        let opts = OpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: None,
            reasoning_summary: None,
            service_tier: None,
        };
        let params = build_params(&model, &context, Some(&opts));
        assert_eq!(params["model"], "gpt-4o");
        assert_eq!(params["stream"], true);
        assert_eq!(params["store"], false);
    }

    #[test]
    fn test_build_params_with_reasoning() {
        let mut model = test_model();
        model.reasoning = true;
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        let opts = OpenAIResponsesOptions {
            base: StreamOptions::default(),
            reasoning_effort: Some("high".to_string()),
            reasoning_summary: Some("auto".to_string()),
            service_tier: None,
        };
        let params = build_params(&model, &context, Some(&opts));
        assert_eq!(params["reasoning"]["effort"], "high");
        assert_eq!(params["reasoning"]["summary"], "auto");
    }

    #[test]
    fn test_openai_tool_call_providers() {
        let providers = openai_tool_call_providers();
        assert!(providers.contains("openai"));
        assert!(providers.contains("openai-codex"));
        assert!(providers.contains("opencode"));
        assert!(!providers.contains("anthropic"));
    }
}
