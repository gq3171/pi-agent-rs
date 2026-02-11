use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Top-level settings structure, compatible with TS settings.json.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Default model ID to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Custom models configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_models: Option<Vec<CustomModelConfig>>,

    /// Compaction settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionSettings>,

    /// Retry settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetrySettings>,

    /// Thinking level setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    /// Temperature setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Max tokens per response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// System prompt override or additions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Additional system prompt to append.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append_system_prompt: Option<String>,

    /// Cache retention strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<String>,

    /// Permissions for tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<PermissionSettings>,

    /// Provider-specific settings (API keys, base URLs, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub providers: Option<HashMap<String, ProviderSettings>>,

    /// Extension configurations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<HashMap<String, Value>>,

    /// Any additional fields not covered above.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Compaction settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
    /// Threshold ratio of context used before triggering compaction (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// Whether to auto-compact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto: Option<bool>,

    /// Model to use for compaction (if different from main model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Retry settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrySettings {
    /// Maximum number of retries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,

    /// Maximum delay between retries in ms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_delay_ms: Option<u64>,

    /// Base delay for exponential backoff in ms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_delay_ms: Option<u64>,
}

/// Permission settings for tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionSettings {
    /// Tools that are always allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow: Option<Vec<String>>,

    /// Tools that are always denied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deny: Option<Vec<String>>,

    /// Default permission mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_mode: Option<String>,
}

/// Provider-specific settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSettings {
    /// API key for this provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Base URL override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Custom headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    /// Any extra fields.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Custom model configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomModelConfig {
    /// Glob or exact model ID pattern.
    pub id: String,

    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// API to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,

    /// Provider to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// Base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Whether model supports reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,

    /// Context window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,

    /// Max output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Custom headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}
