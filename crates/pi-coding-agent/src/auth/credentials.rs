use serde::{Deserialize, Serialize};

/// Represents an authentication credential for an API provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AuthCredential {
    /// A plain API key.
    #[serde(rename = "apiKey")]
    ApiKey { key: String },

    /// An OAuth credential.
    #[serde(rename = "oauth")]
    OAuth {
        access_token: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        refresh_token: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        expires_at: Option<i64>,
    },
}

impl AuthCredential {
    /// Create a new API key credential.
    pub fn api_key(key: impl Into<String>) -> Self {
        AuthCredential::ApiKey { key: key.into() }
    }

    /// Create a new OAuth credential.
    pub fn oauth(access_token: impl Into<String>) -> Self {
        AuthCredential::OAuth {
            access_token: access_token.into(),
            refresh_token: None,
            expires_at: None,
        }
    }

    /// Get the token/key string to use for API calls.
    pub fn token(&self) -> &str {
        match self {
            AuthCredential::ApiKey { key } => key,
            AuthCredential::OAuth { access_token, .. } => access_token,
        }
    }

    /// Check if this credential is expired (always false for API keys).
    pub fn is_expired(&self) -> bool {
        match self {
            AuthCredential::ApiKey { .. } => false,
            AuthCredential::OAuth { expires_at, .. } => {
                if let Some(exp) = expires_at {
                    let now = chrono::Utc::now().timestamp();
                    now >= *exp
                } else {
                    false
                }
            }
        }
    }
}
