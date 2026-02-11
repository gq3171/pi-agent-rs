use serde::{Deserialize, Serialize};

/// OAuth provider identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OAuthProvider {
    Anthropic,
    GithubCopilot,
    OpenaiCodex,
    GoogleAntigravity,
    GoogleGeminiCli,
}

impl OAuthProvider {
    /// Get the provider display name.
    pub fn display_name(&self) -> &str {
        match self {
            OAuthProvider::Anthropic => "Anthropic",
            OAuthProvider::GithubCopilot => "GitHub Copilot",
            OAuthProvider::OpenaiCodex => "OpenAI Codex",
            OAuthProvider::GoogleAntigravity => "Google Antigravity",
            OAuthProvider::GoogleGeminiCli => "Google Gemini CLI",
        }
    }

    /// Get the corresponding pi-agent-core provider key.
    pub fn provider_key(&self) -> &str {
        match self {
            OAuthProvider::Anthropic => "anthropic",
            OAuthProvider::GithubCopilot => "github-copilot",
            OAuthProvider::OpenaiCodex => "openai-codex",
            OAuthProvider::GoogleAntigravity => "google-antigravity",
            OAuthProvider::GoogleGeminiCli => "google-gemini-cli",
        }
    }
}

/// OAuth credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredentials {
    /// Access token for API calls.
    pub access_token: String,
    /// Refresh token (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Expiration timestamp (Unix seconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// Token type (usually "Bearer").
    #[serde(default = "default_token_type")]
    pub token_type: String,
    /// Scopes granted.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// The provider this credential is for.
    pub provider: OAuthProvider,
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

impl OAuthCredentials {
    /// Create `OAuthCredentials` from a raw `TokenResponse`.
    pub fn from_token_response(token: TokenResponse, provider: OAuthProvider) -> Self {
        let expires_at = token
            .expires_in
            .map(|secs| chrono::Utc::now().timestamp() + secs as i64);

        let scopes = token
            .scope
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default();

        Self {
            access_token: token.access_token,
            refresh_token: token.refresh_token,
            expires_at,
            token_type: token.token_type,
            scopes,
            provider,
        }
    }

    /// Check if the credentials are expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = chrono::Utc::now().timestamp();
            // Consider expired 60 seconds before actual expiry
            now >= (expires_at - 60)
        } else {
            false
        }
    }

    /// Get the Authorization header value.
    pub fn authorization_header(&self) -> String {
        format!("{} {}", self.token_type, self.access_token)
    }
}

/// OAuth token response from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<u64>,
    #[serde(default = "default_token_type")]
    pub token_type: String,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Get available OAuth providers.
pub fn get_oauth_providers() -> Vec<OAuthProvider> {
    vec![
        OAuthProvider::Anthropic,
        OAuthProvider::GithubCopilot,
        OAuthProvider::OpenaiCodex,
        OAuthProvider::GoogleAntigravity,
        OAuthProvider::GoogleGeminiCli,
    ]
}

/// Get the OAuth API key for a provider from the stored credentials.
///
/// This looks up the stored OAuth credentials for the given provider
/// and returns the access token as the API key.
pub fn get_oauth_api_key(
    provider: &OAuthProvider,
    credentials: &OAuthCredentials,
) -> Option<String> {
    if credentials.provider == *provider && !credentials.is_expired() {
        Some(credentials.access_token.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oauth_providers() {
        let providers = get_oauth_providers();
        assert_eq!(providers.len(), 5);
    }

    #[test]
    fn test_credentials_not_expired() {
        let creds = OAuthCredentials {
            access_token: "test-token".to_string(),
            refresh_token: None,
            expires_at: Some(chrono::Utc::now().timestamp() + 3600),
            token_type: "Bearer".to_string(),
            scopes: vec![],
            provider: OAuthProvider::Anthropic,
        };
        assert!(!creds.is_expired());
        assert_eq!(creds.authorization_header(), "Bearer test-token");
    }

    #[test]
    fn test_credentials_expired() {
        let creds = OAuthCredentials {
            access_token: "old-token".to_string(),
            refresh_token: None,
            expires_at: Some(1000000000),
            token_type: "Bearer".to_string(),
            scopes: vec![],
            provider: OAuthProvider::Anthropic,
        };
        assert!(creds.is_expired());
    }

    #[test]
    fn test_get_oauth_api_key() {
        let creds = OAuthCredentials {
            access_token: "valid-key".to_string(),
            refresh_token: None,
            expires_at: Some(chrono::Utc::now().timestamp() + 3600),
            token_type: "Bearer".to_string(),
            scopes: vec![],
            provider: OAuthProvider::Anthropic,
        };

        assert_eq!(
            get_oauth_api_key(&OAuthProvider::Anthropic, &creds),
            Some("valid-key".to_string())
        );
        assert_eq!(get_oauth_api_key(&OAuthProvider::OpenaiCodex, &creds), None);
    }

    #[test]
    fn test_from_token_response() {
        let token = TokenResponse {
            access_token: "access-123".to_string(),
            refresh_token: Some("refresh-456".to_string()),
            expires_in: Some(3600),
            token_type: "Bearer".to_string(),
            scope: Some("read write".to_string()),
        };

        let creds = OAuthCredentials::from_token_response(token, OAuthProvider::OpenaiCodex);
        assert_eq!(creds.access_token, "access-123");
        assert_eq!(creds.refresh_token.as_deref(), Some("refresh-456"));
        assert!(creds.expires_at.is_some());
        assert_eq!(creds.token_type, "Bearer");
        assert_eq!(creds.scopes, vec!["read", "write"]);
        assert_eq!(creds.provider, OAuthProvider::OpenaiCodex);
        assert!(!creds.is_expired());
    }

    #[test]
    fn test_from_token_response_minimal() {
        let token = TokenResponse {
            access_token: "tok".to_string(),
            refresh_token: None,
            expires_in: None,
            token_type: "Bearer".to_string(),
            scope: None,
        };

        let creds = OAuthCredentials::from_token_response(token, OAuthProvider::Anthropic);
        assert_eq!(creds.access_token, "tok");
        assert!(creds.refresh_token.is_none());
        assert!(creds.expires_at.is_none());
        assert!(creds.scopes.is_empty());
        assert!(!creds.is_expired()); // no expiry means not expired
    }
}
