use crate::oauth::pkce::{OAuthFlowStart, build_authorization_url, exchange_authorization_code};
use crate::oauth::types::{OAuthCredentials, OAuthProvider};

/// Anthropic OAuth configuration.
pub struct AnthropicOAuthConfig {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
}

impl AnthropicOAuthConfig {
    pub fn new(client_id: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            ..Default::default()
        }
    }
}

impl Default for AnthropicOAuthConfig {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            auth_url: "https://console.anthropic.com/oauth/authorize".to_string(),
            token_url: "https://console.anthropic.com/oauth/token".to_string(),
            redirect_uri: "http://localhost:8787/oauth/callback".to_string(),
        }
    }
}

/// Start the Anthropic OAuth flow.
///
/// Returns the authorization URL, PKCE challenge, and state parameter.
/// The caller must:
/// 1. Open `auth_url` in a browser
/// 2. Receive the callback with `code` and `state`
/// 3. Verify `state` matches the returned state
/// 4. Call `exchange_anthropic_code()` with the code and `pkce.code_verifier`
pub fn start_anthropic_oauth(
    config: &AnthropicOAuthConfig,
) -> Result<OAuthFlowStart, Box<dyn std::error::Error + Send + Sync>> {
    build_authorization_url(
        &config.auth_url,
        &config.client_id,
        &config.redirect_uri,
        "api",
        &[],
    )
}

/// Exchange an authorization code for Anthropic OAuth credentials.
pub async fn exchange_anthropic_code(
    config: &AnthropicOAuthConfig,
    code: &str,
    code_verifier: &str,
) -> Result<OAuthCredentials, Box<dyn std::error::Error + Send + Sync>> {
    let token = exchange_authorization_code(
        &config.token_url,
        &config.client_id,
        &config.redirect_uri,
        code,
        code_verifier,
        &[],
        &[],
    )
    .await?;

    Ok(OAuthCredentials::from_token_response(
        token,
        OAuthProvider::Anthropic,
    ))
}
