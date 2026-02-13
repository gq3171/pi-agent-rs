use crate::oauth::pkce::{OAuthFlowStart, build_authorization_url, exchange_authorization_code};
use crate::oauth::types::{OAuthCredentials, OAuthProvider};

/// OpenAI Codex OAuth configuration.
pub struct OpenAICodexOAuthConfig {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
}

impl OpenAICodexOAuthConfig {
    pub fn new(client_id: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            ..Default::default()
        }
    }
}

impl Default for OpenAICodexOAuthConfig {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            auth_url: "https://auth.openai.com/authorize".to_string(),
            token_url: "https://auth.openai.com/oauth/token".to_string(),
            redirect_uri: "http://localhost:8787/oauth/callback".to_string(),
        }
    }
}

/// Start the OpenAI Codex OAuth flow.
pub fn start_openai_codex_oauth(
    config: &OpenAICodexOAuthConfig,
) -> Result<OAuthFlowStart, Box<dyn std::error::Error + Send + Sync>> {
    build_authorization_url(
        &config.auth_url,
        &config.client_id,
        &config.redirect_uri,
        "openai.api",
        &[],
    )
}

/// Login to OpenAI Codex via OAuth — high-level function for OpenClaw.
///
/// Exchanges an authorization code for OAuth credentials.
/// The caller is responsible for:
/// 1. Starting the flow with `start_openai_codex_oauth()`
/// 2. Running a local HTTP server for the callback
/// 3. Opening the browser to the auth URL
/// 4. Verifying the state parameter in the callback
/// 5. Passing the received code and stored code_verifier here
pub async fn login_openai_codex(
    config: &OpenAICodexOAuthConfig,
    code: &str,
    code_verifier: &str,
) -> Result<OAuthCredentials, Box<dyn std::error::Error + Send + Sync>> {
    exchange_openai_codex_code(config, code, code_verifier).await
}

/// Exchange an authorization code for OpenAI Codex OAuth credentials.
pub async fn exchange_openai_codex_code(
    config: &OpenAICodexOAuthConfig,
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
        OAuthProvider::OpenaiCodex,
    ))
}
