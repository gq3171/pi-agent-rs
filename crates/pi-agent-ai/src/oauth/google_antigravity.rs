use crate::oauth::pkce::{OAuthFlowStart, build_authorization_url, exchange_authorization_code};
use crate::oauth::types::{OAuthCredentials, OAuthProvider};

/// Google Antigravity OAuth configuration.
pub struct GoogleAntigravityOAuthConfig {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
}

impl GoogleAntigravityOAuthConfig {
    pub fn new(client_id: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            ..Default::default()
        }
    }
}

impl Default for GoogleAntigravityOAuthConfig {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            redirect_uri: "http://localhost:8787/oauth/callback".to_string(),
        }
    }
}

/// Start the Google Antigravity OAuth flow.
pub fn start_google_antigravity_oauth(
    config: &GoogleAntigravityOAuthConfig,
) -> Result<OAuthFlowStart, Box<dyn std::error::Error + Send + Sync>> {
    build_authorization_url(
        &config.auth_url,
        &config.client_id,
        &config.redirect_uri,
        "https://www.googleapis.com/auth/generative-language",
        &[("access_type", "offline")],
    )
}

/// Exchange an authorization code for Google Antigravity OAuth credentials.
pub async fn exchange_google_antigravity_code(
    config: &GoogleAntigravityOAuthConfig,
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
        OAuthProvider::GoogleAntigravity,
    ))
}
