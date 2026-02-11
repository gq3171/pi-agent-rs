use crate::oauth::pkce::{build_authorization_url, exchange_authorization_code, OAuthFlowStart};
use crate::oauth::types::{OAuthCredentials, OAuthProvider};

/// GitHub Copilot OAuth configuration.
pub struct GithubCopilotOAuthConfig {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
}

impl GithubCopilotOAuthConfig {
    pub fn new(client_id: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            ..Default::default()
        }
    }
}

impl Default for GithubCopilotOAuthConfig {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            auth_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            redirect_uri: "http://localhost:8787/oauth/callback".to_string(),
        }
    }
}

/// Start the GitHub Copilot OAuth flow.
pub fn start_github_copilot_oauth(config: &GithubCopilotOAuthConfig) -> OAuthFlowStart {
    build_authorization_url(
        &config.auth_url,
        &config.client_id,
        &config.redirect_uri,
        "copilot",
        &[],
    )
}

/// Exchange an authorization code for GitHub Copilot OAuth credentials.
pub async fn exchange_github_copilot_code(
    config: &GithubCopilotOAuthConfig,
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
        &[("Accept", "application/json")],
    )
    .await?;

    Ok(OAuthCredentials::from_token_response(token, OAuthProvider::GithubCopilot))
}
