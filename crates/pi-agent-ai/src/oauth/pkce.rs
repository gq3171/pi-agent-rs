use base64::Engine;
use sha2::{Digest, Sha256};

/// PKCE (Proof Key for Code Exchange) parameters for OAuth 2.0 authorization code flow.
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    /// The code verifier (random string).
    pub code_verifier: String,
    /// The code challenge (SHA256 hash of verifier, base64url encoded).
    pub code_challenge: String,
    /// The challenge method (always "S256").
    pub code_challenge_method: String,
}

impl PkceChallenge {
    /// Generate a new PKCE challenge pair using cryptographic randomness.
    pub fn new() -> Result<Self, getrandom::Error> {
        let code_verifier = generate_verifier()?;
        let code_challenge = generate_challenge(&code_verifier);
        Ok(Self {
            code_verifier,
            code_challenge,
            code_challenge_method: "S256".to_string(),
        })
    }
}

/// Generate a cryptographically random state parameter for CSRF protection.
pub fn generate_state() -> Result<String, getrandom::Error> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// Generate a random code verifier (43-128 characters, URL-safe)
/// using cryptographic randomness.
fn generate_verifier() -> Result<String, getrandom::Error> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// Generate the code challenge from a code verifier (SHA256 + base64url).
fn generate_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let hash = hasher.finalize();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

/// URL-encode a string (percent encoding).
pub fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            _ => {
                result.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    result
}

/// Result of starting an OAuth flow — contains everything the caller needs.
#[derive(Debug, Clone)]
pub struct OAuthFlowStart {
    /// The authorization URL to open in a browser.
    pub auth_url: String,
    /// The PKCE challenge (caller needs code_verifier for the token exchange).
    pub pkce: PkceChallenge,
    /// The state parameter (caller must verify this in the callback).
    pub state: String,
}

/// Build an OAuth 2.0 authorization URL with PKCE.
pub fn build_authorization_url(
    auth_endpoint: &str,
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    extra_params: &[(&str, &str)],
) -> Result<OAuthFlowStart, Box<dyn std::error::Error + Send + Sync>> {
    let pkce = PkceChallenge::new().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("PKCE generation failed: {e}").into()
    })?;
    let state = generate_state().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("State generation failed: {e}").into()
    })?;

    let mut url = format!(
        "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}&code_challenge={}&code_challenge_method=S256",
        auth_endpoint,
        url_encode(client_id),
        url_encode(redirect_uri),
        url_encode(scope),
        url_encode(&state),
        url_encode(&pkce.code_challenge),
    );

    for (key, value) in extra_params {
        url.push_str(&format!("&{}={}", url_encode(key), url_encode(value)));
    }

    Ok(OAuthFlowStart {
        auth_url: url,
        pkce,
        state,
    })
}

/// Exchange an authorization code for tokens (generic for all providers).
///
/// Returns the raw `TokenResponse`. Caller converts to `OAuthCredentials`.
pub async fn exchange_authorization_code(
    token_endpoint: &str,
    client_id: &str,
    redirect_uri: &str,
    code: &str,
    code_verifier: &str,
    extra_form: &[(&str, &str)],
    extra_headers: &[(&str, &str)],
) -> Result<super::types::TokenResponse, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();

    let mut form_data: Vec<(&str, &str)> = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", client_id),
        ("code_verifier", code_verifier),
    ];
    form_data.extend_from_slice(extra_form);

    let mut builder = client.post(token_endpoint).form(&form_data);
    for (key, value) in extra_headers {
        builder = builder.header(*key, *value);
    }

    let response = builder.send().await?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OAuth token exchange failed (HTTP {status}): {body}").into());
    }

    let token: super::types::TokenResponse = response.json().await?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkce_challenge_generation() {
        let pkce = PkceChallenge::new().unwrap();
        assert!(!pkce.code_verifier.is_empty());
        assert!(!pkce.code_challenge.is_empty());
        assert_eq!(pkce.code_challenge_method, "S256");
        assert_ne!(pkce.code_verifier, pkce.code_challenge);
    }

    #[test]
    fn test_pkce_uniqueness() {
        let a = PkceChallenge::new().unwrap();
        let b = PkceChallenge::new().unwrap();
        assert_ne!(a.code_verifier, b.code_verifier);
    }

    #[test]
    fn test_pkce_deterministic_challenge() {
        let verifier = "test_verifier_string";
        let challenge = generate_challenge(verifier);
        let challenge2 = generate_challenge(verifier);
        assert_eq!(challenge, challenge2);
    }

    #[test]
    fn test_generate_state_uniqueness() {
        let s1 = generate_state().unwrap();
        let s2 = generate_state().unwrap();
        assert_ne!(s1, s2);
        assert!(!s1.is_empty());
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello"), "hello");
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a=b&c=d"), "a%3Db%26c%3Dd");
        assert_eq!(url_encode("http://localhost:8080"), "http%3A%2F%2Flocalhost%3A8080");
    }

    #[test]
    fn test_build_authorization_url() {
        let flow = build_authorization_url(
            "https://auth.example.com/authorize",
            "my-client",
            "http://localhost:8080/callback",
            "read write",
            &[],
        )
        .unwrap();
        assert!(flow.auth_url.starts_with("https://auth.example.com/authorize?"));
        assert!(flow.auth_url.contains("client_id=my-client"));
        assert!(flow.auth_url.contains("response_type=code"));
        assert!(flow.auth_url.contains("scope=read%20write"));
        assert!(flow.auth_url.contains(&format!("state={}", url_encode(&flow.state))));
        assert!(!flow.pkce.code_verifier.is_empty());
        assert!(!flow.state.is_empty());
    }

    #[test]
    fn test_build_authorization_url_with_extra_params() {
        let flow = build_authorization_url(
            "https://auth.example.com/authorize",
            "my-client",
            "http://localhost:8080/callback",
            "read",
            &[("access_type", "offline"), ("prompt", "consent")],
        )
        .unwrap();
        assert!(flow.auth_url.contains("access_type=offline"));
        assert!(flow.auth_url.contains("prompt=consent"));
    }
}
