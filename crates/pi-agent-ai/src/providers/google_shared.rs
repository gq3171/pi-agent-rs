//! Google shared utilities for OAuth and API access across Google providers
//! (Google Generative AI, Google Vertex, Google Gemini CLI, Google Antigravity).

/// Google OAuth scopes.
pub mod scopes {
    pub const GENERATIVE_LANGUAGE: &str = "https://www.googleapis.com/auth/generative-language";
    pub const CLOUD_PLATFORM: &str = "https://www.googleapis.com/auth/cloud-platform";
}

/// Google API endpoints.
pub mod endpoints {
    pub const GENERATIVE_AI_BASE: &str = "https://generativelanguage.googleapis.com";
    pub const VERTEX_AI_BASE: &str = "https://{region}-aiplatform.googleapis.com";
    pub const OAUTH_AUTH: &str = "https://accounts.google.com/o/oauth2/v2/auth";
    pub const OAUTH_TOKEN: &str = "https://oauth2.googleapis.com/token";
}

/// Build a Google Vertex AI endpoint URL.
pub fn vertex_endpoint(region: &str) -> String {
    endpoints::VERTEX_AI_BASE.replace("{region}", region)
}

/// Get the appropriate API key environment variable for a Google provider.
pub fn google_api_key_env(provider: &str) -> &'static str {
    match provider {
        "google-gemini-cli" => "GEMINI_API_KEY",
        "google-vertex" => "GOOGLE_APPLICATION_CREDENTIALS",
        _ => "GOOGLE_API_KEY",
    }
}

/// Check if a model ID is a Google model.
pub fn is_google_model(model_id: &str) -> bool {
    let id = model_id.to_lowercase();
    id.starts_with("gemini-")
        || id.starts_with("models/gemini-")
        || id.starts_with("google/")
        || id.starts_with("google-")
}
