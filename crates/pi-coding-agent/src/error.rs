use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodingAgentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Session error: {0}")]
    Session(String),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Auth error: {0}")]
    Auth(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Compaction error: {0}")]
    Compaction(String),

    #[error("{0}")]
    Other(String),
}

impl From<String> for CodingAgentError {
    fn from(s: String) -> Self {
        CodingAgentError::Other(s)
    }
}

impl From<&str> for CodingAgentError {
    fn from(s: &str) -> Self {
        CodingAgentError::Other(s.to_string())
    }
}
