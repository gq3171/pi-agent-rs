use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pi_agent_core::agent_types::AgentMessage;

/// Defines a tool that can be provided by an extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    /// Tool name (must be unique).
    pub name: String,
    /// Display label.
    pub label: String,
    /// Description for the LLM.
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: Value,
}

/// Context provided to extensions.
#[derive(Debug, Clone)]
pub struct ExtensionContext {
    /// Working directory.
    pub working_dir: PathBuf,
    /// Session ID (if available).
    pub session_id: Option<String>,
    /// Current model ID.
    pub model_id: Option<String>,
    /// Extension-specific configuration from settings.
    pub config: Value,
}

/// API surface available to extensions.
#[async_trait]
pub trait ExtensionAPI: Send + Sync {
    /// Get the current conversation messages.
    fn messages(&self) -> &[AgentMessage];

    /// Get the working directory.
    fn working_dir(&self) -> &std::path::Path;

    /// Log a message (visible in debug output).
    fn log(&self, message: &str);
}

/// Events that extensions can receive.
#[derive(Debug, Clone)]
pub enum ContextEvent {
    /// A new turn started.
    TurnStart,
    /// A turn ended.
    TurnEnd,
    /// A file was read.
    FileRead { path: String },
    /// A file was written.
    FileWritten { path: String },
    /// A file was edited.
    FileEdited { path: String },
    /// A command was executed.
    CommandExecuted { command: String, exit_code: Option<i32> },
}

/// File operations interface for extensions.
#[async_trait]
pub trait FileOperations: Send + Sync {
    /// Read a file.
    async fn read_file(&self, path: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    /// Write a file.
    async fn write_file(
        &self,
        path: &str,
        content: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    /// Check if a file exists.
    async fn file_exists(&self, path: &str) -> bool;
    /// List files in a directory.
    async fn list_files(
        &self,
        dir: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Extension trait — implement this to create a coding agent extension.
///
/// NOTE: This is intentionally minimal (interface only). The actual extension
/// runner will be implemented by OpenClaw or other consumers.
#[async_trait]
pub trait Extension: Send + Sync {
    /// Extension name.
    fn name(&self) -> &str;

    /// Initialize the extension with context.
    async fn init(&mut self, context: ExtensionContext) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Get tool definitions provided by this extension.
    fn tools(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    /// Handle a tool call from the LLM.
    async fn handle_tool_call(
        &self,
        _tool_name: &str,
        _params: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("Not implemented".into())
    }

    /// Handle a context event.
    async fn on_event(&self, _event: ContextEvent) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Shutdown the extension.
    async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}
