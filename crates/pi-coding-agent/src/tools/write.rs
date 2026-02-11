use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::path_utils;

/// Trait for write operations — allows mocking in tests.
pub trait WriteOperations: Send + Sync {
    fn write_file(
        &self,
        path: &std::path::Path,
        content: &str,
    ) -> Result<WriteOutput, Box<dyn std::error::Error + Send + Sync>>;
}

/// Output of a write operation.
#[derive(Debug, Clone)]
pub struct WriteOutput {
    pub bytes_written: usize,
    pub created: bool,
}

/// Default file writer.
pub struct DefaultFileWriter;

impl WriteOperations for DefaultFileWriter {
    fn write_file(
        &self,
        path: &std::path::Path,
        content: &str,
    ) -> Result<WriteOutput, Box<dyn std::error::Error + Send + Sync>> {
        let created = !path.exists();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;

        Ok(WriteOutput {
            bytes_written: content.len(),
            created,
        })
    }
}

/// The Write tool for creating/overwriting files.
pub struct WriteTool {
    working_dir: PathBuf,
    writer: Arc<dyn WriteOperations>,
}

impl WriteTool {
    pub fn new(working_dir: PathBuf, writer: Arc<dyn WriteOperations>) -> Self {
        Self {
            working_dir,
            writer,
        }
    }

    pub fn with_default_writer(working_dir: PathBuf) -> Self {
        Self::new(working_dir, Arc::new(DefaultFileWriter))
    }
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn label(&self) -> &str {
        "Write"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "write".to_string(),
            description: "Write content to a file, creating it if it doesn't exist.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }),
        });
        &TOOL
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let file_path = params
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'file_path' parameter")?;
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' parameter")?;

        let resolved = path_utils::resolve_path(file_path, &self.working_dir);

        // Security: verify path is within working directory
        if !path_utils::is_within(&resolved, &self.working_dir) {
            return Err(format!(
                "Access denied: {} is outside the working directory",
                resolved.display()
            )
            .into());
        }

        let writer = self.writer.clone();
        let resolved_clone = resolved.clone();
        let content = content.to_string();
        let output = tokio::task::spawn_blocking(move || {
            writer.write_file(&resolved_clone, &content)
        })
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Task join error: {e}").into()
        })??;

        let action = if output.created { "Created" } else { "Updated" };
        let text = format!(
            "{action} file: {} ({} bytes)",
            resolved.display(),
            output.bytes_written
        );

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({
                "bytesWritten": output.bytes_written,
                "created": output.created,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_tool_create_file() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteTool::with_default_writer(tmp.path().to_path_buf());
        let file_path = tmp.path().join("new_file.txt");

        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "content": "Hello, World!"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("Created"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_tool_overwrite_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("existing.txt");
        std::fs::write(&file_path, "old content").unwrap();

        let tool = WriteTool::with_default_writer(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "content": "new content"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("Updated"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_write_tool_create_nested_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("a/b/c/deep.txt");

        let tool = WriteTool::with_default_writer(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "content": "deep content"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        assert!(result.content[0].as_text().unwrap().text.contains("Created"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "deep content");
    }
}
