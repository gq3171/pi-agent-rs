use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, ImageContent, TextContent, Tool};

use base64::Engine;

use crate::tools::path_utils;
use crate::tools::truncate;

/// Trait for read operations — allows mocking in tests.
pub trait ReadOperations: Send + Sync {
    fn read_file(
        &self,
        path: &std::path::Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<ReadOutput, Box<dyn std::error::Error + Send + Sync>>;
}

/// Output of a read operation.
#[derive(Debug, Clone)]
pub struct ReadOutput {
    pub content: String,
    pub total_lines: usize,
    pub is_binary: bool,
    pub is_image: bool,
    /// Base64 image data (only populated for images).
    pub image_data: Option<String>,
    pub mime_type: Option<String>,
}

/// Default file reader.
pub struct DefaultFileReader;

/// Maximum file size for text files (10 MB).
const MAX_TEXT_FILE_SIZE: u64 = 10 * 1024 * 1024;
/// Maximum file size for image files (20 MB).
const MAX_IMAGE_FILE_SIZE: u64 = 20 * 1024 * 1024;

impl ReadOperations for DefaultFileReader {
    fn read_file(
        &self,
        path: &std::path::Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<ReadOutput, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Err(format!("File not found: {}", path.display()).into());
        }

        // Reject non-regular files (FIFO, device, socket) to prevent blocking
        let metadata = std::fs::metadata(path)?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "Not a regular file: {} (type: {:?})",
                path.display(),
                metadata.file_type()
            )
            .into());
        }

        // Check file size before reading to prevent memory exhaustion
        let file_size = metadata.len();

        if path_utils::is_image(path) && file_size > MAX_IMAGE_FILE_SIZE {
            return Err(format!(
                "Image too large: {} bytes (limit: {} bytes)",
                file_size, MAX_IMAGE_FILE_SIZE
            )
            .into());
        }

        if !path_utils::is_image(path)
            && !path_utils::is_likely_binary(path)
            && file_size > MAX_TEXT_FILE_SIZE
        {
            return Err(format!(
                "File too large: {} bytes (limit: {} bytes)",
                file_size, MAX_TEXT_FILE_SIZE
            )
            .into());
        }

        if path_utils::is_image(path) {
            let data = std::fs::read(path)?;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
            let mime = match ext.to_lowercase().as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "svg" => "image/svg+xml",
                "bmp" => "image/bmp",
                "ico" => "image/x-icon",
                _ => "application/octet-stream",
            };

            return Ok(ReadOutput {
                content: String::new(),
                total_lines: 0,
                is_binary: true,
                is_image: true,
                image_data: Some(b64),
                mime_type: Some(mime.to_string()),
            });
        }

        if path_utils::is_likely_binary(path) {
            return Ok(ReadOutput {
                content: format!("[Binary file: {}]", path.display()),
                total_lines: 0,
                is_binary: true,
                is_image: false,
                image_data: None,
                mime_type: None,
            });
        }

        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(truncate::DEFAULT_MAX_LINES);

        let selected: Vec<&str> = lines.iter().skip(offset).take(limit).copied().collect();

        // Format with line numbers
        let numbered: Vec<String> = selected
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>6}\t{line}", offset + i + 1))
            .collect();
        let result = numbered.join("\n");

        Ok(ReadOutput {
            content: result,
            total_lines,
            is_binary: false,
            is_image: false,
            image_data: None,
            mime_type: None,
        })
    }
}

/// The Read tool for reading files.
pub struct ReadTool {
    working_dir: PathBuf,
    reader: Arc<dyn ReadOperations>,
}

impl ReadTool {
    pub fn new(working_dir: PathBuf, reader: Arc<dyn ReadOperations>) -> Self {
        Self {
            working_dir,
            reader,
        }
    }

    pub fn with_default_reader(working_dir: PathBuf) -> Self {
        Self::new(working_dir, Arc::new(DefaultFileReader))
    }
}

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn label(&self) -> &str {
        "Read"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "read".to_string(),
            description: "Read a file from the filesystem.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from (0-based)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of lines to read"
                    }
                },
                "required": ["file_path"]
            }),
        });
        &TOOL
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        cancel: CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let file_path = params
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'file_path' parameter")?;
        let offset = params
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let resolved = path_utils::resolve_path(file_path, &self.working_dir);

        // Security: verify path is within working directory
        if !path_utils::is_within(&resolved, &self.working_dir) {
            return Err(format!(
                "Access denied: {} is outside the working directory",
                resolved.display()
            )
            .into());
        }

        let reader = self.reader.clone();
        let resolved_clone = resolved.clone();
        let io_task =
            tokio::task::spawn_blocking(move || reader.read_file(&resolved_clone, offset, limit));

        let output = tokio::select! {
            result = io_task => {
                result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("Task join error: {e}").into()
                })??
            }
            _ = cancel.cancelled() => {
                return Err("Operation cancelled".into());
            }
        };

        if output.is_image {
            if let (Some(data), Some(mime)) = (output.image_data, output.mime_type) {
                return Ok(AgentToolResult {
                    content: vec![ContentBlock::Image(ImageContent {
                        data,
                        mime_type: mime,
                    })],
                    details: None,
                });
            }
        }

        // Truncate text output
        let truncated = truncate::truncate_output(&output.content, None, None);

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: truncated.content,
                text_signature: None,
            })],
            details: Some(json!({
                "totalLines": output.total_lines,
                "wasTruncated": truncated.was_truncated,
                "isBinary": output.is_binary,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_tool_text_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").unwrap();

        let tool = ReadTool::with_default_reader(tmp.path().to_path_buf());
        let params = json!({"file_path": file_path.to_str().unwrap()});
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("line one"));
        assert!(text.contains("line two"));
    }

    #[tokio::test]
    async fn test_read_tool_file_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadTool::with_default_reader(tmp.path().to_path_buf());
        let params = json!({"file_path": "/nonexistent/file.txt"});
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_tool_with_offset_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");
        let content: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadTool::with_default_reader(tmp.path().to_path_buf());
        let params = json!({"file_path": file_path.to_str().unwrap(), "offset": 10, "limit": 5});
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("line 11"));
        assert!(text.contains("line 15"));
        assert!(!text.contains("line 16"));
    }
}
