use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::edit_diff;
use crate::tools::path_utils;

/// Trait for edit operations — allows mocking in tests.
pub trait EditOperations: Send + Sync {
    fn edit_file(
        &self,
        path: &std::path::Path,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> Result<EditOutput, Box<dyn std::error::Error + Send + Sync>>;
}

/// Output of an edit operation.
#[derive(Debug, Clone)]
pub struct EditOutput {
    pub replacements: usize,
    pub diff: String,
    pub original_content: String,
    pub new_content: String,
}

/// Default file editor.
pub struct DefaultFileEditor;

impl EditOperations for DefaultFileEditor {
    fn edit_file(
        &self,
        path: &std::path::Path,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> Result<EditOutput, Box<dyn std::error::Error + Send + Sync>> {
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

        if old_string.is_empty() {
            return Err("old_string must not be empty".into());
        }

        let original_content = std::fs::read_to_string(path)?;

        if old_string == new_string {
            return Err("old_string and new_string are identical".into());
        }

        let count = original_content.matches(old_string).count();
        if count == 0 {
            return Err(
                "old_string not found in file. Make sure it matches exactly (including whitespace)."
                    .into(),
            );
        }

        if !replace_all && count > 1 {
            return Err(format!(
                "old_string found {count} times in the file. Use replace_all=true to replace all, \
                or provide more context to make the match unique."
            )
            .into());
        }

        let new_content = if replace_all {
            original_content.replace(old_string, new_string)
        } else {
            // Replace only the first occurrence
            original_content.replacen(old_string, new_string, 1)
        };

        let diff = edit_diff::generate_diff(&original_content, &new_content, 3);

        std::fs::write(path, &new_content)?;

        Ok(EditOutput {
            replacements: if replace_all { count } else { 1 },
            diff,
            original_content,
            new_content,
        })
    }
}

/// The Edit tool for find-and-replace in files.
pub struct EditTool {
    working_dir: PathBuf,
    editor: Arc<dyn EditOperations>,
}

impl EditTool {
    pub fn new(working_dir: PathBuf, editor: Arc<dyn EditOperations>) -> Self {
        Self {
            working_dir,
            editor,
        }
    }

    pub fn with_default_editor(working_dir: PathBuf) -> Self {
        Self::new(working_dir, Arc::new(DefaultFileEditor))
    }
}

#[async_trait]
impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn label(&self) -> &str {
        "Edit"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "edit".to_string(),
            description: "Perform exact string replacement in a file.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
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
        let old_string = params
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'old_string' parameter")?;
        let new_string = params
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'new_string' parameter")?;
        let replace_all = params
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let resolved = path_utils::resolve_path(file_path, &self.working_dir);

        // Security: verify path is within working directory
        if !path_utils::is_within(&resolved, &self.working_dir) {
            return Err(format!(
                "Access denied: {} is outside the working directory",
                resolved.display()
            )
            .into());
        }

        let editor = self.editor.clone();
        let resolved_clone = resolved.clone();
        let old_string = old_string.to_string();
        let new_string = new_string.to_string();
        let io_task = tokio::task::spawn_blocking(move || {
            editor.edit_file(&resolved_clone, &old_string, &new_string, replace_all)
        });

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

        let text = format!(
            "Replaced {} occurrence(s) in {}\n\n{}",
            output.replacements,
            resolved.display(),
            output.diff
        );

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({
                "replacements": output.replacements,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_edit_tool_single_replace() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

        let tool = EditTool::with_default_editor(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "old_string": "println!(\"hello\")",
            "new_string": "println!(\"world\")"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("Replaced 1 occurrence"));

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("println!(\"world\")"));
    }

    #[tokio::test]
    async fn test_edit_tool_replace_all() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").unwrap();

        let tool = EditTool::with_default_editor(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": true
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await.unwrap();
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("Replaced 3 occurrence"));
    }

    #[tokio::test]
    async fn test_edit_tool_not_unique() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "foo bar foo").unwrap();

        let tool = EditTool::with_default_editor(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "baz"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 times"));
    }

    #[tokio::test]
    async fn test_edit_tool_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = EditTool::with_default_editor(tmp.path().to_path_buf());
        let params = json!({
            "file_path": file_path.to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "something"
        });
        let cancel = CancellationToken::new();

        let result = tool.execute("call_1", params, cancel, None).await;
        assert!(result.is_err());
    }
}
