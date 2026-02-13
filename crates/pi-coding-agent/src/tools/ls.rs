use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::path_utils;
use crate::tools::truncate;

const DEFAULT_LIMIT: usize = 500;

fn to_display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub struct LsTool {
    working_dir: PathBuf,
}

impl LsTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[async_trait]
impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn label(&self) -> &str {
        "Ls"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "ls".to_string(),
            description: "List directory contents, appending '/' for directories.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of entries to return (default: 500)"
                    }
                }
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
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let resolved = path_utils::resolve_path(path, &self.working_dir);

        if !path_utils::is_within(&resolved, &self.working_dir) {
            return Err(format!(
                "Access denied: {} is outside the working directory",
                resolved.display()
            )
            .into());
        }

        if cancel.is_cancelled() {
            return Err("Operation cancelled".into());
        }

        let resolved_clone = resolved.clone();
        let cancel_clone = cancel.clone();
        let output = tokio::task::spawn_blocking(move || {
            if !resolved_clone.exists() {
                return Err(format!("Path not found: {}", resolved_clone.display()).into());
            }

            if !resolved_clone.is_dir() {
                return Err(format!("Not a directory: {}", resolved_clone.display()).into());
            }

            let mut entries = Vec::new();
            for entry in std::fs::read_dir(&resolved_clone)? {
                if cancel_clone.is_cancelled() {
                    return Err("Operation cancelled".into());
                }

                let entry = entry?;
                let file_name = entry.file_name().to_string_lossy().to_string();
                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    entries.push(format!("{file_name}/"));
                } else {
                    entries.push(file_name);
                }
            }

            entries.sort_by_key(|name| name.to_lowercase());

            let mut limited = entries;
            let entry_limit_reached = limited.len() > limit;
            if entry_limit_reached {
                limited.truncate(limit);
            }

            let raw_output = if limited.is_empty() {
                "(empty directory)".to_string()
            } else {
                limited.join("\n")
            };

            let truncated = truncate::truncate_output(&raw_output, None, None);
            let mut final_output = truncated.content;
            let mut notices = Vec::new();

            if entry_limit_reached {
                notices.push(format!("{limit} entries limit reached"));
            }
            if truncated.was_truncated {
                notices.push("output byte/line limit reached".to_string());
            }
            if !notices.is_empty() {
                final_output.push_str(&format!("\n\n[{}]", notices.join(". ")));
            }

            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                final_output,
                entry_limit_reached,
                truncated.was_truncated,
            ))
        })
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Task join error: {e}").into()
        })??;

        let details = json!({
            "path": to_display_path(&resolved),
            "entryLimitReached": output.1,
            "wasTruncated": output.2,
        });

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: output.0,
                text_signature: None,
            })],
            details: Some(details),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ls_tool_lists_entries() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "a").unwrap();
        std::fs::create_dir_all(tmp.path().join("dir1")).unwrap();

        let tool = LsTool::new(tmp.path().to_path_buf());
        let result = tool
            .execute("call_1", json!({}), CancellationToken::new(), None)
            .await
            .unwrap();

        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("a.txt"));
        assert!(text.contains("dir1/"));
    }
}
