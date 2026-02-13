use std::path::{Path, PathBuf};

use async_trait::async_trait;
use glob::Pattern;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::path_utils;
use crate::tools::truncate;

const DEFAULT_LIMIT: usize = 1000;

fn normalize_rel_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn should_skip_path(rel: &str) -> bool {
    rel.split('/').any(|p| p == ".git" || p == "node_modules")
}

pub struct FindTool {
    working_dir: PathBuf,
}

impl FindTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[async_trait]
impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn label(&self) -> &str {
        "Find"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "find".to_string(),
            description: "Search for files and directories using a glob pattern.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.ts' or 'src/**/*.rs'"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 1000)"
                    }
                },
                "required": ["pattern"]
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
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' parameter")?;
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

        let pattern = Pattern::new(pattern).map_err(|e| format!("Invalid glob pattern: {e}"))?;
        let resolved_clone = resolved.clone();
        let cancel_clone = cancel.clone();

        let output = tokio::task::spawn_blocking(move || {
            if !resolved_clone.exists() {
                return Err(format!("Path not found: {}", resolved_clone.display()).into());
            }
            if !resolved_clone.is_dir() {
                return Err(format!("Not a directory: {}", resolved_clone.display()).into());
            }

            let mut matches = Vec::new();
            let walker = ignore::WalkBuilder::new(&resolved_clone)
                .hidden(false)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .build();

            for entry in walker {
                if cancel_clone.is_cancelled() {
                    return Err("Operation cancelled".into());
                }

                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                let full_path = entry.path();
                if full_path == resolved_clone {
                    continue;
                }

                let rel_path = match full_path.strip_prefix(&resolved_clone) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let mut rel = normalize_rel_path(rel_path);
                if rel.is_empty() {
                    continue;
                }

                if should_skip_path(&rel) {
                    continue;
                }

                let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);

                // Match against both forms for convenience.
                let mut is_match = pattern.matches(&rel) || pattern.matches_path(rel_path);
                if !is_match && is_dir {
                    let rel_with_slash = format!("{rel}/");
                    is_match = pattern.matches(&rel_with_slash);
                    if is_match {
                        rel = rel_with_slash;
                    }
                }

                if is_match {
                    matches.push(rel);
                    if matches.len() >= limit {
                        break;
                    }
                }
            }

            matches.sort_by_key(|p| p.to_lowercase());

            if matches.is_empty() {
                return Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                    "No files found matching pattern".to_string(),
                    false,
                    false,
                ));
            }

            let raw = matches.join("\n");
            let truncated = truncate::truncate_output(&raw, None, None);
            let mut result = truncated.content;
            let mut notices = Vec::new();

            let result_limit_reached = matches.len() >= limit;
            if result_limit_reached {
                notices.push(format!("{limit} results limit reached"));
            }
            if truncated.was_truncated {
                notices.push("output byte/line limit reached".to_string());
            }

            if !notices.is_empty() {
                result.push_str(&format!("\n\n[{}]", notices.join(". ")));
            }

            Ok((result, result_limit_reached, truncated.was_truncated))
        })
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Task join error: {e}").into()
        })??;

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: output.0,
                text_signature: None,
            })],
            details: Some(json!({
                "searchPath": resolved.display().to_string(),
                "resultLimitReached": output.1,
                "wasTruncated": output.2,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_find_tool_matches_glob() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(tmp.path().join("src/lib.rs"), "pub fn x() {}\n").unwrap();

        let tool = FindTool::new(tmp.path().to_path_buf());
        let result = tool
            .execute(
                "call_1",
                json!({"pattern": "src/*.rs"}),
                CancellationToken::new(),
                None,
            )
            .await
            .unwrap();

        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("src/main.rs"));
        assert!(text.contains("src/lib.rs"));
    }
}
