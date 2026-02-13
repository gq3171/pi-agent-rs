use std::path::{Path, PathBuf};

use async_trait::async_trait;
use glob::Pattern;
use regex::RegexBuilder;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::path_utils;
use crate::tools::truncate;

const DEFAULT_LIMIT: usize = 100;
const DEFAULT_CONTEXT: usize = 0;
const GREP_MAX_LINE_CHARS: usize = 2000;

fn normalize_rel_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn should_skip_path(rel: &str) -> bool {
    rel.split('/').any(|p| p == ".git" || p == "node_modules")
}

pub struct GrepTool {
    working_dir: PathBuf,
}

impl GrepTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn label(&self) -> &str {
        "Grep"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "grep".to_string(),
            description: "Search file contents for a pattern and return matching lines."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex by default)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: current directory)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file glob filter, e.g. '*.rs'"
                    },
                    "ignoreCase": {
                        "type": "boolean",
                        "description": "Case-insensitive search"
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as literal string"
                    },
                    "context": {
                        "type": "number",
                        "description": "Context lines before/after each match"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of matches (default: 100)"
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
        let glob = params
            .get("glob")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let ignore_case = params
            .get("ignoreCase")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let literal = params
            .get("literal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let context = params
            .get("context")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_CONTEXT);
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT)
            .max(1);

        let resolved = path_utils::resolve_path(path, &self.working_dir);

        if !path_utils::is_within(&resolved, &self.working_dir) {
            return Err(format!(
                "Access denied: {} is outside the working directory",
                resolved.display()
            )
            .into());
        }

        let regex_source = if literal {
            regex::escape(pattern)
        } else {
            pattern.to_string()
        };
        let regex = RegexBuilder::new(&regex_source)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|e| format!("Invalid pattern: {e}"))?;
        let glob_pattern = match glob {
            Some(g) => Some(Pattern::new(&g).map_err(|e| format!("Invalid glob: {e}"))?),
            None => None,
        };

        let resolved_clone = resolved.clone();
        let cancel_clone = cancel.clone();

        let output = tokio::task::spawn_blocking(move || {
            if !resolved_clone.exists() {
                return Err(format!("Path not found: {}", resolved_clone.display()).into());
            }

            let mut files: Vec<PathBuf> = Vec::new();

            if resolved_clone.is_file() {
                files.push(resolved_clone.clone());
            } else {
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

                    let path = entry.path();
                    if path == resolved_clone {
                        continue;
                    }

                    let file_type = match entry.file_type() {
                        Some(t) => t,
                        None => continue,
                    };
                    if !file_type.is_file() {
                        continue;
                    }

                    let rel = match path.strip_prefix(&resolved_clone) {
                        Ok(p) => normalize_rel_path(p),
                        Err(_) => continue,
                    };

                    if rel.is_empty() || should_skip_path(&rel) {
                        continue;
                    }

                    if let Some(pattern) = &glob_pattern {
                        if !(pattern.matches(&rel) || pattern.matches_path(Path::new(&rel))) {
                            continue;
                        }
                    }

                    files.push(path.to_path_buf());
                }
            }

            files.sort_by_key(|p| p.to_string_lossy().to_lowercase());

            let mut output_lines: Vec<String> = Vec::new();
            let mut match_count = 0usize;
            let mut match_limit_reached = false;
            let mut lines_truncated = false;

            'files: for file in files {
                if cancel_clone.is_cancelled() {
                    return Err("Operation cancelled".into());
                }

                if path_utils::is_likely_binary(&file) {
                    continue;
                }

                let raw = match std::fs::read_to_string(&file) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
                let lines: Vec<&str> = normalized.split('\n').collect();

                let rel = if resolved_clone.is_file() {
                    file.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| file.display().to_string())
                } else {
                    match file.strip_prefix(&resolved_clone) {
                        Ok(p) => normalize_rel_path(p),
                        Err(_) => file.display().to_string(),
                    }
                };

                for (idx, line) in lines.iter().enumerate() {
                    if cancel_clone.is_cancelled() {
                        return Err("Operation cancelled".into());
                    }

                    if !regex.is_match(line) {
                        continue;
                    }

                    match_count += 1;

                    let start = idx.saturating_sub(context);
                    let end = (idx + context).min(lines.len().saturating_sub(1));

                    for i in start..=end {
                        let text = lines[i];
                        let truncated_text = if text.chars().count() > GREP_MAX_LINE_CHARS {
                            lines_truncated = true;
                            let mut s = text.chars().take(GREP_MAX_LINE_CHARS).collect::<String>();
                            s.push_str("...[truncated]");
                            s
                        } else {
                            text.to_string()
                        };

                        if i == idx {
                            output_lines.push(format!("{}:{}: {}", rel, i + 1, truncated_text));
                        } else {
                            output_lines.push(format!("{}-{}- {}", rel, i + 1, truncated_text));
                        }
                    }

                    if match_count >= limit {
                        match_limit_reached = true;
                        break 'files;
                    }
                }
            }

            if output_lines.is_empty() {
                return Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                    "No matches found".to_string(),
                    false,
                    false,
                    false,
                ));
            }

            let raw_output = output_lines.join("\n");
            let truncated = truncate::truncate_output(&raw_output, None, None);
            let mut output = truncated.content;

            let mut notices = Vec::new();
            if match_limit_reached {
                notices.push(format!("{limit} matches limit reached"));
            }
            if truncated.was_truncated {
                notices.push("output byte/line limit reached".to_string());
            }
            if lines_truncated {
                notices.push(format!(
                    "some lines truncated to {GREP_MAX_LINE_CHARS} chars"
                ));
            }
            if !notices.is_empty() {
                output.push_str(&format!("\n\n[{}]", notices.join(". ")));
            }

            Ok((
                output,
                match_limit_reached,
                truncated.was_truncated,
                lines_truncated,
            ))
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
                "path": resolved.display().to_string(),
                "matchLimitReached": output.1,
                "wasTruncated": output.2,
                "linesTruncated": output.3,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grep_tool_finds_matches() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/a.rs"), "fn main() {}\nlet x = 1;\n").unwrap();
        std::fs::write(tmp.path().join("src/b.rs"), "pub fn helper() {}\n").unwrap();

        let tool = GrepTool::new(tmp.path().to_path_buf());
        let result = tool
            .execute(
                "call_1",
                json!({"pattern": "fn", "glob": "src/*.rs"}),
                CancellationToken::new(),
                None,
            )
            .await
            .unwrap();

        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("src/a.rs:1:"));
        assert!(text.contains("src/b.rs:1:"));
    }
}
