use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};

use crate::tools::truncate;

/// Trait for bash execution operations — allows mocking in tests.
#[async_trait]
pub trait BashOperations: Send + Sync {
    async fn execute_command(
        &self,
        command: &str,
        working_dir: &std::path::Path,
        timeout_ms: Option<u64>,
        cancel: CancellationToken,
    ) -> Result<BashOutput, Box<dyn std::error::Error + Send + Sync>>;
}

/// Output of a bash command.
#[derive(Debug, Clone)]
pub struct BashOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub was_cancelled: bool,
    pub duration_ms: u64,
}

/// Default bash executor using tokio::process::Command.
pub struct DefaultBashExecutor;

#[async_trait]
impl BashOperations for DefaultBashExecutor {
    async fn execute_command(
        &self,
        command: &str,
        working_dir: &std::path::Path,
        timeout_ms: Option<u64>,
        cancel: CancellationToken,
    ) -> Result<BashOutput, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms.unwrap_or(120_000));

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        // Create a new process group so we can kill all child processes
        #[cfg(unix)]
        cmd.process_group(0);

        let child = cmd.spawn()?;

        let child_id = child.id();
        let wait_fut = child.wait_with_output();

        let result = tokio::select! {
            result = wait_fut => {
                let output = result?;
                BashOutput {
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    exit_code: output.status.code(),
                    was_cancelled: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                }
            }
            _ = tokio::time::sleep(timeout) => {
                // Best effort kill via signal
                if let Some(pid) = child_id {
                    let _ = kill_process(pid);
                }
                BashOutput {
                    stdout: String::new(),
                    stderr: format!("Command timed out after {}ms", timeout.as_millis()),
                    exit_code: None,
                    was_cancelled: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                }
            }
            _ = cancel.cancelled() => {
                if let Some(pid) = child_id {
                    let _ = kill_process(pid);
                }
                BashOutput {
                    stdout: String::new(),
                    stderr: "Command cancelled".to_string(),
                    exit_code: None,
                    was_cancelled: true,
                    duration_ms: start.elapsed().as_millis() as u64,
                }
            }
        };

        Ok(result)
    }
}

/// Kill a process by PID (best effort).
fn kill_process(pid: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        // Send SIGKILL to process group
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(-(pid as i32)),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F", "/T"])
            .output();
    }
    Ok(())
}

/// The Bash tool for executing shell commands.
pub struct BashTool {
    working_dir: PathBuf,
    executor: Arc<dyn BashOperations>,
}

impl BashTool {
    pub fn new(working_dir: PathBuf, executor: Arc<dyn BashOperations>) -> Self {
        Self {
            working_dir,
            executor,
        }
    }

    pub fn with_default_executor(working_dir: PathBuf) -> Self {
        Self::new(working_dir, Arc::new(DefaultBashExecutor))
    }
}

#[async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn label(&self) -> &str {
        "Bash"
    }

    fn definition(&self) -> &Tool {
        static TOOL: once_cell::sync::Lazy<Tool> = once_cell::sync::Lazy::new(|| Tool {
            name: "bash".to_string(),
            description: "Execute a bash command.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds"
                    }
                },
                "required": ["command"]
            }),
        });
        &TOOL
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        cancel: CancellationToken,
        on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'command' parameter")?;
        let timeout_ms = params.get("timeout").and_then(|v| v.as_u64());

        let output = self
            .executor
            .execute_command(command, &self.working_dir, timeout_ms, cancel)
            .await?;

        // Build result text
        let mut text = String::new();
        if !output.stdout.is_empty() {
            text.push_str(&output.stdout);
        }
        if !output.stderr.is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&format!("STDERR:\n{}", output.stderr));
        }

        if let Some(code) = output.exit_code {
            if code != 0 && text.is_empty() {
                text = format!("Exit code: {code}");
            }
        }

        // Truncate output
        let truncated = truncate::truncate_output(&text, None, None);

        let details = json!({
            "exitCode": output.exit_code,
            "durationMs": output.duration_ms,
            "wasCancelled": output.was_cancelled,
            "wasTruncated": truncated.was_truncated,
        });

        // Send update if callback provided
        if let Some(cb) = &on_update {
            cb(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: truncated.content.clone(),
                    text_signature: None,
                })],
                details: Some(details.clone()),
            });
        }

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: truncated.content,
                text_signature: None,
            })],
            details: Some(details),
        })
    }
}
