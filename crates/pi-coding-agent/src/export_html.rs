use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::error::CodingAgentError;
use crate::session::types::{SessionEntry, SessionHeader};
use pi_agent_core::types::{ContentBlock, Message, UserContent};

#[derive(Debug, Clone, Default)]
pub struct ExportHtmlOptions {
    pub output_path: Option<PathBuf>,
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn render_content_blocks(blocks: &[ContentBlock]) -> String {
    let mut rendered = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text(text) => {
                rendered.push_str(&format!(
                    "<div class=\"block text\">{}</div>",
                    escape_html(&text.text)
                ));
            }
            ContentBlock::Thinking(thinking) => {
                rendered.push_str(&format!(
                    "<details class=\"block thinking\"><summary>Thinking</summary><pre>{}</pre></details>",
                    escape_html(&thinking.thinking)
                ));
            }
            ContentBlock::ToolCall(call) => {
                rendered.push_str(&format!(
                    "<details class=\"block tool-call\"><summary>Tool Call: {}</summary><pre>{}</pre></details>",
                    escape_html(&call.name),
                    escape_html(&call.arguments.to_string())
                ));
            }
            ContentBlock::Image(image) => {
                rendered.push_str(&format!(
                    "<div class=\"block image\">[image: {}; {} bytes]</div>",
                    escape_html(&image.mime_type),
                    image.data.len()
                ));
            }
        }
    }
    rendered
}

fn render_message(message: &Message) -> String {
    match message {
        Message::User(user) => {
            let content = match &user.content {
                UserContent::Text(text) => {
                    format!("<div class=\"block text\">{}</div>", escape_html(text))
                }
                UserContent::Blocks(blocks) => render_content_blocks(blocks),
            };
            format!("<div class=\"entry user\"><h3>User</h3>{content}</div>")
        }
        Message::Assistant(assistant) => {
            let content = render_content_blocks(&assistant.content);
            let meta = format!(
                "<div class=\"meta\">{}/{}</div>",
                escape_html(&assistant.provider),
                escape_html(&assistant.model)
            );
            format!("<div class=\"entry assistant\"><h3>Assistant</h3>{meta}{content}</div>")
        }
        Message::ToolResult(tool) => {
            let content = render_content_blocks(&tool.content);
            let status = if tool.is_error { "error" } else { "ok" };
            format!(
                "<div class=\"entry tool-result\"><h3>Tool Result: {} ({})</h3>{}</div>",
                escape_html(&tool.tool_name),
                status,
                content
            )
        }
    }
}

fn render_entry(entry: &SessionEntry) -> Option<String> {
    match entry {
        SessionEntry::Message { message, .. } => Some(render_message(message)),
        SessionEntry::Compaction { summary, .. } | SessionEntry::LegacySummary { summary, .. } => {
            Some(format!(
                "<div class=\"entry system\"><h3>Compaction</h3><pre>{}</pre></div>",
                escape_html(summary)
            ))
        }
        SessionEntry::ModelChange {
            provider, model_id, ..
        } => Some(format!(
            "<div class=\"entry system\"><h3>Model Change</h3><div class=\"meta\">{}/{}</div></div>",
            escape_html(provider),
            escape_html(model_id)
        )),
        SessionEntry::ThinkingLevelChange { thinking_level, .. } => Some(format!(
            "<div class=\"entry system\"><h3>Thinking Level</h3><div class=\"meta\">{}</div></div>",
            escape_html(thinking_level)
        )),
        SessionEntry::BranchSummary { summary, .. } => Some(format!(
            "<div class=\"entry system\"><h3>Branch Summary</h3><pre>{}</pre></div>",
            escape_html(summary)
        )),
        _ => None,
    }
}

fn read_session(input_path: &Path) -> Result<(SessionHeader, Vec<SessionEntry>), CodingAgentError> {
    let file = std::fs::File::open(input_path)?;
    let mut lines = BufReader::new(file).lines();

    let header_line = lines
        .next()
        .ok_or_else(|| CodingAgentError::Session("Empty session file".to_string()))??;
    let header = serde_json::from_str::<SessionHeader>(&header_line)?;

    let mut entries = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<SessionEntry>(&line) {
            entries.push(entry);
        }
    }

    Ok((header, entries))
}

fn build_html(header: &SessionHeader, entries: &[SessionEntry]) -> String {
    let title = header
        .title
        .clone()
        .unwrap_or_else(|| format!("Session {}", header.id));
    let mut body = String::new();
    for entry in entries {
        if let Some(html) = render_entry(entry) {
            body.push_str(&html);
            body.push('\n');
        }
    }

    format!(
        "<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{}</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ margin: 0; font-family: ui-monospace, Menlo, Consolas, monospace; background: #0f172a; color: #e2e8f0; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
    .sub {{ margin-bottom: 20px; color: #94a3b8; }}
    .entry {{ border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 12px; background: #111827; }}
    .entry.user {{ border-color: #0ea5e9; }}
    .entry.assistant {{ border-color: #22c55e; }}
    .entry.tool-result {{ border-color: #f59e0b; }}
    .entry.system {{ border-color: #a78bfa; }}
    h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .meta {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; }}
    .block {{ margin: 8px 0; white-space: pre-wrap; word-break: break-word; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
    details > summary {{ cursor: pointer; color: #93c5fd; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>{}</h1>
    <div class=\"sub\">sessionId: {} | created: {}</div>
    {}
  </div>
</body>
</html>",
        escape_html(&title),
        escape_html(&title),
        escape_html(&header.id),
        escape_html(&header.timestamp),
        body
    )
}

pub fn export_session_to_html(
    input_path: &Path,
    options: ExportHtmlOptions,
) -> Result<PathBuf, CodingAgentError> {
    if !input_path.exists() {
        return Err(CodingAgentError::Session(format!(
            "Session file not found: {}",
            input_path.display()
        )));
    }

    let (header, entries) = read_session(input_path)?;
    let html = build_html(&header, &entries);
    let output = options.output_path.unwrap_or_else(|| {
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session");
        PathBuf::from(format!("pi-agent-session-{stem}.html"))
    });
    std::fs::write(&output, html)?;
    Ok(output)
}
