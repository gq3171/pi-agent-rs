use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::agent_session::session::{AgentSession, PromptOptions};
use crate::compaction::compaction::CompactionSettings;
use crate::error::CodingAgentError;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RpcCommand {
    Prompt { id: Option<String>, message: String },
    Abort { id: Option<String> },
    Stats { id: Option<String> },
    SetModel { id: Option<String>, model: String },
    Models { id: Option<String> },
    NewSession { id: Option<String> },
    Compact { id: Option<String> },
    Shutdown { id: Option<String> },
}

#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub response_type: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn write_json(value: &serde_json::Value) {
    println!("{}", serde_json::to_string(value).unwrap_or_default());
    let _ = io::stdout().flush();
}

fn ok(id: Option<String>, data: Option<serde_json::Value>) -> RpcResponse {
    RpcResponse {
        id,
        response_type: "response".to_string(),
        success: true,
        data,
        error: None,
    }
}

fn err(id: Option<String>, message: impl Into<String>) -> RpcResponse {
    RpcResponse {
        id,
        response_type: "response".to_string(),
        success: false,
        data: None,
        error: Some(message.into()),
    }
}

pub async fn run_rpc_mode(session: &mut AgentSession) -> Result<(), CodingAgentError> {
    session.subscribe(Box::new(|event| {
        write_json(&serde_json::json!({
            "type": "event",
            "eventType": event.event_type(),
            "debug": format!("{event:?}")
        }));
    }));

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while let Some(line) = lines.next() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                write_json(&serde_json::to_value(err(None, e.to_string())).unwrap_or_default());
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let cmd: RpcCommand = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                write_json(&serde_json::to_value(err(None, e.to_string())).unwrap_or_default());
                continue;
            }
        };

        let response = match cmd {
            RpcCommand::Prompt { id, message } => {
                match session.prompt(&message, PromptOptions::default()).await {
                    Ok(_) => ok(id, None),
                    Err(e) => err(id, e.to_string()),
                }
            }
            RpcCommand::Abort { id } => {
                session.abort();
                ok(id, None)
            }
            RpcCommand::Stats { id } => {
                let stats = session.get_stats();
                let context_usage = session.get_context_usage();
                ok(
                    id,
                    Some(serde_json::json!({
                        "sessionId": stats.session_id,
                        "messageCount": stats.message_count,
                        "turnCount": stats.turn_count,
                        "estimatedTokens": stats.estimated_tokens,
                        "contextUsage": context_usage.as_ref().map(|usage| serde_json::json!({
                            "tokens": usage.tokens,
                            "contextWindow": usage.context_window,
                            "percent": usage.percent
                        }))
                    })),
                )
            }
            RpcCommand::SetModel { id, model } => {
                let found = session.model_registry().find(&model).cloned();
                if let Some(found) = found {
                    let provider = found.provider.clone();
                    let model_id = found.id.clone();
                    session.set_model(found);
                    ok(
                        id,
                        Some(serde_json::json!({
                            "provider": provider,
                            "modelId": model_id
                        })),
                    )
                } else {
                    err(id, format!("Model not found: {model}"))
                }
            }
            RpcCommand::Models { id } => {
                let models = session
                    .model_registry()
                    .all_models()
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "provider": m.provider,
                            "id": m.id,
                            "name": m.name
                        })
                    })
                    .collect::<Vec<_>>();
                ok(id, Some(serde_json::json!({ "models": models })))
            }
            RpcCommand::NewSession { id } => {
                session.reset_session();
                ok(id, None)
            }
            RpcCommand::Compact { id } => {
                match session.compact(Some(&CompactionSettings::default())).await {
                    Ok(result) => ok(
                        id,
                        Some(serde_json::json!({
                            "messagesBefore": result.messages_before,
                            "messagesAfter": result.messages_after,
                            "tokensBefore": result.tokens_before,
                            "tokensAfter": result.tokens_after,
                        })),
                    ),
                    Err(e) => err(id, e.to_string()),
                }
            }
            RpcCommand::Shutdown { id } => {
                let response = ok(id, None);
                write_json(&serde_json::to_value(&response).unwrap_or_default());
                break;
            }
        };

        write_json(&serde_json::to_value(response).unwrap_or_default());
    }

    Ok(())
}
