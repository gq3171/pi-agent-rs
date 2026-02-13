use std::sync::{Arc, Mutex};

use crate::agent_session::events::AgentSessionEvent;
use crate::agent_session::session::{AgentSession, PromptOptions};
use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::{ContentBlock, Message, StopReason};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintOutputMode {
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub struct PrintModeOptions {
    pub mode: PrintOutputMode,
    pub initial_message: Option<String>,
    pub messages: Vec<String>,
}

fn event_to_json(event: &AgentSessionEvent) -> serde_json::Value {
    serde_json::json!({
        "type": event.event_type(),
        "debug": format!("{event:?}")
    })
}

fn assistant_text(messages: &[AgentMessage]) -> Option<String> {
    let last_assistant = messages.iter().rev().find_map(|msg| match msg {
        AgentMessage::Llm(Message::Assistant(m)) => Some(m),
        _ => None,
    })?;

    if matches!(
        last_assistant.stop_reason,
        StopReason::Error | StopReason::Aborted
    ) {
        return Some(
            last_assistant
                .error_message
                .clone()
                .unwrap_or_else(|| "Assistant request failed".to_string()),
        );
    }

    let text = last_assistant
        .content
        .iter()
        .filter_map(|content| match content {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    Some(text)
}

pub async fn run_print_mode(
    session: &mut AgentSession,
    options: PrintModeOptions,
) -> Result<(), crate::error::CodingAgentError> {
    if options.mode == PrintOutputMode::Json {
        let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let sink = events.clone();
        session.subscribe(Box::new(move |event| {
            if let Ok(mut e) = sink.lock() {
                e.push(event_to_json(&event));
            }
        }));

        if let Some(initial) = &options.initial_message {
            session.prompt(initial, PromptOptions::default()).await?;
        }
        for message in &options.messages {
            session.prompt(message, PromptOptions::default()).await?;
        }

        if let Ok(events) = events.lock() {
            for event in events.iter() {
                println!("{}", serde_json::to_string(event).unwrap_or_default());
            }
        }
        return Ok(());
    }

    if let Some(initial) = &options.initial_message {
        session.prompt(initial, PromptOptions::default()).await?;
    }
    for message in &options.messages {
        session.prompt(message, PromptOptions::default()).await?;
    }

    if let Some(text) = assistant_text(session.messages()) {
        println!("{text}");
    }

    Ok(())
}
