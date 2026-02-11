use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::Message;

/// Convert agent messages to LLM messages, filtering out custom messages.
///
/// This is the default `convert_to_llm` implementation used by AgentLoopConfig.
/// Custom messages are skipped since the LLM doesn't understand them.
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            AgentMessage::Llm(m) => Some(m.clone()),
            AgentMessage::Custom(_) => None,
        })
        .collect()
}

/// Convert agent messages to LLM messages, including custom messages
/// as user messages with their JSON content.
pub fn convert_to_llm_with_custom(messages: &[AgentMessage]) -> Vec<Message> {
    use pi_agent_core::types::{UserContent, UserMessage};

    messages
        .iter()
        .map(|msg| match msg {
            AgentMessage::Llm(m) => m.clone(),
            AgentMessage::Custom(value) => {
                // Convert custom messages to user messages with JSON content
                let text = serde_json::to_string(value).unwrap_or_default();
                Message::User(UserMessage {
                    content: UserContent::Text(format!("[System Event]: {text}")),
                    timestamp: value
                        .get("timestamp")
                        .and_then(|v| v.as_i64())
                        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis()),
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::*;

    #[test]
    fn test_convert_to_llm_filters_custom() {
        let messages = vec![
            AgentMessage::user("Hello"),
            AgentMessage::Custom(serde_json::json!({"type": "metadata"})),
            AgentMessage::Llm(Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Hi!".to_string(),
                    text_signature: None,
                })],
                api: "test".to_string(),
                provider: "test".to_string(),
                model: "test".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            })),
        ];

        let llm = convert_to_llm(&messages);
        assert_eq!(llm.len(), 2); // user + assistant, custom filtered out
    }

    #[test]
    fn test_convert_to_llm_with_custom() {
        let messages = vec![
            AgentMessage::user("Hello"),
            AgentMessage::Custom(serde_json::json!({"type": "test", "timestamp": 1000})),
        ];

        let llm = convert_to_llm_with_custom(&messages);
        assert_eq!(llm.len(), 2); // both included
        assert_eq!(llm[0].role(), "user");
        assert_eq!(llm[1].role(), "user"); // custom becomes user message
    }
}
