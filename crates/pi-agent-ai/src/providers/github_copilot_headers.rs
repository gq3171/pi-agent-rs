use std::collections::HashMap;

use pi_agent_core::types::{Message, UserContent};

/// Copilot expects X-Initiator to indicate whether the request is user-initiated
/// or agent-initiated (e.g. follow-up after assistant/tool messages).
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    let last = messages.last();
    if last.is_some_and(|m| m.role() != "user") {
        "agent"
    } else {
        "user"
    }
}

/// Copilot requires Copilot-Vision-Request header when sending images.
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|msg| match msg {
        Message::User(user_msg) => match &user_msg.content {
            UserContent::Blocks(blocks) => blocks.iter().any(|c| c.as_image().is_some()),
            UserContent::Text(_) => false,
        },
        Message::ToolResult(tool_result) => {
            tool_result.content.iter().any(|c| c.as_image().is_some())
        }
        _ => false,
    })
}

pub fn build_copilot_dynamic_headers(
    messages: &[Message],
    has_images: bool,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert(
        "X-Initiator".to_string(),
        infer_copilot_initiator(messages).to_string(),
    );
    headers.insert(
        "Openai-Intent".to_string(),
        "conversation-edits".to_string(),
    );
    if has_images {
        headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
    }
    headers
}
