use std::collections::{HashMap, HashSet};

use crate::types::*;

type NormalizeToolCallIdFn<'a> = dyn Fn(&str, &Model, &AssistantMessage) -> String + 'a;

/// Transform messages for cross-provider compatibility.
/// Handles thinking block conversion, tool call ID normalization, and orphaned tool calls.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<&NormalizeToolCallIdFn<'_>>,
) -> Vec<Message> {
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();

    // First pass: transform messages (thinking blocks, tool call ID normalization)
    let transformed: Vec<Message> = messages
        .iter()
        .map(|msg| match msg {
            Message::User(_) => msg.clone(),
            Message::ToolResult(tr) => {
                if let Some(normalized_id) = tool_call_id_map.get(&tr.tool_call_id) {
                    if normalized_id != &tr.tool_call_id {
                        let mut new_tr = tr.clone();
                        new_tr.tool_call_id = normalized_id.clone();
                        return Message::ToolResult(new_tr);
                    }
                }
                msg.clone()
            }
            Message::Assistant(assistant_msg) => {
                let is_same_model = assistant_msg.provider == model.provider
                    && assistant_msg.api == model.api
                    && assistant_msg.model == model.id;

                let transformed_content: Vec<ContentBlock> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Thinking(t) => {
                            // Same model with signature: keep
                            if is_same_model && t.thinking_signature.is_some() {
                                return Some(block.clone());
                            }
                            // Empty thinking: skip
                            if t.thinking.trim().is_empty() {
                                return None;
                            }
                            // Same model without signature: keep as-is
                            if is_same_model {
                                return Some(block.clone());
                            }
                            // Cross-model: convert to text
                            Some(ContentBlock::Text(TextContent {
                                text: t.thinking.clone(),
                                text_signature: None,
                            }))
                        }
                        ContentBlock::Text(t) => {
                            if is_same_model {
                                Some(block.clone())
                            } else {
                                // Strip textSignature for cross-model
                                Some(ContentBlock::Text(TextContent {
                                    text: t.text.clone(),
                                    text_signature: None,
                                }))
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            let mut normalized = tc.clone();

                            // Strip thoughtSignature for cross-model
                            if !is_same_model && normalized.thought_signature.is_some() {
                                normalized.thought_signature = None;
                            }

                            // Normalize tool call ID for cross-model
                            if !is_same_model {
                                if let Some(normalize_fn) = normalize_tool_call_id {
                                    let new_id = normalize_fn(&tc.id, model, assistant_msg);
                                    if new_id != tc.id {
                                        tool_call_id_map.insert(tc.id.clone(), new_id.clone());
                                        normalized.id = new_id;
                                    }
                                }
                            }

                            Some(ContentBlock::ToolCall(normalized))
                        }
                        _ => Some(block.clone()),
                    })
                    .collect();

                Message::Assistant(AssistantMessage {
                    content: transformed_content,
                    ..assistant_msg.clone()
                })
            }
        })
        .collect();

    // Second pass: insert synthetic empty tool results for orphaned tool calls
    let mut result: Vec<Message> = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: HashSet<String> = HashSet::new();

    for msg in &transformed {
        match msg {
            Message::Assistant(assistant_msg) => {
                // Insert synthetic results for previous orphaned tool calls
                if !pending_tool_calls.is_empty() {
                    for tc in &pending_tool_calls {
                        if !existing_tool_result_ids.contains(&tc.id) {
                            result.push(Message::ToolResult(ToolResultMessage {
                                tool_call_id: tc.id.clone(),
                                tool_name: tc.name.clone(),
                                content: vec![ContentBlock::Text(TextContent {
                                    text: "No result provided".to_string(),
                                    text_signature: None,
                                })],
                                details: None,
                                is_error: true,
                                timestamp: chrono::Utc::now().timestamp_millis(),
                            }));
                        }
                    }
                    pending_tool_calls.clear();
                    existing_tool_result_ids.clear();
                }

                // Skip errored/aborted assistant messages
                if assistant_msg.stop_reason == StopReason::Error
                    || assistant_msg.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                // Track tool calls from this assistant message
                let tool_calls: Vec<ToolCall> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| b.as_tool_call().cloned())
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids.clear();
                }

                result.push(msg.clone());
            }
            Message::ToolResult(tr) => {
                existing_tool_result_ids.insert(tr.tool_call_id.clone());
                result.push(msg.clone());
            }
            Message::User(_) => {
                // User message interrupts tool flow
                if !pending_tool_calls.is_empty() {
                    for tc in &pending_tool_calls {
                        if !existing_tool_result_ids.contains(&tc.id) {
                            result.push(Message::ToolResult(ToolResultMessage {
                                tool_call_id: tc.id.clone(),
                                tool_name: tc.name.clone(),
                                content: vec![ContentBlock::Text(TextContent {
                                    text: "No result provided".to_string(),
                                    text_signature: None,
                                })],
                                details: None,
                                is_error: true,
                                timestamp: chrono::Utc::now().timestamp_millis(),
                            }));
                        }
                    }
                    pending_tool_calls.clear();
                    existing_tool_result_ids.clear();
                }
                result.push(msg.clone());
            }
        }
    }

    // Flush any remaining pending tool calls at end of message list
    if !pending_tool_calls.is_empty() {
        for tc in &pending_tool_calls {
            if !existing_tool_result_ids.contains(&tc.id) {
                result.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content: vec![ContentBlock::Text(TextContent {
                        text: "No result provided".to_string(),
                        text_signature: None,
                    })],
                    details: None,
                    is_error: true,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                }));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> Model {
        Model {
            id: "claude-sonnet-4".to_string(),
            name: "Claude Sonnet".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            reasoning: true,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 200000,
            max_tokens: 8192,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_same_model_preserves_thinking() {
        let model = test_model();
        let messages = vec![Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "thinking...".to_string(),
                    thinking_signature: Some("sig".to_string()),
                }),
                ContentBlock::Text(TextContent {
                    text: "hello".to_string(),
                    text_signature: None,
                }),
            ],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })];

        let result = transform_messages(&messages, &model, None);
        assert_eq!(result.len(), 1);
        if let Message::Assistant(a) = &result[0] {
            assert_eq!(a.content.len(), 2);
            assert!(a.content[0].as_thinking().is_some());
        } else {
            panic!("expected assistant");
        }
    }

    #[test]
    fn test_cross_model_converts_thinking_to_text() {
        let model = test_model();
        let messages = vec![Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Thinking(ThinkingContent {
                thinking: "deep thought".to_string(),
                thinking_signature: None,
            })],
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })];

        let result = transform_messages(&messages, &model, None);
        assert_eq!(result.len(), 1);
        if let Message::Assistant(a) = &result[0] {
            assert_eq!(a.content.len(), 1);
            assert!(a.content[0].as_text().is_some());
            assert_eq!(a.content[0].as_text().unwrap().text, "deep thought");
        }
    }

    #[test]
    fn test_orphaned_tool_calls_get_synthetic_results() {
        let model = test_model();
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    arguments: serde_json::json!({}),
                    thought_signature: None,
                })],
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 0,
            }),
            // No tool result, directly another assistant message
            Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "ok".to_string(),
                    text_signature: None,
                })],
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
        ];

        let result = transform_messages(&messages, &model, None);
        // Should be: assistant, synthetic toolResult, assistant
        assert_eq!(result.len(), 3);
        assert_eq!(result[1].role(), "toolResult");
    }

    #[test]
    fn test_error_assistant_messages_skipped() {
        let model = test_model();
        let messages = vec![
            Message::User(UserMessage {
                content: UserContent::Text("hello".to_string()),
                timestamp: 0,
            }),
            Message::Assistant(AssistantMessage {
                content: vec![],
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::Error,
                error_message: Some("rate limit".to_string()),
                timestamp: 0,
            }),
        ];

        let result = transform_messages(&messages, &model, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role(), "user");
    }
}
