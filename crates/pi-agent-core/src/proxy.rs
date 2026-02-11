use serde::Deserialize;
use serde_json::Value;

use crate::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::json_parse::parse_streaming_json;
use crate::types::*;

// ---------- ProxyAssistantMessageEvent ----------

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProxyAssistantMessageEvent {
    Start,
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature")]
        content_signature: Option<String>,
    },
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature")]
        content_signature: Option<String>,
    },
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    Done {
        reason: StopReason,
        usage: Usage,
    },
    Error {
        reason: StopReason,
        #[serde(rename = "errorMessage")]
        error_message: Option<String>,
        usage: Usage,
    },
}

// ---------- ProxyStreamOptions ----------

#[derive(Debug, Clone)]
pub struct ProxyStreamOptions {
    pub base: SimpleStreamOptions,
    pub auth_token: String,
    pub proxy_url: String,
}

/// Stream function that proxies through a server.
/// The server strips the partial field from delta events to reduce bandwidth.
/// We reconstruct the partial message client-side.
pub fn stream_proxy(
    model: &Model,
    context: &Context,
    options: ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let mut partial = AssistantMessage::empty(&model);
        partial.content = Vec::new();

        let request_body = serde_json::json!({
            "model": model,
            "context": context,
            "options": {
                "temperature": options.base.base.temperature,
                "maxTokens": options.base.base.max_tokens,
                "reasoning": options.base.reasoning,
            }
        });

        let client = reqwest::Client::new();
        let url = format!("{}/api/stream", options.proxy_url);

        let response = match client
            .post(&url)
            .header("Authorization", format!("Bearer {}", options.auth_token))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                partial.stop_reason = StopReason::Error;
                partial.error_message = Some(format!("Proxy error: {e}"));
                stream_clone.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: partial,
                });
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            partial.stop_reason = StopReason::Error;
            partial.error_message = Some(format!("Proxy error: {status} {error_text}"));
            stream_clone.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: partial,
            });
            return;
        }

        // Stream SSE events from response body
        use futures::StreamExt;
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut partial_json_map: std::collections::HashMap<usize, String> = std::collections::HashMap::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    partial.stop_reason = StopReason::Error;
                    partial.error_message = Some(format!("Stream read error: {e}"));
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: partial,
                    });
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));
            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<ProxyAssistantMessageEvent>(data) {
                        Ok(proxy_event) => {
                            if let Some(event) =
                                process_proxy_event(proxy_event, &mut partial, &mut partial_json_map)
                            {
                                stream_clone.push(event);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse proxy event: {e}");
                        }
                    }
                }
            }
        }
    });

    stream
}

fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: &mut AssistantMessage,
    partial_json_map: &mut std::collections::HashMap<usize, String>,
) -> Option<AssistantMessageEvent> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => Some(AssistantMessageEvent::Start {
            partial: partial.clone(),
        }),

        ProxyAssistantMessageEvent::TextStart { content_index } => {
            // Ensure content vec is large enough
            while partial.content.len() <= content_index {
                partial.content.push(ContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
            }
            partial.content[content_index] = ContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            });
            Some(AssistantMessageEvent::TextStart {
                content_index,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => {
            if let Some(ContentBlock::Text(t)) = partial.content.get_mut(content_index) {
                t.text.push_str(&delta);
            }
            Some(AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => {
            let content = if let Some(ContentBlock::Text(t)) = partial.content.get_mut(content_index) {
                t.text_signature = content_signature;
                t.text.clone()
            } else {
                String::new()
            };
            Some(AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            while partial.content.len() <= content_index {
                partial.content.push(ContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
            }
            partial.content[content_index] = ContentBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
            });
            Some(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => {
            if let Some(ContentBlock::Thinking(t)) = partial.content.get_mut(content_index) {
                t.thinking.push_str(&delta);
            }
            Some(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            content_signature,
        } => {
            let content = if let Some(ContentBlock::Thinking(t)) = partial.content.get_mut(content_index) {
                t.thinking_signature = content_signature;
                t.thinking.clone()
            } else {
                String::new()
            };
            Some(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ToolcallStart {
            content_index,
            id,
            tool_name,
        } => {
            while partial.content.len() <= content_index {
                partial.content.push(ContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
            }
            partial.content[content_index] = ContentBlock::ToolCall(ToolCall {
                id,
                name: tool_name,
                arguments: Value::Object(Default::default()),
                thought_signature: None,
            });
            partial_json_map.insert(content_index, String::new());
            Some(AssistantMessageEvent::ToolCallStart {
                content_index,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index,
            delta,
        } => {
            if let Some(partial_json) = partial_json_map.get_mut(&content_index) {
                partial_json.push_str(&delta);
                if let Some(ContentBlock::ToolCall(tc)) = partial.content.get_mut(content_index) {
                    tc.arguments = parse_streaming_json(partial_json);
                }
            }
            Some(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::ToolcallEnd { content_index } => {
            partial_json_map.remove(&content_index);
            let tool_call = if let Some(ContentBlock::ToolCall(tc)) = partial.content.get(content_index) {
                tc.clone()
            } else {
                return None;
            };
            Some(AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                partial: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::Done { reason, usage } => {
            partial.stop_reason = reason.clone();
            partial.usage = usage;
            Some(AssistantMessageEvent::Done {
                reason,
                message: partial.clone(),
            })
        }

        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage,
        } => {
            partial.stop_reason = reason.clone();
            partial.error_message = error_message;
            partial.usage = usage;
            Some(AssistantMessageEvent::Error {
                reason,
                error: partial.clone(),
            })
        }
    }
}
