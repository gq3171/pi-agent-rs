use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::agent_types::*;
use crate::event_stream::EventStream;
use crate::types::*;
use crate::validation::validate_tool_arguments;

/// Start an agent loop with new prompt messages.
/// The prompts are added to the context and events are emitted for them.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
    stream_fn: Option<StreamFnBox>,
) -> EventStream<AgentEvent, Vec<AgentMessage>> {
    let stream = create_agent_stream();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let new_messages: Vec<AgentMessage> = prompts.clone();
        let mut current_context = AgentContext {
            system_prompt: context.system_prompt,
            messages: {
                let mut msgs = context.messages;
                msgs.extend(prompts.clone());
                msgs
            },
            tools: context.tools,
        };

        stream_clone.push(AgentEvent::AgentStart);
        stream_clone.push(AgentEvent::TurnStart);
        for prompt in &prompts {
            stream_clone.push(AgentEvent::MessageStart {
                message: prompt.clone(),
            });
            stream_clone.push(AgentEvent::MessageEnd {
                message: prompt.clone(),
            });
        }

        run_loop(
            &mut current_context,
            new_messages,
            &config,
            cancel,
            &stream_clone,
            stream_fn.as_ref(),
        )
        .await;
    });

    stream
}

/// Continue an agent loop from the current context without adding a new message.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
    stream_fn: Option<StreamFnBox>,
) -> Result<EventStream<AgentEvent, Vec<AgentMessage>>, String> {
    if context.messages.is_empty() {
        return Err("Cannot continue: no messages in context".to_string());
    }

    if let Some(last) = context.messages.last() {
        if last.role() == Some("assistant") {
            return Err(format!(
                "Cannot continue from message role: assistant"
            ));
        }
    }

    let stream = create_agent_stream();

    let stream_clone = stream.clone();
    tokio::spawn(async move {
        let new_messages: Vec<AgentMessage> = Vec::new();
        let mut current_context = AgentContext {
            system_prompt: context.system_prompt,
            messages: context.messages,
            tools: context.tools,
        };

        stream_clone.push(AgentEvent::AgentStart);
        stream_clone.push(AgentEvent::TurnStart);

        run_loop(
            &mut current_context,
            new_messages,
            &config,
            cancel,
            &stream_clone,
            stream_fn.as_ref(),
        )
        .await;
    });

    Ok(stream)
}

fn create_agent_stream() -> EventStream<AgentEvent, Vec<AgentMessage>> {
    EventStream::new(
        |event: &AgentEvent| matches!(event, AgentEvent::AgentEnd { .. }),
        |event: &AgentEvent| match event {
            AgentEvent::AgentEnd { messages } => messages.clone(),
            _ => vec![],
        },
    )
}

/// Main loop logic shared by agent_loop and agent_loop_continue.
async fn run_loop(
    current_context: &mut AgentContext,
    mut new_messages: Vec<AgentMessage>,
    config: &AgentLoopConfig,
    cancel: CancellationToken,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    stream_fn: Option<&StreamFnBox>,
) {
    let mut first_turn = true;

    // Check for steering messages at start
    let mut pending_messages: Vec<AgentMessage> = if let Some(get_steering) = &config.get_steering_messages {
        get_steering().await
    } else {
        vec![]
    };

    // Outer loop: continues when queued follow-up messages arrive
    loop {
        let mut has_more_tool_calls = true;
        let mut steering_after_tools: Option<Vec<AgentMessage>> = None;

        // Inner loop: process tool calls and steering messages
        while has_more_tool_calls || !pending_messages.is_empty() {
            if cancel.is_cancelled() {
                break;
            }

            if !first_turn {
                stream.push(AgentEvent::TurnStart);
            } else {
                first_turn = false;
            }

            // Process pending messages
            if !pending_messages.is_empty() {
                for message in pending_messages.drain(..) {
                    stream.push(AgentEvent::MessageStart {
                        message: message.clone(),
                    });
                    stream.push(AgentEvent::MessageEnd {
                        message: message.clone(),
                    });
                    current_context.messages.push(message.clone());
                    new_messages.push(message);
                }
            }

            // Stream assistant response
            let message = stream_assistant_response(current_context, config, cancel.clone(), stream, stream_fn).await;
            new_messages.push(message.clone().into());

            if message.stop_reason == StopReason::Error || message.stop_reason == StopReason::Aborted {
                stream.push(AgentEvent::TurnEnd {
                    message: message.clone().into(),
                    tool_results: vec![],
                });
                stream.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                stream.end(Some(new_messages));
                return;
            }

            // Check for tool calls
            let tool_calls: Vec<ToolCall> = message
                .content
                .iter()
                .filter_map(|c| c.as_tool_call().cloned())
                .collect();
            has_more_tool_calls = !tool_calls.is_empty();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            if has_more_tool_calls {
                let execution = execute_tool_calls(
                    &current_context.tools,
                    &message,
                    cancel.clone(),
                    stream,
                    config.get_steering_messages.as_ref(),
                )
                .await;
                tool_results = execution.tool_results;
                steering_after_tools = execution.steering_messages;

                for result in &tool_results {
                    current_context.messages.push(result.clone().into());
                    new_messages.push(result.clone().into());
                }
            }

            stream.push(AgentEvent::TurnEnd {
                message: message.into(),
                tool_results,
            });

            // Get steering messages after turn completes
            if let Some(steering) = steering_after_tools.take() {
                if !steering.is_empty() {
                    pending_messages = steering;
                    continue;
                }
            }

            pending_messages = if let Some(get_steering) = &config.get_steering_messages {
                get_steering().await
            } else {
                vec![]
            };
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_up_messages = if let Some(get_follow_up) = &config.get_follow_up_messages {
            get_follow_up().await
        } else {
            vec![]
        };

        if !follow_up_messages.is_empty() {
            pending_messages = follow_up_messages;
            continue;
        }

        break;
    }

    stream.push(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    });
    stream.end(Some(new_messages));
}

/// Stream an assistant response from the LLM.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    cancel: CancellationToken,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    stream_fn: Option<&StreamFnBox>,
) -> AssistantMessage {
    // Apply context transform if configured
    let messages = if let Some(transform) = &config.transform_context {
        transform(context.messages.clone(), cancel.clone()).await
    } else {
        context.messages.clone()
    };

    // Convert to LLM-compatible messages
    let llm_messages = (config.convert_to_llm)(&messages).await;

    // Build LLM context
    let tools: Option<Vec<Tool>> = if context.tools.is_empty() {
        None
    } else {
        Some(context.tools.iter().map(|t| t.definition().clone()).collect())
    };

    let llm_context = crate::types::Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools,
    };

    // Resolve API key
    let resolved_api_key = if let Some(get_key) = &config.get_api_key {
        get_key(&config.model.provider).await
    } else {
        None
    };
    let api_key = resolved_api_key.or_else(|| config.api_key.clone());

    let options = SimpleStreamOptions {
        base: StreamOptions {
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            api_key,
            cache_retention: config.cache_retention.clone(),
            session_id: config.session_id.clone(),
            headers: config.headers.clone(),
            max_retry_delay_ms: config.max_retry_delay_ms,
        },
        reasoning: config.reasoning.clone(),
        thinking_budgets: config.thinking_budgets.clone(),
    };

    // Call stream function
    let default_stream_fn: StreamFnBox = Arc::new(|_model, _context, _options| {
        panic!("No stream function provided and no default available. Use fishbot-ai to provide one.");
    });

    let actual_stream_fn = stream_fn.unwrap_or(&default_stream_fn);
    let mut response = Box::pin(actual_stream_fn(&config.model, &llm_context, &options));

    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    while let Some(event) = response.next().await {
        match &event {
            AssistantMessageEvent::Start { partial } => {
                partial_message = Some(partial.clone());
                context.messages.push(partial.clone().into());
                added_partial = true;
                stream.push(AgentEvent::MessageStart {
                    message: partial.clone().into(),
                });
            }
            AssistantMessageEvent::TextStart { .. }
            | AssistantMessageEvent::TextDelta { .. }
            | AssistantMessageEvent::TextEnd { .. }
            | AssistantMessageEvent::ThinkingStart { .. }
            | AssistantMessageEvent::ThinkingDelta { .. }
            | AssistantMessageEvent::ThinkingEnd { .. }
            | AssistantMessageEvent::ToolCallStart { .. }
            | AssistantMessageEvent::ToolCallDelta { .. }
            | AssistantMessageEvent::ToolCallEnd { .. } => {
                // Extract partial from event
                let new_partial = extract_partial(&event).cloned();
                if let Some(p) = new_partial {
                    partial_message = Some(p.clone());
                    if added_partial {
                        let len = context.messages.len();
                        context.messages[len - 1] = p.clone().into();
                    }
                    stream.push(AgentEvent::MessageUpdate {
                        assistant_message_event: event,
                        message: p.into(),
                    });
                }
            }
            AssistantMessageEvent::Done { message, .. } | AssistantMessageEvent::Error { error: message, .. } => {
                let final_message = message.clone();
                if added_partial {
                    let len = context.messages.len();
                    context.messages[len - 1] = final_message.clone().into();
                } else {
                    context.messages.push(final_message.clone().into());
                }
                if !added_partial {
                    stream.push(AgentEvent::MessageStart {
                        message: final_message.clone().into(),
                    });
                }
                stream.push(AgentEvent::MessageEnd {
                    message: final_message.clone().into(),
                });
                return final_message;
            }
        }
    }

    // Fallback - should not normally reach here
    partial_message.unwrap_or_else(|| AssistantMessage::empty(&config.model))
}

fn extract_partial(event: &AssistantMessageEvent) -> Option<&AssistantMessage> {
    match event {
        AssistantMessageEvent::Start { partial } => Some(partial),
        AssistantMessageEvent::TextStart { partial, .. } => Some(partial),
        AssistantMessageEvent::TextDelta { partial, .. } => Some(partial),
        AssistantMessageEvent::TextEnd { partial, .. } => Some(partial),
        AssistantMessageEvent::ThinkingStart { partial, .. } => Some(partial),
        AssistantMessageEvent::ThinkingDelta { partial, .. } => Some(partial),
        AssistantMessageEvent::ThinkingEnd { partial, .. } => Some(partial),
        AssistantMessageEvent::ToolCallStart { partial, .. } => Some(partial),
        AssistantMessageEvent::ToolCallDelta { partial, .. } => Some(partial),
        AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
        AssistantMessageEvent::Done { message, .. } => Some(message),
        AssistantMessageEvent::Error { error, .. } => Some(error),
    }
}

struct ToolExecutionResult {
    tool_results: Vec<ToolResultMessage>,
    steering_messages: Option<Vec<AgentMessage>>,
}

/// Execute tool calls from an assistant message.
async fn execute_tool_calls(
    tools: &[Arc<dyn AgentTool>],
    assistant_message: &AssistantMessage,
    cancel: CancellationToken,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    get_steering_messages: Option<
        &Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync>,
    >,
) -> ToolExecutionResult {
    let tool_calls: Vec<&ToolCall> = assistant_message
        .content
        .iter()
        .filter_map(|c| c.as_tool_call())
        .collect();

    let mut results: Vec<ToolResultMessage> = Vec::new();
    let mut steering_messages: Option<Vec<AgentMessage>> = None;

    for (index, tool_call) in tool_calls.iter().enumerate() {
        let tool = tools.iter().find(|t| t.name() == tool_call.name);

        stream.push(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        });

        let (result, is_error) = match tool {
            None => {
                let err_result = AgentToolResult {
                    content: vec![ContentBlock::Text(TextContent {
                        text: format!("Tool {} not found", tool_call.name),
                        text_signature: None,
                    })],
                    details: Some(serde_json::json!({})),
                };
                (err_result, true)
            }
            Some(tool) => {
                // Validate arguments
                let validated_args = match validate_tool_arguments(tool.definition(), tool_call) {
                    Ok(args) => args,
                    Err(err_msg) => {
                        let error_text = format!("Validation failed: {err_msg}");
                        let err_result = AgentToolResult {
                            content: vec![ContentBlock::Text(TextContent {
                                text: error_text.clone(),
                                text_signature: None,
                            })],
                            details: Some(serde_json::json!({})),
                        };
                        stream.push(AgentEvent::ToolExecutionEnd {
                            tool_call_id: tool_call.id.clone(),
                            tool_name: tool_call.name.clone(),
                            result: err_result,
                            is_error: true,
                        });

                        let tool_result_msg = ToolResultMessage {
                            tool_call_id: tool_call.id.clone(),
                            tool_name: tool_call.name.clone(),
                            content: vec![ContentBlock::Text(TextContent {
                                text: error_text,
                                text_signature: None,
                            })],
                            details: Some(serde_json::json!({})),
                            is_error: true,
                            timestamp: chrono::Utc::now().timestamp_millis(),
                        };
                        results.push(tool_result_msg.clone());
                        stream.push(AgentEvent::MessageStart {
                            message: tool_result_msg.clone().into(),
                        });
                        stream.push(AgentEvent::MessageEnd {
                            message: tool_result_msg.into(),
                        });

                        // Check steering
                        if let Some(get_steering) = get_steering_messages {
                            let steering = get_steering().await;
                            if !steering.is_empty() {
                                steering_messages = Some(steering);
                                let remaining = &tool_calls[index + 1..];
                                for skipped in remaining {
                                    results.push(skip_tool_call(skipped, stream));
                                }
                                return ToolExecutionResult {
                                    tool_results: results,
                                    steering_messages,
                                };
                            }
                        }
                        continue;
                    }
                };

                let stream_clone_for_update = stream.clone();
                let tc_id = tool_call.id.clone();
                let tc_name = tool_call.name.clone();
                let tc_args = tool_call.arguments.clone();

                let on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>> =
                    Some(Box::new(move |partial_result: AgentToolResult| {
                        stream_clone_for_update.push(AgentEvent::ToolExecutionUpdate {
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            args: tc_args.clone(),
                            partial_result,
                        });
                    }));

                match tool
                    .execute(&tool_call.id, validated_args, cancel.clone(), on_update)
                    .await
                {
                    Ok(result) => (result, false),
                    Err(e) => {
                        let err_result = AgentToolResult {
                            content: vec![ContentBlock::Text(TextContent {
                                text: e.to_string(),
                                text_signature: None,
                            })],
                            details: Some(serde_json::json!({})),
                        };
                        (err_result, true)
                    }
                }
            }
        };

        stream.push(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: result.clone(),
            is_error,
        });

        let tool_result_msg = ToolResultMessage {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            content: result.content,
            details: result.details,
            is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        results.push(tool_result_msg.clone());
        stream.push(AgentEvent::MessageStart {
            message: tool_result_msg.clone().into(),
        });
        stream.push(AgentEvent::MessageEnd {
            message: tool_result_msg.into(),
        });

        // Check for steering messages - skip remaining tools if user interrupted
        if let Some(get_steering) = get_steering_messages {
            let steering = get_steering().await;
            if !steering.is_empty() {
                steering_messages = Some(steering);
                let remaining = &tool_calls[index + 1..];
                for skipped in remaining {
                    results.push(skip_tool_call(skipped, stream));
                }
                break;
            }
        }
    }

    ToolExecutionResult {
        tool_results: results,
        steering_messages,
    }
}

fn skip_tool_call(
    tool_call: &ToolCall,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
) -> ToolResultMessage {
    let result = AgentToolResult {
        content: vec![ContentBlock::Text(TextContent {
            text: "Skipped due to queued user message.".to_string(),
            text_signature: None,
        })],
        details: Some(serde_json::json!({})),
    };

    stream.push(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: tool_call.arguments.clone(),
    });
    stream.push(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        result: result.clone(),
        is_error: true,
    });

    let tool_result_msg = ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: result.content,
        details: Some(serde_json::json!({})),
        is_error: true,
        timestamp: chrono::Utc::now().timestamp_millis(),
    };

    stream.push(AgentEvent::MessageStart {
        message: tool_result_msg.clone().into(),
    });
    stream.push(AgentEvent::MessageEnd {
        message: tool_result_msg.clone().into(),
    });

    tool_result_msg
}
