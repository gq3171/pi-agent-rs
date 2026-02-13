use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool, GetApiKeyFn, StreamFnBox,
};
use pi_agent_core::types::{Message, Model, StopReason, ThinkingLevel};

use crate::agent_session::events::AgentSessionEvent;
use crate::auth::storage::AuthStorage;
use crate::compaction::compaction;
use crate::error::CodingAgentError;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::ContextEvent;
use crate::messages::convert::convert_to_llm;
use crate::model::registry::ModelRegistry;
use crate::retry::{self, RetryConfig};
use crate::session::manager::SessionManager;
use crate::session::types::{SessionEntry, now_iso_timestamp};
use crate::settings::manager::SettingsManager;

/// Options for prompting the agent.
#[derive(Debug, Clone, Default)]
pub struct PromptOptions {
    /// Override model for this prompt.
    pub model: Option<Model>,
    /// Custom system prompt additions.
    pub system_prompt: Option<String>,
}

/// Parsed `<skill ...>` block from user input.
#[derive(Debug, Clone)]
pub struct ParsedSkillBlock {
    pub name: String,
    pub location: String,
    pub content: String,
    pub user_message: Option<String>,
}

/// Parse a skill block from text.
///
/// Expected format:
/// <skill name=\"...\" location=\"...\">\n...\n</skill>\n\n(optional user message)
pub fn parse_skill_block(text: &str) -> Option<ParsedSkillBlock> {
    let re = regex::Regex::new(
        r#"(?s)^<skill name="([^"]+)" location="([^"]+)">\n(.*?)\n</skill>(?:\n\n(.*))?$"#,
    )
    .ok()?;
    let caps = re.captures(text)?;

    let name = caps.get(1)?.as_str().to_string();
    let location = caps.get(2)?.as_str().to_string();
    let content = caps.get(3)?.as_str().to_string();
    let user_message = caps
        .get(4)
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty());

    Some(ParsedSkillBlock {
        name,
        location,
        content,
        user_message,
    })
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary: String,
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// Result of a fork operation.
#[derive(Debug, Clone)]
pub struct ForkResult {
    pub new_session_id: String,
    pub forked_entries: usize,
}

/// Session statistics.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub session_id: Option<String>,
    pub message_count: usize,
    pub estimated_tokens: u64,
    pub turn_count: usize,
}

/// Estimated context usage for the active model.
#[derive(Debug, Clone, Default)]
pub struct ContextUsage {
    /// Estimated context tokens, or `None` if unknown.
    pub tokens: Option<u64>,
    pub context_window: u64,
    /// Usage percentage, or `None` when `tokens` is unknown.
    pub percent: Option<f64>,
}

/// Type alias for event listener callbacks.
pub type EventListener = Box<dyn Fn(AgentSessionEvent) + Send + Sync>;

/// Type alias for summary generation function.
/// Takes conversation context (messages to summarize) and an optional previous
/// summary for incremental summarization. Returns a summary string.
pub type SummaryFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            Option<String>,
        ) -> Pin<Box<dyn Future<Output = Result<String, CodingAgentError>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for the convert-to-LLM function used by AgentLoopConfig.
type ConvertToLlmFn = Arc<
    dyn Fn(&[AgentMessage]) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> + Send + Sync,
>;

/// Core session orchestrator that ties together the agent, session persistence,
/// auth, model management, and tools.
pub struct AgentSession {
    /// Current session ID.
    session_id: Option<String>,
    /// Working directory.
    working_dir: PathBuf,
    /// Current model.
    model: Option<Model>,
    /// System prompt.
    system_prompt: String,
    /// Agent messages (conversation context).
    messages: Vec<AgentMessage>,
    /// Tools available to the agent.
    tools: Vec<Arc<dyn AgentTool>>,
    /// LLM stream function.
    stream_fn: Option<StreamFnBox>,
    /// Session manager for persistence.
    session_manager: SessionManager,
    /// Auth storage.
    auth_storage: Arc<AuthStorage>,
    /// Model registry.
    model_registry: Arc<ModelRegistry>,
    /// Settings manager.
    settings_manager: Arc<SettingsManager>,
    /// Summary generation function for compaction.
    summary_fn: Option<SummaryFn>,
    /// Retry configuration for transient errors.
    retry_config: RetryConfig,
    /// Current retry attempt counter (reset per prompt).
    retry_attempt: u32,
    /// Cancellation token.
    cancel: CancellationToken,
    /// Event listeners.
    listeners: Vec<EventListener>,
    /// Turn counter.
    turn_count: usize,
    /// Extension runner (if configured).
    extension_runner: Option<Arc<ExtensionRunner>>,
    /// Default reasoning/thinking level.
    thinking_level: Option<ThinkingLevel>,
}

impl AgentSession {
    /// Create a new AgentSession.
    pub fn new(
        working_dir: PathBuf,
        session_manager: SessionManager,
        auth_storage: Arc<AuthStorage>,
        model_registry: Arc<ModelRegistry>,
        settings_manager: Arc<SettingsManager>,
    ) -> Self {
        Self {
            session_id: None,
            working_dir,
            model: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            stream_fn: None,
            session_manager,
            auth_storage,
            model_registry,
            settings_manager,
            summary_fn: None,
            retry_config: RetryConfig::default(),
            retry_attempt: 0,
            cancel: CancellationToken::new(),
            listeners: Vec::new(),
            turn_count: 0,
            extension_runner: None,
            thinking_level: None,
        }
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    /// Set the tools available to the agent.
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn AgentTool>>) {
        self.tools = tools;
    }

    /// Set the LLM stream function.
    pub fn set_stream_fn(&mut self, stream_fn: StreamFnBox) {
        self.stream_fn = Some(stream_fn);
    }

    /// Set the summary generation function for compaction.
    pub fn set_summary_fn(&mut self, summary_fn: SummaryFn) {
        self.summary_fn = Some(summary_fn);
    }

    /// Set the retry configuration for transient errors.
    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
    }

    /// Set extension runner for hook dispatch and tool interception.
    pub fn set_extension_runner(&mut self, runner: Arc<ExtensionRunner>) {
        self.extension_runner = Some(runner);
    }

    /// Set default thinking level from string.
    pub fn set_thinking_level_str(&mut self, level: &str) {
        self.thinking_level = match level {
            "off" => None,
            "minimal" => Some(ThinkingLevel::Minimal),
            "low" => Some(ThinkingLevel::Low),
            "medium" => Some(ThinkingLevel::Medium),
            "high" => Some(ThinkingLevel::High),
            "xhigh" => Some(ThinkingLevel::Xhigh),
            _ => self.thinking_level.clone(),
        };
    }

    /// Set default thinking level.
    pub fn set_thinking_level(&mut self, level: Option<ThinkingLevel>) {
        self.thinking_level = level;
    }

    /// Send a prompt to the agent.
    ///
    /// This is the main entry point for interacting with the agent.
    /// It handles session creation/continuation, message persistence,
    /// agent loop execution, and event emission.
    pub async fn prompt(
        &mut self,
        text: &str,
        options: PromptOptions,
    ) -> Result<(), CodingAgentError> {
        // Validate stream_fn is configured before starting
        if self.stream_fn.is_none() {
            return Err(CodingAgentError::Config(
                "No stream function configured. Call set_stream_fn() before prompt().".to_string(),
            ));
        }

        // Ensure we have a session
        if self.session_id.is_none() {
            let session_id = uuid::Uuid::new_v4().to_string();
            self.session_manager.create(&session_id, None)?;
            self.session_id = Some(session_id.clone());
            self.emit(AgentSessionEvent::SessionStart {
                session_id,
                is_new: true,
            });
        }

        // Apply model override if provided
        if let Some(model) = options.model {
            self.set_model(model);
        }

        // Build system prompt with optional additions
        let system_prompt = if let Some(addition) = &options.system_prompt {
            format!("{}\n\n{}", self.system_prompt, addition)
        } else {
            self.system_prompt.clone()
        };

        // Validate model before any side effects (persist, turn count, etc.)
        let model = self
            .model
            .clone()
            .ok_or_else(|| CodingAgentError::Model("No model set for agent session".to_string()))?;

        // Build user message (do NOT push to self.messages yet — agent_loop will add it)
        let user_msg = AgentMessage::user(text);

        // Persist user entry (after validation passes)
        if let Some(session_id) = &self.session_id {
            let entry = SessionEntry::Message {
                id: SessionEntry::new_id(),
                parent_id: None,
                timestamp: now_iso_timestamp(),
                message: Message::User(pi_agent_core::types::UserMessage {
                    content: pi_agent_core::types::UserContent::Text(text.to_string()),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                }),
            };
            if let Err(e) = self.session_manager.append_entry(session_id, &entry) {
                tracing::warn!("Failed to persist user entry: {e}");
            }
        }

        self.turn_count += 1;
        if let Some(runner) = &self.extension_runner {
            if let Err(e) = runner.emit_event(ContextEvent::TurnStart).await {
                tracing::warn!("Extension turn_start event failed: {e}");
            }
        }

        let convert_fn: ConvertToLlmFn = Arc::new(|msgs: &[AgentMessage]| {
            let msgs = msgs.to_vec();
            Box::pin(async move { convert_to_llm(&msgs) })
        });

        // Wire auth_storage into the get_api_key closure so saved/runtime
        // credentials flow through to provider requests.
        let auth = self.auth_storage.clone();
        let get_api_key_fn: Arc<GetApiKeyFn> = Arc::new(move |provider: &str| {
            let auth = auth.clone();
            let provider = provider.to_string();
            Box::pin(async move { auth.get_api_key(&provider) })
        });

        // Reset retry attempt counter for this prompt
        self.retry_attempt = 0;
        let context_window = model.context_window;

        loop {
            let config = AgentLoopConfig {
                model: model.clone(),
                reasoning: if model.reasoning {
                    self.thinking_level.clone()
                } else {
                    None
                },
                thinking_budgets: None,
                temperature: None,
                max_tokens: None,
                api_key: None,
                cache_retention: None,
                session_id: self.session_id.clone(),
                headers: None,
                max_retry_delay_ms: None,
                convert_to_llm: convert_fn.clone(),
                transform_context: None,
                get_api_key: Some(get_api_key_fn.clone()),
                get_steering_messages: None,
                get_follow_up_messages: None,
            };

            // Reset cancellation for this prompt attempt
            self.cancel = CancellationToken::new();

            let event_stream = pi_agent_core::agent_loop::agent_loop(
                vec![user_msg.clone()],
                AgentContext {
                    system_prompt: system_prompt.clone(),
                    messages: self.messages.clone(),
                    tools: self.tools.clone(),
                },
                config,
                self.cancel.clone(),
                self.stream_fn.clone(),
            );

            // Consume events from the agent loop, forwarding to listeners
            let mut pinned = Box::pin(event_stream.clone());
            while let Some(event) = pinned.next().await {
                // Persist assistant entries on message end
                if let AgentEvent::MessageEnd {
                    message: AgentMessage::Llm(Message::Assistant(assistant_msg)),
                } = &event
                {
                    if let Some(session_id) = &self.session_id {
                        let message_value = match serde_json::to_value(assistant_msg) {
                            Ok(val) => val,
                            Err(e) => {
                                tracing::error!("Failed to serialize assistant message: {e}");
                                serde_json::json!({"error": format!("Serialization failed: {e}")})
                            }
                        };
                        let entry = SessionEntry::Message {
                            id: SessionEntry::new_id(),
                            parent_id: None,
                            timestamp: now_iso_timestamp(),
                            message: match serde_json::from_value::<Message>(message_value) {
                                Ok(message) => message,
                                Err(_) => Message::Assistant(assistant_msg.clone()),
                            },
                        };
                        if let Err(e) = self.session_manager.append_entry(session_id, &entry) {
                            tracing::warn!("Failed to persist assistant entry: {e}");
                        }
                    }
                }

                if let Some(runner) = &self.extension_runner {
                    let extension_event = match &event {
                        AgentEvent::MessageStart { message } => Some(ContextEvent::MessageStart {
                            message: message.clone(),
                        }),
                        AgentEvent::MessageUpdate {
                            message,
                            assistant_message_event,
                        } => Some(ContextEvent::MessageUpdate {
                            message: message.clone(),
                            assistant_message_event: assistant_message_event.clone(),
                        }),
                        AgentEvent::MessageEnd { message } => Some(ContextEvent::MessageEnd {
                            message: message.clone(),
                        }),
                        AgentEvent::ToolExecutionStart {
                            tool_call_id,
                            tool_name,
                            args,
                        } => Some(ContextEvent::ToolExecutionStart {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: tool_name.clone(),
                            args: args.clone(),
                        }),
                        AgentEvent::ToolExecutionUpdate {
                            tool_call_id,
                            tool_name,
                            args,
                            partial_result,
                        } => Some(ContextEvent::ToolExecutionUpdate {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: tool_name.clone(),
                            args: args.clone(),
                            partial_result: partial_result.clone(),
                        }),
                        AgentEvent::ToolExecutionEnd {
                            tool_call_id,
                            tool_name,
                            result,
                            is_error,
                        } => Some(ContextEvent::ToolExecutionEnd {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: tool_name.clone(),
                            result: result.clone(),
                            is_error: *is_error,
                        }),
                        _ => None,
                    };
                    if let Some(extension_event) = extension_event
                        && let Err(e) = runner.emit_event(extension_event).await
                    {
                        tracing::warn!("Extension event dispatch failed: {e}");
                    }
                }

                self.emit(AgentSessionEvent::Agent(event));
            }

            // Get the final messages from the agent loop result
            match event_stream.result().await {
                Some(new_messages) => {
                    // Check the last assistant message for retryable errors
                    let should_retry = self.retry_config.enabled
                        && self.retry_attempt < self.retry_config.max_retries
                        && Self::check_last_message_retryable(&new_messages, context_window);

                    if should_retry {
                        self.retry_attempt += 1;
                        let delay_ms =
                            retry::calculate_delay(&self.retry_config, self.retry_attempt);
                        let error_msg =
                            Self::extract_last_error_message(&new_messages).unwrap_or_default();

                        self.emit(AgentSessionEvent::RetryStart {
                            attempt: self.retry_attempt,
                            max_attempts: self.retry_config.max_retries,
                            delay_ms,
                            error_message: error_msg.clone(),
                        });

                        tracing::warn!(
                            attempt = self.retry_attempt,
                            max = self.retry_config.max_retries,
                            delay_ms,
                            error = %error_msg,
                            "Retrying after transient error"
                        );

                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

                        // Do NOT extend self.messages with error messages — retry from scratch
                        continue;
                    }

                    // Success or non-retryable — commit messages
                    if self.retry_attempt > 0 {
                        self.emit(AgentSessionEvent::RetryEnd {
                            attempt: self.retry_attempt,
                            success: true,
                        });
                    }

                    self.messages.extend(new_messages);
                    if let Some(session_id) = &self.session_id {
                        self.emit(AgentSessionEvent::SessionEnd {
                            session_id: session_id.clone(),
                            messages: self.messages.clone(),
                        });
                    }
                    if let Some(runner) = &self.extension_runner {
                        if let Err(e) = runner.emit_event(ContextEvent::TurnEnd).await {
                            tracing::warn!("Extension turn_end event failed: {e}");
                        }
                    }
                }
                None => {
                    if self.retry_attempt > 0 {
                        self.emit(AgentSessionEvent::RetryEnd {
                            attempt: self.retry_attempt,
                            success: false,
                        });
                    }
                    let err_msg = "Agent loop ended without producing a result".to_string();
                    self.emit(AgentSessionEvent::Error {
                        message: err_msg.clone(),
                    });
                    return Err(CodingAgentError::Agent(err_msg));
                }
            }

            break;
        }

        Ok(())
    }

    /// Check if the last assistant message in new_messages contains a retryable error.
    fn check_last_message_retryable(messages: &[AgentMessage], context_window: u64) -> bool {
        // Find the last assistant message
        for msg in messages.iter().rev() {
            if let AgentMessage::Llm(Message::Assistant(assistant)) = msg {
                if assistant.stop_reason == StopReason::Error {
                    if let Some(ref error_msg) = assistant.error_message {
                        return retry::is_retryable_error(error_msg, context_window);
                    }
                }
                break;
            }
        }
        false
    }

    /// Extract the error message from the last assistant message, if any.
    fn extract_last_error_message(messages: &[AgentMessage]) -> Option<String> {
        for msg in messages.iter().rev() {
            if let AgentMessage::Llm(Message::Assistant(assistant)) = msg {
                return assistant.error_message.clone();
            }
        }
        None
    }

    /// Abort the current operation.
    pub fn abort(&self) {
        self.cancel.cancel();
    }

    /// Subscribe to session events. Returns a listener index.
    pub fn subscribe(&mut self, listener: EventListener) -> usize {
        let idx = self.listeners.len();
        self.listeners.push(listener);
        idx
    }

    /// Get the current model.
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Set the current model.
    pub fn set_model(&mut self, model: Model) {
        let old = self.model.take();
        self.emit(AgentSessionEvent::ModelSwitched {
            from: old,
            to: model.clone(),
        });
        self.model = Some(model);
    }

    /// Find the most recent compaction summary from persisted session entries.
    ///
    /// Walks the JSONL entries in reverse to find the last `Summary` entry,
    /// which represents the previous compaction result.
    fn find_last_compaction_summary(&self) -> Option<String> {
        let session_id = self.session_id.as_ref()?;
        let (_header, entries) = self.session_manager.open(session_id).ok()?;
        entries.iter().rev().find_map(|entry| match entry {
            SessionEntry::Compaction { summary, .. }
            | SessionEntry::LegacySummary { summary, .. } => Some(summary.clone()),
            _ => None,
        })
    }

    fn find_latest_compaction_timestamp_ms(&self) -> Option<i64> {
        let session_id = self.session_id.as_ref()?;
        let (_header, entries) = self.session_manager.open(session_id).ok()?;
        entries.iter().rev().find_map(|entry| match entry {
            SessionEntry::Compaction { timestamp, .. } => {
                chrono::DateTime::parse_from_rfc3339(timestamp)
                    .ok()
                    .map(|value| value.timestamp_millis())
            }
            SessionEntry::LegacySummary { timestamp, .. } => Some(*timestamp),
            _ => None,
        })
    }

    /// Compact the conversation using the given settings.
    ///
    /// Uses token-based cut point detection (aligned with pi-mono) to decide
    /// how much context to keep. If `settings` is `None`, uses defaults
    /// (`reserve_tokens: 16384`, `keep_recent_tokens: 20000`).
    ///
    /// If a `summary_fn` is set on the session, it will be used to generate
    /// an LLM-based summary. The previous compaction summary (if any) is
    /// passed to the summary function for incremental summarization.
    pub async fn compact(
        &mut self,
        settings: Option<&compaction::CompactionSettings>,
    ) -> Result<CompactionResult, CodingAgentError> {
        let default_settings = compaction::CompactionSettings::default();
        let settings = settings.unwrap_or(&default_settings);

        let tokens_before = compaction::estimate_messages_tokens(&self.messages);
        let messages_before = self.messages.len();

        let (to_summarize, to_keep) = compaction::prepare_compaction(&self.messages, settings);

        if to_summarize.is_empty() {
            return Err(CodingAgentError::Compaction(
                "Not enough messages to compact".to_string(),
            ));
        }

        // Look up the previous compaction summary for incremental summarization
        let previous_summary = self.find_last_compaction_summary();

        // Generate summary — use LLM-based summary_fn if available
        let summary = if let Some(summary_fn) = &self.summary_fn {
            summary_fn(to_summarize.to_vec(), previous_summary).await?
        } else {
            // Fallback: use structured context extraction (no LLM)
            let summary_context =
                crate::compaction::branch_summary::serialize_conversation(&to_summarize);
            let max_len = 500;
            let end = summary_context
                .char_indices()
                .take_while(|&(i, _)| i <= max_len)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            format!("Conversation summary: {}", &summary_context[..end])
        };

        self.messages = compaction::apply_compaction(&summary, to_keep);

        let tokens_after = compaction::estimate_messages_tokens(&self.messages);
        let messages_after = self.messages.len();

        let result = CompactionResult {
            summary: summary.clone(),
            messages_before,
            messages_after,
            tokens_before,
            tokens_after,
        };

        self.emit(AgentSessionEvent::Compacted {
            messages_before,
            messages_after,
            tokens_before,
            tokens_after,
        });

        // Persist compaction summary
        if let Some(session_id) = &self.session_id {
            let entry = SessionEntry::Compaction {
                id: SessionEntry::new_id(),
                parent_id: None,
                timestamp: now_iso_timestamp(),
                summary,
                first_kept_entry_id: None,
                tokens_before,
                details: None,
                from_hook: None,
            };
            if let Err(e) = self.session_manager.append_entry(session_id, &entry) {
                tracing::warn!("Failed to persist compaction summary: {e}");
            }
        }

        Ok(result)
    }

    /// Restore a previously-persisted session by loading its JSONL history.
    ///
    /// This rebuilds the in-memory `messages` and `turn_count` from the
    /// session entries stored on disk, allowing the agent to continue a
    /// conversation across process restarts.
    pub fn restore_session(&mut self, session_id: &str) -> Result<(), CodingAgentError> {
        let (_header, entries) = self.session_manager.open(session_id)?;
        self.session_id = Some(session_id.to_string());
        self.messages = crate::session::context::build_session_context(&entries);
        self.turn_count = self
            .messages
            .iter()
            .filter(|m| matches!(m, AgentMessage::Llm(Message::User(_))))
            .count();
        Ok(())
    }

    /// Fork the session from a specific entry.
    pub async fn fork(&mut self, entry_id: &str) -> Result<ForkResult, CodingAgentError> {
        let source_id = self
            .session_id
            .as_ref()
            .ok_or_else(|| CodingAgentError::Session("No active session to fork".to_string()))?;

        let new_session_id = uuid::Uuid::new_v4().to_string();
        let (header, entries) =
            self.session_manager
                .fork_from(source_id, entry_id, &new_session_id)?;

        let forked_entries = entries.len();

        self.emit(AgentSessionEvent::Forked {
            source_session_id: source_id.clone(),
            new_session_id: new_session_id.clone(),
            fork_entry_id: entry_id.to_string(),
        });

        // Switch to the new session
        self.session_id = Some(header.id);
        // Rebuild context from forked entries
        self.messages = crate::session::context::build_session_context(&entries);

        Ok(ForkResult {
            new_session_id,
            forked_entries,
        })
    }

    /// Get session statistics.
    pub fn get_stats(&self) -> SessionStats {
        SessionStats {
            session_id: self.session_id.clone(),
            message_count: self.messages.len(),
            estimated_tokens: compaction::estimate_messages_tokens(&self.messages),
            turn_count: self.turn_count,
        }
    }

    /// Estimate current context usage for the active model.
    ///
    /// After compaction, usage is unknown until we receive a successful assistant
    /// response with usage metrics produced after the latest compaction boundary.
    pub fn get_context_usage(&self) -> Option<ContextUsage> {
        let model = self.model.as_ref()?;
        let context_window = model.context_window;
        if context_window == 0 {
            return None;
        }

        if let Some(compaction_ts) = self.find_latest_compaction_timestamp_ms() {
            let mut has_post_compaction_usage = false;

            for message in self.messages.iter().rev() {
                let AgentMessage::Llm(Message::Assistant(assistant)) = message else {
                    continue;
                };
                if assistant.timestamp <= compaction_ts {
                    break;
                }
                if assistant.stop_reason == StopReason::Aborted
                    || assistant.stop_reason == StopReason::Error
                {
                    continue;
                }
                let context_tokens = assistant.usage.input
                    + assistant.usage.output
                    + assistant.usage.cache_read
                    + assistant.usage.cache_write;
                if context_tokens > 0 {
                    has_post_compaction_usage = true;
                }
                break;
            }

            if !has_post_compaction_usage {
                return Some(ContextUsage {
                    tokens: None,
                    context_window,
                    percent: None,
                });
            }
        }

        let tokens = compaction::estimate_messages_tokens(&self.messages);
        let percent = (tokens as f64 / context_window as f64) * 100.0;
        Some(ContextUsage {
            tokens: Some(tokens),
            context_window,
            percent: Some(percent),
        })
    }

    /// Get the current session ID.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Reset current in-memory conversation and force next prompt to start a new session.
    pub fn reset_session(&mut self) {
        self.session_id = None;
        self.messages.clear();
        self.turn_count = 0;
    }

    /// Get the working directory.
    pub fn working_dir(&self) -> &std::path::Path {
        &self.working_dir
    }

    /// Get current messages.
    pub fn messages(&self) -> &[AgentMessage] {
        &self.messages
    }

    /// Get current tools.
    pub fn tools(&self) -> &[Arc<dyn AgentTool>] {
        &self.tools
    }

    /// Get mutable messages.
    pub fn messages_mut(&mut self) -> &mut Vec<AgentMessage> {
        &mut self.messages
    }

    /// Get the auth storage.
    pub fn auth_storage(&self) -> &AuthStorage {
        &self.auth_storage
    }

    /// Get the session manager.
    pub fn session_manager(&self) -> &SessionManager {
        &self.session_manager
    }

    /// Get the model registry.
    pub fn model_registry(&self) -> &ModelRegistry {
        &self.model_registry
    }

    /// Get the settings manager.
    pub fn settings_manager(&self) -> &SettingsManager {
        &self.settings_manager
    }

    /// Get the cancellation token.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Add an agent event (to be called from external agent loop integration).
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        self.emit(AgentSessionEvent::Agent(event));
    }

    /// Emit an event to all listeners.
    fn emit(&self, event: AgentSessionEvent) {
        for listener in &self.listeners {
            listener(event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::{AssistantMessage, Message, Usage, UsageCost};

    fn create_test_session() -> (tempfile::TempDir, AgentSession) {
        let tmp = tempfile::tempdir().unwrap();
        let base_dir = tmp.path().join("agent");
        std::fs::create_dir_all(&base_dir).unwrap();

        let session_manager = SessionManager::new(&base_dir);
        session_manager.create("test-session", None).unwrap();

        let auth_storage = Arc::new(AuthStorage::new(&base_dir));
        let model_registry = Arc::new(ModelRegistry::new());
        let settings_manager = Arc::new(SettingsManager::new(&base_dir));

        let mut session = AgentSession::new(
            tmp.path().to_path_buf(),
            session_manager,
            auth_storage,
            model_registry.clone(),
            settings_manager,
        );
        session.session_id = Some("test-session".to_string());
        session.set_model(model_registry.all_models()[0].clone());

        (tmp, session)
    }

    #[test]
    fn test_get_context_usage_unknown_after_compaction_without_post_usage() {
        let (_tmp, session) = create_test_session();
        let compaction_entry = SessionEntry::Compaction {
            id: SessionEntry::new_id(),
            parent_id: None,
            timestamp: "2026-02-12T00:00:00Z".to_string(),
            summary: "summary".to_string(),
            first_kept_entry_id: None,
            tokens_before: 1000,
            details: None,
            from_hook: None,
        };
        session
            .session_manager
            .append_entry("test-session", &compaction_entry)
            .unwrap();

        let usage = session.get_context_usage().unwrap();
        assert_eq!(usage.tokens, None);
        assert_eq!(usage.percent, None);
        assert!(usage.context_window > 0);
    }

    #[test]
    fn test_get_context_usage_known_with_post_compaction_usage() {
        let (_tmp, mut session) = create_test_session();
        let compaction_entry = SessionEntry::Compaction {
            id: SessionEntry::new_id(),
            parent_id: None,
            timestamp: "2026-02-12T00:00:00Z".to_string(),
            summary: "summary".to_string(),
            first_kept_entry_id: None,
            tokens_before: 1000,
            details: None,
            from_hook: None,
        };
        session
            .session_manager
            .append_entry("test-session", &compaction_entry)
            .unwrap();

        session
            .messages
            .push(AgentMessage::Llm(Message::Assistant(AssistantMessage {
                content: Vec::new(),
                api: "openai-responses".to_string(),
                provider: "openai".to_string(),
                model: "gpt-5".to_string(),
                usage: Usage {
                    input: 120,
                    output: 80,
                    cache_read: 0,
                    cache_write: 0,
                    total_tokens: 200,
                    cost: UsageCost::default(),
                },
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: chrono::DateTime::parse_from_rfc3339("2026-02-12T00:00:01Z")
                    .unwrap()
                    .timestamp_millis(),
            })));

        let usage = session.get_context_usage().unwrap();
        assert!(usage.tokens.is_some());
        assert!(usage.percent.is_some());
    }
}
