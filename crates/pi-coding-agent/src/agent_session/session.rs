use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool, StreamFnBox,
};
use pi_agent_core::types::{Message, Model, StopReason};

use crate::agent_session::events::AgentSessionEvent;
use crate::auth::storage::AuthStorage;
use crate::compaction::compaction;
use crate::error::CodingAgentError;
use crate::messages::convert::convert_to_llm;
use crate::model::registry::ModelRegistry;
use crate::retry::{self, RetryConfig};
use crate::session::manager::SessionManager;
use crate::session::types::SessionEntry;
use crate::settings::manager::SettingsManager;

/// Options for prompting the agent.
#[derive(Debug, Clone, Default)]
pub struct PromptOptions {
    /// Override model for this prompt.
    pub model: Option<Model>,
    /// Custom system prompt additions.
    pub system_prompt: Option<String>,
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

/// Type alias for event listener callbacks.
pub type EventListener = Box<dyn Fn(AgentSessionEvent) + Send + Sync>;

/// Type alias for summary generation function.
/// Takes conversation context (messages to summarize) and an optional previous
/// summary for incremental summarization. Returns a summary string.
pub type SummaryFn = Arc<
    dyn Fn(Vec<AgentMessage>, Option<String>) -> Pin<Box<dyn Future<Output = Result<String, CodingAgentError>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for the convert-to-LLM function used by AgentLoopConfig.
type ConvertToLlmFn = Arc<
    dyn Fn(&[AgentMessage]) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>>
        + Send
        + Sync,
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
        let model = self.model.clone().ok_or_else(|| {
            CodingAgentError::Model("No model set for agent session".to_string())
        })?;

        // Build user message (do NOT push to self.messages yet — agent_loop will add it)
        let user_msg = AgentMessage::user(text);

        // Persist user entry (after validation passes)
        if let Some(session_id) = &self.session_id {
            let entry = SessionEntry::User {
                id: SessionEntry::new_id(),
                parent_id: None,
                timestamp: chrono::Utc::now().timestamp_millis(),
                content: text.to_string(),
            };
            if let Err(e) = self.session_manager.append_entry(session_id, &entry) {
                tracing::warn!("Failed to persist user entry: {e}");
            }
        }

        self.turn_count += 1;

        let convert_fn: ConvertToLlmFn = Arc::new(|msgs: &[AgentMessage]| {
            let msgs = msgs.to_vec();
            Box::pin(async move { convert_to_llm(&msgs) })
        });

        // Wire auth_storage into the get_api_key closure so saved/runtime
        // credentials flow through to provider requests.
        let auth = self.auth_storage.clone();
        let get_api_key_fn: Arc<
            dyn Fn(&str) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync,
        > = Arc::new(move |provider: &str| {
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
                reasoning: None,
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
                    message: AgentMessage::Llm(Message::Assistant(ref assistant_msg)),
                } = event
                {
                    if let Some(session_id) = &self.session_id {
                        let message_value = match serde_json::to_value(assistant_msg) {
                            Ok(val) => val,
                            Err(e) => {
                                tracing::error!("Failed to serialize assistant message: {e}");
                                serde_json::json!({"error": format!("Serialization failed: {e}")})
                            }
                        };
                        let entry = SessionEntry::Assistant {
                            id: SessionEntry::new_id(),
                            parent_id: None,
                            timestamp: chrono::Utc::now().timestamp_millis(),
                            message: message_value,
                        };
                        if let Err(e) = self.session_manager.append_entry(session_id, &entry) {
                            tracing::warn!("Failed to persist assistant entry: {e}");
                        }
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
                        let error_msg = Self::extract_last_error_message(&new_messages)
                            .unwrap_or_default();

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
        entries.iter().rev().find_map(|entry| {
            if let SessionEntry::Summary { summary, .. } = entry {
                Some(summary.clone())
            } else {
                None
            }
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
            let entry = SessionEntry::Summary {
                id: SessionEntry::new_id(),
                parent_id: None,
                timestamp: chrono::Utc::now().timestamp_millis(),
                summary,
                summarized_ids: Vec::new(),
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
        let source_id = self.session_id.as_ref().ok_or_else(|| {
            CodingAgentError::Session("No active session to fork".to_string())
        })?;

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
        self.session_id = Some(header.session_id);
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

    /// Get the current session ID.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Get the working directory.
    pub fn working_dir(&self) -> &std::path::Path {
        &self.working_dir
    }

    /// Get current messages.
    pub fn messages(&self) -> &[AgentMessage] {
        &self.messages
    }

    /// Get mutable messages.
    pub fn messages_mut(&mut self) -> &mut Vec<AgentMessage> {
        &mut self.messages
    }

    /// Get the auth storage.
    pub fn auth_storage(&self) -> &AuthStorage {
        &self.auth_storage
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
