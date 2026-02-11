use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use pi_agent_core::agent_types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool, StreamFnBox,
};
use pi_agent_core::types::{Message, Model};

use crate::agent_session::events::AgentSessionEvent;
use crate::auth::storage::AuthStorage;
use crate::compaction::compaction;
use crate::error::CodingAgentError;
use crate::messages::convert::convert_to_llm;
use crate::model::registry::ModelRegistry;
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
/// Takes conversation context (messages to summarize) and returns a summary string.
pub type SummaryFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Result<String, CodingAgentError>> + Send>>
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

        let context = AgentContext {
            system_prompt,
            messages: self.messages.clone(),
            tools: self.tools.clone(),
        };

        let convert_fn: ConvertToLlmFn = Arc::new(|msgs: &[AgentMessage]| {
            let msgs = msgs.to_vec();
            Box::pin(async move { convert_to_llm(&msgs) })
        });

        let config = AgentLoopConfig {
            model,
            reasoning: None,
            thinking_budgets: None,
            temperature: None,
            max_tokens: None,
            api_key: None,
            cache_retention: None,
            session_id: self.session_id.clone(),
            headers: None,
            max_retry_delay_ms: None,
            convert_to_llm: convert_fn,
            transform_context: None,
            get_api_key: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
        };

        // Reset cancellation for this prompt
        self.cancel = CancellationToken::new();

        let event_stream = pi_agent_core::agent_loop::agent_loop(
            vec![user_msg],
            context,
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
                    let message_value = serde_json::to_value(assistant_msg)
                        .unwrap_or(serde_json::Value::Null);
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
        // agent_loop returns only the NEW messages (prompts + responses)
        match event_stream.result().await {
            Some(new_messages) => {
                self.messages.extend(new_messages);
                if let Some(session_id) = &self.session_id {
                    self.emit(AgentSessionEvent::SessionEnd {
                        session_id: session_id.clone(),
                        messages: self.messages.clone(),
                    });
                }
            }
            None => {
                let err_msg = "Agent loop ended without producing a result".to_string();
                self.emit(AgentSessionEvent::Error {
                    message: err_msg.clone(),
                });
                return Err(CodingAgentError::Agent(err_msg));
            }
        }

        Ok(())
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

    /// Compact the conversation.
    ///
    /// If a `summary_fn` is set on the session, it will be used to generate
    /// an LLM-based summary. Otherwise, a context-based summary is produced.
    pub async fn compact(&mut self) -> Result<CompactionResult, CodingAgentError> {
        let tokens_before = compaction::estimate_messages_tokens(&self.messages);
        let messages_before = self.messages.len();

        // Keep the last 6 messages (3 turns) by default
        let (to_summarize, to_keep) = compaction::prepare_compaction(&self.messages, 6);

        if to_summarize.is_empty() {
            return Err(CodingAgentError::Compaction(
                "Not enough messages to compact".to_string(),
            ));
        }

        // Generate summary — use LLM-based summary_fn if available
        let summary = if let Some(summary_fn) = &self.summary_fn {
            summary_fn(to_summarize.to_vec()).await?
        } else {
            // Fallback: use structured context extraction (no LLM)
            let summary_context =
                crate::compaction::branch_summary::generate_summary_context(&to_summarize);
            format!(
                "Conversation summary: {}",
                &summary_context[..summary_context.len().min(500)]
            )
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
