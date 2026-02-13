use pi_agent_core::agent_types::{AgentEvent, AgentMessage};
use pi_agent_core::types::Model;

/// Events emitted by AgentSession, extending AgentEvent with session-level events.
#[derive(Debug, Clone)]
pub enum AgentSessionEvent {
    /// An agent event (delegated from the inner agent loop).
    Agent(AgentEvent),

    /// Session was created or resumed.
    SessionStart { session_id: String, is_new: bool },

    /// Session completed (all turns done).
    SessionEnd {
        session_id: String,
        messages: Vec<AgentMessage>,
    },

    /// Model was switched.
    ModelSwitched { from: Option<Model>, to: Model },

    /// Compaction occurred.
    Compacted {
        messages_before: usize,
        messages_after: usize,
        tokens_before: u64,
        tokens_after: u64,
    },

    /// Session was forked.
    Forked {
        source_session_id: String,
        new_session_id: String,
        fork_entry_id: String,
    },

    /// A retry attempt is starting due to a transient error.
    RetryStart {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error_message: String,
    },

    /// A retry attempt has completed.
    RetryEnd { attempt: u32, success: bool },

    /// Error occurred at session level.
    Error { message: String },
}

impl AgentSessionEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            AgentSessionEvent::Agent(e) => e.event_type(),
            AgentSessionEvent::SessionStart { .. } => "session_start",
            AgentSessionEvent::SessionEnd { .. } => "session_end",
            AgentSessionEvent::ModelSwitched { .. } => "model_switched",
            AgentSessionEvent::Compacted { .. } => "compacted",
            AgentSessionEvent::Forked { .. } => "forked",
            AgentSessionEvent::RetryStart { .. } => "retry_start",
            AgentSessionEvent::RetryEnd { .. } => "retry_end",
            AgentSessionEvent::Error { .. } => "session_error",
        }
    }
}
