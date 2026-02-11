pub mod error;
pub mod config;
pub mod settings;
pub mod auth;
pub mod session;
pub mod messages;
pub mod tools;
pub mod model;
pub mod compaction;
pub mod resources;
pub mod system_prompt;
pub mod agent_session;
pub mod extensions;

// ========== Public API re-exports ==========
// These match the OpenClaw import requirements.

// Error
pub use error::CodingAgentError;

// Config
pub use config::paths;

// Session
pub use session::types::CURRENT_SESSION_VERSION;
pub use session::manager::SessionManager;

// Settings
pub use settings::manager::SettingsManager;

// Auth
pub use auth::storage::AuthStorage;

// Model
pub use model::registry::ModelRegistry;

// AgentSession
pub use agent_session::session::{
    AgentSession, PromptOptions, CompactionResult, ForkResult,
    SessionStats, EventListener, SummaryFn,
};
pub use agent_session::events::AgentSessionEvent;
pub use agent_session::sdk::create_agent_session;

// Extensions
pub use extensions::types::{
    Extension, ExtensionAPI, ExtensionContext,
    ToolDefinition, ContextEvent, FileOperations,
};

// Resources
pub use resources::skills::{Skill, load_skills_from_dir};

// Compaction
pub use compaction::compaction::{estimate_tokens, estimate_messages_tokens};
pub use compaction::branch_summary::generate_summary_context;

// Tools
pub use tools::{
    create_read_tool, create_write_tool, create_edit_tool,
    create_bash_tool, create_coding_tools, create_read_only_tools,
    read_tool, coding_tools,
};
