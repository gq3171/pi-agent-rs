pub mod agent_session;
pub mod auth;
pub mod cli;
pub mod compaction;
pub mod config;
pub mod error;
pub mod export_html;
pub mod extensions;
pub mod keybindings;
pub mod messages;
pub mod model;
pub mod modes;
pub mod resources;
pub mod retry;
pub mod session;
pub mod settings;
pub mod slash_commands;
pub mod system_prompt;
pub mod tools;

// ========== Public API re-exports ==========
// These match the OpenClaw import requirements.

// Error
pub use error::CodingAgentError;
pub use export_html::{ExportHtmlOptions, export_session_to_html};

// Config
pub use config::paths;

// Session
pub use session::manager::SessionManager;
pub use session::types::CURRENT_SESSION_VERSION;

// Settings
pub use settings::manager::SettingsManager;

// Auth
pub use auth::storage::AuthStorage;

// Model
pub use model::registry::ModelRegistry;

// AgentSession
pub use agent_session::events::AgentSessionEvent;
pub use agent_session::sdk::{
    CreateSessionOptions, CreateSessionResult, CreateSessionWithExtensionsOptions,
    create_agent_session, create_agent_session_with_extensions,
};
pub use agent_session::session::{
    AgentSession, CompactionResult, EventListener, ForkResult, ParsedSkillBlock, PromptOptions,
    SessionStats, SummaryFn, parse_skill_block,
};

// Extensions
pub use extensions::types::{
    ContextEvent, ContextUsage, Extension, ExtensionAPI, ExtensionContext, ExtensionFactory,
    FileOperations, ToolCallDecision, ToolDefinition,
};
pub use extensions::{
    ExtensionLoadError, ExtensionRunner, ExtensionRuntime, LoadExtensionsResult,
    create_extension_tools, discover_and_load_extensions, load_extensions_from_factories,
    load_extensions_from_paths, wrap_tool_with_extensions, wrap_tools_with_extensions,
};

// Slash commands
pub use slash_commands::{
    SlashCommandInfo, SlashCommandLocation, SlashCommandSource, builtin_slash_commands,
};

// Keybindings
pub use keybindings::{AppAction, KeybindingsManager, default_app_keybindings};

// Resources
pub use resources::loader::{
    ContextFile, DefaultResourceLoader, DefaultResourceLoaderOptions, PathMetadata,
    ResourceDiagnostic, ResourceDiagnosticType, ResourceExtensionPaths, ResourceLoader,
};
pub use resources::package_manager::{PackageManager, PackageRecord};
pub use resources::prompts::{PromptTemplate, load_prompts_from_dir};
pub use resources::skills::{Skill, load_skills_from_dir};
pub use resources::themes::{Theme, load_themes_from_dir};

// Modes
pub use modes::{
    InteractiveMode, InteractiveModeOptions, PrintModeOptions, PrintOutputMode, RpcCommand,
    RpcResponse, ScopedModelConfig, run_print_mode, run_rpc_mode,
};

// Compaction
#[allow(deprecated)]
pub use compaction::branch_summary::generate_summary_context;
pub use compaction::branch_summary::serialize_conversation;
pub use compaction::compaction::{estimate_messages_tokens, estimate_tokens};

// Tools
pub use tools::{
    all_tools, coding_tools, create_all_tools, create_bash_tool, create_coding_tools,
    create_edit_tool, create_find_tool, create_grep_tool, create_ls_tool, create_read_only_tools,
    create_read_tool, create_write_tool, find_tool, grep_tool, ls_tool, read_tool,
};
