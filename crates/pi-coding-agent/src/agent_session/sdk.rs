use std::sync::Arc;

use crate::agent_session::session::AgentSession;
use crate::auth::storage::AuthStorage;
use crate::config::paths;
use crate::error::CodingAgentError;
use crate::extensions::types::{ExtensionContext, ExtensionFactory};
use crate::extensions::{
    ExtensionRunner, create_extension_tools, discover_and_load_extensions,
    load_extensions_from_paths, wrap_tools_with_extensions,
};
use crate::model::registry::ModelRegistry;
use crate::resources::loader::{
    DefaultResourceLoader, DefaultResourceLoaderOptions, ResourceLoader,
};
use crate::session::manager::SessionManager;
use crate::settings::manager::SettingsManager;
use crate::settings::types::Settings;
use crate::system_prompt::builder::{SystemPromptOptions, build_system_prompt};
use crate::tools::{create_all_tools, create_coding_tools};
use pi_agent_core::types::{AssistantMessage, AssistantMessageEvent, StopReason};
use tokio_util::sync::CancellationToken;

/// Options for creating an agent session.
#[derive(Debug, Clone, Default)]
pub struct CreateSessionOptions {
    /// Override the base config directory (default: ~/.pi/agent/).
    pub config_dir: Option<std::path::PathBuf>,
    /// Working directory for tools.
    pub working_dir: std::path::PathBuf,
    /// Project-level settings to merge.
    pub project_settings: Option<Settings>,
    /// Initial model ID to use.
    pub model_id: Option<String>,
    /// Custom models to register.
    pub custom_models: Option<Vec<crate::settings::types::CustomModelConfig>>,
    /// Optional thinking level: off|minimal|low|medium|high|xhigh.
    pub thinking_level: Option<String>,
    /// Optional system prompt override.
    pub system_prompt: Option<String>,
    /// Optional append system prompt.
    pub append_system_prompt: Option<String>,
    /// Selected built-in tool names. Default is coding tools.
    pub tool_names: Option<Vec<String>>,
    /// Additional skill paths.
    pub skill_paths: Vec<std::path::PathBuf>,
    /// Additional prompt template paths.
    pub prompt_template_paths: Vec<std::path::PathBuf>,
    /// Additional theme paths.
    pub theme_paths: Vec<std::path::PathBuf>,
    /// Disable skills loading.
    pub no_skills: bool,
    /// Disable prompt templates loading.
    pub no_prompt_templates: bool,
    /// Disable themes loading.
    pub no_themes: bool,
}

/// Options for async session creation with extension loading.
#[derive(Clone, Default)]
pub struct CreateSessionWithExtensionsOptions {
    pub base: CreateSessionOptions,
    pub extension_factories: Vec<ExtensionFactory>,
    pub extension_paths: Vec<std::path::PathBuf>,
    pub discover_extensions: bool,
    pub extension_config: Option<serde_json::Value>,
}

/// Result of session creation with optional extension runner.
pub struct CreateSessionResult {
    pub session: AgentSession,
    pub extension_runner: Option<Arc<ExtensionRunner>>,
    pub extension_errors: Vec<crate::extensions::ExtensionLoadError>,
}

/// Create and initialize an AgentSession — the main SDK entry point.
///
/// This function:
/// 1. Resolves the config directory
/// 2. Loads and merges settings
/// 3. Initializes auth storage
/// 4. Builds the model registry (built-in + custom models)
/// 5. Creates the session manager
/// 6. Resolves the initial model
/// 7. Returns a ready-to-use AgentSession
pub fn create_agent_session(
    options: CreateSessionOptions,
) -> Result<AgentSession, CodingAgentError> {
    // 1. Resolve config directory
    let base_dir = paths::resolve_base_dir(options.config_dir.as_deref());
    paths::ensure_dir(&base_dir)?;

    // 2. Load and merge settings
    let mut settings_manager = SettingsManager::new(&base_dir);
    let project_base = options.working_dir.join(paths::CONFIG_DIR_NAME);
    settings_manager.load_and_merge(options.project_settings.as_ref(), Some(&project_base))?;

    // 3. Initialize auth storage
    let auth_storage = AuthStorage::new(&base_dir);

    // 4. Build model registry
    let mut model_registry = ModelRegistry::new();

    // Load custom models from models.json
    let models_path = paths::models_file(&base_dir);
    if let Err(e) = model_registry.load_custom_models(&models_path) {
        tracing::warn!(
            "Failed to load custom models from {}: {e}",
            models_path.display()
        );
    }

    // Add custom models from settings
    if let Some(custom_models) = &settings_manager.settings().custom_models {
        model_registry.add_custom_models(custom_models);
    }

    // Add custom models from options
    if let Some(custom_models) = &options.custom_models {
        model_registry.add_custom_models(custom_models);
    }

    // 5. Create session manager
    let session_manager = SessionManager::new(&base_dir);

    // 6. Create the session
    let mut session = AgentSession::new(
        options.working_dir,
        session_manager,
        Arc::new(auth_storage),
        Arc::new(model_registry),
        Arc::new(settings_manager),
    );

    // 7. Resolve initial model
    if let Some(model_id) = &options.model_id {
        if let Some(model) = session.model_registry().find(model_id) {
            session.set_model(model.clone());
        }
    } else if let Some(model_id) = &session.settings_manager().settings().model {
        if let Some(model) = session.model_registry().find(model_id) {
            session.set_model(model.clone());
        }
    } else if let Some(default_model) = session.model_registry().default_model() {
        session.set_model(default_model.clone());
    }

    // 8. Configure tools
    let tools = if let Some(selected_names) = &options.tool_names {
        let all = create_all_tools(session.working_dir());
        selected_names
            .iter()
            .filter_map(|name| all.get(name).cloned())
            .collect::<Vec<_>>()
    } else {
        create_coding_tools(session.working_dir())
    };
    session.set_tools(tools.clone());

    // 8.1 Configure default stream function from built-in AI providers.
    let registry = Arc::new(pi_agent_ai::register::create_default_registry());
    session.set_stream_fn(Arc::new(move |model, context, options| {
        match pi_agent_ai::stream::stream_simple(
            model,
            context,
            options,
            &registry,
            CancellationToken::new(),
        ) {
            Ok(stream) => stream,
            Err(error) => {
                let stream = pi_agent_core::event_stream::create_assistant_message_event_stream();
                let mut message = AssistantMessage::empty(model);
                message.stop_reason = StopReason::Error;
                message.error_message = Some(error);
                stream.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: message,
                });
                stream
            }
        }
    }));

    // 9. Build default system prompt with skills and tool list
    if let Some(model) = session.model().cloned() {
        let mut resource_loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: session.working_dir().to_path_buf(),
            agent_dir: Some(base_dir.clone()),
            additional_skill_paths: options.skill_paths.clone(),
            additional_prompt_template_paths: options.prompt_template_paths.clone(),
            additional_theme_paths: options.theme_paths.clone(),
            no_skills: options.no_skills,
            no_prompt_templates: options.no_prompt_templates,
            no_themes: options.no_themes,
            system_prompt: options
                .system_prompt
                .clone()
                .or_else(|| session.settings_manager().settings().system_prompt.clone()),
            append_system_prompt: options.append_system_prompt.clone().or_else(|| {
                session
                    .settings_manager()
                    .settings()
                    .append_system_prompt
                    .clone()
            }),
            package_sources: session.settings_manager().settings().packages.clone(),
            ..Default::default()
        });

        if let Err(e) = resource_loader.reload() {
            tracing::warn!("Failed to load resources: {e}");
        }
        let (skills, _) = resource_loader.get_skills();
        let context_files = resource_loader.get_agents_files();

        let tool_names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
        let mut effective_settings = session.settings_manager().settings().clone();
        if let Some(system_prompt) = resource_loader.get_system_prompt() {
            effective_settings.system_prompt = Some(system_prompt.to_string());
        }
        if let Some(append) = resource_loader.get_append_system_prompt().first() {
            effective_settings.append_system_prompt = Some(append.clone());
        }
        let prompt = build_system_prompt(&SystemPromptOptions {
            model: &model,
            settings: &effective_settings,
            working_dir: &session.working_dir().display().to_string(),
            skills,
            context_files,
            tool_names: &tool_names,
            custom_instructions: None,
        });
        session.set_system_prompt(prompt);
    }

    // 10. Apply thinking override.
    if let Some(level) = &options.thinking_level {
        session.set_thinking_level_str(level);
    } else if let Some(level) = session.settings_manager().settings().thinking.clone() {
        session.set_thinking_level_str(&level);
    }

    Ok(session)
}

/// Async session creation that also loads extension factories and wires tool hooks.
pub async fn create_agent_session_with_extensions(
    options: CreateSessionWithExtensionsOptions,
) -> Result<CreateSessionResult, CodingAgentError> {
    let mut session = create_agent_session(options.base.clone())?;

    let has_explicit_paths = !options.extension_paths.is_empty();
    if options.extension_factories.is_empty() && !options.discover_extensions && !has_explicit_paths
    {
        return Ok(CreateSessionResult {
            session,
            extension_runner: None,
            extension_errors: Vec::new(),
        });
    }

    let context = ExtensionContext {
        working_dir: session.working_dir().to_path_buf(),
        session_id: session.session_id().map(ToString::to_string),
        model_id: session.model().map(|m| m.id.clone()),
        config: options
            .extension_config
            .unwrap_or_else(|| serde_json::json!({})),
    };

    let load_result = if options.discover_extensions {
        discover_and_load_extensions(
            &options.extension_paths,
            session.working_dir(),
            context.clone(),
        )
        .await
    } else if has_explicit_paths {
        load_extensions_from_paths(
            &options.extension_paths,
            session.working_dir(),
            context.clone(),
        )
        .await
    } else {
        crate::extensions::LoadExtensionsResult {
            runner: ExtensionRunner::new(context.clone()),
            errors: Vec::new(),
        }
    };

    let mut runner = load_result.runner;
    let mut extension_errors = load_result.errors;

    for (index, factory) in options.extension_factories.iter().enumerate() {
        let extension = factory();
        if let Err(error) = runner.add_extension(extension).await {
            extension_errors.push(crate::extensions::ExtensionLoadError {
                source: format!("factory:{index}"),
                error,
            });
        }
    }
    let runner = Arc::new(runner);

    let mut tools = wrap_tools_with_extensions(session.tools().to_vec(), runner.clone());
    tools.extend(create_extension_tools(runner.clone()));
    session.set_tools(tools.clone());
    session.set_extension_runner(runner.clone());
    runner
        .runtime()
        .set_active_tools(tools.iter().map(|tool| tool.name().to_string()));

    Ok(CreateSessionResult {
        session,
        extension_runner: Some(runner),
        extension_errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;

    use crate::extensions::types::{Extension, ExtensionContext, ToolDefinition};

    #[test]
    fn test_create_agent_session() {
        let tmp = tempfile::tempdir().unwrap();
        let session = create_agent_session(CreateSessionOptions {
            config_dir: Some(tmp.path().to_path_buf()),
            working_dir: tmp.path().to_path_buf(),
            ..Default::default()
        });

        assert!(session.is_ok());
        let session = session.unwrap();
        assert!(session.session_id().is_none()); // No session until first prompt
    }

    #[test]
    fn test_create_agent_session_with_model() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::new();
        let first_model = registry.all_models().first().cloned();

        if let Some(model) = first_model {
            let session = create_agent_session(CreateSessionOptions {
                config_dir: Some(tmp.path().to_path_buf()),
                working_dir: tmp.path().to_path_buf(),
                model_id: Some(model.id.clone()),
                ..Default::default()
            })
            .unwrap();

            assert!(session.model().is_some());
            assert_eq!(session.model().unwrap().id, model.id);
        }
    }

    struct EchoExtension;

    #[async_trait]
    impl Extension for EchoExtension {
        fn name(&self) -> &str {
            "echo-extension"
        }

        async fn init(
            &mut self,
            _context: ExtensionContext,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }

        fn tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "echo_tool".to_string(),
                label: "Echo".to_string(),
                description: "Echo input".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            }]
        }

        async fn handle_tool_call(
            &self,
            _tool_name: &str,
            params: serde_json::Value,
        ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
            let text = params
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(json!({ "content": format!("echo: {text}") }))
        }
    }

    #[tokio::test]
    async fn test_create_agent_session_with_extensions() {
        let tmp = tempfile::tempdir().unwrap();
        let result = create_agent_session_with_extensions(CreateSessionWithExtensionsOptions {
            base: CreateSessionOptions {
                config_dir: Some(tmp.path().to_path_buf()),
                working_dir: tmp.path().to_path_buf(),
                ..Default::default()
            },
            extension_factories: vec![Arc::new(|| Box::new(EchoExtension))],
            extension_paths: Vec::new(),
            discover_extensions: false,
            extension_config: None,
        })
        .await
        .unwrap();

        assert!(result.extension_runner.is_some());
        assert!(result.extension_errors.is_empty());
        assert!(
            result
                .session
                .tools()
                .iter()
                .any(|tool| tool.name() == "echo_tool")
        );
    }

    #[tokio::test]
    async fn test_create_session_loads_explicit_extension_paths_without_discovery() {
        let tmp = tempfile::tempdir().unwrap();
        let ext_dir = tempfile::tempdir().unwrap();
        let manifest_path = ext_dir.path().join("extension.json");
        std::fs::write(
            &manifest_path,
            serde_json::json!({
                "name": "manifest-echo",
                "command": "echo",
                "args": [],
                "tools": [{
                    "name": "manifest_echo_tool",
                    "label": "Manifest Echo",
                    "description": "Echo from manifest extension",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" }
                        }
                    }
                }]
            })
            .to_string(),
        )
        .unwrap();

        let result = create_agent_session_with_extensions(CreateSessionWithExtensionsOptions {
            base: CreateSessionOptions {
                config_dir: Some(tmp.path().to_path_buf()),
                working_dir: tmp.path().to_path_buf(),
                ..Default::default()
            },
            extension_factories: vec![],
            extension_paths: vec![ext_dir.path().to_path_buf()],
            discover_extensions: false,
            extension_config: None,
        })
        .await
        .unwrap();

        assert!(result.extension_runner.is_some());
        assert!(result.extension_errors.is_empty());
        assert!(
            result
                .session
                .tools()
                .iter()
                .any(|tool| tool.name() == "manifest_echo_tool")
        );
    }
}
