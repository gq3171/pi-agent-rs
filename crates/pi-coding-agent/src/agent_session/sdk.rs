use std::sync::Arc;

use crate::agent_session::session::AgentSession;
use crate::auth::storage::AuthStorage;
use crate::config::paths;
use crate::error::CodingAgentError;
use crate::model::registry::ModelRegistry;
use crate::session::manager::SessionManager;
use crate::settings::manager::SettingsManager;
use crate::settings::types::Settings;

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
    settings_manager.load_and_merge(options.project_settings.as_ref())?;

    // 3. Initialize auth storage
    let auth_storage = AuthStorage::new(&base_dir);

    // 4. Build model registry
    let mut model_registry = ModelRegistry::new();

    // Load custom models from models.json
    let models_path = paths::models_file(&base_dir);
    let _ = model_registry.load_custom_models(&models_path);

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
    }

    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
