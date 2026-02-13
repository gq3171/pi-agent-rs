use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;

pub const APP_NAME: &str = "pi-agent";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CONFIG_DIR_NAME: &str = ".pi";

pub const SESSIONS_DIR_NAME: &str = "sessions";
pub const MODELS_FILE_NAME: &str = "models.json";
pub const AUTH_FILE_NAME: &str = "auth.json";
pub const SETTINGS_FILE_NAME: &str = "settings.json";
pub const SKILLS_DIR_NAME: &str = "skills";

/// Default base directory: ~/.pi/agent/
pub static DEFAULT_BASE_DIR: Lazy<PathBuf> = Lazy::new(|| {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR_NAME)
        .join("agent")
});

/// Resolve a config directory, using the provided override or the default.
pub fn resolve_base_dir(override_dir: Option<&Path>) -> PathBuf {
    override_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| DEFAULT_BASE_DIR.clone())
}

/// Get the sessions directory path.
pub fn sessions_dir(base: &Path) -> PathBuf {
    base.join(SESSIONS_DIR_NAME)
}

/// Get the models.json file path.
pub fn models_file(base: &Path) -> PathBuf {
    base.join(MODELS_FILE_NAME)
}

/// Get the auth.json file path.
pub fn auth_file(base: &Path) -> PathBuf {
    base.join(AUTH_FILE_NAME)
}

/// Get the settings.json file path.
pub fn settings_file(base: &Path) -> PathBuf {
    base.join(SETTINGS_FILE_NAME)
}

/// Get the skills directory path.
pub fn skills_dir(base: &Path) -> PathBuf {
    base.join(SKILLS_DIR_NAME)
}

/// Ensure a directory exists, creating it if needed.
pub fn ensure_dir(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}
