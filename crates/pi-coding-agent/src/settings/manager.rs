use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config::paths;
use crate::error::CodingAgentError;
use crate::settings::types::Settings;

/// Manages loading, merging, and saving of settings.
pub struct SettingsManager {
    base_dir: PathBuf,
    settings: Settings,
}

impl SettingsManager {
    /// Create a new SettingsManager with given base directory.
    pub fn new(base_dir: &Path) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            settings: Settings::default(),
        }
    }

    /// Load settings from the settings.json file.
    /// If the file doesn't exist, returns default settings.
    pub fn load(&mut self) -> Result<&Settings, CodingAgentError> {
        let path = paths::settings_file(&self.base_dir);
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let loaded: Settings = serde_json::from_str(&content)?;
            self.settings = loaded;
        }
        Ok(&self.settings)
    }

    /// Load settings from a file, then merge with project-level settings.
    pub fn load_and_merge(
        &mut self,
        project_settings: Option<&Settings>,
    ) -> Result<&Settings, CodingAgentError> {
        self.load()?;
        if let Some(project) = project_settings {
            let base = serde_json::to_value(&self.settings)?;
            let overlay = serde_json::to_value(project)?;
            let merged = deep_merge(base, overlay);
            self.settings = serde_json::from_value(merged)?;
        }
        Ok(&self.settings)
    }

    /// Save current settings to settings.json atomically (temp file + rename).
    /// On Unix, restricts permissions to owner-only (0600) since settings may contain API keys.
    pub fn save(&self) -> Result<(), CodingAgentError> {
        let path = paths::settings_file(&self.base_dir);
        paths::ensure_dir(&self.base_dir)?;
        let content = serde_json::to_string_pretty(&self.settings)?;

        // Write to a unique temp file then rename for crash safety
        let unique = uuid::Uuid::new_v4();
        let tmp_path = path.with_file_name(format!(".settings.{unique}.tmp"));

        // Create with restrictive permissions from the start
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
                CodingAgentError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to create temp settings file: {e}"),
                ))
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Err(e) = file
                    .set_permissions(std::fs::Permissions::from_mode(0o600))
                {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(CodingAgentError::Io(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        format!("Failed to restrict permissions on temp settings file: {e}"),
                    )));
                }
            }

            file.write_all(content.as_bytes()).map_err(|e| {
                let _ = std::fs::remove_file(&tmp_path);
                CodingAgentError::Io(e)
            })?;
        }

        std::fs::rename(&tmp_path, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            CodingAgentError::Io(e)
        })?;
        Ok(())
    }

    /// Get a reference to the current settings.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Get a mutable reference to the current settings.
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }

    /// Update settings by merging in a partial settings object.
    pub fn update(&mut self, partial: &Settings) -> Result<(), CodingAgentError> {
        let base = serde_json::to_value(&self.settings)?;
        let overlay = serde_json::to_value(partial)?;
        let merged = deep_merge(base, overlay);
        self.settings = serde_json::from_value(merged)?;
        Ok(())
    }
}

/// Deep merge two JSON values. `overlay` values take precedence.
/// Objects are recursively merged; other types are replaced.
pub fn deep_merge(base: Value, overlay: Value) -> Value {
    match (base, overlay) {
        (Value::Object(mut base_map), Value::Object(overlay_map)) => {
            for (key, overlay_val) in overlay_map {
                let merged = if let Some(base_val) = base_map.remove(&key) {
                    deep_merge(base_val, overlay_val)
                } else {
                    overlay_val
                };
                base_map.insert(key, merged);
            }
            Value::Object(base_map)
        }
        (_, overlay) => overlay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deep_merge_simple() {
        let base = json!({"a": 1, "b": 2});
        let overlay = json!({"b": 3, "c": 4});
        let result = deep_merge(base, overlay);
        assert_eq!(result, json!({"a": 1, "b": 3, "c": 4}));
    }

    #[test]
    fn test_deep_merge_nested() {
        let base = json!({"a": {"x": 1, "y": 2}, "b": 10});
        let overlay = json!({"a": {"y": 3, "z": 4}});
        let result = deep_merge(base, overlay);
        assert_eq!(result, json!({"a": {"x": 1, "y": 3, "z": 4}, "b": 10}));
    }

    #[test]
    fn test_deep_merge_replace_non_object() {
        let base = json!({"a": [1, 2]});
        let overlay = json!({"a": [3, 4, 5]});
        let result = deep_merge(base, overlay);
        assert_eq!(result, json!({"a": [3, 4, 5]}));
    }

    #[test]
    fn test_settings_round_trip() {
        let settings = Settings {
            model: Some("claude-sonnet-4".to_string()),
            temperature: Some(0.7),
            ..Default::default()
        };
        let json = serde_json::to_string_pretty(&settings).unwrap();
        let loaded: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.model, Some("claude-sonnet-4".to_string()));
        assert_eq!(loaded.temperature, Some(0.7));
    }

    #[test]
    fn test_settings_manager_load_save() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = SettingsManager::new(tmp.path());

        // Load from non-existent file returns defaults
        let settings = mgr.load().unwrap();
        assert!(settings.model.is_none());

        // Set and save
        mgr.settings_mut().model = Some("test-model".to_string());
        mgr.save().unwrap();

        // Re-load
        let mut mgr2 = SettingsManager::new(tmp.path());
        let settings = mgr2.load().unwrap();
        assert_eq!(settings.model, Some("test-model".to_string()));
    }
}
