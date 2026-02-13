use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config::paths;
use crate::error::CodingAgentError;
use crate::resources::source_identity::{normalize_source_for_scope, source_match_key_for_scope};
use crate::settings::types::{PackageSource, Settings};

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
        project_base_dir: Option<&Path>,
    ) -> Result<&Settings, CodingAgentError> {
        self.load()?;
        let global_packages = self.settings.packages.clone();
        if let Some(project) = project_settings {
            let base = serde_json::to_value(&self.settings)?;
            let overlay = serde_json::to_value(project)?;
            let merged = deep_merge(base, overlay);
            self.settings = serde_json::from_value(merged)?;
            let project_base = project_base_dir.unwrap_or(self.base_dir.as_path());
            self.settings.packages = merge_package_sources(
                global_packages,
                project.packages.clone(),
                &self.base_dir,
                project_base,
            );
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

        // Create with restrictive permissions from the start (no permission window)
        {
            use std::io::Write;

            #[cfg(unix)]
            let mut file = {
                use std::os::unix::fs::OpenOptionsExt;
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&tmp_path)
                    .map_err(|e| {
                        CodingAgentError::Io(std::io::Error::new(
                            e.kind(),
                            format!("Failed to create temp settings file: {e}"),
                        ))
                    })?
            };
            #[cfg(not(unix))]
            let mut file = {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&tmp_path)
                    .map_err(|e| {
                        CodingAgentError::Io(std::io::Error::new(
                            e.kind(),
                            format!("Failed to create temp settings file: {e}"),
                        ))
                    })?
            };

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

fn merge_package_sources(
    global_packages: Option<Vec<PackageSource>>,
    project_packages: Option<Vec<PackageSource>>,
    global_base: &Path,
    project_base: &Path,
) -> Option<Vec<PackageSource>> {
    match (global_packages, project_packages) {
        (None, None) => None,
        (Some(global), None) => Some(
            global
                .into_iter()
                .map(|package| normalize_package_source(package, global_base))
                .collect(),
        ),
        (None, Some(project)) => Some(
            project
                .into_iter()
                .map(|package| normalize_package_source(package, project_base))
                .collect(),
        ),
        (Some(global), Some(project)) => {
            let mut merged = Vec::<PackageSource>::new();
            let mut seen = std::collections::HashSet::<String>::new();

            for package in project {
                let package = normalize_package_source(package, project_base);
                let key = source_match_key_for_scope(project_base, package.source());
                if seen.insert(key) {
                    merged.push(package);
                }
            }
            for package in global {
                let package = normalize_package_source(package, global_base);
                let key = source_match_key_for_scope(global_base, package.source());
                if seen.insert(key) {
                    merged.push(package);
                }
            }
            Some(merged)
        }
    }
}

fn normalize_package_source(package: PackageSource, scope_base: &Path) -> PackageSource {
    match package {
        PackageSource::Source(source) => {
            PackageSource::Source(normalize_source_for_scope(scope_base, &source))
        }
        PackageSource::Filtered(mut filter) => {
            filter.source = normalize_source_for_scope(scope_base, &filter.source);
            PackageSource::Filtered(filter)
        }
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
    use crate::settings::types::{PackageSource, PackageSourceFilter};
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

    #[test]
    fn test_merge_package_sources_project_overrides_global() {
        let global = Some(vec![
            PackageSource::Source("npm:a".to_string()),
            PackageSource::Source("npm:b".to_string()),
        ]);
        let project = Some(vec![
            PackageSource::Filtered(PackageSourceFilter {
                source: "npm:b".to_string(),
                extensions: Some(vec![]),
                skills: None,
                prompts: None,
                themes: None,
            }),
            PackageSource::Source("npm:c".to_string()),
        ]);

        let merged =
            merge_package_sources(global, project, Path::new("/user"), Path::new("/project"))
                .unwrap();
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].source(), "npm:b");
        assert_eq!(merged[1].source(), "npm:c");
        assert_eq!(merged[2].source(), "npm:a");
    }

    #[test]
    fn test_merge_package_sources_uses_identity_for_local_paths() {
        let global = Some(vec![PackageSource::Source(
            "../../work/proj/pkg".to_string(),
        )]);
        let project = Some(vec![PackageSource::Source("../pkg".to_string())]);

        let merged = merge_package_sources(
            global,
            project,
            Path::new("/home/user/.pi/agent"),
            Path::new("/home/user/work/proj/.pi"),
        )
        .unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(
            crate::resources::source_identity::source_match_key_for_scope(
                Path::new("/home/user/work/proj/.pi"),
                merged[0].source()
            ),
            crate::resources::source_identity::source_match_key_for_scope(
                Path::new("/home/user/work/proj/.pi"),
                "../pkg"
            )
        );
    }
}
