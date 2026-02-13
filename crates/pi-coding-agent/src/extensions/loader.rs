use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::paths;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::{Extension, ExtensionContext, ExtensionFactory, ToolDefinition};

/// An extension load error.
#[derive(Debug, Clone)]
pub struct ExtensionLoadError {
    pub source: String,
    pub error: String,
}

/// Result of loading extensions.
pub struct LoadExtensionsResult {
    pub runner: ExtensionRunner,
    pub errors: Vec<ExtensionLoadError>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtensionManifest {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    tools: Vec<ToolDefinition>,
    #[serde(default)]
    config: Value,
}

struct ExternalCommandExtension {
    manifest: ExtensionManifest,
    context: RwLock<Option<ExtensionContext>>,
}

#[async_trait::async_trait]
impl Extension for ExternalCommandExtension {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    async fn init(
        &mut self,
        context: ExtensionContext,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Ok(mut slot) = self.context.write() {
            *slot = Some(context);
        }
        Ok(())
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        self.manifest.tools.clone()
    }

    async fn handle_tool_call(
        &self,
        tool_name: &str,
        params: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let context = self.context.read().ok().and_then(|s| s.clone());
        let input = json!({
            "type": "tool_call",
            "toolName": tool_name,
            "params": params,
            "context": context,
            "config": self.manifest.config,
        });

        let output = tokio::process::Command::new(&self.manifest.command)
            .args(&self.manifest.args)
            .arg(input.to_string())
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(format!(
                "Extension command '{}' failed: {}",
                self.manifest.command, stderr
            )
            .into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            return Ok(json!({ "content": "" }));
        }

        match serde_json::from_str::<Value>(&stdout) {
            Ok(value) => Ok(value),
            Err(_) => Ok(json!({ "content": stdout })),
        }
    }
}

fn discover_manifest_files(
    configured_paths: &[PathBuf],
    cwd: &Path,
    agent_dir: &Path,
    include_default_roots: bool,
) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut roots = Vec::new();
    if include_default_roots {
        roots.push(agent_dir.join("extensions"));
        roots.push(cwd.join(".pi").join("extensions"));
    }
    roots.extend_from_slice(configured_paths);

    for root in roots {
        if !root.exists() {
            continue;
        }

        if root.is_file()
            && root
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.ends_with(".extension.json") || n == "extension.json")
        {
            let key = root.display().to_string();
            if seen.insert(key) {
                files.push(root);
            }
            continue;
        }

        let mut candidates = Vec::new();
        if root.is_dir() {
            let direct_manifest = root.join("extension.json");
            if direct_manifest.exists() {
                candidates.push(direct_manifest);
            }

            let entries = match std::fs::read_dir(&root) {
                Ok(v) => v,
                Err(_) => continue,
            };
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() {
                    let nested = path.join("extension.json");
                    if nested.exists() {
                        candidates.push(nested);
                    }
                } else if path.is_file()
                    && path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|n| n.ends_with(".extension.json") || n == "extension.json")
                {
                    candidates.push(path);
                }
            }
        }

        for manifest_path in candidates {
            let key = manifest_path.display().to_string();
            if seen.insert(key) {
                files.push(manifest_path);
            }
        }
    }

    files
}

fn load_manifest(path: &Path) -> Result<ExtensionManifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str::<ExtensionManifest>(&content)
        .map_err(|e| format!("Invalid manifest {}: {e}", path.display()))
}

/// Load extensions from in-memory factories and initialize them.
pub async fn load_extensions_from_factories(
    factories: &[ExtensionFactory],
    context: ExtensionContext,
) -> LoadExtensionsResult {
    let mut runner = ExtensionRunner::new(context);
    let mut errors = Vec::new();

    for (index, factory) in factories.iter().enumerate() {
        let extension = factory();
        if let Err(error) = runner.add_extension(extension).await {
            errors.push(ExtensionLoadError {
                source: format!("factory:{index}"),
                error,
            });
        }
    }

    LoadExtensionsResult { runner, errors }
}

/// Discover and load external-command extensions from standard paths and configured paths.
pub async fn discover_and_load_extensions(
    configured_paths: &[PathBuf],
    cwd: &Path,
    context: ExtensionContext,
) -> LoadExtensionsResult {
    let agent_dir = paths::DEFAULT_BASE_DIR.clone();
    let manifests = discover_manifest_files(configured_paths, cwd, &agent_dir, true);
    load_extensions_from_manifests(manifests, context).await
}

/// Load external-command extensions from explicit paths only.
pub async fn load_extensions_from_paths(
    configured_paths: &[PathBuf],
    cwd: &Path,
    context: ExtensionContext,
) -> LoadExtensionsResult {
    let manifests = discover_manifest_files(configured_paths, cwd, cwd, false);
    load_extensions_from_manifests(manifests, context).await
}

async fn load_extensions_from_manifests(
    manifests: Vec<PathBuf>,
    context: ExtensionContext,
) -> LoadExtensionsResult {
    let mut runner = ExtensionRunner::new(context);
    let mut errors = Vec::new();

    for manifest_path in manifests {
        let manifest = match load_manifest(&manifest_path) {
            Ok(m) => m,
            Err(error) => {
                errors.push(ExtensionLoadError {
                    source: manifest_path.display().to_string(),
                    error,
                });
                continue;
            }
        };

        let extension: Box<dyn Extension + Send + Sync> = Box::new(ExternalCommandExtension {
            manifest,
            context: RwLock::new(None),
        });

        if let Err(error) = runner.add_extension(extension).await {
            errors.push(ExtensionLoadError {
                source: manifest_path.display().to_string(),
                error,
            });
        }
    }

    LoadExtensionsResult { runner, errors }
}
