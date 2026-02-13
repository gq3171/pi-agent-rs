use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::config::paths;
use crate::error::CodingAgentError;
use crate::resources::package_manager::PackageManager;
use crate::resources::patterns::{apply_patterns, to_posix_string};
use crate::resources::prompts::{PromptTemplate, load_prompts_from_dir};
use crate::resources::skills::{Skill, load_skills_from_dir};
use crate::resources::source_identity::source_match_key_for_scope;
use crate::resources::themes::{Theme, load_themes_from_dir};
use crate::settings::types::PackageSource;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceDiagnosticType {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct ResourceDiagnostic {
    pub diagnostic_type: ResourceDiagnosticType,
    pub message: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct PathMetadata {
    pub source: String,
    pub scope: String,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceExtensionPaths {
    pub skill_paths: Vec<PathBuf>,
    pub prompt_paths: Vec<PathBuf>,
    pub theme_paths: Vec<PathBuf>,
}

pub trait ResourceLoader {
    fn get_skills(&self) -> (&[Skill], &[ResourceDiagnostic]);
    fn get_prompts(&self) -> (&[PromptTemplate], &[ResourceDiagnostic]);
    fn get_themes(&self) -> (&[Theme], &[ResourceDiagnostic]);
    fn get_agents_files(&self) -> &[ContextFile];
    fn get_system_prompt(&self) -> Option<&str>;
    fn get_append_system_prompt(&self) -> &[String];
    fn get_path_metadata(&self) -> &HashMap<String, PathMetadata>;
    fn extend_resources(&mut self, paths: ResourceExtensionPaths);
    fn reload(&mut self) -> Result<(), CodingAgentError>;
}

#[derive(Debug, Clone)]
pub struct DefaultResourceLoaderOptions {
    pub cwd: PathBuf,
    pub agent_dir: Option<PathBuf>,
    pub additional_skill_paths: Vec<PathBuf>,
    pub additional_prompt_template_paths: Vec<PathBuf>,
    pub additional_theme_paths: Vec<PathBuf>,
    pub no_skills: bool,
    pub no_prompt_templates: bool,
    pub no_themes: bool,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<String>,
    pub package_sources: Option<Vec<PackageSource>>,
}

impl Default for DefaultResourceLoaderOptions {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            agent_dir: None,
            additional_skill_paths: Vec::new(),
            additional_prompt_template_paths: Vec::new(),
            additional_theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            system_prompt: None,
            append_system_prompt: None,
            package_sources: None,
        }
    }
}

#[derive(Debug, Clone)]
struct PackageResourcePath {
    path: PathBuf,
    config: Option<PackageSource>,
}

pub struct DefaultResourceLoader {
    cwd: PathBuf,
    agent_dir: PathBuf,
    additional_skill_paths: Vec<PathBuf>,
    additional_prompt_template_paths: Vec<PathBuf>,
    additional_theme_paths: Vec<PathBuf>,
    no_skills: bool,
    no_prompt_templates: bool,
    no_themes: bool,
    system_prompt_source: Option<String>,
    append_system_prompt_source: Option<String>,
    package_sources: Option<Vec<PackageSource>>,
    skills: Vec<Skill>,
    prompts: Vec<PromptTemplate>,
    themes: Vec<Theme>,
    diagnostics: Vec<ResourceDiagnostic>,
    agents_files: Vec<ContextFile>,
    system_prompt: Option<String>,
    append_system_prompt: Vec<String>,
    path_metadata: HashMap<String, PathMetadata>,
}

impl DefaultResourceLoader {
    pub fn new(options: DefaultResourceLoaderOptions) -> Self {
        let resolved_agent_dir = options
            .agent_dir
            .unwrap_or_else(|| paths::DEFAULT_BASE_DIR.clone());
        Self {
            cwd: options.cwd,
            agent_dir: resolved_agent_dir,
            additional_skill_paths: options.additional_skill_paths,
            additional_prompt_template_paths: options.additional_prompt_template_paths,
            additional_theme_paths: options.additional_theme_paths,
            no_skills: options.no_skills,
            no_prompt_templates: options.no_prompt_templates,
            no_themes: options.no_themes,
            system_prompt_source: options.system_prompt,
            append_system_prompt_source: options.append_system_prompt,
            package_sources: options.package_sources,
            skills: Vec::new(),
            prompts: Vec::new(),
            themes: Vec::new(),
            diagnostics: Vec::new(),
            agents_files: Vec::new(),
            system_prompt: None,
            append_system_prompt: Vec::new(),
            path_metadata: HashMap::new(),
        }
    }

    fn resolve_prompt_input(input: &str, description: &str) -> Result<String, CodingAgentError> {
        let path = Path::new(input);
        if path.exists() {
            return std::fs::read_to_string(path).map_err(|e| {
                CodingAgentError::Config(format!(
                    "Failed to read {description} file {}: {e}",
                    path.display()
                ))
            });
        }
        Ok(input.to_string())
    }

    fn load_context_file_from_dir(dir: &Path) -> Option<ContextFile> {
        for file_name in ["AGENTS.md", "CLAUDE.md"] {
            let path = dir.join(file_name);
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        return Some(ContextFile {
                            path: path.display().to_string(),
                            content,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read context file {}: {e}", path.display());
                    }
                }
            }
        }
        None
    }

    fn load_project_context_files(&self) -> Vec<ContextFile> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        if let Some(global_context) = Self::load_context_file_from_dir(&self.agent_dir) {
            seen.insert(global_context.path.clone());
            result.push(global_context);
        }

        let mut stack = Vec::new();
        let mut current = self.cwd.clone();

        loop {
            if let Some(context_file) = Self::load_context_file_from_dir(&current) {
                if seen.insert(context_file.path.clone()) {
                    stack.push(context_file);
                }
            }

            let Some(parent) = current.parent() else {
                break;
            };
            if parent == current {
                break;
            }
            current = parent.to_path_buf();
        }

        stack.reverse();
        result.extend(stack);
        result
    }

    fn discover_package_resource_paths(&mut self) -> Vec<PackageResourcePath> {
        let mut discovered = Vec::new();
        let user_manager = PackageManager::new(&self.agent_dir);
        let project_scope_dir = self.cwd.join(".pi");
        let project_manager = PackageManager::new(&project_scope_dir);
        let user_records = user_manager.list().unwrap_or_default();
        let project_records = project_manager.list().unwrap_or_default();
        if user_records.is_empty() && project_records.is_empty() {
            return discovered;
        }

        // Project records win over user records for the same source.
        let mut records = Vec::<(crate::resources::package_manager::PackageRecord, PathBuf)>::new();
        let mut seen = HashSet::<String>::new();
        for (record, scope_dir) in project_records
            .into_iter()
            .map(|record| (record, project_scope_dir.as_path()))
            .chain(
                user_records
                    .into_iter()
                    .map(|record| (record, self.agent_dir.as_path())),
            )
        {
            let key = source_match_key_for_scope(scope_dir, &record.source);
            if seen.insert(key) {
                records.push((record, scope_dir.to_path_buf()));
            }
        }

        let package_filter_active = self.package_sources.is_some();
        let package_config = self.package_sources.clone().unwrap_or_default();

        for (record, record_scope_dir) in records {
            let record_key = source_match_key_for_scope(&record_scope_dir, &record.source);
            let matched_config = package_config
                .iter()
                .find(|candidate| {
                    source_match_key_for_scope(&self.agent_dir, candidate.source()) == record_key
                        || source_match_key_for_scope(&project_scope_dir, candidate.source())
                            == record_key
                })
                .cloned();
            if package_filter_active && matched_config.is_none() {
                continue;
            }

            let base = PathBuf::from(&record.installed_path);
            if !base.exists() {
                continue;
            }
            self.path_metadata.insert(
                base.display().to_string(),
                PathMetadata {
                    source: record.source.clone(),
                    scope: "package".to_string(),
                },
            );
            discovered.push(PackageResourcePath {
                path: base,
                config: matched_config,
            });
        }
        discovered
    }

    fn package_resource_patterns<'a>(
        pkg: &'a PackageResourcePath,
        resource_kind: &str,
    ) -> Option<&'a [String]> {
        pkg.config
            .as_ref()
            .and_then(|config| config.resource_patterns(resource_kind))
    }

    fn filter_package_items<T, FSource>(
        &self,
        package: &PackageResourcePath,
        resource_kind: &str,
        items: Vec<T>,
        source_fn: FSource,
    ) -> Vec<T>
    where
        FSource: Fn(&T) -> &str,
    {
        let Some(patterns) = Self::package_resource_patterns(package, resource_kind) else {
            return items;
        };
        if patterns.is_empty() {
            return Vec::new();
        }

        let all_paths = items
            .iter()
            .map(|item| PathBuf::from(source_fn(item)))
            .collect::<Vec<_>>();
        let enabled = apply_patterns(&all_paths, patterns, &package.path);

        items
            .into_iter()
            .filter(|item| enabled.contains(&to_posix_string(Path::new(source_fn(item)))))
            .collect()
    }

    fn merge_by_name<T, FName, FSource>(
        &mut self,
        target: &mut Vec<T>,
        items: Vec<T>,
        seen: &mut HashMap<String, String>,
        name_fn: FName,
        source_fn: FSource,
        resource_kind: &str,
    ) where
        FName: Fn(&T) -> &str,
        FSource: Fn(&T) -> &str,
    {
        for item in items {
            let name = name_fn(&item).to_string();
            let source = source_fn(&item).to_string();
            if let Some(existing) = seen.get(&name) {
                self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!(
                        "Duplicate {resource_kind} '{name}' ignored (winner: {existing}, loser: {source})"
                    ),
                    path: Some(source),
                });
                continue;
            }
            seen.insert(name, source.clone());
            self.path_metadata.insert(
                source.clone(),
                PathMetadata {
                    source: source.clone(),
                    scope: "resource".to_string(),
                },
            );
            target.push(item);
        }
    }

    fn load_all_skills(&mut self, package_paths: &[PackageResourcePath]) {
        if self.no_skills {
            self.skills.clear();
            return;
        }

        let mut all = Vec::new();
        let mut seen = HashMap::<String, String>::new();
        let mut paths = vec![
            paths::skills_dir(&self.agent_dir),
            self.cwd.join(".pi").join("skills"),
        ];
        paths.extend(self.additional_skill_paths.clone());

        for path in paths {
            match load_skills_from_dir(&path) {
                Ok(items) => self.merge_by_name(
                    &mut all,
                    items,
                    &mut seen,
                    |s| &s.name,
                    |s| &s.source,
                    "skill",
                ),
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load skills from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        for package in package_paths {
            let path = package.path.join("skills");
            match load_skills_from_dir(&path) {
                Ok(items) => {
                    let items = self.filter_package_items(package, "skills", items, |s| &s.source);
                    self.merge_by_name(
                        &mut all,
                        items,
                        &mut seen,
                        |s| &s.name,
                        |s| &s.source,
                        "skill",
                    );
                }
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load skills from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        all.sort_by(|a, b| a.name.cmp(&b.name));
        self.skills = all;
    }

    fn load_all_prompts(&mut self, package_paths: &[PackageResourcePath]) {
        if self.no_prompt_templates {
            self.prompts.clear();
            return;
        }

        let mut all = Vec::new();
        let mut seen = HashMap::<String, String>::new();
        let mut paths = vec![
            self.agent_dir.join("prompts"),
            self.cwd.join(".pi").join("prompts"),
        ];
        paths.extend(self.additional_prompt_template_paths.clone());

        for path in paths {
            match load_prompts_from_dir(&path) {
                Ok(items) => self.merge_by_name(
                    &mut all,
                    items,
                    &mut seen,
                    |p| &p.name,
                    |p| &p.source,
                    "prompt",
                ),
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load prompts from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        for package in package_paths {
            let path = package.path.join("prompts");
            match load_prompts_from_dir(&path) {
                Ok(items) => {
                    let items = self.filter_package_items(package, "prompts", items, |p| &p.source);
                    self.merge_by_name(
                        &mut all,
                        items,
                        &mut seen,
                        |p| &p.name,
                        |p| &p.source,
                        "prompt",
                    );
                }
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load prompts from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        all.sort_by(|a, b| a.name.cmp(&b.name));
        self.prompts = all;
    }

    fn load_all_themes(&mut self, package_paths: &[PackageResourcePath]) {
        if self.no_themes {
            self.themes.clear();
            return;
        }

        let mut all = Vec::new();
        let mut seen = HashMap::<String, String>::new();
        let mut paths = vec![
            self.agent_dir.join("themes"),
            self.cwd.join(".pi").join("themes"),
        ];
        paths.extend(self.additional_theme_paths.clone());

        for path in paths {
            match load_themes_from_dir(&path) {
                Ok(items) => self.merge_by_name(
                    &mut all,
                    items,
                    &mut seen,
                    |t| &t.name,
                    |t| &t.source,
                    "theme",
                ),
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load themes from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        for package in package_paths {
            let path = package.path.join("themes");
            match load_themes_from_dir(&path) {
                Ok(items) => {
                    let items = self.filter_package_items(package, "themes", items, |t| &t.source);
                    self.merge_by_name(
                        &mut all,
                        items,
                        &mut seen,
                        |t| &t.name,
                        |t| &t.source,
                        "theme",
                    );
                }
                Err(e) => self.diagnostics.push(ResourceDiagnostic {
                    diagnostic_type: ResourceDiagnosticType::Warning,
                    message: format!("Failed to load themes from {}: {e}", path.display()),
                    path: Some(path.display().to_string()),
                }),
            }
        }

        all.sort_by(|a, b| a.name.cmp(&b.name));
        self.themes = all;
    }
}

impl ResourceLoader for DefaultResourceLoader {
    fn get_skills(&self) -> (&[Skill], &[ResourceDiagnostic]) {
        (&self.skills, &self.diagnostics)
    }

    fn get_prompts(&self) -> (&[PromptTemplate], &[ResourceDiagnostic]) {
        (&self.prompts, &self.diagnostics)
    }

    fn get_themes(&self) -> (&[Theme], &[ResourceDiagnostic]) {
        (&self.themes, &self.diagnostics)
    }

    fn get_agents_files(&self) -> &[ContextFile] {
        &self.agents_files
    }

    fn get_system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    fn get_append_system_prompt(&self) -> &[String] {
        &self.append_system_prompt
    }

    fn get_path_metadata(&self) -> &HashMap<String, PathMetadata> {
        &self.path_metadata
    }

    fn extend_resources(&mut self, paths: ResourceExtensionPaths) {
        self.additional_skill_paths.extend(paths.skill_paths);
        self.additional_prompt_template_paths
            .extend(paths.prompt_paths);
        self.additional_theme_paths.extend(paths.theme_paths);
    }

    fn reload(&mut self) -> Result<(), CodingAgentError> {
        self.diagnostics.clear();
        self.path_metadata.clear();
        let package_paths = self.discover_package_resource_paths();
        self.load_all_skills(&package_paths);
        self.load_all_prompts(&package_paths);
        self.load_all_themes(&package_paths);
        self.agents_files = self.load_project_context_files();

        self.system_prompt = match &self.system_prompt_source {
            Some(source) => Some(Self::resolve_prompt_input(source, "system prompt")?),
            None => None,
        };

        self.append_system_prompt.clear();
        if let Some(source) = &self.append_system_prompt_source {
            self.append_system_prompt
                .push(Self::resolve_prompt_input(source, "append system prompt")?);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::types::{PackageSource, PackageSourceFilter};

    #[test]
    fn test_load_project_context_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();

        std::fs::write(root.join("AGENTS.md"), "root").unwrap();
        std::fs::write(root.join("a").join("CLAUDE.md"), "a").unwrap();

        let loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: nested,
            agent_dir: Some(root.to_path_buf()),
            ..Default::default()
        });

        let files = loader.load_project_context_files();
        assert!(!files.is_empty());
        assert!(files.iter().any(|f| f.content == "root"));
        assert!(files.iter().any(|f| f.content == "a"));
    }

    #[test]
    fn test_package_sources_empty_list_disables_package_resources() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let pkg_root = agent_dir.join("packages").join("pkg_a");
        std::fs::create_dir_all(pkg_root.join("skills")).unwrap();
        std::fs::write(pkg_root.join("skills").join("hello.md"), "hello").unwrap();
        std::fs::create_dir_all(agent_dir.join("packages")).unwrap();
        std::fs::write(
            agent_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "pkg_a",
                "installedPath": pkg_root.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: tmp.path().to_path_buf(),
            agent_dir: Some(agent_dir),
            package_sources: Some(vec![]),
            ..Default::default()
        });
        loader.reload().unwrap();

        assert!(loader.get_skills().0.is_empty());
    }

    #[test]
    fn test_package_sources_filter_enables_selected_package() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let pkg_root = agent_dir.join("packages").join("pkg_a");
        std::fs::create_dir_all(pkg_root.join("skills")).unwrap();
        std::fs::write(pkg_root.join("skills").join("hello.md"), "hello").unwrap();
        std::fs::create_dir_all(agent_dir.join("packages")).unwrap();
        std::fs::write(
            agent_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "pkg_a",
                "installedPath": pkg_root.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: tmp.path().to_path_buf(),
            agent_dir: Some(agent_dir),
            package_sources: Some(vec![PackageSource::Source("pkg_a".to_string())]),
            ..Default::default()
        });
        loader.reload().unwrap();

        assert_eq!(loader.get_skills().0.len(), 1);
        assert_eq!(loader.get_skills().0[0].name, "hello");
    }

    #[test]
    fn test_package_source_patterns_filter_skills() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let pkg_root = agent_dir.join("packages").join("pkg_a");
        std::fs::create_dir_all(pkg_root.join("skills")).unwrap();
        std::fs::write(pkg_root.join("skills").join("hello.md"), "hello").unwrap();
        std::fs::write(pkg_root.join("skills").join("bye.md"), "bye").unwrap();
        std::fs::create_dir_all(agent_dir.join("packages")).unwrap();
        std::fs::write(
            agent_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "pkg_a",
                "installedPath": pkg_root.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: tmp.path().to_path_buf(),
            agent_dir: Some(agent_dir),
            package_sources: Some(vec![PackageSource::Filtered(PackageSourceFilter {
                source: "pkg_a".to_string(),
                extensions: None,
                skills: Some(vec!["*.md".to_string(), "!bye.md".to_string()]),
                prompts: None,
                themes: None,
            })]),
            ..Default::default()
        });
        loader.reload().unwrap();

        assert_eq!(loader.get_skills().0.len(), 1);
        assert_eq!(loader.get_skills().0[0].name, "hello");
    }

    #[test]
    fn test_package_source_patterns_force_include_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let pkg_root = agent_dir.join("packages").join("pkg_a");
        std::fs::create_dir_all(pkg_root.join("skills")).unwrap();
        std::fs::write(pkg_root.join("skills").join("hello.md"), "hello").unwrap();
        std::fs::write(pkg_root.join("skills").join("bye.md"), "bye").unwrap();
        std::fs::create_dir_all(agent_dir.join("packages")).unwrap();
        std::fs::write(
            agent_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "pkg_a",
                "installedPath": pkg_root.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: tmp.path().to_path_buf(),
            agent_dir: Some(agent_dir),
            package_sources: Some(vec![PackageSource::Filtered(PackageSourceFilter {
                source: "pkg_a".to_string(),
                extensions: None,
                skills: Some(vec!["!*.md".to_string(), "+skills/bye.md".to_string()]),
                prompts: None,
                themes: None,
            })]),
            ..Default::default()
        });
        loader.reload().unwrap();

        assert_eq!(loader.get_skills().0.len(), 1);
        assert_eq!(loader.get_skills().0[0].name, "bye");
    }

    #[test]
    fn test_project_package_overrides_user_package_with_same_source() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        let agent_dir = tmp.path().join("agent");

        let user_pkg = agent_dir.join("packages").join("pkg_same_user");
        std::fs::create_dir_all(user_pkg.join("skills")).unwrap();
        std::fs::write(user_pkg.join("skills").join("user_only.md"), "user").unwrap();
        std::fs::create_dir_all(agent_dir.join("packages")).unwrap();
        std::fs::write(
            agent_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_same",
                "installedPath": user_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let project_pkg_root = cwd.join(".pi").join("packages");
        let project_pkg = project_pkg_root.join("pkg_same_project");
        std::fs::create_dir_all(project_pkg.join("skills")).unwrap();
        std::fs::write(
            project_pkg.join("skills").join("project_only.md"),
            "project",
        )
        .unwrap();
        std::fs::create_dir_all(&project_pkg_root).unwrap();
        std::fs::write(
            project_pkg_root.join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_same",
                "installedPath": project_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            cwd: cwd.clone(),
            agent_dir: Some(agent_dir),
            package_sources: Some(vec![PackageSource::Source("npm:pkg_same".to_string())]),
            ..Default::default()
        });
        loader.reload().unwrap();

        assert_eq!(loader.get_skills().0.len(), 1);
        assert_eq!(loader.get_skills().0[0].name, "project_only");
    }
}
