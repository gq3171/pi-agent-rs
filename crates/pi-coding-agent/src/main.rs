use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::{ContentBlock, Message, Model};
use pi_coding_agent::agent_session::sdk::{
    CreateSessionOptions, CreateSessionWithExtensionsOptions, create_agent_session_with_extensions,
};
use pi_coding_agent::agent_session::session::PromptOptions;
use pi_coding_agent::auth::credentials::AuthCredential;
use pi_coding_agent::cli::args::{Args, Mode, is_valid_thinking_level, parse_args, print_help};
use pi_coding_agent::config::paths::{self, APP_NAME, CONFIG_DIR_NAME};
use pi_coding_agent::export_html::{ExportHtmlOptions, export_session_to_html};
use pi_coding_agent::model::registry::ModelRegistry;
use pi_coding_agent::model::resolver::{parse_model_pattern, resolve_cli_model};
use pi_coding_agent::modes::{
    InteractiveMode, InteractiveModeOptions, PrintModeOptions, PrintOutputMode, ScopedModelConfig,
    run_print_mode, run_rpc_mode,
};
use pi_coding_agent::resources::package_manager::PackageManager;
use pi_coding_agent::resources::patterns::{apply_patterns, to_posix_string};
use pi_coding_agent::resources::source_identity::{
    normalize_source_for_scope, source_match_key_for_input, source_match_key_for_scope,
};
use pi_coding_agent::session::manager::SessionManager;
use pi_coding_agent::settings::manager::SettingsManager;
use pi_coding_agent::settings::types::{PackageSource, PackageSourceFilter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageCommand {
    Install,
    Remove,
    Update,
    List,
    Config,
}

#[derive(Debug, Clone)]
struct PackageCommandOptions {
    command: PackageCommand,
    source: Option<String>,
    local: bool,
    help: bool,
    invalid_option: Option<String>,
}

#[derive(Debug, Clone)]
struct ScopedModelChoice {
    model: Model,
    thinking_level: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageScope {
    User,
    Project,
}

impl PackageScope {
    fn from_local(local: bool) -> Self {
        if local { Self::Project } else { Self::User }
    }

    fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
        }
    }
}

fn parse_package_command(args: &[String]) -> Option<PackageCommandOptions> {
    let command_raw = args.first()?;

    let command = match command_raw.as_str() {
        "install" => PackageCommand::Install,
        "remove" => PackageCommand::Remove,
        "update" => PackageCommand::Update,
        "list" => PackageCommand::List,
        "config" => PackageCommand::Config,
        _ => return None,
    };

    let mut local = false;
    let mut help = false;
    let mut invalid_option = None;
    let mut source = None;

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => help = true,
            "-l" | "--local" => {
                if matches!(command, PackageCommand::Install | PackageCommand::Remove) {
                    local = true;
                } else if invalid_option.is_none() {
                    invalid_option = Some(arg.clone());
                }
            }
            v if v.starts_with('-') => {
                if invalid_option.is_none() {
                    invalid_option = Some(v.to_string());
                }
            }
            _ if source.is_none() => source = Some(arg.clone()),
            _ => {}
        }
    }

    Some(PackageCommandOptions {
        command,
        source,
        local,
        help,
        invalid_option,
    })
}

fn package_usage(command: PackageCommand) -> &'static str {
    match command {
        PackageCommand::Install => "install <source> [-l]",
        PackageCommand::Remove => "remove <source> [-l]",
        PackageCommand::Update => "update [source]",
        PackageCommand::List => "list",
        PackageCommand::Config => "config",
    }
}

fn print_package_help(command: PackageCommand) {
    println!("Usage:\n  {} {}", APP_NAME, package_usage(command));
}

fn scope_base_dir(cwd: &Path, base_dir: &Path, scope: PackageScope) -> PathBuf {
    match scope {
        PackageScope::User => base_dir.to_path_buf(),
        PackageScope::Project => cwd.join(CONFIG_DIR_NAME),
    }
}

fn load_settings_manager_at(base_dir: &Path) -> Result<SettingsManager, String> {
    let mut manager = SettingsManager::new(base_dir);
    manager.load().map_err(|e| e.to_string())?;
    Ok(manager)
}

fn load_project_settings(cwd: &Path) -> Option<pi_coding_agent::settings::types::Settings> {
    let project_dir = cwd.join(CONFIG_DIR_NAME);
    let mut manager = SettingsManager::new(&project_dir);
    match manager.load() {
        Ok(_) => Some(manager.settings().clone()),
        Err(_) => None,
    }
}

fn upsert_package_source(
    cwd: &Path,
    base_dir: &Path,
    source: &str,
    scope: PackageScope,
) -> Result<(), String> {
    let scope_dir = scope_base_dir(cwd, base_dir, scope);
    let mut settings = load_settings_manager_at(&scope_dir)?;
    let packages = settings
        .settings_mut()
        .packages
        .get_or_insert_with(Vec::new);
    let input_key = source_match_key_for_input(cwd, source);
    if !packages
        .iter()
        .any(|item| source_match_key_for_scope(&scope_dir, item.source()) == input_key)
    {
        packages.push(PackageSource::Source(source.to_string()));
    }
    settings.save().map_err(|e| e.to_string())
}

fn remove_package_source(
    cwd: &Path,
    base_dir: &Path,
    source: &str,
    scope: PackageScope,
) -> Result<bool, String> {
    let scope_dir = scope_base_dir(cwd, base_dir, scope);
    let mut settings = load_settings_manager_at(&scope_dir)?;
    let mut removed = false;
    let input_key = source_match_key_for_input(cwd, source);
    if let Some(packages) = settings.settings_mut().packages.as_mut() {
        let before = packages.len();
        packages.retain(|pkg| source_match_key_for_scope(&scope_dir, pkg.source()) != input_key);
        removed = packages.len() != before;
    }
    settings.save().map_err(|e| e.to_string())?;
    Ok(removed)
}

fn package_enabled(
    package_sources: &[PackageSource],
    scope_dir: &Path,
    cwd: &Path,
    source: &str,
) -> bool {
    let input_key = source_match_key_for_input(cwd, source);
    package_sources
        .iter()
        .any(|pkg| source_match_key_for_scope(scope_dir, pkg.source()) == input_key)
}

#[derive(Debug, Clone, Copy)]
enum PackageResourceKind {
    Extensions,
    Skills,
    Prompts,
    Themes,
}

impl PackageResourceKind {
    fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "e" | "ext" | "extension" | "extensions" => Some(Self::Extensions),
            "s" | "skill" | "skills" => Some(Self::Skills),
            "p" | "prompt" | "prompts" => Some(Self::Prompts),
            "t" | "theme" | "themes" => Some(Self::Themes),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Extensions => "extensions",
            Self::Skills => "skills",
            Self::Prompts => "prompts",
            Self::Themes => "themes",
        }
    }

    fn short(self) -> &'static str {
        match self {
            Self::Extensions => "e",
            Self::Skills => "s",
            Self::Prompts => "p",
            Self::Themes => "t",
        }
    }
}

fn package_resource_enabled(config: &PackageSource, kind: PackageResourceKind) -> bool {
    match kind {
        PackageResourceKind::Extensions => config.extensions_enabled(),
        PackageResourceKind::Skills => config.skills_enabled(),
        PackageResourceKind::Prompts => config.prompts_enabled(),
        PackageResourceKind::Themes => config.themes_enabled(),
    }
}

fn source_resource_enabled(
    package_sources: &[PackageSource],
    scope_dir: &Path,
    cwd: &Path,
    source: &str,
    kind: PackageResourceKind,
) -> bool {
    let input_key = source_match_key_for_input(cwd, source);
    package_sources
        .iter()
        .find(|pkg| source_match_key_for_scope(scope_dir, pkg.source()) == input_key)
        .is_some_and(|pkg| package_resource_enabled(pkg, kind))
}

fn package_resource_patterns(
    config: &PackageSource,
    kind: PackageResourceKind,
) -> Option<&[String]> {
    config.resource_patterns(kind.label())
}

fn empty_package_filter(source: &str) -> PackageSourceFilter {
    PackageSourceFilter {
        source: source.to_string(),
        extensions: None,
        skills: None,
        prompts: None,
        themes: None,
    }
}

fn normalize_package_source_for_scope(package: PackageSource, scope_base: &Path) -> PackageSource {
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

fn package_filter_all_enabled(filter: &PackageSourceFilter) -> bool {
    filter.extensions.is_none()
        && filter.skills.is_none()
        && filter.prompts.is_none()
        && filter.themes.is_none()
}

fn filter_patterns_mut(
    filter: &mut PackageSourceFilter,
    kind: PackageResourceKind,
) -> &mut Option<Vec<String>> {
    match kind {
        PackageResourceKind::Extensions => &mut filter.extensions,
        PackageResourceKind::Skills => &mut filter.skills,
        PackageResourceKind::Prompts => &mut filter.prompts,
        PackageResourceKind::Themes => &mut filter.themes,
    }
}

fn normalize_pattern_entry(value: &str) -> String {
    let stripped = value
        .strip_prefix('!')
        .or_else(|| value.strip_prefix('+'))
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    if stripped.starts_with("./") || stripped.starts_with(".\\") {
        stripped[2..].replace('\\', "/")
    } else {
        stripped.replace('\\', "/")
    }
}

fn is_extension_manifest_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|v| v.to_str())
        .is_some_and(|name| name == "extension.json" || name.ends_with(".extension.json"))
}

fn collect_package_extension_manifest_paths(package_root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::<PathBuf>::new();
    let mut seen = std::collections::HashSet::<String>::new();

    let mut push_unique = |path: PathBuf| {
        let key = to_posix_string(&path);
        if seen.insert(key) {
            result.push(path);
        }
    };

    if !package_root.exists() {
        return result;
    }

    if package_root.is_file() && is_extension_manifest_path(package_root) {
        push_unique(package_root.to_path_buf());
        return result;
    }

    if !package_root.is_dir() {
        return result;
    }

    let direct = package_root.join("extension.json");
    if direct.exists() {
        push_unique(direct);
    }

    let entries = match std::fs::read_dir(package_root) {
        Ok(entries) => entries,
        Err(_) => return result,
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            let nested = path.join("extension.json");
            if nested.exists() {
                push_unique(nested);
            }
            continue;
        }

        if path.is_file() && is_extension_manifest_path(&path) {
            push_unique(path);
        }
    }

    result.sort_by_key(|a| to_posix_string(a));
    result
}

fn collect_directory_files_with_extension(dir: &Path, extension: &str) -> Vec<PathBuf> {
    if !dir.exists() || !dir.is_dir() {
        return Vec::new();
    }

    let mut files = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|a| to_posix_string(a));
    files
}

fn collect_package_resource_paths(package_root: &Path, kind: PackageResourceKind) -> Vec<PathBuf> {
    match kind {
        PackageResourceKind::Extensions => collect_package_extension_manifest_paths(package_root),
        PackageResourceKind::Skills => {
            collect_directory_files_with_extension(&package_root.join("skills"), "md")
        }
        PackageResourceKind::Prompts => {
            collect_directory_files_with_extension(&package_root.join("prompts"), "md")
        }
        PackageResourceKind::Themes => {
            collect_directory_files_with_extension(&package_root.join("themes"), "json")
        }
    }
}

#[derive(Debug, Clone)]
struct ConfigResourceItem {
    relative_path: String,
    enabled: bool,
}

fn build_config_resource_items(
    package_root: &Path,
    package_config: Option<&PackageSource>,
    kind: PackageResourceKind,
) -> Vec<ConfigResourceItem> {
    let all_paths = collect_package_resource_paths(package_root, kind);
    if all_paths.is_empty() {
        return Vec::new();
    }

    let enabled_set = match package_config.and_then(|cfg| package_resource_patterns(cfg, kind)) {
        None => all_paths
            .iter()
            .map(|path| to_posix_string(path))
            .collect::<std::collections::HashSet<_>>(),
        Some([]) => std::collections::HashSet::new(),
        Some(patterns) => apply_patterns(&all_paths, patterns, package_root),
    };

    all_paths
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(package_root)
                .ok()
                .map(to_posix_string)
                .unwrap_or_else(|| to_posix_string(&path));
            let path_key = to_posix_string(&path);
            ConfigResourceItem {
                relative_path: relative,
                enabled: enabled_set.contains(&path_key),
            }
        })
        .collect()
}

fn toggle_package_resource(
    package_sources: &mut [PackageSource],
    scope_dir: &Path,
    cwd: &Path,
    source: &str,
    kind: PackageResourceKind,
) -> Result<bool, String> {
    let input_key = source_match_key_for_input(cwd, source);
    let Some(index) = package_sources
        .iter()
        .position(|pkg| source_match_key_for_scope(scope_dir, pkg.source()) == input_key)
    else {
        return Err(format!("Package '{source}' 当前未启用。"));
    };

    let mut filter = match &package_sources[index] {
        PackageSource::Source(source) => empty_package_filter(source),
        PackageSource::Filtered(filter) => filter.clone(),
    };

    let current_enabled = package_resource_enabled(&package_sources[index], kind);
    let new_enabled = !current_enabled;
    let next_value = if new_enabled { None } else { Some(Vec::new()) };

    *filter_patterns_mut(&mut filter, kind) = next_value;

    if package_filter_all_enabled(&filter) {
        package_sources[index] = PackageSource::Source(filter.source);
    } else {
        package_sources[index] = PackageSource::Filtered(filter);
    }

    Ok(new_enabled)
}

fn toggle_package_resource_item(
    package_sources: &mut [PackageSource],
    scope_dir: &Path,
    cwd: &Path,
    source: &str,
    kind: PackageResourceKind,
    relative_path: &str,
    currently_enabled: bool,
) -> Result<bool, String> {
    let input_key = source_match_key_for_input(cwd, source);
    let Some(index) = package_sources
        .iter()
        .position(|pkg| source_match_key_for_scope(scope_dir, pkg.source()) == input_key)
    else {
        return Err(format!("Package '{source}' 当前未启用。"));
    };

    let mut filter = match &package_sources[index] {
        PackageSource::Source(source) => empty_package_filter(source),
        PackageSource::Filtered(filter) => filter.clone(),
    };

    let normalized_path = normalize_pattern_entry(relative_path);
    let patterns_slot = filter_patterns_mut(&mut filter, kind);
    let mut patterns = patterns_slot.take().unwrap_or_default();
    patterns.retain(|entry| normalize_pattern_entry(entry) != normalized_path);
    patterns.push(format!(
        "{}{}",
        if currently_enabled { "-" } else { "+" },
        normalized_path
    ));
    *patterns_slot = Some(patterns);

    if package_filter_all_enabled(&filter) {
        package_sources[index] = PackageSource::Source(filter.source);
    } else {
        package_sources[index] = PackageSource::Filtered(filter);
    }

    Ok(!currently_enabled)
}

fn mark(enabled: bool) -> &'static str {
    if enabled { "x" } else { " " }
}

#[derive(Debug, Clone)]
struct ScopedPackageRecord {
    scope: PackageScope,
    source: String,
    installed_path: String,
}

fn run_config_command(cwd: &Path, base_dir: &Path) -> Result<i32, String> {
    let user_manager = PackageManager::new(base_dir);
    let project_manager =
        PackageManager::new(&scope_base_dir(cwd, base_dir, PackageScope::Project));
    let user_records = user_manager.list().map_err(|e| e.to_string())?;
    let project_records = project_manager.list().map_err(|e| e.to_string())?;

    let records = user_records
        .into_iter()
        .map(|record| ScopedPackageRecord {
            scope: PackageScope::User,
            source: record.source,
            installed_path: record.installed_path,
        })
        .chain(
            project_records
                .into_iter()
                .map(|record| ScopedPackageRecord {
                    scope: PackageScope::Project,
                    source: record.source,
                    installed_path: record.installed_path,
                }),
        )
        .collect::<Vec<_>>();

    if records.is_empty() {
        println!("No installed packages.");
        return Ok(0);
    }

    let mut user_settings = load_settings_manager_at(base_dir)?;
    let project_settings_dir = scope_base_dir(cwd, base_dir, PackageScope::Project);
    let mut project_settings = load_settings_manager_at(&project_settings_dir)?;

    let mut user_package_sources = user_settings
        .settings()
        .packages
        .clone()
        .unwrap_or_else(|| {
            records
                .iter()
                .filter(|record| record.scope == PackageScope::User)
                .map(|record| PackageSource::Source(record.source.clone()))
                .collect()
        });
    let mut project_package_sources =
        project_settings
            .settings()
            .packages
            .clone()
            .unwrap_or_else(|| {
                records
                    .iter()
                    .filter(|record| record.scope == PackageScope::Project)
                    .map(|record| PackageSource::Source(record.source.clone()))
                    .collect()
            });

    let resource_kinds = [
        PackageResourceKind::Extensions,
        PackageResourceKind::Skills,
        PackageResourceKind::Prompts,
        PackageResourceKind::Themes,
    ];

    let print_config =
        |user_sources: &[PackageSource], project_sources: &[PackageSource], include_items: bool| {
            println!("Installed packages:");
            for (idx, record) in records.iter().enumerate() {
                let (package_sources, scope_dir) = match record.scope {
                    PackageScope::User => (user_sources, base_dir),
                    PackageScope::Project => (project_sources, project_settings_dir.as_path()),
                };
                let enabled = package_enabled(package_sources, scope_dir, cwd, &record.source);
                let ext = if enabled {
                    mark(source_resource_enabled(
                        package_sources,
                        scope_dir,
                        cwd,
                        &record.source,
                        PackageResourceKind::Extensions,
                    ))
                } else {
                    " "
                };
                let skills = if enabled {
                    mark(source_resource_enabled(
                        package_sources,
                        scope_dir,
                        cwd,
                        &record.source,
                        PackageResourceKind::Skills,
                    ))
                } else {
                    " "
                };
                let prompts = if enabled {
                    mark(source_resource_enabled(
                        package_sources,
                        scope_dir,
                        cwd,
                        &record.source,
                        PackageResourceKind::Prompts,
                    ))
                } else {
                    " "
                };
                let themes = if enabled {
                    mark(source_resource_enabled(
                        package_sources,
                        scope_dir,
                        cwd,
                        &record.source,
                        PackageResourceKind::Themes,
                    ))
                } else {
                    " "
                };
                println!(
                    "{}. [{}] {} ({})  e:[{}] s:[{}] p:[{}] t:[{}]",
                    idx + 1,
                    mark(enabled),
                    record.source,
                    record.scope.label(),
                    ext,
                    skills,
                    prompts,
                    themes
                );

                if include_items && enabled {
                    let record_key = source_match_key_for_input(cwd, &record.source);
                    let config = package_sources.iter().find(|pkg| {
                        source_match_key_for_scope(scope_dir, pkg.source()) == record_key
                    });
                    let package_root = PathBuf::from(&record.installed_path);
                    for kind in resource_kinds {
                        let items = build_config_resource_items(&package_root, config, kind);
                        if items.is_empty() {
                            continue;
                        }
                        for (item_idx, item) in items.iter().enumerate() {
                            println!(
                                "    {}:{}:{} [{}] {}",
                                idx + 1,
                                kind.short(),
                                item_idx + 1,
                                mark(item.enabled),
                                item.relative_path
                            );
                        }
                    }
                }
            }
        };

    if !std::io::stdin().is_terminal() {
        print_config(&user_package_sources, &project_package_sources, false);
        return Ok(0);
    }

    println!(
        "Package config (编号切换包；编号:e|s|p|t 切类型；编号:e|s|p|t:资源编号 切具体资源；s 保存；q 退出):"
    );
    loop {
        print_config(&user_package_sources, &project_package_sources, true);
        print!("> ");
        std::io::stdout().flush().map_err(|e| e.to_string())?;

        let mut line = String::new();
        let read = std::io::stdin()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if read == 0 {
            return Ok(0);
        }

        let input = line.trim();
        if input.eq_ignore_ascii_case("q") {
            println!("Canceled.");
            return Ok(0);
        }
        if input.eq_ignore_ascii_case("s") {
            user_settings.settings_mut().packages = Some(user_package_sources.clone());
            user_settings.save().map_err(|e| e.to_string())?;
            project_settings.settings_mut().packages = Some(project_package_sources.clone());
            project_settings.save().map_err(|e| e.to_string())?;
            println!("Saved.");
            return Ok(0);
        }

        let parts = input.split(':').map(|part| part.trim()).collect::<Vec<_>>();
        let Ok(index) = parts.first().unwrap_or(&"").parse::<usize>() else {
            println!("请输入编号、编号:e|s|p|t、编号:e|s|p|t:资源编号、s 或 q");
            continue;
        };
        if index == 0 || index > records.len() {
            println!("编号超出范围");
            continue;
        }

        let selected_record = &records[index - 1];
        let source = &selected_record.source;
        let (package_sources, scope_dir) = match selected_record.scope {
            PackageScope::User => (&mut user_package_sources, base_dir),
            PackageScope::Project => (&mut project_package_sources, project_settings_dir.as_path()),
        };
        match parts.len() {
            1 => {
                if package_enabled(package_sources, scope_dir, cwd, source) {
                    let source_key = source_match_key_for_input(cwd, source);
                    package_sources.retain(|pkg| {
                        source_match_key_for_scope(scope_dir, pkg.source()) != source_key
                    });
                } else {
                    package_sources.push(PackageSource::Source(source.clone()));
                }
            }
            2 => {
                let Some(resource_kind) = PackageResourceKind::parse(parts[1]) else {
                    println!("未知资源类型 '{}', 使用 e/s/p/t", parts[1]);
                    continue;
                };
                if !package_enabled(package_sources, scope_dir, cwd, source) {
                    println!("请先启用该 package（先输入编号 {index}）");
                    continue;
                }
                match toggle_package_resource(
                    package_sources,
                    scope_dir,
                    cwd,
                    source,
                    resource_kind,
                ) {
                    Ok(enabled) => {
                        println!(
                            "{} {} {}",
                            source,
                            resource_kind.label(),
                            if enabled { "enabled" } else { "disabled" }
                        );
                    }
                    Err(message) => println!("{message}"),
                }
            }
            3 => {
                let Some(resource_kind) = PackageResourceKind::parse(parts[1]) else {
                    println!("未知资源类型 '{}', 使用 e/s/p/t", parts[1]);
                    continue;
                };
                let Ok(item_index) = parts[2].parse::<usize>() else {
                    println!("资源编号无效: {}", parts[2]);
                    continue;
                };
                if item_index == 0 {
                    println!("资源编号从 1 开始");
                    continue;
                }
                if !package_enabled(package_sources, scope_dir, cwd, source) {
                    println!("请先启用该 package（先输入编号 {index}）");
                    continue;
                }

                let package_root = PathBuf::from(&selected_record.installed_path);
                let source_key = source_match_key_for_input(cwd, source);
                let package_config = package_sources
                    .iter()
                    .find(|pkg| source_match_key_for_scope(scope_dir, pkg.source()) == source_key);
                let items =
                    build_config_resource_items(&package_root, package_config, resource_kind);
                if item_index > items.len() {
                    println!("资源编号超出范围");
                    continue;
                }

                let item = &items[item_index - 1];
                match toggle_package_resource_item(
                    package_sources,
                    scope_dir,
                    cwd,
                    source,
                    resource_kind,
                    &item.relative_path,
                    item.enabled,
                ) {
                    Ok(enabled) => {
                        println!(
                            "{} {} {} {}",
                            source,
                            resource_kind.label(),
                            item.relative_path,
                            if enabled { "enabled" } else { "disabled" }
                        );
                    }
                    Err(message) => println!("{message}"),
                }
            }
            _ => {
                println!("格式错误，使用 编号、编号:e|s|p|t、编号:e|s|p|t:资源编号");
            }
        }
        println!();
    }
}

fn handle_package_command(raw_args: &[String], cwd: &Path, base_dir: &Path) -> Option<i32> {
    let options = parse_package_command(raw_args)?;

    if options.help {
        print_package_help(options.command);
        return Some(0);
    }

    if let Some(invalid) = options.invalid_option {
        eprintln!(
            "Unknown option {} for command {}",
            invalid,
            package_usage(options.command)
        );
        return Some(1);
    }

    match options.command {
        PackageCommand::Install => {
            let Some(source) = options.source else {
                eprintln!("Missing install source");
                eprintln!(
                    "Usage: {} {}",
                    APP_NAME,
                    package_usage(PackageCommand::Install)
                );
                return Some(1);
            };

            let scope = PackageScope::from_local(options.local);
            let scope_dir = scope_base_dir(cwd, base_dir, scope);
            let manager = PackageManager::new(&scope_dir);
            match manager.install_source(cwd, &source) {
                Ok(record) => {
                    if let Err(error) = upsert_package_source(cwd, base_dir, &record.source, scope)
                    {
                        eprintln!("Warning: 安装成功但写入 settings 失败: {error}");
                    }
                    println!("Installed ({}): {}", scope.label(), record.source);
                    println!("Path: {}", record.installed_path);
                    Some(0)
                }
                Err(e) => {
                    eprintln!("Install failed: {e}");
                    Some(1)
                }
            }
        }
        PackageCommand::Remove => {
            let Some(source) = options.source else {
                eprintln!("Missing remove source");
                eprintln!(
                    "Usage: {} {}",
                    APP_NAME,
                    package_usage(PackageCommand::Remove)
                );
                return Some(1);
            };

            let scope = PackageScope::from_local(options.local);
            let scope_dir = scope_base_dir(cwd, base_dir, scope);
            let manager = PackageManager::new(&scope_dir);
            let input_key = source_match_key_for_input(cwd, &source);
            let records = match manager.list() {
                Ok(records) => records,
                Err(e) => {
                    eprintln!("Remove failed: {e}");
                    return Some(1);
                }
            };
            let matched_source = records.into_iter().find_map(|record| {
                (source_match_key_for_scope(&scope_dir, &record.source) == input_key)
                    .then_some(record.source)
            });
            let removed = match matched_source {
                Some(record_source) => manager.remove(&record_source),
                None => Ok(false),
            };
            match removed {
                Ok(true) => {
                    if let Err(error) = remove_package_source(cwd, base_dir, &source, scope) {
                        eprintln!("Warning: 删除成功但更新 settings 失败: {error}");
                    }
                    println!("Removed ({}): {source}", scope.label());
                    Some(0)
                }
                Ok(false) => {
                    eprintln!("No package record matched source: {source}");
                    Some(1)
                }
                Err(e) => {
                    eprintln!("Remove failed: {e}");
                    Some(1)
                }
            }
        }
        PackageCommand::Update => {
            let mut failed = false;
            let mut matched_any = false;
            let scopes = [PackageScope::User, PackageScope::Project];
            for scope in scopes {
                let scope_dir = scope_base_dir(cwd, base_dir, scope);
                let manager = PackageManager::new(&scope_dir);
                let records = match manager.list() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Update failed ({}): {e}", scope.label());
                        failed = true;
                        continue;
                    }
                };

                let sources: Vec<String> = if let Some(source) = &options.source {
                    let input_key = source_match_key_for_input(cwd, source);
                    records
                        .iter()
                        .filter(|record| {
                            source_match_key_for_scope(&scope_dir, &record.source) == input_key
                        })
                        .map(|record| record.source.clone())
                        .collect()
                } else {
                    records.iter().map(|r| r.source.clone()).collect()
                };
                if sources.is_empty() {
                    continue;
                }
                matched_any = true;

                for source in sources {
                    if let Err(e) = manager.update_source(cwd, &source) {
                        eprintln!("Update failed for {} ({}): {}", source, scope.label(), e);
                        failed = true;
                    } else {
                        println!("Updated ({}): {source}", scope.label());
                    }
                }
            }

            if options.source.is_some() && !matched_any {
                eprintln!(
                    "No installed package record matched source: {}",
                    options.source.as_deref().unwrap_or_default()
                );
                return Some(1);
            }

            Some(if failed { 1 } else { 0 })
        }
        PackageCommand::List => {
            let user_manager = PackageManager::new(base_dir);
            let project_manager =
                PackageManager::new(&scope_base_dir(cwd, base_dir, PackageScope::Project));

            let user_records = match user_manager.list() {
                Ok(records) => records,
                Err(e) => {
                    eprintln!("List failed (user): {e}");
                    return Some(1);
                }
            };
            let project_records = match project_manager.list() {
                Ok(records) => records,
                Err(e) => {
                    eprintln!("List failed (project): {e}");
                    return Some(1);
                }
            };

            if user_records.is_empty() && project_records.is_empty() {
                println!("No installed packages.");
                return Some(0);
            }

            if !user_records.is_empty() {
                println!("User packages:");
                for record in user_records {
                    println!("{}", record.source);
                    println!("  {}", record.installed_path);
                }
            }
            if !project_records.is_empty() {
                println!("Project packages:");
                for record in project_records {
                    println!("{}", record.source);
                    println!("  {}", record.installed_path);
                }
            }
            Some(0)
        }
        PackageCommand::Config => match run_config_command(cwd, base_dir) {
            Ok(code) => Some(code),
            Err(error) => {
                eprintln!("Config failed: {error}");
                Some(1)
            }
        },
    }
}

fn resolve_session_id(manager: &SessionManager, input: &str) -> String {
    if input.contains('/') || input.contains('\\') || input.ends_with(".jsonl") {
        return PathBuf::from(input)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| input.to_string());
    }

    if manager.exists(input) {
        return input.to_string();
    }

    if let Ok(ids) = manager.list_all() {
        if let Some(matched) = ids
            .into_iter()
            .filter(|id| id.starts_with(input))
            .min_by(|a, b| a.cmp(b))
        {
            return matched;
        }
    }

    input.to_string()
}

fn print_models(search: Option<&str>) {
    let registry = ModelRegistry::new();
    let query = search.unwrap_or("").trim().to_lowercase();
    for model in registry.all_models() {
        let key = format!("{}/{} {}", model.provider, model.id, model.name).to_lowercase();
        if query.is_empty() || key.contains(&query) {
            println!("{}/{} - {}", model.provider, model.id, model.name);
        }
    }
}

fn read_piped_stdin() -> Result<Option<String>, std::io::Error> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Ok(None);
    }

    let mut content = String::new();
    std::io::stdin().read_to_string(&mut content)?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

fn prepare_initial_message(file_args: &[String], messages: &mut Vec<String>) -> Option<String> {
    if file_args.is_empty() {
        return None;
    }

    let mut file_text = String::new();
    for file_arg in file_args {
        let path = PathBuf::from(file_arg);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                file_text.push_str(&format!("File: {}\n{}\n\n", path.display(), content));
            }
            Err(e) => {
                eprintln!("Warning: 无法读取 @{file_arg}: {e}");
            }
        }
    }

    if file_text.is_empty() {
        return None;
    }

    if !messages.is_empty() {
        let first = messages.remove(0);
        Some(format!("{file_text}{first}"))
    } else {
        Some(file_text)
    }
}

fn contains_glob_pattern(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

fn wildcard_match_case_insensitive(pattern: &str, target: &str) -> bool {
    let pattern_lc = pattern.to_lowercase();
    let target_lc = target.to_lowercase();
    glob::Pattern::new(&pattern_lc)
        .map(|glob_pattern| glob_pattern.matches(&target_lc))
        .unwrap_or(false)
}

fn resolve_scoped_models(registry: &ModelRegistry, patterns: &[String]) -> Vec<ScopedModelChoice> {
    let mut scoped_models = Vec::new();
    let mut seen = std::collections::HashSet::<String>::new();

    for pattern in patterns {
        if contains_glob_pattern(pattern) {
            let mut glob_pattern = pattern.as_str();
            let mut thinking_level = None;

            if let Some((prefix, suffix)) = pattern.rsplit_once(':')
                && is_valid_thinking_level(suffix)
            {
                glob_pattern = prefix;
                thinking_level = Some(suffix.to_string());
            }

            let mut matches = 0usize;
            for model in registry.all_models() {
                let full_id = format!("{}/{}", model.provider, model.id);
                if wildcard_match_case_insensitive(glob_pattern, &full_id)
                    || wildcard_match_case_insensitive(glob_pattern, &model.id)
                {
                    let key = format!("{}/{}", model.provider, model.id);
                    if seen.insert(key) {
                        scoped_models.push(ScopedModelChoice {
                            model: model.clone(),
                            thinking_level: thinking_level.clone(),
                        });
                    }
                    matches += 1;
                }
            }

            if matches == 0 {
                eprintln!("Warning: No models match pattern \"{pattern}\"");
            }
            continue;
        }

        let parsed = parse_model_pattern(pattern, registry.all_models(), true);
        if let Some(warning) = parsed.warning {
            eprintln!("Warning: {warning}");
        }
        let Some(model) = parsed.model else {
            eprintln!("Warning: No models match pattern \"{pattern}\"");
            continue;
        };

        let key = format!("{}/{}", model.provider, model.id);
        if seen.insert(key) {
            scoped_models.push(ScopedModelChoice {
                model,
                thinking_level: parsed.thinking_level,
            });
        }
    }

    scoped_models
}

fn resolve_package_extension_paths(cwd: &Path, base_dir: &Path) -> Vec<PathBuf> {
    let project_base = scope_base_dir(cwd, base_dir, PackageScope::Project);

    let user_manager = PackageManager::new(base_dir);
    let project_manager = PackageManager::new(&project_base);
    let user_records = user_manager.list().unwrap_or_default();
    let project_records = project_manager.list().unwrap_or_default();
    if user_records.is_empty() && project_records.is_empty() {
        return Vec::new();
    }

    let mut records = Vec::new();
    let mut seen_sources = std::collections::HashSet::<String>::new();
    for (record, scope_dir) in project_records
        .into_iter()
        .map(|record| (record, project_base.as_path()))
        .chain(user_records.into_iter().map(|record| (record, base_dir)))
    {
        let key = source_match_key_for_scope(scope_dir, &record.source);
        if seen_sources.insert(key) {
            records.push((record, scope_dir.to_path_buf()));
        }
    }

    let user_sources = load_settings_manager_at(base_dir)
        .ok()
        .and_then(|manager| manager.settings().packages.clone());
    let project_sources = load_settings_manager_at(&project_base)
        .ok()
        .and_then(|manager| manager.settings().packages.clone());
    let has_package_filter = user_sources.is_some() || project_sources.is_some();
    let mut package_sources = Vec::<PackageSource>::new();
    let mut seen = std::collections::HashSet::<String>::new();
    if let Some(project) = project_sources {
        for package in project {
            let package = normalize_package_source_for_scope(package, &project_base);
            let key = source_match_key_for_scope(&project_base, package.source());
            if seen.insert(key) {
                package_sources.push(package);
            }
        }
    }
    if let Some(user) = user_sources {
        for package in user {
            let package = normalize_package_source_for_scope(package, base_dir);
            let key = source_match_key_for_scope(base_dir, package.source());
            if seen.insert(key) {
                package_sources.push(package);
            }
        }
    }

    let mut result = Vec::new();
    let mut seen_paths = std::collections::HashSet::<String>::new();

    for (record, record_scope_dir) in records {
        let record_key = source_match_key_for_scope(&record_scope_dir, &record.source);
        let config = package_sources
            .iter()
            .find(|source| source_match_key_for_input(cwd, source.source()) == record_key);
        if has_package_filter && config.is_none() {
            continue;
        }

        let root = PathBuf::from(&record.installed_path);
        if !root.exists() {
            continue;
        }

        let mut selected_paths = Vec::<PathBuf>::new();
        if let Some(config) = config {
            if let Some(patterns) =
                package_resource_patterns(config, PackageResourceKind::Extensions)
            {
                if patterns.is_empty() {
                    continue;
                }
                let all_paths = collect_package_extension_manifest_paths(&root);
                let enabled = apply_patterns(&all_paths, patterns, &root);
                selected_paths.extend(
                    all_paths
                        .into_iter()
                        .filter(|path| enabled.contains(&to_posix_string(path))),
                );
            } else if config.extensions_enabled() {
                selected_paths.push(root.clone());
            }
        } else {
            selected_paths.push(root.clone());
        }

        for path in selected_paths {
            let key = to_posix_string(&path);
            if seen_paths.insert(key) {
                result.push(path);
            }
        }
    }

    result
}

fn assistant_text(messages: &[AgentMessage]) -> Option<String> {
    let last_assistant = messages.iter().rev().find_map(|msg| match msg {
        AgentMessage::Llm(Message::Assistant(m)) => Some(m),
        _ => None,
    })?;

    let text = last_assistant
        .content
        .iter()
        .filter_map(|content| match content {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if text.is_empty() { None } else { Some(text) }
}

async fn run_interactive_bootstrap(
    session: &mut pi_coding_agent::AgentSession,
    initial_message: Option<&str>,
    messages: &[String],
) -> Result<(), String> {
    if let Some(initial) = initial_message {
        session
            .prompt(initial, PromptOptions::default())
            .await
            .map_err(|e| e.to_string())?;
        if let Some(text) = assistant_text(session.messages()) {
            println!("{text}");
        }
    }

    for message in messages {
        session
            .prompt(message, PromptOptions::default())
            .await
            .map_err(|e| e.to_string())?;
        if let Some(text) = assistant_text(session.messages()) {
            println!("{text}");
        }
    }

    Ok(())
}

fn model_query(args: &Args) -> Option<String> {
    match (&args.provider, &args.model) {
        (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
        (None, Some(model)) => Some(model.clone()),
        _ => None,
    }
}

#[tokio::main]
async fn main() {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let base_dir = paths::resolve_base_dir(None);

    if let Some(exit_code) = handle_package_command(&raw_args, &cwd, &base_dir) {
        if exit_code != 0 {
            std::process::exit(exit_code);
        }
        return;
    }

    let mut args = parse_args(&raw_args, None);

    if args.help {
        print_help(APP_NAME);
        return;
    }
    if args.version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if let Some(search) = &args.list_models {
        let search = if search.is_empty() {
            None
        } else {
            Some(search.as_str())
        };
        print_models(search);
        return;
    }

    if args.mode != Some(Mode::Rpc) {
        match read_piped_stdin() {
            Ok(Some(stdin_content)) => {
                args.print = true;
                args.messages.insert(0, stdin_content);
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("读取标准输入失败: {e}");
                std::process::exit(1);
            }
        }
    }

    if args.mode == Some(Mode::Rpc) && !args.file_args.is_empty() {
        eprintln!("Error: @file 参数在 rpc 模式下不支持");
        std::process::exit(1);
    }

    if let Some(export_source) = args.export.as_deref() {
        let output = args.messages.first().map(PathBuf::from);
        match export_session_to_html(
            Path::new(export_source),
            ExportHtmlOptions {
                output_path: output,
            },
        ) {
            Ok(path) => println!("Exported to: {}", path.display()),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.no_session {
        eprintln!("Warning: --no-session 当前版本尚未完全实现，将继续使用持久化会话。");
    }

    let initial_message = prepare_initial_message(&args.file_args, &mut args.messages);
    let project_settings = load_project_settings(&cwd);

    let tool_names = if args.no_tools {
        Some(Vec::new())
    } else {
        args.tools.clone()
    };
    let mut extension_paths: Vec<PathBuf> = args.extensions.iter().map(PathBuf::from).collect();
    if !args.no_extensions {
        extension_paths.extend(resolve_package_extension_paths(&cwd, &base_dir));
    }
    let mut extension_seen = std::collections::HashSet::<String>::new();
    extension_paths.retain(|path| extension_seen.insert(path.display().to_string()));

    let created = match create_agent_session_with_extensions(CreateSessionWithExtensionsOptions {
        base: CreateSessionOptions {
            config_dir: args.session_dir.as_ref().map(PathBuf::from),
            working_dir: cwd.clone(),
            project_settings: project_settings.clone(),
            model_id: model_query(&args),
            custom_models: None,
            thinking_level: args.thinking.clone(),
            system_prompt: args.system_prompt.clone(),
            append_system_prompt: args.append_system_prompt.clone(),
            tool_names,
            skill_paths: args.skills.iter().map(PathBuf::from).collect(),
            prompt_template_paths: args.prompt_templates.iter().map(PathBuf::from).collect(),
            theme_paths: args.themes.iter().map(PathBuf::from).collect(),
            no_skills: args.no_skills,
            no_prompt_templates: args.no_prompt_templates,
            no_themes: args.no_themes,
        },
        extension_factories: Vec::new(),
        extension_paths,
        discover_extensions: !args.no_extensions,
        extension_config: None,
    })
    .await
    {
        Ok(result) => result,
        Err(e) => {
            eprintln!("创建会话失败: {e}");
            std::process::exit(1);
        }
    };

    for error in &created.extension_errors {
        eprintln!("扩展加载失败 [{}]: {}", error.source, error.error);
    }

    if let Some(runner) = &created.extension_runner {
        for (name, value) in &args.unknown_flags {
            runner
                .runtime()
                .set_flag(name.clone(), serde_json::Value::String(value.clone()));
        }
    }

    let mut session = created.session;
    let model_patterns = args
        .models
        .clone()
        .or_else(|| session.settings_manager().settings().enabled_models.clone());
    let scoped_models = model_patterns
        .as_ref()
        .map(|patterns| resolve_scoped_models(session.model_registry(), patterns))
        .unwrap_or_default();

    if !scoped_models.is_empty()
        && args.provider.is_none()
        && args.model.is_none()
        && args.session.is_none()
        && !args.continue_session
        && !args.resume
    {
        session.set_model(scoped_models[0].model.clone());
        if args.thinking.is_none()
            && let Some(level) = &scoped_models[0].thinking_level
        {
            session.set_thinking_level_str(level);
        }
    }

    if args.model.is_some() {
        let resolved = resolve_cli_model(
            args.provider.as_deref(),
            args.model.as_deref(),
            session.model_registry(),
        );
        if let Some(warning) = resolved.warning {
            eprintln!("Warning: {warning}");
        }
        if let Some(error) = resolved.error {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if let Some(model) = resolved.model {
            session.set_model(model);
            if args.thinking.is_none() {
                if let Some(level) = resolved.thinking_level {
                    session.set_thinking_level_str(&level);
                }
            }
        }
    }

    if let Some(provider) = &args.provider {
        if args.model.is_none() {
            if let Some(model) = session
                .model_registry()
                .models_for_provider(provider)
                .first()
            {
                session.set_model((**model).clone());
            } else {
                eprintln!("Warning: provider '{provider}' 未找到可用模型");
            }
        }
    }

    if let Some(api_key) = &args.api_key {
        let provider = args
            .provider
            .clone()
            .or_else(|| session.model().map(|m| m.provider.clone()));
        if let Some(provider_name) = provider {
            session
                .auth_storage()
                .set_runtime_credential(&provider_name, AuthCredential::api_key(api_key.clone()));
        } else {
            eprintln!("--api-key 需要 --provider 或可解析的当前模型");
            std::process::exit(1);
        }
    }

    if let Some(session_input) = &args.session {
        let session_id = resolve_session_id(session.session_manager(), session_input);
        if let Err(e) = session.restore_session(&session_id) {
            eprintln!("恢复会话失败 ({session_id}): {e}");
            std::process::exit(1);
        }
    } else if args.continue_session || args.resume {
        match session.session_manager().continue_recent() {
            Ok(Some((header, _entries))) => {
                if let Err(e) = session.restore_session(&header.id) {
                    eprintln!("恢复最近会话失败 ({}): {e}", header.id);
                    std::process::exit(1);
                }
            }
            Ok(None) => {
                if args.resume {
                    eprintln!("没有可恢复的会话。");
                }
            }
            Err(e) => {
                eprintln!("读取最近会话失败: {e}");
                std::process::exit(1);
            }
        }
    }

    let is_interactive = !args.print && args.mode.is_none();
    let mode = args.mode.unwrap_or(Mode::Text);
    if is_interactive && args.verbose && !scoped_models.is_empty() {
        let scoped = scoped_models
            .iter()
            .map(|item| {
                if let Some(level) = &item.thinking_level {
                    format!(
                        "{}/{}:{}",
                        item.model.provider.as_str(),
                        item.model.id.as_str(),
                        level
                    )
                } else {
                    format!(
                        "{}/{}",
                        item.model.provider.as_str(),
                        item.model.id.as_str()
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("Model scope: {scoped}");
    }

    let result = if is_interactive {
        if let Err(e) =
            run_interactive_bootstrap(&mut session, initial_message.as_deref(), &args.messages)
                .await
        {
            eprintln!("{e}");
            std::process::exit(1);
        }
        let interactive_scoped_models = scoped_models
            .iter()
            .map(|item| ScopedModelConfig {
                provider: item.model.provider.clone(),
                model_id: item.model.id.clone(),
                thinking_level: item.thinking_level.clone(),
            })
            .collect();
        InteractiveMode::new(InteractiveModeOptions {
            scoped_models: interactive_scoped_models,
            ..InteractiveModeOptions::default()
        })
        .run(&mut session)
        .await
    } else {
        match mode {
            Mode::Text => {
                run_print_mode(
                    &mut session,
                    PrintModeOptions {
                        mode: PrintOutputMode::Text,
                        initial_message,
                        messages: args.messages.clone(),
                    },
                )
                .await
            }
            Mode::Json => {
                run_print_mode(
                    &mut session,
                    PrintModeOptions {
                        mode: PrintOutputMode::Json,
                        initial_message,
                        messages: args.messages.clone(),
                    },
                )
                .await
            }
            Mode::Rpc => run_rpc_mode(&mut session).await,
        }
    };

    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_resource_kind_parse_aliases() {
        assert!(matches!(
            PackageResourceKind::parse("extensions"),
            Some(PackageResourceKind::Extensions)
        ));
        assert!(matches!(
            PackageResourceKind::parse("s"),
            Some(PackageResourceKind::Skills)
        ));
        assert!(matches!(
            PackageResourceKind::parse("prompt"),
            Some(PackageResourceKind::Prompts)
        ));
        assert!(matches!(
            PackageResourceKind::parse("t"),
            Some(PackageResourceKind::Themes)
        ));
        assert!(PackageResourceKind::parse("unknown").is_none());
    }

    #[test]
    fn test_toggle_package_resource_round_trip() {
        let scope = tempfile::tempdir().unwrap();
        let mut package_sources = vec![PackageSource::Source("pkg-a".to_string())];

        assert!(source_resource_enabled(
            &package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Extensions
        ));

        let enabled = toggle_package_resource(
            &mut package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Extensions,
        )
        .unwrap();
        assert!(!enabled);
        assert!(!source_resource_enabled(
            &package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Extensions
        ));
        assert!(matches!(package_sources[0], PackageSource::Filtered(_)));

        let enabled = toggle_package_resource(
            &mut package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Extensions,
        )
        .unwrap();
        assert!(enabled);
        assert!(source_resource_enabled(
            &package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Extensions
        ));
        assert!(matches!(
            &package_sources[0],
            PackageSource::Source(source) if source == "pkg-a"
        ));
    }

    #[test]
    fn test_toggle_package_resource_item_round_trip() {
        let scope = tempfile::tempdir().unwrap();
        let mut package_sources = vec![PackageSource::Source("pkg-a".to_string())];

        let enabled = toggle_package_resource_item(
            &mut package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Skills,
            "skills/hello.md",
            true,
        )
        .unwrap();
        assert!(!enabled);
        let PackageSource::Filtered(filter) = &package_sources[0] else {
            panic!("expected filtered config");
        };
        assert_eq!(
            filter.skills.as_ref(),
            Some(&vec!["-skills/hello.md".to_string()])
        );

        let enabled = toggle_package_resource_item(
            &mut package_sources,
            scope.path(),
            scope.path(),
            "pkg-a",
            PackageResourceKind::Skills,
            "skills/hello.md",
            false,
        )
        .unwrap();
        assert!(enabled);
        let PackageSource::Filtered(filter) = &package_sources[0] else {
            panic!("expected filtered config");
        };
        assert_eq!(
            filter.skills.as_ref(),
            Some(&vec!["+skills/hello.md".to_string()])
        );
    }

    #[test]
    fn test_build_config_resource_items_with_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        let package_root = tmp.path();
        std::fs::create_dir_all(package_root.join("skills")).unwrap();
        std::fs::write(package_root.join("skills").join("hello.md"), "hello").unwrap();
        std::fs::write(package_root.join("skills").join("bye.md"), "bye").unwrap();

        let config = PackageSource::Filtered(PackageSourceFilter {
            source: "pkg-a".to_string(),
            extensions: None,
            skills: Some(vec!["*.md".to_string(), "!bye.md".to_string()]),
            prompts: None,
            themes: None,
        });
        let items =
            build_config_resource_items(package_root, Some(&config), PackageResourceKind::Skills);
        assert_eq!(items.len(), 2);
        assert!(
            items
                .iter()
                .any(|item| item.relative_path == "skills/hello.md" && item.enabled)
        );
        assert!(
            items
                .iter()
                .any(|item| item.relative_path == "skills/bye.md" && !item.enabled)
        );
    }

    #[test]
    fn test_collect_package_extension_manifest_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("extension.json"), "{}").unwrap();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("nested").join("extension.json"), "{}").unwrap();
        std::fs::write(root.join("extra.extension.json"), "{}").unwrap();

        let entries = collect_package_extension_manifest_paths(root);
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().any(|path| path.ends_with("extension.json")));
        assert!(
            entries
                .iter()
                .any(|path| path.ends_with("nested/extension.json"))
        );
        assert!(
            entries
                .iter()
                .any(|path| path.ends_with("extra.extension.json"))
        );
    }

    #[test]
    fn test_resolve_package_extension_paths_project_config_overrides_user() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        let base_dir = tmp.path().join("agent");

        let user_pkg = base_dir.join("packages").join("pkg_user");
        std::fs::create_dir_all(&user_pkg).unwrap();
        std::fs::write(user_pkg.join("extension.json"), "{}").unwrap();
        std::fs::create_dir_all(base_dir.join("packages")).unwrap();
        std::fs::write(
            base_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_shared",
                "installedPath": user_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let project_pkg_root = cwd.join(CONFIG_DIR_NAME).join("packages");
        let project_pkg = project_pkg_root.join("pkg_project");
        std::fs::create_dir_all(&project_pkg).unwrap();
        std::fs::write(project_pkg.join("extension.json"), "{}").unwrap();
        std::fs::create_dir_all(&project_pkg_root).unwrap();
        std::fs::write(
            project_pkg_root.join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_shared",
                "installedPath": project_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        std::fs::create_dir_all(&base_dir).unwrap();
        std::fs::write(
            base_dir.join("settings.json"),
            serde_json::json!({
                "packages": ["npm:pkg_shared"]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME)).unwrap();
        std::fs::write(
            cwd.join(CONFIG_DIR_NAME).join("settings.json"),
            serde_json::json!({
                "packages": [{
                    "source": "npm:pkg_shared",
                    "extensions": []
                }]
            })
            .to_string(),
        )
        .unwrap();

        let resolved = resolve_package_extension_paths(&cwd, &base_dir);
        assert!(resolved.is_empty());
    }

    #[test]
    fn test_resolve_package_extension_paths_prefers_project_record() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        let base_dir = tmp.path().join("agent");

        let user_pkg = base_dir.join("packages").join("pkg_user");
        std::fs::create_dir_all(&user_pkg).unwrap();
        std::fs::write(user_pkg.join("extension.json"), "{}").unwrap();
        std::fs::create_dir_all(base_dir.join("packages")).unwrap();
        std::fs::write(
            base_dir.join("packages").join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_shared",
                "installedPath": user_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let project_pkg_root = cwd.join(CONFIG_DIR_NAME).join("packages");
        let project_pkg = project_pkg_root.join("pkg_project");
        std::fs::create_dir_all(&project_pkg).unwrap();
        std::fs::write(project_pkg.join("extension.json"), "{}").unwrap();
        std::fs::create_dir_all(&project_pkg_root).unwrap();
        std::fs::write(
            project_pkg_root.join("packages.json"),
            serde_json::json!([{
                "source": "npm:pkg_shared",
                "installedPath": project_pkg.display().to_string()
            }])
            .to_string(),
        )
        .unwrap();

        let resolved = resolve_package_extension_paths(&cwd, &base_dir);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], project_pkg);
    }
}
