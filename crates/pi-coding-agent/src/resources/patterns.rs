use std::collections::HashSet;
use std::path::{Path, PathBuf};

use glob::Pattern;

fn to_posix_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn normalize_exact_pattern(pattern: &str) -> String {
    if pattern.starts_with("./") || pattern.starts_with(".\\") {
        pattern[2..].replace('\\', "/")
    } else {
        pattern.replace('\\', "/")
    }
}

fn path_relative_to_base(path: &Path, base_dir: &Path) -> PathBuf {
    path.strip_prefix(base_dir)
        .map(PathBuf::from)
        .unwrap_or_else(|_| path.to_path_buf())
}

fn glob_matches(target: &str, pattern: &str) -> bool {
    Pattern::new(pattern)
        .map(|compiled| compiled.matches(target))
        .unwrap_or(false)
}

fn matches_any_pattern(file_path: &Path, patterns: &[String], base_dir: &Path) -> bool {
    let rel = to_posix_path(&path_relative_to_base(file_path, base_dir));
    let name = file_path
        .file_name()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_default();
    let full = to_posix_path(file_path);

    let is_skill_file = name == "SKILL.md";
    let parent = file_path.parent();
    let parent_rel = parent.map(|p| to_posix_path(&path_relative_to_base(p, base_dir)));
    let parent_name = parent
        .and_then(|p| p.file_name())
        .map(|v| v.to_string_lossy().to_string());
    let parent_full = parent.map(to_posix_path);

    patterns.iter().any(|pattern| {
        if glob_matches(&rel, pattern)
            || glob_matches(&name, pattern)
            || glob_matches(&full, pattern)
        {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        parent_rel
            .as_ref()
            .is_some_and(|value| glob_matches(value, pattern))
            || parent_name
                .as_ref()
                .is_some_and(|value| glob_matches(value, pattern))
            || parent_full
                .as_ref()
                .is_some_and(|value| glob_matches(value, pattern))
    })
}

fn matches_any_exact_pattern(file_path: &Path, patterns: &[String], base_dir: &Path) -> bool {
    if patterns.is_empty() {
        return false;
    }

    let rel = to_posix_path(&path_relative_to_base(file_path, base_dir));
    let full = to_posix_path(file_path);
    let name = file_path
        .file_name()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_default();
    let is_skill_file = name == "SKILL.md";
    let parent = file_path.parent();
    let parent_rel = parent.map(|p| to_posix_path(&path_relative_to_base(p, base_dir)));
    let parent_full = parent.map(to_posix_path);

    patterns.iter().any(|pattern| {
        let normalized = normalize_exact_pattern(pattern);
        if normalized == rel || normalized == full {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        parent_rel
            .as_ref()
            .is_some_and(|value| *value == normalized)
            || parent_full
                .as_ref()
                .is_some_and(|value| *value == normalized)
    })
}

/// Apply pi-mono compatible include/exclude patterns.
///
/// Rules:
/// - plain pattern: include
/// - `!pattern`: exclude
/// - `+path`: force include exact path
/// - `-path`: force exclude exact path
pub fn apply_patterns(
    all_paths: &[PathBuf],
    patterns: &[String],
    base_dir: &Path,
) -> HashSet<String> {
    let mut includes = Vec::<String>::new();
    let mut excludes = Vec::<String>::new();
    let mut force_includes = Vec::<String>::new();
    let mut force_excludes = Vec::<String>::new();

    for pattern in patterns {
        if let Some(value) = pattern.strip_prefix('+') {
            force_includes.push(value.to_string());
        } else if let Some(value) = pattern.strip_prefix('-') {
            force_excludes.push(value.to_string());
        } else if let Some(value) = pattern.strip_prefix('!') {
            excludes.push(value.to_string());
        } else {
            includes.push(pattern.clone());
        }
    }

    let mut result = if includes.is_empty() {
        all_paths
            .iter()
            .map(|path| to_posix_path(path))
            .collect::<HashSet<_>>()
    } else {
        all_paths
            .iter()
            .filter(|path| matches_any_pattern(path, &includes, base_dir))
            .map(|path| to_posix_path(path))
            .collect::<HashSet<_>>()
    };

    if !excludes.is_empty() {
        result.retain(|path| !matches_any_pattern(Path::new(path), &excludes, base_dir));
    }

    if !force_includes.is_empty() {
        for path in all_paths {
            if matches_any_exact_pattern(path, &force_includes, base_dir) {
                result.insert(to_posix_path(path));
            }
        }
    }

    if !force_excludes.is_empty() {
        result
            .retain(|path| !matches_any_exact_pattern(Path::new(path), &force_excludes, base_dir));
    }

    result
}

pub fn to_posix_string(path: &Path) -> String {
    to_posix_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_patterns_include_exclude() {
        let root = PathBuf::from("/tmp/pkg");
        let all = vec![
            root.join("skills").join("a.md"),
            root.join("skills").join("b.md"),
        ];
        let patterns = vec!["*.md".to_string(), "!b.md".to_string()];
        let enabled = apply_patterns(&all, &patterns, &root);
        assert!(enabled.contains("/tmp/pkg/skills/a.md"));
        assert!(!enabled.contains("/tmp/pkg/skills/b.md"));
    }

    #[test]
    fn test_apply_patterns_force_include_and_exclude() {
        let root = PathBuf::from("/tmp/pkg");
        let all = vec![
            root.join("skills").join("a.md"),
            root.join("skills").join("b.md"),
        ];
        let patterns = vec![
            "!*.md".to_string(),
            "+skills/b.md".to_string(),
            "-skills/b.md".to_string(),
        ];
        let enabled = apply_patterns(&all, &patterns, &root);
        assert!(enabled.is_empty());
    }
}
