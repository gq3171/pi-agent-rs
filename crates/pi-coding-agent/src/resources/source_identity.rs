use std::path::{Component, Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;

static NPM_SPEC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(@?[^@]+(?:/[^@]+)?)(?:@(.+))?$").expect("valid npm regex"));
static SCP_GIT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^git@([^:]+):(.+)$").expect("valid scp git regex"));
static SCHEME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z][a-zA-Z0-9+.-]*://").expect("valid scheme regex"));
static GITHUB_SHORTHAND_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:@[A-Za-z0-9._/-]+)?$")
        .expect("valid github shorthand regex")
});

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParsedSource {
    Npm { name: String },
    Git { host: String, path: String },
    Local { path: String },
}

pub fn source_match_key_for_input(cwd: &Path, source: &str) -> String {
    source_match_key_with_base(cwd, source)
}

pub fn source_match_key_for_scope(scope_base: &Path, source: &str) -> String {
    source_match_key_with_base(scope_base, source)
}

pub fn normalize_source_for_scope(scope_base: &Path, source: &str) -> String {
    match parse_source(source) {
        ParsedSource::Npm { .. } | ParsedSource::Git { .. } => source.to_string(),
        ParsedSource::Local { path } => {
            if is_windows_absolute_path(&path) {
                return normalize_windows_path(&path);
            }
            let resolved = resolve_local_path(scope_base, &path);
            to_posix_path(&resolved)
        }
    }
}

fn source_match_key_with_base(base_dir: &Path, source: &str) -> String {
    match parse_source(source) {
        ParsedSource::Npm { name } => format!("npm:{name}"),
        ParsedSource::Git { host, path } => format!("git:{host}/{path}"),
        ParsedSource::Local { path } => {
            if is_windows_absolute_path(&path) {
                return format!("local:{}", normalize_windows_path(&path));
            }
            let resolved = resolve_local_path(base_dir, &path);
            format!("local:{}", to_posix_path(&resolved))
        }
    }
}

fn parse_source(source: &str) -> ParsedSource {
    if let Some(name) = parse_npm_name(source) {
        return ParsedSource::Npm { name };
    }

    let trimmed = source.trim();
    let is_local_path_like = trimmed.starts_with('.')
        || trimmed.starts_with('/')
        || trimmed == "~"
        || trimmed.starts_with("~/")
        || trimmed.starts_with("~\\")
        || is_windows_absolute_path(trimmed);
    if is_local_path_like {
        return ParsedSource::Local {
            path: source.to_string(),
        };
    }

    if let Some((host, path)) = parse_git_source(source) {
        return ParsedSource::Git { host, path };
    }

    ParsedSource::Local {
        path: source.to_string(),
    }
}

fn parse_npm_name(source: &str) -> Option<String> {
    let value = source.trim();
    let spec = value.strip_prefix("npm:")?.trim();
    if spec.is_empty() {
        return Some(String::new());
    }
    let name = NPM_SPEC_RE
        .captures(spec)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str())
        .unwrap_or(spec);
    Some(name.to_string())
}

fn parse_git_source(source: &str) -> Option<(String, String)> {
    let raw = source
        .trim()
        .strip_prefix("git:")
        .map(str::trim)
        .unwrap_or_else(|| source.trim());
    if raw.is_empty() {
        return None;
    }

    if let Some(captures) = SCP_GIT_RE.captures(raw) {
        let host = captures.get(1).map(|m| m.as_str().to_ascii_lowercase())?;
        let path_with_ref = captures.get(2).map(|m| m.as_str())?;
        let path = normalize_git_path(strip_git_ref_suffix(path_with_ref))?;
        return Some((host, path));
    }

    if SCHEME_RE.is_match(raw)
        && let Some((authority, path_with_ref)) = raw.split_once("://").and_then(|(_, rest)| {
            let mut parts = rest.splitn(2, '/');
            let authority = parts.next()?;
            let path = parts.next()?;
            Some((authority, path))
        })
    {
        let host_with_port = authority.rsplit('@').next().unwrap_or(authority);
        let host = host_with_port
            .split(':')
            .next()
            .unwrap_or(host_with_port)
            .to_ascii_lowercase();
        if host.is_empty() {
            return None;
        }
        let path = normalize_git_path(strip_git_ref_suffix(path_with_ref))?;
        return Some((host, path));
    }

    if let Some((host, path_with_ref)) = raw.split_once('/') {
        if host.contains('.') || host.eq_ignore_ascii_case("localhost") {
            let path = normalize_git_path(strip_git_ref_suffix(path_with_ref))?;
            return Some((host.to_ascii_lowercase(), path));
        }
    }

    if GITHUB_SHORTHAND_RE.is_match(raw) {
        let path = normalize_git_path(strip_git_ref_suffix(raw))?;
        return Some(("github.com".to_string(), path));
    }

    None
}

fn strip_git_ref_suffix(path: &str) -> &str {
    let path = path.split('#').next().unwrap_or(path);
    if let Some((repo_path, _)) = path.rsplit_once('@') {
        if !repo_path.is_empty() {
            return repo_path;
        }
    }
    path
}

fn normalize_git_path(path: &str) -> Option<String> {
    let trimmed = path.trim().trim_start_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let mut normalized = trimmed.to_string();
    if normalized.ends_with(".git") {
        normalized.truncate(normalized.len().saturating_sub(4));
    }
    while normalized.ends_with('/') {
        normalized.pop();
    }
    let segments = normalized
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.len() < 2 {
        return None;
    }
    Some(segments.join("/"))
}

fn resolve_local_path(base_dir: &Path, raw_path: &str) -> PathBuf {
    let expanded = expand_tilde(raw_path);
    let path = PathBuf::from(&expanded);
    let absolute = if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    };
    std::fs::canonicalize(&absolute).unwrap_or_else(|_| normalize_path(&absolute))
}

fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|home| home.display().to_string())
            .unwrap_or_else(|| path.to_string());
    }

    let tail = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\"));
    if let Some(suffix) = tail
        && let Some(home) = dirs::home_dir()
    {
        return home.join(suffix).display().to_string();
    }
    path.to_string()
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn is_windows_absolute_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    let drive_prefixed = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    drive_prefixed || path.starts_with("\\\\")
}

fn normalize_windows_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn to_posix_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npm_identity_ignores_version() {
        let cwd = tempfile::tempdir().unwrap();
        let a = source_match_key_for_input(cwd.path(), "npm:@foo/bar");
        let b = source_match_key_for_input(cwd.path(), "npm:@foo/bar@1.2.3");
        assert_eq!(a, b);
    }

    #[test]
    fn test_git_identity_normalizes_transport() {
        let cwd = tempfile::tempdir().unwrap();
        let https = source_match_key_for_input(cwd.path(), "https://github.com/foo/bar.git");
        let ssh = source_match_key_for_input(cwd.path(), "git@github.com:foo/bar");
        let prefixed = source_match_key_for_input(cwd.path(), "git:github.com/foo/bar@main");
        let with_hash = source_match_key_for_input(cwd.path(), "https://github.com/foo/bar#dev");
        assert_eq!(https, ssh);
        assert_eq!(ssh, prefixed);
        assert_eq!(prefixed, with_hash);
    }

    #[test]
    fn test_local_identity_matches_relative_and_absolute() {
        let cwd = tempfile::tempdir().unwrap();
        let pkg = cwd.path().join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();

        let abs = source_match_key_for_input(cwd.path(), &pkg.display().to_string());
        let rel = source_match_key_for_input(cwd.path(), "./pkg");
        assert_eq!(abs, rel);
    }

    #[test]
    fn test_local_identity_normalizes_dotdot_when_missing() {
        let cwd = tempfile::tempdir().unwrap();
        let key = source_match_key_for_input(cwd.path(), "./a/../b");
        assert!(key.ends_with("/b"));
    }
}
