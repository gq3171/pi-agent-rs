use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::error::CodingAgentError;
use crate::resources::source_identity::{
    normalize_source_for_scope, source_match_key_for_input, source_match_key_for_scope,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageRecord {
    pub source: String,
    pub installed_path: String,
}

#[derive(Debug, Clone)]
pub struct PackageManager {
    packages_root: PathBuf,
    records_file: PathBuf,
}

impl PackageManager {
    pub fn new(agent_dir: &Path) -> Self {
        let packages_root = agent_dir.join("packages");
        let records_file = packages_root.join("packages.json");
        Self {
            packages_root,
            records_file,
        }
    }

    pub fn install_local_path(
        &self,
        source_path: &Path,
    ) -> Result<PackageRecord, CodingAgentError> {
        let base = source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        self.install_source(&base, &source_path.display().to_string())
    }

    pub fn install_source(
        &self,
        cwd: &Path,
        source: &str,
    ) -> Result<PackageRecord, CodingAgentError> {
        let parsed = parse_source(cwd, source)?;
        let record = match parsed {
            ParsedSource::Local {
                normalized_source,
                source_path,
            } => self.install_local(cwd, &normalized_source, &source_path)?,
            ParsedSource::Npm {
                normalized_source,
                spec,
                name,
            } => self.install_npm(cwd, &normalized_source, &spec, &name)?,
            ParsedSource::Git {
                normalized_source,
                clone_url,
                reference,
                identity_key,
            } => self.install_git(
                cwd,
                &normalized_source,
                &clone_url,
                reference.as_deref(),
                &identity_key,
            )?,
        };
        Ok(record)
    }

    pub fn update_source(
        &self,
        cwd: &Path,
        source: &str,
    ) -> Result<PackageRecord, CodingAgentError> {
        self.install_source(cwd, source)
    }

    pub fn remove(&self, source: &str) -> Result<bool, CodingAgentError> {
        let mut records = self.list()?;
        if let Some(index) = records.iter().position(|r| r.source == source) {
            let record = records.remove(index);
            let path = PathBuf::from(record.installed_path);
            if path.exists() {
                std::fs::remove_dir_all(path)?;
            }
            self.save(&records)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn list(&self) -> Result<Vec<PackageRecord>, CodingAgentError> {
        if !self.records_file.exists() {
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&self.records_file)?;
        let records: Vec<PackageRecord> = serde_json::from_str(&content)?;
        Ok(records)
    }

    fn upsert_record(&self, cwd: &Path, record: PackageRecord) -> Result<(), CodingAgentError> {
        let mut records = self.list()?;
        let new_key = source_match_key_for_scope(cwd, &record.source);
        if let Some(existing) = records
            .iter_mut()
            .find(|r| source_match_key_for_scope(cwd, &r.source) == new_key)
        {
            *existing = record;
        } else {
            records.push(record);
        }
        self.save(&records)
    }

    fn save(&self, records: &[PackageRecord]) -> Result<(), CodingAgentError> {
        std::fs::create_dir_all(&self.packages_root)?;
        let content = serde_json::to_string_pretty(records)?;
        std::fs::write(&self.records_file, content)?;
        Ok(())
    }

    fn install_local(
        &self,
        cwd: &Path,
        normalized_source: &str,
        source_path: &Path,
    ) -> Result<PackageRecord, CodingAgentError> {
        if !source_path.exists() || !source_path.is_dir() {
            return Err(CodingAgentError::Config(format!(
                "Package source path does not exist: {}",
                source_path.display()
            )));
        }

        std::fs::create_dir_all(&self.packages_root)?;

        let local_root = self.packages_root.join("local");
        std::fs::create_dir_all(&local_root)?;
        let identity_key = source_match_key_for_scope(cwd, normalized_source);
        let target = local_root.join(sanitize_identity(&identity_key));
        if target.exists() {
            std::fs::remove_dir_all(&target)?;
        }
        copy_dir_recursive(source_path, &target)?;

        let record = PackageRecord {
            source: normalized_source.to_string(),
            installed_path: target.display().to_string(),
        };
        self.upsert_record(cwd, record.clone())?;
        Ok(record)
    }

    fn install_npm(
        &self,
        cwd: &Path,
        normalized_source: &str,
        spec: &str,
        name: &str,
    ) -> Result<PackageRecord, CodingAgentError> {
        let install_root = self.packages_root.join("npm");
        std::fs::create_dir_all(&install_root)?;

        run_command(
            "npm",
            &[
                "install",
                spec,
                "--prefix",
                &install_root.display().to_string(),
            ],
            None,
        )?;

        let installed_path = install_root.join("node_modules").join(PathBuf::from(name));
        if !installed_path.exists() {
            return Err(CodingAgentError::Config(format!(
                "npm installed package not found at {}",
                installed_path.display()
            )));
        }

        let record = PackageRecord {
            source: normalized_source.to_string(),
            installed_path: installed_path.display().to_string(),
        };
        self.upsert_record(cwd, record.clone())?;
        Ok(record)
    }

    fn install_git(
        &self,
        cwd: &Path,
        normalized_source: &str,
        clone_url: &str,
        reference: Option<&str>,
        identity_key: &str,
    ) -> Result<PackageRecord, CodingAgentError> {
        let git_root = self.packages_root.join("git");
        std::fs::create_dir_all(&git_root)?;

        let target = git_root.join(sanitize_identity(identity_key));
        if target.exists() {
            std::fs::remove_dir_all(&target)?;
        }

        run_command(
            "git",
            &["clone", clone_url, &target.display().to_string()],
            None,
        )?;

        if let Some(reference) = reference {
            run_command(
                "git",
                &["-C", &target.display().to_string(), "checkout", reference],
                None,
            )?;
        }

        if target.join("package.json").exists() {
            run_command("npm", &["install"], Some(&target))?;
        }

        let record = PackageRecord {
            source: normalized_source.to_string(),
            installed_path: target.display().to_string(),
        };
        self.upsert_record(cwd, record.clone())?;
        Ok(record)
    }
}

#[derive(Debug, Clone)]
enum ParsedSource {
    Local {
        normalized_source: String,
        source_path: PathBuf,
    },
    Npm {
        normalized_source: String,
        spec: String,
        name: String,
    },
    Git {
        normalized_source: String,
        clone_url: String,
        reference: Option<String>,
        identity_key: String,
    },
}

fn parse_source(cwd: &Path, source: &str) -> Result<ParsedSource, CodingAgentError> {
    let source = source.trim();
    if source.is_empty() {
        return Err(CodingAgentError::Config(
            "Package source cannot be empty".to_string(),
        ));
    }

    if let Some(spec) = source.strip_prefix("npm:") {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(CodingAgentError::Config(
                "npm source requires package spec, e.g. npm:@scope/name".to_string(),
            ));
        }
        let name = parse_npm_package_name(spec);
        return Ok(ParsedSource::Npm {
            normalized_source: format!("npm:{spec}"),
            spec: spec.to_string(),
            name,
        });
    }

    let key = source_match_key_for_input(cwd, source);
    if key.starts_with("git:") {
        let normalized_source = source.to_string();
        let (clone_url, reference) = parse_git_clone_source(source)?;
        return Ok(ParsedSource::Git {
            normalized_source,
            clone_url,
            reference,
            identity_key: key,
        });
    }

    let normalized_source = normalize_source_for_scope(cwd, source);
    Ok(ParsedSource::Local {
        source_path: PathBuf::from(&normalized_source),
        normalized_source,
    })
}

fn parse_npm_package_name(spec: &str) -> String {
    if spec.starts_with('@') {
        if let Some((name, _)) = spec.rsplit_once('@') {
            // @scope/name@version
            if name.contains('/') {
                return name.to_string();
            }
        }
        return spec.to_string();
    }
    if let Some((name, _)) = spec.rsplit_once('@') {
        return name.to_string();
    }
    spec.to_string()
}

fn parse_git_clone_source(source: &str) -> Result<(String, Option<String>), CodingAgentError> {
    let raw = source
        .trim()
        .strip_prefix("git:")
        .map(str::trim)
        .unwrap_or_else(|| source.trim());

    if let Some((host, path_ref)) = raw
        .strip_prefix("git@")
        .and_then(|rest| rest.split_once(':'))
    {
        let (repo_path, reference) = split_git_repo_and_ref(path_ref);
        return Ok((format!("git@{host}:{repo_path}"), reference));
    }

    // https://host/path@ref, ssh://host/path@ref, host/path@ref
    if let Some((prefix, tail)) = raw.split_once("://")
        && let Some((host, path_ref)) = tail.split_once('/')
    {
        let (repo_path, reference) = split_git_repo_and_ref(path_ref);
        return Ok((format!("{prefix}://{host}/{repo_path}"), reference));
    }

    if let Some((host, path_ref)) = raw.split_once('/') {
        let (repo_path, reference) = split_git_repo_and_ref(path_ref);
        let clone = if host.contains('.') || host.eq_ignore_ascii_case("localhost") {
            format!("https://{host}/{repo_path}")
        } else {
            // github shorthand: user/repo(@ref)
            format!("https://github.com/{host}/{repo_path}")
        };
        return Ok((clone, reference));
    }

    Err(CodingAgentError::Config(format!(
        "Unsupported git source format: {source}"
    )))
}

fn split_git_repo_and_ref(path_ref: &str) -> (String, Option<String>) {
    if let Some((repo_path, reference)) = path_ref.split_once('#') {
        if !repo_path.is_empty() && !reference.is_empty() {
            return (repo_path.to_string(), Some(reference.to_string()));
        }
    }
    if let Some((repo_path, reference)) = path_ref.rsplit_once('@') {
        if !repo_path.is_empty() && !reference.is_empty() {
            return (repo_path.to_string(), Some(reference.to_string()));
        }
    }
    (path_ref.to_string(), None)
}

fn sanitize_identity(identity: &str) -> String {
    let mut value = String::new();
    for ch in identity.chars() {
        if ch.is_ascii_alphanumeric() {
            value.push(ch);
        } else {
            value.push('_');
        }
    }
    while value.contains("__") {
        value = value.replace("__", "_");
    }
    value.trim_matches('_').to_string()
}

fn run_command(program: &str, args: &[&str], cwd: Option<&Path>) -> Result<(), CodingAgentError> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().map_err(|e| {
        CodingAgentError::Config(format!(
            "Failed to run `{}`: {}",
            format_args!("{program} {}", sanitize_args_for_log(args).join(" ")),
            e
        ))
    })?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit code {:?}", output.status.code())
    };
    Err(CodingAgentError::Config(format!(
        "`{}` failed: {}",
        format_args!("{program} {}", sanitize_args_for_log(args).join(" ")),
        detail
    )))
}

fn sanitize_args_for_log(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| sanitize_arg_for_log(arg)).collect()
}

fn sanitize_arg_for_log(arg: &str) -> String {
    let mut value = arg.to_string();

    if let Some((scheme, rest)) = value.split_once("://")
        && let Some((userinfo, host_rest)) = rest.split_once('@')
        && !userinfo.is_empty()
    {
        value = format!("{scheme}://***@{host_rest}");
    }

    // Best-effort redaction for query/kv secrets.
    for marker in [
        "token=",
        "access_token=",
        "auth=",
        "password=",
        "passwd=",
        "apikey=",
        "api_key=",
    ] {
        if let Some(pos) = value.to_ascii_lowercase().find(marker) {
            let key_len = marker.len();
            let value_start = pos + key_len;
            let value_end = value[value_start..]
                .find(|ch: char| ch == '&' || ch == ';' || ch == ' ')
                .map(|idx| value_start + idx)
                .unwrap_or(value.len());
            value.replace_range(value_start..value_end, "***");
        }
    }

    value
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<(), CodingAgentError> {
    std::fs::create_dir_all(target)?;
    for entry in WalkDir::new(source).into_iter().filter_map(Result::ok) {
        let rel = entry
            .path()
            .strip_prefix(source)
            .map_err(|e| CodingAgentError::Other(e.to_string()))?;
        let dest = target.join(rel);

        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dest)?;
        } else if entry.file_type().is_file() {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_and_remove_local_package() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("package.json"), "{\"name\":\"x\"}").unwrap();

        let manager = PackageManager::new(tmp.path());
        let record = manager
            .install_source(tmp.path(), &source.path().display().to_string())
            .unwrap();
        assert!(Path::new(&record.installed_path).exists());

        let removed = manager.remove(&record.source).unwrap();
        assert!(removed);
    }

    #[test]
    fn test_parse_source_npm() {
        let cwd = tempfile::tempdir().unwrap();
        let parsed = parse_source(cwd.path(), "npm:@scope/pkg@1.2.3").unwrap();
        match parsed {
            ParsedSource::Npm {
                normalized_source,
                spec,
                name,
            } => {
                assert_eq!(normalized_source, "npm:@scope/pkg@1.2.3");
                assert_eq!(spec, "@scope/pkg@1.2.3");
                assert_eq!(name, "@scope/pkg");
            }
            _ => panic!("expected npm source"),
        }
    }

    #[test]
    fn test_parse_source_git() {
        let cwd = tempfile::tempdir().unwrap();
        let parsed = parse_source(cwd.path(), "git@github.com:foo/bar@main").unwrap();
        match parsed {
            ParsedSource::Git {
                clone_url,
                reference,
                ..
            } => {
                assert_eq!(clone_url, "git@github.com:foo/bar");
                assert_eq!(reference.as_deref(), Some("main"));
            }
            _ => panic!("expected git source"),
        }
    }

    #[test]
    fn test_parse_source_git_hash_ref() {
        let cwd = tempfile::tempdir().unwrap();
        let parsed = parse_source(cwd.path(), "https://github.com/foo/bar#main").unwrap();
        match parsed {
            ParsedSource::Git {
                clone_url,
                reference,
                ..
            } => {
                assert_eq!(clone_url, "https://github.com/foo/bar");
                assert_eq!(reference.as_deref(), Some("main"));
            }
            _ => panic!("expected git source"),
        }
    }

    #[test]
    fn test_install_local_same_basename_does_not_collide() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = PackageManager::new(tmp.path());

        let first = tmp.path().join("first").join("pkg");
        let second = tmp.path().join("second").join("pkg");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(first.join("marker.txt"), "first").unwrap();
        std::fs::write(second.join("marker.txt"), "second").unwrap();

        let record_first = manager
            .install_source(tmp.path(), &first.display().to_string())
            .unwrap();
        let record_second = manager
            .install_source(tmp.path(), &second.display().to_string())
            .unwrap();

        assert_ne!(record_first.installed_path, record_second.installed_path);
        assert!(
            Path::new(&record_first.installed_path)
                .join("marker.txt")
                .exists()
        );
        assert!(
            Path::new(&record_second.installed_path)
                .join("marker.txt")
                .exists()
        );
    }

    #[test]
    fn test_sanitize_arg_for_log_redacts_credentials() {
        let sanitized =
            sanitize_arg_for_log("https://token-123@github.com/org/repo.git?access_token=abc");
        assert!(!sanitized.contains("token-123"));
        assert!(!sanitized.contains("access_token=abc"));
        assert!(sanitized.contains("***@github.com"));
    }
}
