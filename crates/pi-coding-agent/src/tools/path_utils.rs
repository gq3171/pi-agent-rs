use std::path::{Path, PathBuf};

/// Resolve a possibly relative path against a working directory.
///
/// - If the path is absolute, returns it as-is.
/// - If relative, joins it with the working directory.
/// - Normalizes the result (removes `.` and `..` components).
pub fn resolve_path(path: &str, working_dir: &Path) -> PathBuf {
    let p = Path::new(path);
    let absolute = if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    };
    normalize_path(&absolute)
}

/// Normalize a path by resolving `.` and `..` components.
/// Does NOT follow symlinks (pure lexical normalization).
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {} // skip "."
            std::path::Component::ParentDir => {
                // Pop last component if possible
                if !components.is_empty() {
                    components.pop();
                }
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Check if a path is within a given root directory (sandbox check).
///
/// Uses `canonicalize` when the path exists (resolves symlinks) and
/// falls back to lexical normalization for paths that don't exist yet.
/// Returns true if `path` is a descendant of (or equal to) `root`.
pub fn is_within(path: &Path, root: &Path) -> bool {
    // Resolve root (should always exist)
    let resolved_root = root.canonicalize().unwrap_or_else(|_| normalize_path(root));

    if path.exists() {
        // Path exists: resolve symlinks
        match path.canonicalize() {
            Ok(resolved) => resolved.starts_with(&resolved_root),
            Err(_) => false,
        }
    } else {
        // Path doesn't exist (e.g. write): check parent directory + lexical check
        if let Some(parent) = path.parent() {
            if parent.exists() {
                match parent.canonicalize() {
                    Ok(resolved_parent) => resolved_parent.starts_with(&resolved_root),
                    Err(_) => false,
                }
            } else {
                // Parent doesn't exist either — fall back to lexical check
                normalize_path(path).starts_with(&resolved_root)
            }
        } else {
            false
        }
    }
}

/// Check if a file looks like a binary based on its extension.
pub fn is_likely_binary(path: &Path) -> bool {
    let binary_extensions = [
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "svg", "mp3", "mp4", "wav", "ogg",
        "avi", "mov", "mkv", "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "exe", "dll", "so",
        "dylib", "o", "a", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "woff", "woff2",
        "ttf", "otf", "eot", "class", "pyc", "pyo", "wasm", "sqlite", "db",
    ];

    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| binary_extensions.contains(&ext.to_lowercase().as_str()))
}

/// Check if a file looks like an image based on its extension.
pub fn is_image(path: &Path) -> bool {
    let image_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "svg"];

    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| image_extensions.contains(&ext.to_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_absolute_path() {
        let result = resolve_path("/home/user/file.txt", Path::new("/work"));
        assert_eq!(result, PathBuf::from("/home/user/file.txt"));
    }

    #[test]
    fn test_resolve_relative_path() {
        let result = resolve_path("src/main.rs", Path::new("/work/project"));
        assert_eq!(result, PathBuf::from("/work/project/src/main.rs"));
    }

    #[test]
    fn test_normalize_path() {
        let result = normalize_path(Path::new("/home/user/../user/./docs/file.txt"));
        assert_eq!(result, PathBuf::from("/home/user/docs/file.txt"));
    }

    #[test]
    fn test_is_within() {
        assert!(is_within(
            Path::new("/home/user/project/src/main.rs"),
            Path::new("/home/user/project")
        ));
        assert!(!is_within(
            Path::new("/home/other/file.txt"),
            Path::new("/home/user/project")
        ));
        // Path traversal attack
        assert!(!is_within(
            Path::new("/home/user/project/../../etc/passwd"),
            Path::new("/home/user/project")
        ));
    }

    #[test]
    fn test_is_likely_binary() {
        assert!(is_likely_binary(Path::new("image.png")));
        assert!(is_likely_binary(Path::new("lib.so")));
        assert!(!is_likely_binary(Path::new("code.rs")));
        assert!(!is_likely_binary(Path::new("readme.md")));
    }

    #[test]
    fn test_is_image() {
        assert!(is_image(Path::new("photo.jpg")));
        assert!(is_image(Path::new("icon.PNG")));
        assert!(!is_image(Path::new("video.mp4")));
    }
}
