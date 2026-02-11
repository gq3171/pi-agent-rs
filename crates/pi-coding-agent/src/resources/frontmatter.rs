use serde::Deserialize;

/// Parse YAML frontmatter from a markdown file.
///
/// Frontmatter is delimited by `---` at the beginning of the file:
/// ```text
/// ---
/// key: value
/// ---
/// body content here
/// ```
///
/// Returns (frontmatter, body) if frontmatter is found, or (None, full_content) otherwise.
pub fn parse_frontmatter<T: for<'de> Deserialize<'de>>(
    content: &str,
) -> Result<(Option<T>, &str), FrontmatterError> {
    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        return Ok((None, content));
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let rest = after_first.trim_start_matches(['\r', '\n']);

    if let Some(end_pos) = rest.find("\n---") {
        let yaml_str = &rest[..end_pos];
        let body_start = end_pos + 4; // skip "\n---"
        let body = rest[body_start..].trim_start_matches(['\r', '\n']);

        let parsed: T =
            serde_yaml::from_str(yaml_str).map_err(|e| FrontmatterError::YamlParse(e.to_string()))?;

        Ok((Some(parsed), body))
    } else {
        // No closing ---, treat entire content as body
        Ok((None, content))
    }
}

/// Error type for frontmatter parsing.
#[derive(Debug)]
pub enum FrontmatterError {
    YamlParse(String),
}

impl std::fmt::Display for FrontmatterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontmatterError::YamlParse(e) => write!(f, "YAML parse error: {e}"),
        }
    }
}

impl std::error::Error for FrontmatterError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestMeta {
        title: String,
        #[serde(default)]
        tags: Vec<String>,
    }

    #[test]
    fn test_parse_with_frontmatter() {
        let content = "---\ntitle: Hello\ntags:\n  - rust\n  - test\n---\nBody content here";
        let (meta, body) = parse_frontmatter::<TestMeta>(content).unwrap();
        let meta = meta.unwrap();
        assert_eq!(meta.title, "Hello");
        assert_eq!(meta.tags, vec!["rust", "test"]);
        assert_eq!(body, "Body content here");
    }

    #[test]
    fn test_parse_without_frontmatter() {
        let content = "Just regular content\nNo frontmatter here.";
        let (meta, body) = parse_frontmatter::<TestMeta>(content).unwrap();
        assert!(meta.is_none());
        assert_eq!(body, content);
    }

    #[test]
    fn test_parse_invalid_yaml() {
        let content = "---\n: invalid: yaml: [[\n---\nBody";
        let result = parse_frontmatter::<TestMeta>(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_closing_delimiter() {
        let content = "---\ntitle: Hello\nNo closing delimiter";
        let (meta, body) = parse_frontmatter::<TestMeta>(content).unwrap();
        assert!(meta.is_none());
        assert_eq!(body, content);
    }
}
