/// Default maximum number of lines for tool output.
pub const DEFAULT_MAX_LINES: usize = 2000;

/// Default maximum bytes for tool output.
pub const DEFAULT_MAX_BYTES: usize = 200_000;

/// Truncation result.
pub struct TruncateResult {
    pub content: String,
    pub was_truncated: bool,
    pub original_lines: usize,
    pub original_bytes: usize,
}

/// Truncate content by line count and byte size.
pub fn truncate_output(
    content: &str,
    max_lines: Option<usize>,
    max_bytes: Option<usize>,
) -> TruncateResult {
    let max_lines = max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = max_bytes.unwrap_or(DEFAULT_MAX_BYTES);
    let original_lines = content.lines().count();
    let original_bytes = content.len();

    if original_lines <= max_lines && original_bytes <= max_bytes {
        return TruncateResult {
            content: content.to_string(),
            was_truncated: false,
            original_lines,
            original_bytes,
        };
    }

    // Truncate by lines first
    let mut lines: Vec<&str> = content.lines().collect();
    let lines_truncated = lines.len() > max_lines;
    if lines_truncated {
        lines.truncate(max_lines);
    }

    let mut result = lines.join("\n");

    // Then truncate by bytes
    let bytes_truncated = result.len() > max_bytes;
    if bytes_truncated {
        // Find a safe UTF-8 boundary
        let mut end = max_bytes;
        while end > 0 && !result.is_char_boundary(end) {
            end -= 1;
        }
        result.truncate(end);
    }

    let was_truncated = lines_truncated || bytes_truncated;
    if was_truncated {
        result.push_str(&format!(
            "\n\n[Output truncated: {original_lines} lines, {original_bytes} bytes total]"
        ));
    }

    TruncateResult {
        content: result,
        was_truncated,
        original_lines,
        original_bytes,
    }
}

/// Truncate a string to a max byte length at a safe boundary.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_truncation_needed() {
        let result = truncate_output("hello\nworld", None, None);
        assert!(!result.was_truncated);
        assert_eq!(result.content, "hello\nworld");
    }

    #[test]
    fn test_truncate_by_lines() {
        let content = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_output(&content, Some(10), None);
        assert!(result.was_truncated);
        assert!(result.content.contains("[Output truncated:"));
    }

    #[test]
    fn test_truncate_by_bytes() {
        let content = "a".repeat(1000);
        let result = truncate_output(&content, None, Some(100));
        assert!(result.was_truncated);
    }

    #[test]
    fn test_truncate_str_safe_boundary() {
        let s = "hello 世界";
        let t = truncate_str(s, 7);
        assert_eq!(t, "hello ");
    }
}
