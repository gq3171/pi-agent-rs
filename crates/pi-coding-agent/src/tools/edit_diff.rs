use similar::{ChangeTag, TextDiff};

/// Generate a unified diff between two strings.
pub fn generate_diff(original: &str, modified: &str, context_lines: usize) -> String {
    let diff = TextDiff::from_lines(original, modified);

    diff.unified_diff()
        .context_radius(context_lines)
        .to_string()
}

/// Count the number of changes in a diff.
pub fn count_changes(original: &str, modified: &str) -> DiffStats {
    let diff = TextDiff::from_lines(original, modified);
    let mut additions = 0;
    let mut deletions = 0;

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => additions += 1,
            ChangeTag::Delete => deletions += 1,
            ChangeTag::Equal => {}
        }
    }

    DiffStats {
        additions,
        deletions,
    }
}

/// Statistics about a diff.
#[derive(Debug, Clone)]
pub struct DiffStats {
    pub additions: usize,
    pub deletions: usize,
}

impl DiffStats {
    pub fn total_changes(&self) -> usize {
        self.additions + self.deletions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_diff() {
        let original = "line 1\nline 2\nline 3\n";
        let modified = "line 1\nline 2 modified\nline 3\nline 4\n";
        let diff = generate_diff(original, modified, 3);
        assert!(diff.contains("-line 2"));
        assert!(diff.contains("+line 2 modified"));
        assert!(diff.contains("+line 4"));
    }

    #[test]
    fn test_count_changes() {
        let original = "a\nb\nc\n";
        let modified = "a\nx\nc\nd\n";
        let stats = count_changes(original, modified);
        assert_eq!(stats.deletions, 1); // "b" removed
        assert_eq!(stats.additions, 2); // "x" and "d" added
    }

    #[test]
    fn test_no_changes() {
        let text = "same\ncontent\n";
        let stats = count_changes(text, text);
        assert_eq!(stats.total_changes(), 0);
    }
}
