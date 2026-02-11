/// Sanitize surrogates from text.
///
/// In Rust, `String` is guaranteed valid UTF-8, so unpaired surrogates
/// cannot exist in a Rust String. This function is a no-op passthrough
/// for Rust, included for API compatibility with the TypeScript SDK.
///
/// If you receive data from external sources (e.g., raw bytes from JSON
/// that might contain WTF-8), you should use `String::from_utf8_lossy`
/// at the boundary instead.
pub fn sanitize_surrogates(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_text() {
        assert_eq!(sanitize_surrogates("Hello World"), "Hello World");
    }

    #[test]
    fn test_emoji() {
        assert_eq!(sanitize_surrogates("Hello 🙈 World"), "Hello 🙈 World");
    }

    #[test]
    fn test_empty() {
        assert_eq!(sanitize_surrogates(""), "");
    }
}
