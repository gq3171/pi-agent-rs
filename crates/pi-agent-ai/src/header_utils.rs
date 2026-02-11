use std::collections::HashMap;

/// Headers that must not be overridden by model or extra headers.
/// These are security-sensitive authentication headers.
const SENSITIVE_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
];

/// Check whether a header key is sensitive (case-insensitive).
fn is_sensitive(key: &str) -> bool {
    SENSITIVE_HEADERS
        .iter()
        .any(|&h| key.eq_ignore_ascii_case(h))
}

/// Check whether `target` already contains a key matching `key` case-insensitively.
fn has_key_ignore_case(target: &HashMap<String, String>, key: &str) -> bool {
    target
        .keys()
        .any(|existing| existing.eq_ignore_ascii_case(key))
}

/// Merge `source` headers into `target`, skipping any sensitive keys
/// that are already present in `target` (case-insensitive match).
pub fn merge_headers_safe(target: &mut HashMap<String, String>, source: &HashMap<String, String>) {
    for (k, v) in source {
        if is_sensitive(k) && has_key_ignore_case(target, k) {
            continue;
        }
        target.insert(k.clone(), v.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_headers_safe_blocks_auth_override() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer secret".to_string());
        headers.insert("content-type".to_string(), "application/json".to_string());

        let mut extra = HashMap::new();
        extra.insert("Authorization".to_string(), "Bearer malicious".to_string());
        extra.insert("x-custom".to_string(), "allowed".to_string());

        merge_headers_safe(&mut headers, &extra);

        assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
        assert!(!headers.contains_key("Authorization"));
        assert_eq!(headers.get("x-custom").unwrap(), "allowed");
    }

    #[test]
    fn test_merge_headers_safe_blocks_api_key_override() {
        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), "real-key".to_string());

        let mut extra = HashMap::new();
        extra.insert("x-api-key".to_string(), "fake-key".to_string());

        merge_headers_safe(&mut headers, &extra);

        assert_eq!(headers.get("x-api-key").unwrap(), "real-key");
    }

    #[test]
    fn test_merge_headers_safe_blocks_mixed_case() {
        // Target has "Authorization" (capitalized), source tries "authorization" (lowercase)
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer real".to_string());

        let mut extra = HashMap::new();
        extra.insert("authorization".to_string(), "Bearer fake".to_string());

        merge_headers_safe(&mut headers, &extra);

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer real");
        assert!(!headers.contains_key("authorization"));
    }

    #[test]
    fn test_merge_headers_safe_blocks_api_key_mixed_case() {
        let mut headers = HashMap::new();
        headers.insert("api-key".to_string(), "real-key".to_string());

        let mut extra = HashMap::new();
        extra.insert("Api-Key".to_string(), "fake-key".to_string());

        merge_headers_safe(&mut headers, &extra);

        assert_eq!(headers.get("api-key").unwrap(), "real-key");
        assert!(!headers.contains_key("Api-Key"));
    }

    #[test]
    fn test_merge_headers_safe_allows_non_sensitive() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer token".to_string());

        let mut extra = HashMap::new();
        extra.insert("x-custom-header".to_string(), "value".to_string());
        extra.insert("user-agent".to_string(), "test/1.0".to_string());

        merge_headers_safe(&mut headers, &extra);

        assert_eq!(headers.get("x-custom-header").unwrap(), "value");
        assert_eq!(headers.get("user-agent").unwrap(), "test/1.0");
    }
}
