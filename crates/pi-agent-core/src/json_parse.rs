use serde_json::Value;

/// Attempts to parse potentially incomplete JSON during streaming.
/// Always returns a valid Value, even if the JSON is incomplete.
pub fn parse_streaming_json(partial_json: &str) -> Value {
    let trimmed = partial_json.trim();
    if trimmed.is_empty() {
        return Value::Object(Default::default());
    }

    // Try standard parsing first (fastest for complete JSON)
    if let Ok(val) = serde_json::from_str(trimmed) {
        return val;
    }

    // Try to fix incomplete JSON by closing open brackets/braces/strings
    if let Some(val) = try_fix_partial_json(trimmed) {
        return val;
    }

    Value::Object(Default::default())
}

fn try_fix_partial_json(input: &str) -> Option<Value> {
    let mut fixed = String::from(input);

    // Track open brackets and string state
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }

        if ch == '"' {
            in_string = !in_string;
            continue;
        }

        if in_string {
            continue;
        }

        match ch {
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }

    // If we're in a string, close it
    if in_string {
        fixed.push('"');
    }

    // Close open brackets/braces in reverse order
    while let Some(closer) = stack.pop() {
        fixed.push(closer);
    }

    serde_json::from_str(&fixed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let result = parse_streaming_json("");
        assert!(result.is_object());
    }

    #[test]
    fn test_complete_json() {
        let result = parse_streaming_json(r#"{"name": "test", "value": 42}"#);
        assert_eq!(result["name"], "test");
        assert_eq!(result["value"], 42);
    }

    #[test]
    fn test_incomplete_object() {
        let result = parse_streaming_json(r#"{"name": "test"#);
        assert!(result.is_object());
    }

    #[test]
    fn test_incomplete_string() {
        let result = parse_streaming_json(r#"{"name": "tes"#);
        assert!(result.is_object());
    }

    #[test]
    fn test_incomplete_array() {
        let result = parse_streaming_json(r#"{"items": [1, 2"#);
        assert!(result.is_object());
    }

    #[test]
    fn test_nested_incomplete() {
        let result = parse_streaming_json(r#"{"outer": {"inner": "val"#);
        assert!(result.is_object());
    }
}
