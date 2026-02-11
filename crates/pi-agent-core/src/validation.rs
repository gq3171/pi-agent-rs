use serde_json::Value;

use crate::types::{Tool, ToolCall};

/// Validates tool call arguments against the tool's JSON Schema.
/// Returns the validated arguments on success, or an error message string.
pub fn validate_tool_arguments(tool: &Tool, tool_call: &ToolCall) -> Result<Value, String> {
    let schema = &tool.parameters;

    // If schema is null or empty object, accept any arguments
    if schema.is_null() || (schema.is_object() && schema.as_object().unwrap().is_empty()) {
        return Ok(tool_call.arguments.clone());
    }

    // Use jsonschema crate for validation
    match jsonschema::options()
        .build(schema)
    {
        Ok(validator) => {
            if validator.is_valid(&tool_call.arguments) {
                Ok(tool_call.arguments.clone())
            } else {
                let errors: Vec<String> = validator
                    .iter_errors(&tool_call.arguments)
                    .map(|err| {
                        let path_str = err.instance_path.to_string();
                        let path = if path_str.is_empty() {
                            "root".to_string()
                        } else {
                            path_str
                        };
                        format!("  - {}: {}", path, err)
                    })
                    .collect();

                let error_message = format!(
                    "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
                    tool_call.name,
                    errors.join("\n"),
                    serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_default()
                );
                Err(error_message)
            }
        }
        Err(e) => {
            // Schema compilation failed - fail closed to prevent unvalidated input
            tracing::error!("Failed to compile JSON schema for tool {}: {}", tool.name, e);
            Err(format!(
                "Internal error: schema compilation failed for tool \"{}\": {}",
                tool.name, e
            ))
        }
    }
}

/// Find a tool by name and validate the tool call arguments.
pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value, String> {
    let tool = tools.iter().find(|t| t.name == tool_call.name);
    match tool {
        Some(tool) => validate_tool_arguments(tool, tool_call),
        None => Err(format!("Tool \"{}\" not found", tool_call.name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_arguments() {
        let tool = Tool {
            name: "search".to_string(),
            description: "Search for something".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: json!({"query": "rust"}),
            thought_signature: None,
        };

        let result = validate_tool_arguments(&tool, &tool_call);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_required_field() {
        let tool = Tool {
            name: "search".to_string(),
            description: "Search for something".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: json!({}),
            thought_signature: None,
        };

        let result = validate_tool_arguments(&tool, &tool_call);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Validation failed"));
    }

    #[test]
    fn test_empty_schema_accepts_all() {
        let tool = Tool {
            name: "test".to_string(),
            description: "test".to_string(),
            parameters: json!({}),
        };

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "test".to_string(),
            arguments: json!({"anything": "goes"}),
            thought_signature: None,
        };

        let result = validate_tool_arguments(&tool, &tool_call);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tool_not_found() {
        let tools: Vec<Tool> = vec![];
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "nonexistent".to_string(),
            arguments: json!({}),
            thought_signature: None,
        };

        let result = validate_tool_call(&tools, &tool_call);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }
}
