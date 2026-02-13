use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::agent_types::{AgentTool, AgentToolResult};
use pi_agent_core::types::{ContentBlock, TextContent, Tool};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::ToolDefinition;

fn text_result(text: impl Into<String>) -> AgentToolResult {
    AgentToolResult {
        content: vec![ContentBlock::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })],
        details: None,
    }
}

fn extension_value_to_tool_result(value: Value) -> AgentToolResult {
    if let Some(obj) = value.as_object() {
        let details = obj.get("details").cloned().or_else(|| Some(value.clone()));

        if let Some(content) = obj.get("content") {
            if let Some(text) = content.as_str() {
                return AgentToolResult {
                    content: text_result(text).content,
                    details,
                };
            }

            if let Ok(content_blocks) = serde_json::from_value::<Vec<ContentBlock>>(content.clone())
            {
                return AgentToolResult {
                    content: content_blocks,
                    details,
                };
            }
        }
    }

    AgentToolResult {
        content: text_result(
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        )
        .content,
        details: Some(value),
    }
}

struct WrappedTool {
    inner: Arc<dyn AgentTool>,
    runner: Arc<ExtensionRunner>,
}

#[async_trait]
impl AgentTool for WrappedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn label(&self) -> &str {
        self.inner.label()
    }

    fn definition(&self) -> &Tool {
        self.inner.definition()
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        cancel: CancellationToken,
        on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        self.runner
            .before_tool_call(self.name(), tool_call_id, &params)
            .await?;

        match self
            .inner
            .execute(tool_call_id, params.clone(), cancel, on_update)
            .await
        {
            Ok(result) => {
                if let Some(replaced) = self
                    .runner
                    .after_tool_result(self.name(), tool_call_id, &result, false)
                    .await?
                {
                    return Ok(replaced);
                }
                Ok(result)
            }
            Err(err) => {
                let _ = self
                    .runner
                    .after_tool_result(
                        self.name(),
                        tool_call_id,
                        &text_result(err.to_string()),
                        true,
                    )
                    .await;
                Err(err)
            }
        }
    }
}

struct ExtensionTool {
    name: String,
    label: String,
    definition: Tool,
    runner: Arc<ExtensionRunner>,
}

impl ExtensionTool {
    fn from_definition(definition: ToolDefinition, runner: Arc<ExtensionRunner>) -> Self {
        let tool = Tool {
            name: definition.name.clone(),
            description: definition.description.clone(),
            parameters: definition.parameters.clone(),
        };
        Self {
            name: definition.name,
            label: definition.label,
            definition: tool,
            runner,
        }
    }
}

#[async_trait]
impl AgentTool for ExtensionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn definition(&self) -> &Tool {
        &self.definition
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        self.runner
            .before_tool_call(self.name(), tool_call_id, &params)
            .await?;

        let value = self
            .runner
            .execute_registered_tool(self.name(), params)
            .await?;
        let result = extension_value_to_tool_result(value);

        if let Some(replaced) = self
            .runner
            .after_tool_result(self.name(), tool_call_id, &result, false)
            .await?
        {
            return Ok(replaced);
        }

        Ok(result)
    }
}

/// Wrap one built-in tool with extension hooks.
pub fn wrap_tool_with_extensions(
    tool: Arc<dyn AgentTool>,
    runner: Arc<ExtensionRunner>,
) -> Arc<dyn AgentTool> {
    Arc::new(WrappedTool {
        inner: tool,
        runner,
    })
}

/// Wrap built-in tools with extension hooks.
pub fn wrap_tools_with_extensions(
    tools: Vec<Arc<dyn AgentTool>>,
    runner: Arc<ExtensionRunner>,
) -> Vec<Arc<dyn AgentTool>> {
    tools
        .into_iter()
        .map(|tool| wrap_tool_with_extensions(tool, runner.clone()))
        .collect()
}

/// Convert all extension-registered tool definitions into callable AgentTools.
pub fn create_extension_tools(runner: Arc<ExtensionRunner>) -> Vec<Arc<dyn AgentTool>> {
    runner
        .registered_tools()
        .into_iter()
        .map(|definition| {
            Arc::new(ExtensionTool::from_definition(definition, runner.clone()))
                as Arc<dyn AgentTool>
        })
        .collect()
}

/// Build a custom message payload for extension-originated session entries.
pub fn extension_custom_entry(tool_name: &str, payload: Value) -> Value {
    json!({
        "type": "extension_tool_result",
        "toolName": tool_name,
        "payload": payload
    })
}
