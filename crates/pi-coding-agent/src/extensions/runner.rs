use std::collections::HashMap;

use pi_agent_core::agent_types::AgentToolResult;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::extensions::runtime::ExtensionRuntime;
use crate::extensions::types::{
    ContextEvent, Extension, ExtensionContext, ToolCallDecision, ToolDefinition,
};

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn boxed_error(message: impl Into<String>) -> DynError {
    Box::new(std::io::Error::other(message.into()))
}

#[derive(Debug, Clone)]
struct RegisteredTool {
    owner_index: usize,
    definition: ToolDefinition,
}

/// Runs extensions, dispatches events, and routes extension-provided tools.
pub struct ExtensionRunner {
    context: ExtensionContext,
    runtime: ExtensionRuntime,
    extensions: Vec<Mutex<Box<dyn Extension + Send + Sync>>>,
    tools: HashMap<String, RegisteredTool>,
}

impl ExtensionRunner {
    pub fn new(context: ExtensionContext) -> Self {
        Self {
            context,
            runtime: ExtensionRuntime::default(),
            extensions: Vec::new(),
            tools: HashMap::new(),
        }
    }

    pub fn runtime(&self) -> &ExtensionRuntime {
        &self.runtime
    }

    pub async fn add_extension(
        &mut self,
        mut extension: Box<dyn Extension + Send + Sync>,
    ) -> Result<(), String> {
        extension
            .init(self.context.clone())
            .await
            .map_err(|e| format!("Failed to initialize extension {}: {e}", extension.name()))?;

        let owner_index = self.extensions.len();
        for definition in extension.tools() {
            if self.tools.contains_key(&definition.name) {
                return Err(format!(
                    "Duplicate extension tool '{}' from extension '{}'",
                    definition.name,
                    extension.name()
                ));
            }
            self.tools.insert(
                definition.name.clone(),
                RegisteredTool {
                    owner_index,
                    definition,
                },
            );
        }

        self.extensions.push(Mutex::new(extension));
        Ok(())
    }

    pub fn tool_definition(&self, tool_name: &str) -> Option<ToolDefinition> {
        self.tools.get(tool_name).map(|t| t.definition.clone())
    }

    pub fn registered_tools(&self) -> Vec<ToolDefinition> {
        let mut tools: Vec<ToolDefinition> = self
            .tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    pub async fn execute_registered_tool(
        &self,
        tool_name: &str,
        params: Value,
    ) -> Result<Value, DynError> {
        let Some(registered) = self.tools.get(tool_name) else {
            return Err(boxed_error(format!(
                "Extension tool not found: {tool_name}"
            )));
        };

        let Some(extension_lock) = self.extensions.get(registered.owner_index) else {
            return Err(boxed_error(format!(
                "Extension owner not found for tool: {tool_name}"
            )));
        };

        let extension = extension_lock.lock().await;
        extension.handle_tool_call(tool_name, params).await
    }

    pub async fn emit_event(&self, event: ContextEvent) -> Result<(), DynError> {
        for extension_lock in &self.extensions {
            let extension = extension_lock.lock().await;
            extension.on_event(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn before_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        params: &Value,
    ) -> Result<(), DynError> {
        self.emit_event(ContextEvent::ToolCall {
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            input: params.clone(),
        })
        .await?;

        for extension_lock in &self.extensions {
            let extension = extension_lock.lock().await;
            match extension
                .on_tool_call(tool_name, tool_call_id, params)
                .await?
            {
                ToolCallDecision::Allow => {}
                ToolCallDecision::Block { reason } => {
                    return Err(boxed_error(reason.unwrap_or_else(|| {
                        format!("Tool call blocked by extension: {tool_name}")
                    })));
                }
            }
        }

        Ok(())
    }

    pub async fn after_tool_result(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        result: &AgentToolResult,
        is_error: bool,
    ) -> Result<Option<AgentToolResult>, DynError> {
        self.emit_event(ContextEvent::ToolResult {
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            is_error,
        })
        .await?;

        let mut current = result.clone();
        let mut replaced = false;

        for extension_lock in &self.extensions {
            let extension = extension_lock.lock().await;
            if let Some(next) = extension
                .on_tool_result(tool_name, tool_call_id, &current, is_error)
                .await?
            {
                current = next;
                replaced = true;
            }
        }

        Ok(replaced.then_some(current))
    }

    pub async fn shutdown(&self) -> Result<(), DynError> {
        for extension_lock in &self.extensions {
            let mut extension = extension_lock.lock().await;
            extension.shutdown().await?;
        }
        Ok(())
    }
}
