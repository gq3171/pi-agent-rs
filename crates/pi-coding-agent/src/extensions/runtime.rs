use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde_json::Value;

#[derive(Debug, Default)]
struct ExtensionRuntimeState {
    session_name: Option<String>,
    labels: HashMap<String, Option<String>>,
    active_tools: HashSet<String>,
    flags: HashMap<String, Value>,
    thinking_level: Option<String>,
}

/// Shared mutable runtime state for extensions.
#[derive(Debug, Clone, Default)]
pub struct ExtensionRuntime {
    state: Arc<RwLock<ExtensionRuntimeState>>,
}

impl ExtensionRuntime {
    pub fn session_name(&self) -> Option<String> {
        self.state.read().ok().and_then(|s| s.session_name.clone())
    }

    pub fn set_session_name(&self, name: Option<String>) {
        if let Ok(mut s) = self.state.write() {
            s.session_name = name;
        }
    }

    pub fn set_label(&self, entry_id: String, label: Option<String>) {
        if let Ok(mut s) = self.state.write() {
            s.labels.insert(entry_id, label);
        }
    }

    pub fn label(&self, entry_id: &str) -> Option<Option<String>> {
        self.state
            .read()
            .ok()
            .and_then(|s| s.labels.get(entry_id).cloned())
    }

    pub fn active_tools(&self) -> Vec<String> {
        self.state
            .read()
            .ok()
            .map(|s| {
                let mut tools: Vec<String> = s.active_tools.iter().cloned().collect();
                tools.sort();
                tools
            })
            .unwrap_or_default()
    }

    pub fn set_active_tools(&self, tool_names: impl IntoIterator<Item = String>) {
        if let Ok(mut s) = self.state.write() {
            s.active_tools = tool_names.into_iter().collect();
        }
    }

    pub fn get_flag(&self, name: &str) -> Option<Value> {
        self.state
            .read()
            .ok()
            .and_then(|s| s.flags.get(name).cloned())
    }

    pub fn set_flag(&self, name: impl Into<String>, value: Value) {
        if let Ok(mut s) = self.state.write() {
            s.flags.insert(name.into(), value);
        }
    }

    pub fn thinking_level(&self) -> Option<String> {
        self.state
            .read()
            .ok()
            .and_then(|s| s.thinking_level.clone())
    }

    pub fn set_thinking_level(&self, level: Option<String>) {
        if let Ok(mut s) = self.state.write() {
            s.thinking_level = level;
        }
    }
}
