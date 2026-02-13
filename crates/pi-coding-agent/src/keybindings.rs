use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CodingAgentError;

/// Application-level key actions aligned with pi-mono.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AppAction {
    Interrupt,
    Clear,
    Exit,
    Suspend,
    CycleThinkingLevel,
    CycleModelForward,
    CycleModelBackward,
    SelectModel,
    ExpandTools,
    ToggleThinking,
    ToggleSessionNamedFilter,
    ExternalEditor,
    FollowUp,
    Dequeue,
    PasteImage,
    NewSession,
    Tree,
    Fork,
    Resume,
}

impl AppAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            AppAction::Interrupt => "interrupt",
            AppAction::Clear => "clear",
            AppAction::Exit => "exit",
            AppAction::Suspend => "suspend",
            AppAction::CycleThinkingLevel => "cycleThinkingLevel",
            AppAction::CycleModelForward => "cycleModelForward",
            AppAction::CycleModelBackward => "cycleModelBackward",
            AppAction::SelectModel => "selectModel",
            AppAction::ExpandTools => "expandTools",
            AppAction::ToggleThinking => "toggleThinking",
            AppAction::ToggleSessionNamedFilter => "toggleSessionNamedFilter",
            AppAction::ExternalEditor => "externalEditor",
            AppAction::FollowUp => "followUp",
            AppAction::Dequeue => "dequeue",
            AppAction::PasteImage => "pasteImage",
            AppAction::NewSession => "newSession",
            AppAction::Tree => "tree",
            AppAction::Fork => "fork",
            AppAction::Resume => "resume",
        }
    }
}

fn parse_action(name: &str) -> Option<AppAction> {
    match name {
        "interrupt" => Some(AppAction::Interrupt),
        "clear" => Some(AppAction::Clear),
        "exit" => Some(AppAction::Exit),
        "suspend" => Some(AppAction::Suspend),
        "cycleThinkingLevel" => Some(AppAction::CycleThinkingLevel),
        "cycleModelForward" => Some(AppAction::CycleModelForward),
        "cycleModelBackward" => Some(AppAction::CycleModelBackward),
        "selectModel" => Some(AppAction::SelectModel),
        "expandTools" => Some(AppAction::ExpandTools),
        "toggleThinking" => Some(AppAction::ToggleThinking),
        "toggleSessionNamedFilter" => Some(AppAction::ToggleSessionNamedFilter),
        "externalEditor" => Some(AppAction::ExternalEditor),
        "followUp" => Some(AppAction::FollowUp),
        "dequeue" => Some(AppAction::Dequeue),
        "pasteImage" => Some(AppAction::PasteImage),
        "newSession" => Some(AppAction::NewSession),
        "tree" => Some(AppAction::Tree),
        "fork" => Some(AppAction::Fork),
        "resume" => Some(AppAction::Resume),
        _ => None,
    }
}

/// Keybindings manager.
#[derive(Debug, Clone, Default)]
pub struct KeybindingsManager {
    bindings: HashMap<AppAction, Vec<String>>,
}

impl KeybindingsManager {
    pub fn with_defaults() -> Self {
        Self {
            bindings: default_app_keybindings(),
        }
    }

    pub fn load_from_file(path: &Path) -> Result<Self, CodingAgentError> {
        let mut manager = Self::with_defaults();
        if !path.exists() {
            return Ok(manager);
        }

        let content = std::fs::read_to_string(path)?;
        let user: HashMap<String, serde_json::Value> = serde_json::from_str(&content)?;
        for (key, value) in user {
            let Some(action) = parse_action(&key) else {
                continue;
            };
            let parsed = if let Some(one) = value.as_str() {
                vec![one.to_string()]
            } else if let Some(arr) = value.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_str().map(ToString::to_string))
                    .collect()
            } else {
                Vec::new()
            };
            manager.bindings.insert(action, parsed);
        }

        Ok(manager)
    }

    pub fn get_keys(&self, action: AppAction) -> &[String] {
        self.bindings.get(&action).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn matches(&self, input: &str, action: AppAction) -> bool {
        let normalized = input.trim().to_lowercase();
        self.get_keys(action)
            .iter()
            .any(|key| key.to_lowercase() == normalized)
    }

    pub fn effective_config(&self) -> HashMap<String, Vec<String>> {
        self.bindings
            .iter()
            .map(|(action, keys)| (action.as_str().to_string(), keys.clone()))
            .collect()
    }
}

/// Default app keybindings aligned with pi-mono.
pub fn default_app_keybindings() -> HashMap<AppAction, Vec<String>> {
    HashMap::from([
        (AppAction::Interrupt, vec!["escape".to_string()]),
        (AppAction::Clear, vec!["ctrl+c".to_string()]),
        (AppAction::Exit, vec!["ctrl+d".to_string()]),
        (AppAction::Suspend, vec!["ctrl+z".to_string()]),
        (AppAction::CycleThinkingLevel, vec!["shift+tab".to_string()]),
        (AppAction::CycleModelForward, vec!["ctrl+p".to_string()]),
        (
            AppAction::CycleModelBackward,
            vec!["shift+ctrl+p".to_string()],
        ),
        (AppAction::SelectModel, vec!["ctrl+l".to_string()]),
        (AppAction::ExpandTools, vec!["ctrl+o".to_string()]),
        (AppAction::ToggleThinking, vec!["ctrl+t".to_string()]),
        (
            AppAction::ToggleSessionNamedFilter,
            vec!["ctrl+n".to_string()],
        ),
        (AppAction::ExternalEditor, vec!["ctrl+g".to_string()]),
        (AppAction::FollowUp, vec!["alt+enter".to_string()]),
        (AppAction::Dequeue, vec!["alt+up".to_string()]),
        (AppAction::PasteImage, vec!["ctrl+v".to_string()]),
        (AppAction::NewSession, Vec::new()),
        (AppAction::Tree, Vec::new()),
        (AppAction::Fork, Vec::new()),
        (AppAction::Resume, Vec::new()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_keybindings() {
        let manager = KeybindingsManager::with_defaults();
        assert!(manager.matches("ctrl+p", AppAction::CycleModelForward));
        assert!(manager.matches("shift+tab", AppAction::CycleThinkingLevel));
    }
}
