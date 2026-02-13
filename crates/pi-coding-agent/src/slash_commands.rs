/// Source of a slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommandSource {
    Extension,
    Prompt,
    Skill,
    Builtin,
}

/// Location where slash command was discovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommandLocation {
    User,
    Project,
    Path,
}

/// Slash command metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandInfo {
    pub name: String,
    pub description: Option<String>,
    pub source: SlashCommandSource,
    pub location: Option<SlashCommandLocation>,
    pub path: Option<String>,
}

/// Built-in slash commands aligned with pi-mono.
pub fn builtin_slash_commands() -> Vec<SlashCommandInfo> {
    let commands = [
        ("settings", "Open settings menu"),
        ("model", "Select model"),
        ("scoped-models", "Enable/disable scoped models"),
        ("export", "Export session"),
        ("share", "Share session"),
        ("copy", "Copy last assistant message"),
        ("name", "Set session display name"),
        ("session", "Show session info and stats"),
        ("changelog", "Show changelog"),
        ("hotkeys", "Show keyboard shortcuts"),
        ("fork", "Create a fork from a previous message"),
        ("tree", "Navigate session tree"),
        ("login", "Login provider"),
        ("logout", "Logout provider"),
        ("new", "Start a new session"),
        ("compact", "Manually compact context"),
        ("resume", "Resume another session"),
        ("reload", "Reload extensions/skills/prompts/themes"),
        ("quit", "Quit"),
    ];

    commands
        .into_iter()
        .map(|(name, description)| SlashCommandInfo {
            name: name.to_string(),
            description: Some(description.to_string()),
            source: SlashCommandSource::Builtin,
            location: None,
            path: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_slash_commands_contains_model() {
        let commands = builtin_slash_commands();
        assert!(commands.iter().any(|cmd| cmd.name == "model"));
    }
}
