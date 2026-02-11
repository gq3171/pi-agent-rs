use pi_agent_core::types::Model;

use crate::resources::skills::Skill;
use crate::settings::types::Settings;

/// Options for building the system prompt.
pub struct SystemPromptOptions<'a> {
    pub model: &'a Model,
    pub settings: &'a Settings,
    pub working_dir: &'a str,
    pub skills: &'a [Skill],
    pub tool_names: &'a [String],
    pub custom_instructions: Option<&'a str>,
}

/// Build the system prompt from components.
///
/// Assembles the system prompt from:
/// - Base instructions
/// - Model-specific instructions
/// - Tool descriptions
/// - Skill descriptions
/// - Custom instructions (from settings or project config)
/// - Working directory context
pub fn build_system_prompt(options: &SystemPromptOptions<'_>) -> String {
    let mut parts = Vec::new();

    // Base instructions
    parts.push(build_base_instructions(options.model));

    // System override from settings
    if let Some(system_prompt) = &options.settings.system_prompt {
        parts.push(system_prompt.clone());
    }

    // Working directory context
    parts.push(format!("Working directory: {}", options.working_dir));

    // Available tools
    if !options.tool_names.is_empty() {
        let tools_list = options.tool_names.join(", ");
        parts.push(format!("Available tools: {tools_list}"));
    }

    // Skills
    if !options.skills.is_empty() {
        let skills_section = build_skills_section(options.skills);
        parts.push(skills_section);
    }

    // Custom instructions
    if let Some(instructions) = options.custom_instructions {
        parts.push(format!("Additional instructions:\n{instructions}"));
    }

    // Append system prompt from settings
    if let Some(append) = &options.settings.append_system_prompt {
        parts.push(append.clone());
    }

    parts.join("\n\n")
}

/// Build base instructions based on the model.
fn build_base_instructions(model: &Model) -> String {
    let model_info = format!(
        "You are a coding assistant powered by {}. You help with software engineering tasks.",
        model.name
    );
    model_info
}

/// Build the skills section of the system prompt.
fn build_skills_section(skills: &[Skill]) -> String {
    let mut section = String::from("Available skills:\n");
    for skill in skills {
        section.push_str(&format!("- {}", skill.name));
        if let Some(desc) = &skill.description {
            section.push_str(&format!(": {desc}"));
        }
        section.push('\n');
    }
    section
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::{Model, ModelCost};

    fn test_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            api: "test".to_string(),
            provider: "test".to_string(),
            base_url: String::new(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: 100_000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn test_build_system_prompt_basic() {
        let model = test_model();
        let settings = Settings::default();

        let prompt = build_system_prompt(&SystemPromptOptions {
            model: &model,
            settings: &settings,
            working_dir: "/home/user/project",
            skills: &[],
            tool_names: &["bash".to_string(), "read".to_string()],
            custom_instructions: None,
        });

        assert!(prompt.contains("Test Model"));
        assert!(prompt.contains("/home/user/project"));
        assert!(prompt.contains("bash, read"));
    }

    #[test]
    fn test_build_system_prompt_with_skills() {
        let model = test_model();
        let settings = Settings::default();

        let skills = vec![Skill {
            name: "greet".to_string(),
            description: Some("Greet the user".to_string()),
            allowed_tools: vec![],
            content: String::new(),
            source: String::new(),
        }];

        let prompt = build_system_prompt(&SystemPromptOptions {
            model: &model,
            settings: &settings,
            working_dir: "/tmp",
            skills: &skills,
            tool_names: &[],
            custom_instructions: None,
        });

        assert!(prompt.contains("greet"));
        assert!(prompt.contains("Greet the user"));
    }

    #[test]
    fn test_build_system_prompt_with_custom_instructions() {
        let model = test_model();
        let settings = Settings::default();

        let prompt = build_system_prompt(&SystemPromptOptions {
            model: &model,
            settings: &settings,
            working_dir: "/tmp",
            skills: &[],
            tool_names: &[],
            custom_instructions: Some("Always use Rust"),
        });

        assert!(prompt.contains("Always use Rust"));
    }
}
