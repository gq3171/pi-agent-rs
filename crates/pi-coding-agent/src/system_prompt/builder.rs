use pi_agent_core::types::Model;

use crate::resources::loader::ContextFile;
use crate::resources::skills::Skill;
use crate::settings::types::Settings;

/// Options for building the system prompt.
pub struct SystemPromptOptions<'a> {
    pub model: &'a Model,
    pub settings: &'a Settings,
    pub working_dir: &'a str,
    pub skills: &'a [Skill],
    pub context_files: &'a [ContextFile],
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
    let mut prompt = if let Some(custom_prompt) = &options.settings.system_prompt {
        custom_prompt.clone()
    } else {
        build_default_prompt(options.model, options.tool_names)
    };

    if let Some(instructions) = options.custom_instructions {
        prompt.push_str("\n\n");
        prompt.push_str(instructions);
    }

    if let Some(append) = &options.settings.append_system_prompt {
        prompt.push_str("\n\n");
        prompt.push_str(append);
    }

    if !options.context_files.is_empty() {
        prompt.push_str("\n\n# Project Context\n\n");
        prompt.push_str("Project-specific instructions and guidelines:\n\n");
        for file in options.context_files {
            prompt.push_str("## ");
            prompt.push_str(&file.path);
            prompt.push_str("\n\n");
            prompt.push_str(&file.content);
            prompt.push_str("\n\n");
        }
    }

    if options.tool_names.iter().any(|t| t == "read") && !options.skills.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&build_skills_section(options.skills));
    }

    let date_time = chrono::Local::now().format("%A, %B %d, %Y %H:%M:%S %Z");
    prompt.push_str("\nCurrent date and time: ");
    prompt.push_str(&date_time.to_string());
    prompt.push_str("\nCurrent working directory: ");
    prompt.push_str(options.working_dir);

    prompt
}

/// Build base instructions based on the model.
fn build_base_instructions(model: &Model) -> String {
    let model_info = format!(
        "You are a coding assistant powered by {}. You help with software engineering tasks.",
        model.name
    );
    model_info
}

fn build_default_prompt(model: &Model, tools: &[String]) -> String {
    let tools_list = if tools.is_empty() {
        "(none)".to_string()
    } else {
        tools
            .iter()
            .map(|tool| {
                let description = match tool.as_str() {
                    "read" => "Read file contents",
                    "bash" => "Execute shell commands",
                    "edit" => "Make targeted file edits",
                    "write" => "Create or overwrite files",
                    "grep" => "Search file contents for patterns",
                    "find" => "Find files by glob pattern",
                    "ls" => "List directory contents",
                    _ => "Custom tool",
                };
                format!("- {tool}: {description}")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let guidelines = [
        "Be concise in your responses.",
        "Show file paths clearly when working with files.",
        "Prefer structured tools over shell commands when available.",
        "Read files before editing them.",
    ]
    .join("\n- ");

    format!(
        "{}\n\nAvailable tools:\n{}\n\nGuidelines:\n- {}",
        build_base_instructions(model),
        tools_list,
        guidelines
    )
}

/// Build the skills section of the system prompt.
fn build_skills_section(skills: &[Skill]) -> String {
    let mut section = String::from(
        "Available skills. Use read to open the skill file when the task matches:\n<available_skills>\n",
    );
    for skill in skills {
        section.push_str("  <skill>\n");
        section.push_str(&format!("    <name>{}</name>\n", skill.name));
        if let Some(desc) = &skill.description {
            section.push_str(&format!("    <description>{desc}</description>\n"));
        }
        section.push_str(&format!("    <location>{}</location>\n", skill.source));
        section.push_str("  </skill>\n");
    }
    section.push_str("</available_skills>");
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
            context_files: &[],
            tool_names: &["bash".to_string(), "read".to_string()],
            custom_instructions: None,
        });

        assert!(prompt.contains("Test Model"));
        assert!(prompt.contains("/home/user/project"));
        assert!(prompt.contains("- bash:"));
        assert!(prompt.contains("- read:"));
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
            context_files: &[],
            tool_names: &["read".to_string()],
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
            context_files: &[],
            tool_names: &[],
            custom_instructions: Some("Always use Rust"),
        });

        assert!(prompt.contains("Always use Rust"));
    }
}
