use std::collections::HashMap;

use pi_agent_core::types::ThinkingLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Text,
    Json,
    Rpc,
}

#[derive(Debug, Clone, Default)]
pub struct Args {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<String>,
    pub thinking: Option<String>,
    pub continue_session: bool,
    pub resume: bool,
    pub help: bool,
    pub version: bool,
    pub mode: Option<Mode>,
    pub no_session: bool,
    pub session: Option<String>,
    pub session_dir: Option<String>,
    pub models: Option<Vec<String>>,
    pub no_tools: bool,
    pub tools: Option<Vec<String>>,
    pub extensions: Vec<String>,
    pub no_extensions: bool,
    pub print: bool,
    pub export: Option<String>,
    pub no_skills: bool,
    pub skills: Vec<String>,
    pub prompt_templates: Vec<String>,
    pub no_prompt_templates: bool,
    pub themes: Vec<String>,
    pub no_themes: bool,
    pub list_models: Option<String>,
    pub verbose: bool,
    pub messages: Vec<String>,
    pub file_args: Vec<String>,
    pub unknown_flags: HashMap<String, String>,
}

pub fn is_valid_thinking_level(level: &str) -> bool {
    matches!(
        level,
        "off" | "minimal" | "low" | "medium" | "high" | "xhigh"
    )
}

pub fn parse_thinking_level(level: &str) -> Option<Option<ThinkingLevel>> {
    match level {
        "off" => Some(None),
        "minimal" => Some(Some(ThinkingLevel::Minimal)),
        "low" => Some(Some(ThinkingLevel::Low)),
        "medium" => Some(Some(ThinkingLevel::Medium)),
        "high" => Some(Some(ThinkingLevel::High)),
        "xhigh" => Some(Some(ThinkingLevel::Xhigh)),
        _ => None,
    }
}

pub fn parse_args(args: &[String], extension_flags: Option<&HashMap<String, String>>) -> Args {
    let mut result = Args::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        match arg.as_str() {
            "--help" | "-h" => result.help = true,
            "--version" | "-v" => result.version = true,
            "--mode" if i + 1 < args.len() => {
                i += 1;
                result.mode = match args[i].as_str() {
                    "text" => Some(Mode::Text),
                    "json" => Some(Mode::Json),
                    "rpc" => Some(Mode::Rpc),
                    _ => None,
                };
            }
            "--continue" | "-c" => result.continue_session = true,
            "--resume" | "-r" => result.resume = true,
            "--provider" if i + 1 < args.len() => {
                i += 1;
                result.provider = Some(args[i].clone());
            }
            "--model" if i + 1 < args.len() => {
                i += 1;
                result.model = Some(args[i].clone());
            }
            "--api-key" if i + 1 < args.len() => {
                i += 1;
                result.api_key = Some(args[i].clone());
            }
            "--system-prompt" if i + 1 < args.len() => {
                i += 1;
                result.system_prompt = Some(args[i].clone());
            }
            "--append-system-prompt" if i + 1 < args.len() => {
                i += 1;
                result.append_system_prompt = Some(args[i].clone());
            }
            "--thinking" if i + 1 < args.len() => {
                i += 1;
                let level = args[i].clone();
                if is_valid_thinking_level(&level) {
                    result.thinking = Some(level);
                }
            }
            "--no-session" => result.no_session = true,
            "--session" if i + 1 < args.len() => {
                i += 1;
                result.session = Some(args[i].clone());
            }
            "--session-dir" if i + 1 < args.len() => {
                i += 1;
                result.session_dir = Some(args[i].clone());
            }
            "--models" if i + 1 < args.len() => {
                i += 1;
                let models = args[i]
                    .split(',')
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<_>>();
                result.models = Some(models);
            }
            "--no-tools" => result.no_tools = true,
            "--tools" if i + 1 < args.len() => {
                i += 1;
                let tools = args[i]
                    .split(',')
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<_>>();
                result.tools = Some(tools);
            }
            "--extension" | "-e" if i + 1 < args.len() => {
                i += 1;
                result.extensions.push(args[i].clone());
            }
            "--no-extensions" | "-ne" => result.no_extensions = true,
            "--skill" if i + 1 < args.len() => {
                i += 1;
                result.skills.push(args[i].clone());
            }
            "--no-skills" | "-ns" => result.no_skills = true,
            "--prompt-template" if i + 1 < args.len() => {
                i += 1;
                result.prompt_templates.push(args[i].clone());
            }
            "--no-prompt-templates" | "-np" => result.no_prompt_templates = true,
            "--theme" if i + 1 < args.len() => {
                i += 1;
                result.themes.push(args[i].clone());
            }
            "--no-themes" => result.no_themes = true,
            "--list-models" => {
                if i + 1 < args.len()
                    && !args[i + 1].starts_with('-')
                    && !args[i + 1].starts_with('@')
                {
                    i += 1;
                    result.list_models = Some(args[i].clone());
                } else {
                    result.list_models = Some(String::new());
                }
            }
            "--print" | "-p" => result.print = true,
            "--export" if i + 1 < args.len() => {
                i += 1;
                result.export = Some(args[i].clone());
            }
            "--verbose" => result.verbose = true,
            v if v.starts_with('@') => {
                result.file_args.push(v.trim_start_matches('@').to_string());
            }
            v if v.starts_with("--") => {
                let key = v.trim_start_matches("--");
                if let Some(flags) = extension_flags {
                    if flags.contains_key(key) {
                        let value = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                            i += 1;
                            args[i].clone()
                        } else {
                            "true".to_string()
                        };
                        result.unknown_flags.insert(key.to_string(), value);
                    }
                }
            }
            v if !v.starts_with('-') => result.messages.push(v.to_string()),
            _ => {}
        }

        i += 1;
    }

    result
}

pub fn print_help(bin_name: &str) {
    println!(
        "{bin_name} - AI coding assistant\n\n\
         Usage:\n  {bin_name} [options] [@files...] [messages...]\n\n\
         Commands:\n  {bin_name} install <source> [-l]\n  {bin_name} remove <source> [-l]\n  {bin_name} update [source]\n  {bin_name} list\n  {bin_name} config\n\n\
         Options:\n  --mode <text|json|rpc>\n  --continue, -c\n  --resume, -r\n  --provider <name>\n  --model <pattern>\n  --api-key <key>\n  --system-prompt <text>\n  --append-system-prompt <text>\n  --thinking <off|minimal|low|medium|high|xhigh>\n  --no-session\n  --session <id>\n  --session-dir <dir>\n  --models <patterns>\n  --no-tools\n  --tools <read,bash,...>\n  --extension, -e <path>\n  --no-extensions\n  --skill <path>\n  --no-skills\n  --prompt-template <path>\n  --no-prompt-templates\n  --theme <path>\n  --no-themes\n  --export <file>\n  --list-models [search]\n  --print, -p\n  --verbose\n  --help, -h\n  --version, -v"
    );
}
