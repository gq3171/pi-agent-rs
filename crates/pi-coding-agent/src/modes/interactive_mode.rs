use std::io::{self, Write};

use crate::agent_session::session::{AgentSession, PromptOptions};
use crate::compaction::compaction::CompactionSettings;
use crate::error::CodingAgentError;
use crate::slash_commands::builtin_slash_commands;
use pi_agent_core::agent_types::AgentMessage;
use pi_agent_core::types::{ContentBlock, Message};

#[derive(Debug, Clone)]
pub struct ScopedModelConfig {
    pub provider: String,
    pub model_id: String,
    pub thinking_level: Option<String>,
}

fn find_scoped_model<'a>(
    scoped_models: &'a [ScopedModelConfig],
    query: &str,
) -> Option<&'a ScopedModelConfig> {
    let query = query.trim();
    if query.is_empty() {
        return None;
    }

    if let Some((provider, model_id)) = query.split_once('/') {
        return scoped_models.iter().find(|item| {
            item.provider.eq_ignore_ascii_case(provider)
                && item.model_id.eq_ignore_ascii_case(model_id)
        });
    }

    let mut matches = scoped_models
        .iter()
        .filter(|item| item.model_id.eq_ignore_ascii_case(query));
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(first)
}

fn print_last_assistant(messages: &[AgentMessage]) {
    let Some(assistant) = messages.iter().rev().find_map(|m| match m {
        AgentMessage::Llm(Message::Assistant(msg)) => Some(msg),
        _ => None,
    }) else {
        return;
    };

    for block in &assistant.content {
        if let ContentBlock::Text(text) = block {
            println!("{}", text.text);
        }
    }
}

#[derive(Debug, Clone)]
pub struct InteractiveModeOptions {
    pub prompt: String,
    pub scoped_models: Vec<ScopedModelConfig>,
}

impl Default for InteractiveModeOptions {
    fn default() -> Self {
        Self {
            prompt: "pi> ".to_string(),
            scoped_models: Vec::new(),
        }
    }
}

pub struct InteractiveMode {
    options: InteractiveModeOptions,
}

impl InteractiveMode {
    pub fn new(options: InteractiveModeOptions) -> Self {
        Self { options }
    }

    pub async fn run(&self, session: &mut AgentSession) -> Result<(), CodingAgentError> {
        println!("Interactive mode started. 输入 /help 查看命令。");

        loop {
            print!("{}", self.options.prompt);
            io::stdout()
                .flush()
                .map_err(|e| CodingAgentError::Other(e.to_string()))?;

            let mut line = String::new();
            let bytes = io::stdin()
                .read_line(&mut line)
                .map_err(|e| CodingAgentError::Other(e.to_string()))?;
            if bytes == 0 {
                break;
            }

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            if input.starts_with('/') {
                let mut parts = input.split_whitespace();
                let command = parts.next().unwrap_or("");
                match command {
                    "/quit" | "/exit" => break,
                    "/help" | "/hotkeys" => {
                        println!("可用命令:");
                        for cmd in builtin_slash_commands() {
                            println!("  /{} - {}", cmd.name, cmd.description.unwrap_or_default());
                        }
                        continue;
                    }
                    "/session" => {
                        let stats = session.get_stats();
                        let context_usage = session.get_context_usage();
                        let context_str = match context_usage {
                            Some(usage) => match usage.tokens {
                                Some(tokens) => format!(
                                    "{} / {} ({:.1}%)",
                                    tokens,
                                    usage.context_window,
                                    usage.percent.unwrap_or(0.0)
                                ),
                                None => format!("? / {}", usage.context_window),
                            },
                            None => "n/a".to_string(),
                        };
                        println!(
                            "session_id={:?}, messages={}, turns={}, estimated_tokens={}, context_usage={}",
                            stats.session_id,
                            stats.message_count,
                            stats.turn_count,
                            stats.estimated_tokens,
                            context_str
                        );
                        continue;
                    }
                    "/model" => {
                        if let Some(query) = parts.next() {
                            match query {
                                "next" | "prev" => {
                                    if self.options.scoped_models.is_empty() {
                                        println!(
                                            "当前没有可轮转模型范围（可通过 --models 配置）。"
                                        );
                                        continue;
                                    }

                                    let current =
                                        session.model().map(|m| (m.provider.clone(), m.id.clone()));
                                    let current_index = current.and_then(|(provider, model_id)| {
                                        self.options.scoped_models.iter().position(|item| {
                                            item.provider == provider && item.model_id == model_id
                                        })
                                    });
                                    let len = self.options.scoped_models.len();
                                    let next_index = if query == "next" {
                                        current_index.map(|idx| (idx + 1) % len).unwrap_or(0)
                                    } else {
                                        current_index
                                            .map(|idx| if idx == 0 { len - 1 } else { idx - 1 })
                                            .unwrap_or(0)
                                    };
                                    let target = &self.options.scoped_models[next_index];
                                    let selected = session
                                        .model_registry()
                                        .find_by_provider(&target.provider, &target.model_id)
                                        .cloned();
                                    if let Some(model) = selected {
                                        session.set_model(model.clone());
                                        if let Some(level) = &target.thinking_level {
                                            session.set_thinking_level_str(level);
                                        }
                                        println!("已切换模型: {}/{}", model.provider, model.id);
                                    } else {
                                        println!(
                                            "模型已不可用: {}/{}",
                                            target.provider, target.model_id
                                        );
                                    }
                                }
                                "list" => {
                                    if self.options.scoped_models.is_empty() {
                                        println!(
                                            "当前没有可轮转模型范围（可通过 --models 配置）。"
                                        );
                                    } else {
                                        println!("模型轮转范围:");
                                        for item in &self.options.scoped_models {
                                            if let Some(level) = &item.thinking_level {
                                                println!(
                                                    "  {}/{}:{}",
                                                    item.provider, item.model_id, level
                                                );
                                            } else {
                                                println!("  {}/{}", item.provider, item.model_id);
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let selected = if self.options.scoped_models.is_empty() {
                                        session.model_registry().find(query).cloned()
                                    } else {
                                        find_scoped_model(&self.options.scoped_models, query)
                                            .and_then(|item| {
                                                session
                                                    .model_registry()
                                                    .find_by_provider(
                                                        &item.provider,
                                                        &item.model_id,
                                                    )
                                                    .cloned()
                                            })
                                    };
                                    if let Some(model) = selected {
                                        let provider = model.provider.clone();
                                        let model_id = model.id.clone();
                                        session.set_model(model);
                                        println!("已切换模型: {provider}/{model_id}");
                                    } else if self.options.scoped_models.is_empty() {
                                        println!("未找到模型: {query}");
                                    } else {
                                        println!("未找到模型（当前 scope 内）: {query}");
                                    }
                                }
                            }
                        } else {
                            if !self.options.scoped_models.is_empty() {
                                println!("提示: /model next | /model prev | /model list");
                                println!("可用模型（scope）:");
                                for item in &self.options.scoped_models {
                                    println!("  {}/{}", item.provider, item.model_id);
                                }
                                continue;
                            }
                            println!("可用模型:");
                            for model in session.model_registry().all_models().iter().take(50) {
                                println!("  {}/{}", model.provider, model.id);
                            }
                        }
                        continue;
                    }
                    "/new" => {
                        session.reset_session();
                        println!("已创建新会话上下文。");
                        continue;
                    }
                    "/compact" => {
                        match session.compact(Some(&CompactionSettings::default())).await {
                            Ok(result) => {
                                println!(
                                    "compacted: messages {} -> {}, tokens {} -> {}",
                                    result.messages_before,
                                    result.messages_after,
                                    result.tokens_before,
                                    result.tokens_after
                                );
                            }
                            Err(e) => println!("compact 失败: {e}"),
                        }
                        continue;
                    }
                    "/reload" => {
                        println!("资源重载入口已预留（当前版本需重建 session 生效）。");
                        continue;
                    }
                    _ => {
                        println!("未知命令: {command}");
                        continue;
                    }
                }
            }

            session.prompt(input, PromptOptions::default()).await?;
            print_last_assistant(session.messages());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scoped_models() -> Vec<ScopedModelConfig> {
        vec![
            ScopedModelConfig {
                provider: "anthropic".to_string(),
                model_id: "claude-sonnet".to_string(),
                thinking_level: None,
            },
            ScopedModelConfig {
                provider: "openai".to_string(),
                model_id: "gpt-4o".to_string(),
                thinking_level: None,
            },
            ScopedModelConfig {
                provider: "test-provider".to_string(),
                model_id: "shared-name".to_string(),
                thinking_level: None,
            },
            ScopedModelConfig {
                provider: "other-provider".to_string(),
                model_id: "shared-name".to_string(),
                thinking_level: None,
            },
        ]
    }

    #[test]
    fn test_find_scoped_model_by_provider_and_id() {
        let models = sample_scoped_models();
        let found = find_scoped_model(&models, "openai/gpt-4o");
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.provider, "openai");
        assert_eq!(found.model_id, "gpt-4o");
    }

    #[test]
    fn test_find_scoped_model_by_unique_id() {
        let models = sample_scoped_models();
        let found = find_scoped_model(&models, "claude-sonnet");
        assert!(found.is_some());
        assert_eq!(found.unwrap().provider, "anthropic");
    }

    #[test]
    fn test_find_scoped_model_returns_none_for_ambiguous_id() {
        let models = sample_scoped_models();
        let found = find_scoped_model(&models, "shared-name");
        assert!(found.is_none());
    }
}
