pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod find;
pub mod grep;
pub mod ls;
pub mod path_utils;
pub mod read;
pub mod truncate;
pub mod write;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use pi_agent_core::agent_types::AgentTool;

use self::bash::BashTool;
use self::edit::EditTool;
use self::find::FindTool;
use self::grep::GrepTool;
use self::ls::LsTool;
use self::read::ReadTool;
use self::write::WriteTool;

/// Create the read tool.
pub fn create_read_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(ReadTool::with_default_reader(working_dir.to_path_buf()))
}

/// Create the write tool.
pub fn create_write_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(WriteTool::with_default_writer(working_dir.to_path_buf()))
}

/// Create the edit tool.
pub fn create_edit_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(EditTool::with_default_editor(working_dir.to_path_buf()))
}

/// Create the bash tool.
pub fn create_bash_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(BashTool::with_default_executor(working_dir.to_path_buf()))
}

/// Create the find tool.
pub fn create_find_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(FindTool::new(working_dir.to_path_buf()))
}

/// Create the grep tool.
pub fn create_grep_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(GrepTool::new(working_dir.to_path_buf()))
}

/// Create the ls tool.
pub fn create_ls_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    Arc::new(LsTool::new(working_dir.to_path_buf()))
}

/// Create the core coding tools (bash, read, write, edit).
pub fn create_coding_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    vec![
        create_read_tool(working_dir),
        create_bash_tool(working_dir),
        create_write_tool(working_dir),
        create_edit_tool(working_dir),
    ]
}

/// Create read-only tools (read, grep, find, ls).
pub fn create_read_only_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    vec![
        create_read_tool(working_dir),
        create_grep_tool(working_dir),
        create_find_tool(working_dir),
        create_ls_tool(working_dir),
    ]
}

/// Create all built-in tools keyed by name.
pub fn create_all_tools(working_dir: &Path) -> HashMap<String, Arc<dyn AgentTool>> {
    HashMap::from([
        ("read".to_string(), create_read_tool(working_dir)),
        ("bash".to_string(), create_bash_tool(working_dir)),
        ("edit".to_string(), create_edit_tool(working_dir)),
        ("write".to_string(), create_write_tool(working_dir)),
        ("grep".to_string(), create_grep_tool(working_dir)),
        ("find".to_string(), create_find_tool(working_dir)),
        ("ls".to_string(), create_ls_tool(working_dir)),
    ])
}

/// Aliases for backward compatibility with OpenClaw imports.
pub fn read_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    create_read_tool(working_dir)
}

pub fn coding_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    create_coding_tools(working_dir)
}

pub fn grep_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    create_grep_tool(working_dir)
}

pub fn find_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    create_find_tool(working_dir)
}

pub fn ls_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    create_ls_tool(working_dir)
}

pub fn all_tools(working_dir: &Path) -> HashMap<String, Arc<dyn AgentTool>> {
    create_all_tools(working_dir)
}
