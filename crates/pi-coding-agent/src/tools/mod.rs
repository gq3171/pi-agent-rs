pub mod truncate;
pub mod path_utils;
pub mod bash;
pub mod read;
pub mod write;
pub mod edit;
pub mod edit_diff;

use std::path::Path;
use std::sync::Arc;

use pi_agent_core::agent_types::AgentTool;

use self::bash::BashTool;
use self::edit::EditTool;
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

/// Create the core coding tools (bash, read, write, edit).
pub fn create_coding_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    vec![
        create_bash_tool(working_dir),
        create_read_tool(working_dir),
        create_write_tool(working_dir),
        create_edit_tool(working_dir),
    ]
}

/// Create read-only tools (just read).
pub fn create_read_only_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    vec![create_read_tool(working_dir)]
}

/// Aliases for backward compatibility with OpenClaw imports.
pub fn read_tool(working_dir: &Path) -> Arc<dyn AgentTool> {
    create_read_tool(working_dir)
}

pub fn coding_tools(working_dir: &Path) -> Vec<Arc<dyn AgentTool>> {
    create_coding_tools(working_dir)
}
