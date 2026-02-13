use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::config::paths;
use crate::error::CodingAgentError;
use crate::session::types::*;

/// Create a new file with restrictive permissions on Unix (0600).
fn create_new_restricted(path: &Path) -> std::io::Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }

    opts.open(path)
}

fn parse_timestamp_value(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(n) => n.as_i64(),
        serde_json::Value::String(s) => {
            if let Ok(ms) = s.parse::<i64>() {
                return Some(ms);
            }
            chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.timestamp_millis())
        }
        _ => None,
    }
}

/// Manages session files in JSONL format.
pub struct SessionManager {
    base_dir: PathBuf,
}

impl SessionManager {
    /// Create a new SessionManager with given base directory.
    pub fn new(base_dir: &Path) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
        }
    }

    /// Get the sessions directory.
    fn sessions_dir(&self) -> PathBuf {
        paths::sessions_dir(&self.base_dir)
    }

    /// Validate that a session ID is safe (no path traversal).
    fn validate_session_id(session_id: &str) -> Result<(), CodingAgentError> {
        if session_id.is_empty() {
            return Err(CodingAgentError::Session(
                "Session ID cannot be empty".to_string(),
            ));
        }

        if !session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(CodingAgentError::Session(format!(
                "Invalid session ID: {session_id} (only [a-zA-Z0-9_-] allowed)"
            )));
        }

        Ok(())
    }

    /// Get path for a session file.
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.jsonl"))
    }

    /// Create a new session, writing the header to disk.
    pub fn create(
        &self,
        session_id: &str,
        title: Option<&str>,
    ) -> Result<SessionHeader, CodingAgentError> {
        Self::validate_session_id(session_id)?;
        let dir = self.sessions_dir();
        paths::ensure_dir(&dir)?;

        let header = SessionHeader {
            entry_type: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: session_id.to_string(),
            timestamp: now_iso_timestamp(),
            cwd: String::new(),
            parent_session: None,
            title: title.map(ToString::to_string),
        };

        let path = self.session_path(session_id);
        let mut file = create_new_restricted(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                CodingAgentError::Session(format!("Session already exists: {session_id}"))
            } else {
                CodingAgentError::Io(e)
            }
        })?;
        let line = serde_json::to_string(&header)?;
        writeln!(file, "{line}")?;

        Ok(header)
    }

    /// Open an existing session, returning header and entries.
    pub fn open(
        &self,
        session_id: &str,
    ) -> Result<(SessionHeader, Vec<SessionEntry>), CodingAgentError> {
        Self::validate_session_id(session_id)?;
        let path = self.session_path(session_id);
        if !path.exists() {
            return Err(CodingAgentError::Session(format!(
                "Session not found: {session_id}"
            )));
        }

        let file = std::fs::File::open(&path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header_line = lines
            .next()
            .ok_or_else(|| CodingAgentError::Session("Empty session file".to_string()))??;
        let header: SessionHeader = serde_json::from_str(&header_line)?;

        if header.entry_type != "session" {
            return Err(CodingAgentError::Session(format!(
                "Invalid session header in {}",
                path.display()
            )));
        }

        let mut entries = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<SessionEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => tracing::warn!("Skipping malformed session entry: {e}"),
            }
        }

        Ok((header, entries))
    }

    /// Append an entry to an existing session file.
    pub fn append_entry(
        &self,
        session_id: &str,
        entry: &SessionEntry,
    ) -> Result<(), CodingAgentError> {
        Self::validate_session_id(session_id)?;
        let path = self.session_path(session_id);
        if !path.exists() {
            return Err(CodingAgentError::Session(format!(
                "Session not found: {session_id}"
            )));
        }

        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{line}")?;

        Ok(())
    }

    /// Append multiple entries.
    pub fn append_entries(
        &self,
        session_id: &str,
        entries: &[SessionEntry],
    ) -> Result<(), CodingAgentError> {
        Self::validate_session_id(session_id)?;
        let path = self.session_path(session_id);
        if !path.exists() {
            return Err(CodingAgentError::Session(format!(
                "Session not found: {session_id}"
            )));
        }

        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        for entry in entries {
            let line = serde_json::to_string(entry)?;
            writeln!(file, "{line}")?;
        }

        Ok(())
    }

    /// Continue the most recent session.
    pub fn continue_recent(
        &self,
    ) -> Result<Option<(SessionHeader, Vec<SessionEntry>)>, CodingAgentError> {
        let sessions = self.list()?;
        if let Some(info) = sessions.first() {
            Ok(Some(self.open(&info.session_id)?))
        } else {
            Ok(None)
        }
    }

    /// Fork a session from a specific entry.
    pub fn fork_from(
        &self,
        source_session_id: &str,
        source_entry_id: &str,
        new_session_id: &str,
    ) -> Result<(SessionHeader, Vec<SessionEntry>), CodingAgentError> {
        Self::validate_session_id(new_session_id)?;
        let (source_header, source_entries) = self.open(source_session_id)?;

        let entry_exists = source_entries.iter().any(|e| e.id() == source_entry_id);
        if !entry_exists {
            return Err(CodingAgentError::Session(format!(
                "Entry not found: {source_entry_id} in session {source_session_id}"
            )));
        }

        let mut forked_entries = Vec::new();
        for entry in &source_entries {
            forked_entries.push(entry.clone());
            if entry.id() == source_entry_id {
                break;
            }
        }

        let dir = self.sessions_dir();
        paths::ensure_dir(&dir)?;

        let header = SessionHeader {
            entry_type: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: new_session_id.to_string(),
            timestamp: now_iso_timestamp(),
            cwd: source_header.cwd,
            parent_session: Some(source_header.id),
            title: source_header.title,
        };

        let path = self.session_path(new_session_id);
        let mut file = create_new_restricted(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                CodingAgentError::Session(format!(
                    "Fork target session already exists: {new_session_id}"
                ))
            } else {
                CodingAgentError::Io(e)
            }
        })?;

        let header_line = serde_json::to_string(&header)?;
        writeln!(file, "{header_line}")?;

        for entry in &forked_entries {
            let line = serde_json::to_string(entry)?;
            writeln!(file, "{line}")?;
        }

        let fork_entry = SessionEntry::LegacyFork {
            id: SessionEntry::new_id(),
            parent_id: Some(source_entry_id.to_string()),
            timestamp: chrono::Utc::now().timestamp_millis(),
            source_session_id: source_session_id.to_string(),
            source_entry_id: source_entry_id.to_string(),
        };

        let fork_line = serde_json::to_string(&fork_entry)?;
        writeln!(file, "{fork_line}")?;
        forked_entries.push(fork_entry);

        Ok((header, forked_entries))
    }

    /// List all sessions sorted by updated_at descending.
    pub fn list(&self) -> Result<Vec<SessionInfo>, CodingAgentError> {
        let dir = self.sessions_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "jsonl") {
                if let Some(info) = self.read_session_info(&path) {
                    sessions.push(info);
                }
            }
        }

        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    /// List all session IDs.
    pub fn list_all(&self) -> Result<Vec<String>, CodingAgentError> {
        let dir = self.sessions_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "jsonl") {
                if let Some(stem) = path.file_stem() {
                    ids.push(stem.to_string_lossy().to_string());
                }
            }
        }

        Ok(ids)
    }

    /// Check if a session exists.
    pub fn exists(&self, session_id: &str) -> bool {
        Self::validate_session_id(session_id).is_ok() && self.session_path(session_id).exists()
    }

    /// Delete a session file.
    pub fn delete(&self, session_id: &str) -> Result<(), CodingAgentError> {
        Self::validate_session_id(session_id)?;
        let path = self.session_path(session_id);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Read lightweight session info from file.
    fn read_session_info(&self, path: &Path) -> Option<SessionInfo> {
        let file = std::fs::File::open(path).ok()?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header_line = lines.next()?.ok()?;
        let header: SessionHeader = serde_json::from_str(&header_line).ok()?;
        if header.entry_type != "session" {
            return None;
        }

        let mut entry_count = 0usize;
        let mut last_timestamp = header.timestamp_ms();

        for line in lines.map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }

            entry_count += 1;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(ts) = val.get("timestamp").and_then(parse_timestamp_value) {
                    last_timestamp = last_timestamp.max(ts);
                }
            }
        }

        let created_at = header.timestamp_ms();
        Some(SessionInfo {
            session_id: header.id,
            title: header.title,
            created_at,
            updated_at: last_timestamp,
            entry_count,
            parent_session_id: header.parent_session,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::{
        AssistantMessage, ContentBlock, Message, StopReason, TextContent, Usage, UserContent,
        UserMessage,
    };

    fn assistant_message() -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "Hello!".to_string(),
                text_signature: None,
            })],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }

    #[test]
    fn test_create_and_open_session() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        let header = mgr.create("test-session-1", Some("Test Session")).unwrap();
        assert_eq!(header.id, "test-session-1");
        assert_eq!(header.title, Some("Test Session".to_string()));
        assert_eq!(header.version, Some(CURRENT_SESSION_VERSION));

        let (loaded_header, entries) = mgr.open("test-session-1").unwrap();
        assert_eq!(loaded_header.id, "test-session-1");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_append_and_read_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        mgr.create("sess-1", None).unwrap();

        let user_entry = SessionEntry::Message {
            id: "e1".to_string(),
            parent_id: None,
            timestamp: now_iso_timestamp(),
            message: Message::User(UserMessage {
                content: UserContent::Text("Hello, agent!".to_string()),
                timestamp: chrono::Utc::now().timestamp_millis(),
            }),
        };
        mgr.append_entry("sess-1", &user_entry).unwrap();

        let assistant_entry = SessionEntry::Message {
            id: "e2".to_string(),
            parent_id: Some("e1".to_string()),
            timestamp: now_iso_timestamp(),
            message: assistant_message(),
        };
        mgr.append_entry("sess-1", &assistant_entry).unwrap();

        let (_, entries) = mgr.open("sess-1").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id(), "e1");
        assert_eq!(entries[1].id(), "e2");
        assert_eq!(entries[1].parent_id(), Some("e1"));
    }

    #[test]
    fn test_list_sessions() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        mgr.create("sess-a", Some("Session A")).unwrap();
        mgr.create("sess-b", Some("Session B")).unwrap();

        let sessions = mgr.list().unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn test_fork_session() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        mgr.create("original", Some("Original")).unwrap();
        let e1 = SessionEntry::LegacyUser {
            id: "e1".to_string(),
            parent_id: None,
            timestamp: 1000,
            content: "First prompt".to_string(),
        };
        let e2 = SessionEntry::LegacyAssistant {
            id: "e2".to_string(),
            parent_id: Some("e1".to_string()),
            timestamp: 1001,
            message: serde_json::to_value(assistant_message()).unwrap(),
        };
        let e3 = SessionEntry::LegacyUser {
            id: "e3".to_string(),
            parent_id: Some("e2".to_string()),
            timestamp: 1002,
            content: "Second prompt".to_string(),
        };
        mgr.append_entries("original", &[e1, e2, e3]).unwrap();

        let (fork_header, fork_entries) = mgr.fork_from("original", "e2", "forked").unwrap();
        assert_eq!(fork_header.parent_session, Some("original".to_string()));
        assert_eq!(fork_entries.len(), 3);
        assert_eq!(fork_entries[0].id(), "e1");
        assert_eq!(fork_entries[1].id(), "e2");
        assert_eq!(fork_entries[2].entry_type(), "fork");
    }

    #[test]
    fn test_session_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        let result = mgr.open("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_session() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        mgr.create("to-delete", None).unwrap();
        assert!(mgr.exists("to-delete"));

        mgr.delete("to-delete").unwrap();
        assert!(!mgr.exists("to-delete"));
    }

    #[test]
    fn test_session_entry_serde_round_trip() {
        let entry = SessionEntry::LegacyToolUse {
            id: "t1".to_string(),
            parent_id: Some("e2".to_string()),
            timestamp: 2000,
            tool_call_id: "call_123".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls -la"}),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let loaded: SessionEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.id(), "t1");
        assert_eq!(loaded.entry_type(), "toolUse");
    }
}
