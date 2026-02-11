use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::config::paths;
use crate::error::CodingAgentError;
use crate::session::types::*;

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
            return Err(CodingAgentError::Session("Session ID cannot be empty".to_string()));
        }
        // Only allow UUID-safe characters: alphanumeric, hyphen, underscore
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
            version: CURRENT_SESSION_VERSION,
            session_id: session_id.to_string(),
            parent_session_id: None,
            parent_entry_id: None,
            created_at: chrono::Utc::now().timestamp_millis(),
            title: title.map(|s| s.to_string()),
        };

        let path = self.session_path(session_id);
        let mut file = std::fs::File::create(&path)?;
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

        // First line is the header
        let header_line = lines
            .next()
            .ok_or_else(|| CodingAgentError::Session("Empty session file".to_string()))??;
        let header: SessionHeader = serde_json::from_str(&header_line)?;

        // Remaining lines are entries
        let mut entries = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<SessionEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("Skipping malformed session entry: {e}");
                }
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

    /// Append multiple entries atomically.
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

        // Validate that the entry_id exists
        let entry_exists = source_entries.iter().any(|e| e.id() == source_entry_id);
        if !entry_exists {
            return Err(CodingAgentError::Session(format!(
                "Entry not found: {source_entry_id} in session {source_session_id}"
            )));
        }

        // Collect entries up to and including the fork point
        let mut forked_entries = Vec::new();
        for entry in &source_entries {
            forked_entries.push(entry.clone());
            if entry.id() == source_entry_id {
                break;
            }
        }

        // Create new session
        let dir = self.sessions_dir();
        paths::ensure_dir(&dir)?;

        let header = SessionHeader {
            version: CURRENT_SESSION_VERSION,
            session_id: new_session_id.to_string(),
            parent_session_id: Some(source_header.session_id.clone()),
            parent_entry_id: Some(source_entry_id.to_string()),
            created_at: chrono::Utc::now().timestamp_millis(),
            title: source_header.title.clone(),
        };

        let path = self.session_path(new_session_id);
        let mut file = std::fs::File::create(&path)?;
        let header_line = serde_json::to_string(&header)?;
        writeln!(file, "{header_line}")?;

        for entry in &forked_entries {
            let line = serde_json::to_string(entry)?;
            writeln!(file, "{line}")?;
        }

        // Add a fork marker entry
        let fork_entry = SessionEntry::Fork {
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

    /// List all sessions sorted by updated_at descending (most recent first).
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

    /// Read session info (header + metadata) from a file without loading all entries.
    fn read_session_info(&self, path: &Path) -> Option<SessionInfo> {
        let file = std::fs::File::open(path).ok()?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header_line = lines.next()?.ok()?;
        let header: SessionHeader = serde_json::from_str(&header_line).ok()?;

        let mut entry_count = 0;
        let mut last_timestamp = header.created_at;
        for line in lines.map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            entry_count += 1;
            // Try to extract timestamp without full deserialization
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(ts) = val.get("timestamp").and_then(|v| v.as_i64()) {
                    if ts > last_timestamp {
                        last_timestamp = ts;
                    }
                }
            }
        }

        Some(SessionInfo {
            session_id: header.session_id,
            title: header.title,
            created_at: header.created_at,
            updated_at: last_timestamp,
            entry_count,
            parent_session_id: header.parent_session_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_open_session() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        let header = mgr.create("test-session-1", Some("Test Session")).unwrap();
        assert_eq!(header.session_id, "test-session-1");
        assert_eq!(header.title, Some("Test Session".to_string()));
        assert_eq!(header.version, CURRENT_SESSION_VERSION);

        let (loaded_header, entries) = mgr.open("test-session-1").unwrap();
        assert_eq!(loaded_header.session_id, "test-session-1");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_append_and_read_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path());

        mgr.create("sess-1", None).unwrap();

        let user_entry = SessionEntry::User {
            id: "e1".to_string(),
            parent_id: None,
            timestamp: 1000,
            content: "Hello, agent!".to_string(),
        };
        mgr.append_entry("sess-1", &user_entry).unwrap();

        let assistant_entry = SessionEntry::Assistant {
            id: "e2".to_string(),
            parent_id: Some("e1".to_string()),
            timestamp: 1001,
            message: serde_json::json!({"text": "Hello!"}),
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
        let e1 = SessionEntry::User {
            id: "e1".to_string(),
            parent_id: None,
            timestamp: 1000,
            content: "First prompt".to_string(),
        };
        let e2 = SessionEntry::Assistant {
            id: "e2".to_string(),
            parent_id: Some("e1".to_string()),
            timestamp: 1001,
            message: serde_json::json!({"text": "Response 1"}),
        };
        let e3 = SessionEntry::User {
            id: "e3".to_string(),
            parent_id: Some("e2".to_string()),
            timestamp: 1002,
            content: "Second prompt".to_string(),
        };
        mgr.append_entries("original", &[e1, e2, e3]).unwrap();

        // Fork from e2 (should include e1, e2 but not e3)
        let (fork_header, fork_entries) = mgr.fork_from("original", "e2", "forked").unwrap();
        assert_eq!(fork_header.parent_session_id, Some("original".to_string()));
        // e1, e2, plus fork marker = 3
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
        let entry = SessionEntry::ToolUse {
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
