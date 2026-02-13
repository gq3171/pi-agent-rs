use std::collections::HashMap;

use crate::session::types::SessionEntry;

/// A node in the session tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub entry: SessionEntry,
    pub children: Vec<String>,
}

/// Session tree structure for navigating branching conversations.
#[derive(Debug)]
pub struct SessionTree {
    nodes: HashMap<String, TreeNode>,
    roots: Vec<String>,
}

impl SessionTree {
    /// Build a tree from a flat list of session entries.
    pub fn from_entries(entries: &[SessionEntry]) -> Self {
        let mut nodes = HashMap::new();
        let mut roots = Vec::new();

        for entry in entries {
            let id = entry.id().to_string();
            nodes.insert(
                id.clone(),
                TreeNode {
                    entry: entry.clone(),
                    children: Vec::new(),
                },
            );

            if let Some(parent_id) = entry.parent_id() {
                if let Some(parent) = nodes.get_mut(parent_id) {
                    parent.children.push(id);
                } else {
                    roots.push(id);
                }
            } else {
                roots.push(id);
            }
        }

        Self { nodes, roots }
    }

    /// Get a node by ID.
    pub fn get(&self, id: &str) -> Option<&TreeNode> {
        self.nodes.get(id)
    }

    /// Get root entry IDs.
    pub fn roots(&self) -> &[String] {
        &self.roots
    }

    /// Get the linear path from root to a given entry (following parent_id chain).
    pub fn path_to(&self, entry_id: &str) -> Vec<&SessionEntry> {
        let mut path = Vec::new();
        let mut current = entry_id;

        while let Some(node) = self.nodes.get(current) {
            path.push(&node.entry);
            if let Some(parent_id) = node.entry.parent_id() {
                current = parent_id;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Get the latest leaf entry (last entry with no children).
    pub fn latest_leaf(&self) -> Option<&SessionEntry> {
        let mut latest: Option<&SessionEntry> = None;
        for node in self.nodes.values() {
            if node.children.is_empty()
                && (latest.is_none() || node.entry.timestamp() > latest.unwrap().timestamp())
            {
                latest = Some(&node.entry);
            }
        }
        latest
    }

    /// Traverse entries in order (DFS, following the main branch — first child).
    pub fn traverse_main_branch(&self) -> Vec<&SessionEntry> {
        let mut result = Vec::new();
        if let Some(root_id) = self.roots.first() {
            self.traverse_main_branch_from(root_id, &mut result);
        }
        result
    }

    fn traverse_main_branch_from<'a>(&'a self, id: &str, result: &mut Vec<&'a SessionEntry>) {
        if let Some(node) = self.nodes.get(id) {
            result.push(&node.entry);
            if let Some(first_child) = node.children.first() {
                self.traverse_main_branch_from(first_child, result);
            }
        }
    }

    /// Get all leaf entries (entries with no children).
    pub fn leaves(&self) -> Vec<&SessionEntry> {
        self.nodes
            .values()
            .filter(|n| n.children.is_empty())
            .map(|n| &n.entry)
            .collect()
    }

    /// Check if an entry has branches (more than one child).
    pub fn has_branches(&self, id: &str) -> bool {
        self.nodes.get(id).is_some_and(|n| n.children.len() > 1)
    }

    /// Count total entries.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::types::SessionEntry;

    fn make_entries() -> Vec<SessionEntry> {
        vec![
            SessionEntry::LegacyUser {
                id: "e1".to_string(),
                parent_id: None,
                timestamp: 1000,
                content: "Hello".to_string(),
            },
            SessionEntry::LegacyAssistant {
                id: "e2".to_string(),
                parent_id: Some("e1".to_string()),
                timestamp: 1001,
                message: serde_json::json!({"text": "Hi!"}),
            },
            SessionEntry::LegacyUser {
                id: "e3".to_string(),
                parent_id: Some("e2".to_string()),
                timestamp: 1002,
                content: "Tell me a joke".to_string(),
            },
        ]
    }

    #[test]
    fn test_tree_from_entries() {
        let entries = make_entries();
        let tree = SessionTree::from_entries(&entries);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.roots(), &["e1"]);
    }

    #[test]
    fn test_path_to() {
        let entries = make_entries();
        let tree = SessionTree::from_entries(&entries);
        let path = tree.path_to("e3");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].id(), "e1");
        assert_eq!(path[1].id(), "e2");
        assert_eq!(path[2].id(), "e3");
    }

    #[test]
    fn test_latest_leaf() {
        let entries = make_entries();
        let tree = SessionTree::from_entries(&entries);
        let leaf = tree.latest_leaf().unwrap();
        assert_eq!(leaf.id(), "e3");
    }

    #[test]
    fn test_traverse_main_branch() {
        let entries = make_entries();
        let tree = SessionTree::from_entries(&entries);
        let branch = tree.traverse_main_branch();
        assert_eq!(branch.len(), 3);
    }
}
