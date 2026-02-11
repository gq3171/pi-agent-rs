use std::path::Path;

use serde::Deserialize;

use crate::resources::frontmatter;

/// A skill loaded from a .md file.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Skill name (derived from filename).
    pub name: String,
    /// Description from frontmatter.
    pub description: Option<String>,
    /// Tools allowed for this skill.
    pub allowed_tools: Vec<String>,
    /// The body content (prompt template).
    pub content: String,
    /// Source file path.
    pub source: String,
}

/// YAML frontmatter for skills.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SkillFrontmatter {
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
}

/// Load all skills from a directory.
///
/// Each `.md` file in the directory becomes a skill.
/// The filename (without extension) is the skill name.
pub fn load_skills_from_dir(dir: &Path) -> Result<Vec<Skill>, Box<dyn std::error::Error>> {
    let mut skills = Vec::new();

    if !dir.exists() {
        return Ok(skills);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "md") {
            if let Some(skill) = load_skill_from_file(&path)? {
                skills.push(skill);
            }
        }
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(skills)
}

/// Load a single skill from a markdown file.
fn load_skill_from_file(
    path: &Path,
) -> Result<Option<Skill>, Box<dyn std::error::Error>> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("Invalid skill filename: {}", path.display()))?
        .to_string();

    let content = std::fs::read_to_string(path)?;
    let (meta, body) = frontmatter::parse_frontmatter::<SkillFrontmatter>(&content)
        .unwrap_or((None, &content));

    let meta = meta.unwrap_or_default();

    Ok(Some(Skill {
        name,
        description: meta.description,
        allowed_tools: meta.allowed_tools.unwrap_or_default(),
        content: body.to_string(),
        source: path.display().to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_skills_from_dir() {
        let tmp = tempfile::tempdir().unwrap();

        // Create a skill file
        std::fs::write(
            tmp.path().join("greet.md"),
            "---\ndescription: Greet the user\nallowedTools:\n  - bash\n---\nHello! How can I help?",
        )
        .unwrap();

        // Create another skill
        std::fs::write(
            tmp.path().join("search.md"),
            "Search for files matching the pattern.",
        )
        .unwrap();

        // Non-md file should be ignored
        std::fs::write(tmp.path().join("notes.txt"), "not a skill").unwrap();

        let skills = load_skills_from_dir(tmp.path()).unwrap();
        assert_eq!(skills.len(), 2);

        let greet = skills.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.description, Some("Greet the user".to_string()));
        assert_eq!(greet.allowed_tools, vec!["bash"]);
        assert_eq!(greet.content, "Hello! How can I help?");

        let search = skills.iter().find(|s| s.name == "search").unwrap();
        assert!(search.description.is_none());
        assert!(search.allowed_tools.is_empty());
    }

    #[test]
    fn test_load_skills_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let skills = load_skills_from_dir(tmp.path()).unwrap();
        assert!(skills.is_empty());
    }

    #[test]
    fn test_load_skills_nonexistent_dir() {
        let skills = load_skills_from_dir(Path::new("/nonexistent/dir")).unwrap();
        assert!(skills.is_empty());
    }
}
