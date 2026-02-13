use std::path::Path;

use serde::Deserialize;

use crate::resources::frontmatter;

/// Prompt template loaded from markdown.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub name: String,
    pub description: Option<String>,
    pub content: String,
    pub source: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFrontmatter {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

pub fn load_prompts_from_dir(
    dir: &Path,
) -> Result<Vec<PromptTemplate>, Box<dyn std::error::Error>> {
    let mut prompts = Vec::new();
    if !dir.exists() {
        return Ok(prompts);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().is_some_and(|ext| ext == "md") {
            continue;
        }

        let raw = std::fs::read_to_string(&path)?;
        let (meta, body) =
            frontmatter::parse_frontmatter::<PromptFrontmatter>(&raw).unwrap_or((None, &raw));
        let meta = meta.unwrap_or_default();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("prompt")
            .to_string();

        prompts.push(PromptTemplate {
            name: meta.name.unwrap_or(stem),
            description: meta.description,
            content: body.to_string(),
            source: path.display().to_string(),
        });
    }

    prompts.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(prompts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_prompts_from_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("fix.md"),
            "---\nname: fix\n---\nPlease fix the issue.",
        )
        .unwrap();

        let prompts = load_prompts_from_dir(tmp.path()).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "fix");
    }
}
