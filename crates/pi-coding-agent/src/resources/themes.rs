use std::collections::HashMap;
use std::path::Path;

/// Theme definition (minimal, generic key-value colors).
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub colors: HashMap<String, String>,
    pub source: String,
}

pub fn load_themes_from_dir(dir: &Path) -> Result<Vec<Theme>, Box<dyn std::error::Error>> {
    let mut themes = Vec::new();
    if !dir.exists() {
        return Ok(themes);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }

        let content = std::fs::read_to_string(&path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        let name = value
            .get("name")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "theme".to_string());

        let colors = value
            .get("colors")
            .and_then(|v| v.as_object())
            .map(|map| {
                map.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();

        themes.push(Theme {
            name,
            colors,
            source: path.display().to_string(),
        });
    }

    themes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(themes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_themes_from_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("ocean.json"),
            r##"{"name":"ocean","colors":{"bg":"#001122","fg":"#ddeeff"}}"##,
        )
        .unwrap();

        let themes = load_themes_from_dir(tmp.path()).unwrap();
        assert_eq!(themes.len(), 1);
        assert_eq!(themes[0].name, "ocean");
    }
}
