use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::auth::credentials::AuthCredential;
use crate::config::paths;
use crate::error::CodingAgentError;

/// Persisted auth.json structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthFile {
    /// Provider -> credential mapping.
    #[serde(default)]
    pub credentials: HashMap<String, AuthCredential>,
}

/// Multi-layer auth credential storage.
///
/// Resolution order:
/// 1. Runtime overrides (set programmatically)
/// 2. auth.json file on disk
/// 3. Environment variables
/// 4. Fallback (if configured)
pub struct AuthStorage {
    base_dir: PathBuf,
    /// Runtime overrides, highest priority.
    runtime: Arc<RwLock<HashMap<String, AuthCredential>>>,
    /// Cached auth file contents.
    file_cache: Arc<RwLock<Option<AuthFile>>>,
    /// Environment variable name mappings: provider -> env var name.
    env_mappings: HashMap<String, String>,
}

impl AuthStorage {
    /// Create a new AuthStorage with a base directory.
    pub fn new(base_dir: &Path) -> Self {
        let mut env_mappings = HashMap::new();
        // Standard env var mappings for providers
        env_mappings.insert("anthropic".to_string(), "ANTHROPIC_API_KEY".to_string());
        env_mappings.insert("openai".to_string(), "OPENAI_API_KEY".to_string());
        env_mappings.insert("google".to_string(), "GOOGLE_API_KEY".to_string());
        env_mappings.insert(
            "google-gemini-cli".to_string(),
            "GOOGLE_API_KEY".to_string(),
        );
        env_mappings.insert("google-vertex".to_string(), "GOOGLE_API_KEY".to_string());
        env_mappings.insert(
            "google-antigravity".to_string(),
            "GOOGLE_API_KEY".to_string(),
        );
        env_mappings.insert(
            "amazon-bedrock".to_string(),
            "AWS_ACCESS_KEY_ID".to_string(),
        );
        env_mappings.insert("xai".to_string(), "XAI_API_KEY".to_string());
        env_mappings.insert("groq".to_string(), "GROQ_API_KEY".to_string());
        env_mappings.insert(
            "github-copilot".to_string(),
            "GITHUB_COPILOT_TOKEN".to_string(),
        );
        env_mappings.insert("openai-codex".to_string(), "OPENAI_API_KEY".to_string());
        env_mappings.insert("mistral".to_string(), "MISTRAL_API_KEY".to_string());
        env_mappings.insert("openrouter".to_string(), "OPENROUTER_API_KEY".to_string());

        Self {
            base_dir: base_dir.to_path_buf(),
            runtime: Arc::new(RwLock::new(HashMap::new())),
            file_cache: Arc::new(RwLock::new(None)),
            env_mappings,
        }
    }

    /// Set a runtime credential override (highest priority).
    pub fn set_runtime_credential(&self, provider: &str, credential: AuthCredential) {
        let mut rt = self.runtime.write().unwrap();
        rt.insert(provider.to_string(), credential);
    }

    /// Remove a runtime credential.
    pub fn remove_runtime_credential(&self, provider: &str) {
        let mut rt = self.runtime.write().unwrap();
        rt.remove(provider);
    }

    /// Get the API key for a provider, following the resolution order.
    pub fn get_api_key(&self, provider: &str) -> Option<String> {
        self.get_credential(provider).map(|c| c.token().to_string())
    }

    /// Get the credential for a provider, following the resolution order.
    pub fn get_credential(&self, provider: &str) -> Option<AuthCredential> {
        // 1. Runtime overrides
        {
            let rt = self.runtime.read().unwrap();
            if let Some(cred) = rt.get(provider) {
                if !cred.is_expired() {
                    return Some(cred.clone());
                }
            }
        }

        // 2. auth.json file
        if let Some(cred) = self.get_from_file(provider) {
            if !cred.is_expired() {
                return Some(cred);
            }
        }

        // 3. Environment variable
        if let Some(env_var) = self.env_mappings.get(provider) {
            if let Ok(val) = std::env::var(env_var) {
                if !val.is_empty() {
                    return Some(AuthCredential::api_key(val));
                }
            }
        }

        // Also try the generic PI_API_KEY
        if let Ok(val) = std::env::var("PI_API_KEY") {
            if !val.is_empty() {
                return Some(AuthCredential::api_key(val));
            }
        }

        None
    }

    /// Load or return cached auth file.
    fn load_auth_file(&self) -> Option<AuthFile> {
        // Check cache first
        {
            let cache = self.file_cache.read().unwrap();
            if cache.is_some() {
                return cache.clone();
            }
        }

        // Load from disk
        let path = paths::auth_file(&self.base_dir);
        if !path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&path).ok()?;
        let auth_file: AuthFile = serde_json::from_str(&content).ok()?;

        // Cache it
        {
            let mut cache = self.file_cache.write().unwrap();
            *cache = Some(auth_file.clone());
        }

        Some(auth_file)
    }

    fn get_from_file(&self, provider: &str) -> Option<AuthCredential> {
        let auth_file = self.load_auth_file()?;
        auth_file.credentials.get(provider).cloned()
    }

    /// Save a credential to auth.json.
    pub fn save_credential(
        &self,
        provider: &str,
        credential: &AuthCredential,
    ) -> Result<(), CodingAgentError> {
        let path = paths::auth_file(&self.base_dir);
        paths::ensure_dir(&self.base_dir)?;

        let mut auth_file = self.load_auth_file().unwrap_or_default();
        auth_file
            .credentials
            .insert(provider.to_string(), credential.clone());

        let content = serde_json::to_string_pretty(&auth_file)?;
        std::fs::write(&path, content)?;

        // Update cache
        {
            let mut cache = self.file_cache.write().unwrap();
            *cache = Some(auth_file);
        }

        Ok(())
    }

    /// Remove a credential from auth.json.
    pub fn remove_credential(&self, provider: &str) -> Result<(), CodingAgentError> {
        let path = paths::auth_file(&self.base_dir);
        if !path.exists() {
            return Ok(());
        }

        let mut auth_file = self.load_auth_file().unwrap_or_default();
        auth_file.credentials.remove(provider);

        let content = serde_json::to_string_pretty(&auth_file)?;
        std::fs::write(&path, content)?;

        // Update cache
        {
            let mut cache = self.file_cache.write().unwrap();
            *cache = Some(auth_file);
        }

        Ok(())
    }

    /// Invalidate the file cache, forcing re-read on next access.
    pub fn invalidate_cache(&self) {
        let mut cache = self.file_cache.write().unwrap();
        *cache = None;
    }

    /// List all providers that have credentials (from all layers).
    pub fn list_providers(&self) -> Vec<String> {
        let mut providers = std::collections::HashSet::new();

        // Runtime
        {
            let rt = self.runtime.read().unwrap();
            for key in rt.keys() {
                providers.insert(key.clone());
            }
        }

        // File
        if let Some(auth_file) = self.load_auth_file() {
            for key in auth_file.credentials.keys() {
                providers.insert(key.clone());
            }
        }

        // Env
        for (provider, env_var) in &self.env_mappings {
            if std::env::var(env_var).is_ok() {
                providers.insert(provider.clone());
            }
        }

        let mut list: Vec<String> = providers.into_iter().collect();
        list.sort();
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_storage_runtime_override() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = AuthStorage::new(tmp.path());

        assert!(storage.get_api_key("anthropic").is_none());

        storage.set_runtime_credential("anthropic", AuthCredential::api_key("sk-test-123"));
        assert_eq!(
            storage.get_api_key("anthropic"),
            Some("sk-test-123".to_string())
        );

        storage.remove_runtime_credential("anthropic");
        assert!(storage.get_api_key("anthropic").is_none());
    }

    #[test]
    fn test_auth_storage_file_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = AuthStorage::new(tmp.path());

        storage
            .save_credential("openai", &AuthCredential::api_key("sk-openai-test"))
            .unwrap();

        // New storage instance should find it
        let storage2 = AuthStorage::new(tmp.path());
        assert_eq!(
            storage2.get_api_key("openai"),
            Some("sk-openai-test".to_string())
        );
    }

    #[test]
    fn test_auth_credential_types() {
        let api_key = AuthCredential::api_key("test-key");
        assert_eq!(api_key.token(), "test-key");
        assert!(!api_key.is_expired());

        let oauth = AuthCredential::OAuth {
            access_token: "access-123".to_string(),
            refresh_token: Some("refresh-456".to_string()),
            expires_at: Some(chrono::Utc::now().timestamp() + 3600),
        };
        assert_eq!(oauth.token(), "access-123");
        assert!(!oauth.is_expired());

        let expired = AuthCredential::OAuth {
            access_token: "old-token".to_string(),
            refresh_token: None,
            expires_at: Some(1000000000), // Way in the past
        };
        assert!(expired.is_expired());
    }

    #[test]
    fn test_auth_resolution_order() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = AuthStorage::new(tmp.path());

        // Save to file
        storage
            .save_credential("anthropic", &AuthCredential::api_key("file-key"))
            .unwrap();

        // File key should be found
        assert_eq!(
            storage.get_api_key("anthropic"),
            Some("file-key".to_string())
        );

        // Runtime override takes precedence
        storage.set_runtime_credential("anthropic", AuthCredential::api_key("runtime-key"));
        assert_eq!(
            storage.get_api_key("anthropic"),
            Some("runtime-key".to_string())
        );
    }
}
