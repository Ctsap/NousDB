//! [`ApiKey`], a credential that never shows up in logs.

use std::fmt;

use super::error::ProviderError;

/// An API key. Its `Debug` output is redacted, so a provider can be logged
/// or printed without leaking it.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Reads the key from an environment variable.
    pub fn from_env(var: &str) -> Result<Self, ProviderError> {
        std::env::var(var)
            .map(Self)
            .map_err(|_| ProviderError::MissingEnv(var.to_string()))
    }

    /// The key itself.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl From<String> for ApiKey {
    fn from(key: String) -> Self {
        Self(key)
    }
}

impl From<&str> for ApiKey {
    fn from(key: &str) -> Self {
        Self(key.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_is_redacted() {
        let key = ApiKey::new("sk-secret");
        assert!(!format!("{key:?}").contains("sk-secret"));
        assert_eq!(key.expose(), "sk-secret");
    }
}
