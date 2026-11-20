//! Errors raised while configuring providers and resolving models.

use std::path::PathBuf;

use datafusion::error::DataFusionError;

/// An error configuring a provider, loading a catalog or resolving a model.
///
/// Failures of individual requests are not errors: they are reported on the
/// [`PredictResult`](super::PredictResult) so an operator can recover.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// A model catalog could not be parsed.
    #[error("invalid model catalog: {0}")]
    Catalog(String),

    /// A model catalog file could not be read.
    #[error("cannot read model catalog {}: {source}", path.display())]
    CatalogIo {
        /// The file that was read.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The provider was configured inconsistently.
    #[error("invalid provider configuration: {0}")]
    Config(String),

    /// An environment variable named in the configuration is not set.
    #[error("environment variable {0} is not set")]
    MissingEnv(String),

    /// The endpoint could not be reached while building the provider.
    #[error("cannot reach {endpoint}: {message}")]
    Connect {
        /// The endpoint that was contacted.
        endpoint: String,
        /// What went wrong.
        message: String,
    },

    /// No registered provider serves the model.
    #[error(
        "no registered provider serves model '{0}'; add a ModelCard for it to a provider's catalog"
    )]
    UnknownModel(String),

    /// No provider is registered under this name.
    #[error("no provider named '{0}' is registered")]
    UnknownProvider(String),

    /// A provider is already registered under this name.
    #[error("a provider named '{0}' is already registered")]
    DuplicateProvider(String),
}

impl From<ProviderError> for DataFusionError {
    fn from(e: ProviderError) -> Self {
        DataFusionError::External(Box::new(e))
    }
}
