#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod context;
pub mod dataframe;

/// The DataFusion version NousDB is built on, so programs can use it without
/// pinning a second copy.
pub use datafusion;

/// Model providers, prompts, token estimation and certification statistics
/// (the `nousdb-core` crate).
pub mod core {
    pub use nousdb_core::*;
}

/// Logical and physical semantic operators (the `nousdb-operators` crate).
pub mod operators {
    pub use nousdb_operators::*;
}

/// Optimizer rules (the `nousdb-optimizer` crate).
pub mod optimizer {
    pub use nousdb_optimizer::*;
}

/// Physical planning (the `nousdb-planner` crate).
pub mod planner {
    pub use nousdb_planner::*;
}

/// SQL functions for the semantic operators (the `nousdb-sql` crate):
/// `EXTEND`, `SEM_FILTER`, `SEM_JOIN` and `GENERATE`.
pub mod sql {
    pub use nousdb_sql::*;
}

pub use context::{NousDBContext, NousDBContextBuilder};
pub use dataframe::{IntoColumnList, IntoOutputSchema, NousDBDataFrame, NousDBDataFrameExt};
pub use nousdb_core::certification::QualityCertConfig;
pub use nousdb_core::prompt::reasoning::ReasoningMode;
#[cfg(feature = "ollama")]
pub use nousdb_core::providers::OllamaProvider;
#[cfg(feature = "openai")]
pub use nousdb_core::providers::OpenAiCompatibleProvider;
pub use nousdb_core::providers::{
    ApiKey, ModelCard, ModelCatalog, ModelProvider, ModelRegistry, ProviderError,
};
pub use nousdb_optimizer::{BatchSizingChoice, SemanticOptimizerConfig, TokenRepartitionChoice};

/// The types most programs need: `use nousdb::prelude::*;`.
///
/// Includes DataFusion's own prelude.
pub mod prelude {
    pub use datafusion::arrow::datatypes::DataType;
    pub use datafusion::prelude::*;

    #[cfg(feature = "ollama")]
    pub use crate::OllamaProvider;
    #[cfg(feature = "openai")]
    pub use crate::OpenAiCompatibleProvider;
    pub use crate::{
        BatchSizingChoice, IntoColumnList, IntoOutputSchema, ModelCard, ModelCatalog,
        ModelRegistry, NousDBContext, NousDBContextBuilder, NousDBDataFrame, NousDBDataFrameExt,
        QualityCertConfig, ReasoningMode, SemanticOptimizerConfig, TokenRepartitionChoice,
    };
}
