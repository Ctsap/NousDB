//! Core building blocks of [NousDB](https://docs.rs/nousdb): model providers,
//! prompt templates, token estimation and the statistics behind quality
//! certification.
//!
//! Most programs should depend on the `nousdb` crate, which re-exports what
//! is needed from here.
//!
//! # Features
//!
//! - `openai` (default): [`OpenAiCompatibleProvider`] for OpenAI-compatible
//!   endpoints.
//! - `azure` (default): Azure OpenAI support in that provider.
//! - `ollama` (default): [`OllamaProvider`] for Ollama's native API.

#![warn(missing_docs)]

pub mod arrow_util;
pub mod certification;
pub mod prompt;
pub mod providers;
pub mod token_estimator;

pub use arrow_util::StringColumn;
pub use certification::{
    CertKey, Certificate, CertificateCache, CertificationSpec, OperatorKind, PackingPolicy,
    QualityCertConfig, RungResult, cp_lower_bound, n_min, s_star,
};
pub use prompt::{
    message::Message, prompt::Prompt, reasoning::ReasoningMode, renderer::BatchRenderer,
};
#[cfg(feature = "ollama")]
pub use providers::OllamaProvider;
#[cfg(feature = "openai")]
pub use providers::OpenAiCompatibleProvider;
pub use providers::{
    ApiKey, ModelCard, ModelCatalog, ModelOutput, ModelProvider, ModelRegistry, ProviderError,
};
pub use token_estimator::{
    BatchBudgetModel, HistogramBucket, HistogramStats, SafeBatchEstimate, SafeBatchSizingConfig,
    TokenEstimator, TokenEstimatorConfig, TokenHistogram, estimate_safe_batch_size,
};
