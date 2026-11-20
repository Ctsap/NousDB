//! Model providers: how NousDB reaches a model, and what it knows about it.
//!
//! - [`ModelCard`] describes one model; [`ModelCatalog`] holds several.
//! - [`ModelProvider`] sends packed requests to an endpoint.
//!   [`OpenAiCompatibleProvider`] and [`OllamaProvider`] are built in.
//! - [`ModelRegistry`] holds the providers a session uses and resolves the
//!   model names queries refer to.

pub mod error;
pub mod metadata;
pub mod metrics;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(any(feature = "openai", feature = "ollama"))]
pub(crate) mod pipeline;
pub mod provider;
pub mod registry;
pub mod schemas;
mod secret;
pub mod trace;
pub mod utils;

pub use error::ProviderError;
pub use metadata::{DEFAULT_MAX_BATCH_SIZE, ModelCard, ModelCatalog, ModelPricing};
pub use metrics::{InferenceMetrics, PredictResult, Rejection};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaBuilder, OllamaProvider};
#[cfg(feature = "openai")]
pub use openai::{OpenAiCompatibleBuilder, OpenAiCompatibleProvider, ProviderKind};
pub use provider::{AdmissionControl, ModelProvider};
pub use registry::ModelRegistry;
pub use schemas::ModelOutput;
pub use secret::ApiKey;
