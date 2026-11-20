//! The [`ModelProvider`] trait.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::prompt::prompt::Prompt;
use crate::prompt::renderer::BatchRenderer;
use crate::providers::metadata::ModelCard;
use crate::providers::metrics::PredictResult;
use crate::providers::utils;

/// A cap on the requests in flight to one endpoint.
///
/// Every operator and partition using a provider shares its
/// `AdmissionControl`, so the cap bounds the load on the server as a whole.
#[derive(Debug, Clone)]
pub struct AdmissionControl {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl AdmissionControl {
    /// Allows at most `max_concurrent` requests in flight (at least one).
    pub fn new(max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// The cap.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Slots free right now.
    pub fn available(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Waits for a free slot. The slot is released when the permit drops.
    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        Arc::clone(&self.semaphore)
            .acquire_owned()
            .await
            .expect("the semaphore is never closed")
    }
}

/// A source of model completions.
///
/// NousDB ships [`OpenAiCompatibleProvider`](super::OpenAiCompatibleProvider)
/// and [`OllamaProvider`](super::OllamaProvider). Implement this trait to
/// serve models from anywhere else.
///
/// A provider sends one *packed* request per call to
/// [`predict`](Self::predict): `batch_size` rows starting at `start_row`,
/// rendered into a single prompt, answered with one JSON array of
/// `batch_size` objects. The helpers in [`utils`] build the
/// JSON schema to constrain the answer with and decode it.
#[async_trait]
pub trait ModelProvider: Send + Sync + std::fmt::Debug {
    /// A short name for logs and errors, e.g. `"vllm"`.
    fn name(&self) -> &str;

    /// The models this provider serves.
    fn models(&self) -> Vec<ModelCard>;

    /// The card for `model_id`, if this provider serves it.
    fn model(&self, model_id: &str) -> Option<ModelCard> {
        self.models().into_iter().find(|m| m.id == model_id)
    }

    /// The JSON schema, as a string, that the answer to a request of
    /// `batch_size` rows must satisfy.
    fn construct_output_schema(&self, prompt: &Prompt, batch_size: usize) -> String {
        utils::construct_output_schema_from_prompt(prompt, batch_size)
    }

    /// Answers one packed request.
    ///
    /// Return `Err` only for failures that should abort the query, such as a
    /// template that does not render. A request that fails in transport, is
    /// truncated or cannot be decoded should come back as `Ok` with the
    /// failure recorded on the [`PredictResult`], so the operator can apply
    /// its recovery policy.
    async fn predict(
        &self,
        model_id: &str,
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult>;

    /// The cap on requests in flight to this provider's endpoint, or `None`
    /// for no cap.
    fn admission_control(&self) -> Option<AdmissionControl> {
        None
    }
}
