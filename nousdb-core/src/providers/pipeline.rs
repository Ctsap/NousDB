//! The request steps every built-in provider shares: rendering the prompt and
//! schema, and turning the endpoint's answer (or its absence) into a
//! [`PredictResult`]. A provider supplies only the transport in between.

use std::fmt::Display;
use std::time::Duration;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};

use crate::prompt::prompt::Prompt;
use crate::prompt::renderer::BatchRenderer;
use crate::providers::metrics::{InferenceMetrics, PredictResult};
use crate::providers::schemas::ModelOutput;
use crate::providers::{trace, utils};

/// One packed request, before and after rendering.
pub(crate) struct Request<'a> {
    pub model_id: &'a str,
    pub prompt: &'a Prompt,
    pub batch: &'a RecordBatch,
    pub start_row: usize,
    pub batch_size: usize,
    pub renderer: &'a (dyn BatchRenderer + Send + Sync),
    pub system: Option<String>,
    pub user: String,
    pub schema: serde_json::Value,
}

impl<'a> Request<'a> {
    /// Renders the system and user prompts. `schema` is the provider's
    /// [`construct_output_schema`](super::ModelProvider::construct_output_schema).
    pub fn render(
        model_id: &'a str,
        prompt: &'a Prompt,
        batch: &'a RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &'a (dyn BatchRenderer + Send + Sync),
        schema: &str,
    ) -> Result<Self> {
        let schema = serde_json::from_str(schema)
            .map_err(|e| DataFusionError::Execution(format!("invalid output schema: {e}")))?;
        Ok(Self {
            model_id,
            prompt,
            batch,
            start_row,
            batch_size,
            renderer,
            system: prompt.render_system()?,
            user: prompt.render_batch(renderer, batch, start_row, batch_size)?,
            schema,
        })
    }

    /// The endpoint never answered. Logged and returned as a transport
    /// failure, which operators recover from.
    pub fn transport_failure(&self, elapsed: Duration, error: impl Display) -> PredictResult {
        let elapsed_ms = elapsed.as_millis() as u64;
        log::warn!(
            "request to model {} with {} rows failed after {:.1}s: {error}",
            self.model_id,
            self.batch_size,
            elapsed.as_secs_f64(),
        );
        self.trace(None, &[], true, false, (0, 0), elapsed_ms);
        PredictResult::transport_failure(elapsed_ms)
    }

    /// The endpoint answered: decode the answer against the prompt's schema.
    pub fn complete(&self, completion: Completion, elapsed: Duration) -> PredictResult {
        if completion.truncated {
            log::warn!(
                "model {} ran out of output tokens ({} used, {} of them reasoning) on {} rows; \
                 lower the batch size or raise the card's max_output_tokens",
                self.model_id,
                completion.output_tokens,
                completion.reasoning_tokens,
                self.batch_size,
            );
        }
        let parsed = utils::parse_json_to_model_outputs(
            &utils::clean_model_response(&completion.text),
            self.prompt,
            self.batch_size,
        );
        self.trace(
            Some(&completion.text),
            &parsed.outputs,
            parsed.row_count_mismatch,
            completion.truncated,
            (completion.input_tokens, completion.output_tokens),
            elapsed.as_millis() as u64,
        );
        let metrics = InferenceMetrics::new(
            completion.input_tokens,
            completion.output_tokens,
            elapsed.as_nanos() as u64,
        )
        .with_reasoning_tokens(completion.reasoning_tokens);
        PredictResult::new(parsed.outputs, metrics, parsed.row_count_mismatch)
            .with_truncation(completion.truncated)
    }

    #[allow(clippy::too_many_arguments)]
    fn trace(
        &self,
        raw_response: Option<&str>,
        outputs: &[ModelOutput],
        row_count_mismatch: bool,
        truncated: bool,
        tokens: (u64, u64),
        latency_ms: u64,
    ) {
        if !trace::enabled() {
            return;
        }
        let mut m = serde_json::Map::new();
        m.insert("operator".into(), self.renderer.name().into());
        m.insert("model".into(), self.model_id.into());
        m.insert("k".into(), self.batch_size.into());
        m.insert(
            "rows".into(),
            trace::rows_as_json(self.batch, self.start_row, self.batch_size).into(),
        );
        m.insert(
            "system_prompt".into(),
            self.system.as_deref().unwrap_or("").into(),
        );
        m.insert("user_prompt".into(), self.user.as_str().into());
        m.insert("raw_response".into(), raw_response.into());
        m.insert(
            "outputs".into(),
            outputs
                .iter()
                .map(trace::output_as_json)
                .collect::<Vec<_>>()
                .into(),
        );
        m.insert("row_count_mismatch".into(), row_count_mismatch.into());
        m.insert("truncated".into(), truncated.into());
        m.insert("input_tokens".into(), tokens.0.into());
        m.insert("output_tokens".into(), tokens.1.into());
        m.insert("latency_ms".into(), latency_ms.into());
        trace::record(m);
    }
}

/// What a transport returns for an answered request.
pub(crate) struct Completion {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Part of `output_tokens` spent on hidden reasoning, when reported.
    pub reasoning_tokens: u64,
    /// The model stopped at its output budget. Under a JSON grammar this
    /// leaves an unterminated array, so it explains the decode failure.
    pub truncated: bool,
}
