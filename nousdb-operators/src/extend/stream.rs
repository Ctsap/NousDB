//! Execution stream for Extend: request packing, in-flight window and output assembly.

use datafusion::arrow::array::{Array, new_null_array};
use datafusion::arrow::compute::{BatchCoalescer, concat_batches};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};

use crate::certification::{
    CalibrationHost, CalibrationParts, CalibrationState, ProbeObservation, ProbeScorer,
    RequestOutcome, RequestOutcomeSink, StreamCertification, poll_calibration,
};
use crate::extend::renderer::ExtendBatchRenderer;
use crate::metrics::{SemanticOperatorMetrics, WallClockTimer};
use nousdb_core::certification::OperatorKind;
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::InferenceMetrics;
use nousdb_core::providers::provider::{AdmissionControl, ModelProvider};
use nousdb_core::providers::schemas::ModelOutput;

use futures::Future;
use futures::stream::FuturesOrdered;
use futures::stream::Stream;
use futures::stream::StreamExt;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

struct ProcessingResult {
    batch: RecordBatch,
    metrics: InferenceMetrics,
    rows_processed: usize,
    failed_rows: Option<usize>,
    /// The model ran out of output budget mid-response. Independent of
    /// `failed_rows`: it is counted even when the response still decoded.
    truncated: bool,
    /// What a probe stream reports to its sink; production ignores it.
    outcome: RequestOutcome,
}

type RequestFuture = Pin<Box<dyn Future<Output = Result<ProcessingResult>> + Send>>;

/// Outcome of trying to assemble the next request-sized input batch.
enum NextRequest {
    Ready(RecordBatch),
    /// Upstream returned `Poll::Pending` (its waker is registered).
    Pending,
    /// Input is exhausted and no buffered rows remain.
    Exhausted,
}

/// Stream that packs input rows into requests, keeps a window of requests in
/// flight, and emits the input rows with the derived columns appended.
pub struct ExtendStream {
    input: SendableRecordBatchStream,
    renderer: ExtendBatchRenderer,
    prompt: Prompt,
    output_schema: SchemaRef,
    model_id: String,
    model_provider: Arc<dyn ModelProvider>,
    model_card: ModelCard,
    schema: SchemaRef,
    metrics: SemanticOperatorMetrics,
    inference_timer: Arc<WallClockTimer>,
    optimal_batch_size: Option<usize>,
    pending_slices: VecDeque<RecordBatch>,
    coalesce_buffer: Vec<RecordBatch>,
    coalesce_rows: usize,
    /// Re-coalesces per-request result batches (at most `optimal_batch_size`
    /// rows each) up to the session's `batch_size` before emitting downstream.
    /// Bypassed in probe mode, which needs one batch per request.
    output_coalescer: BatchCoalescer,
    /// Max concurrent in-flight requests for this partition (1 = serial).
    /// Completed results are emitted in submission order.
    request_window: usize,
    /// Endpoint-global permit pool shared across partitions and operators.
    admission: Option<AdmissionControl>,
    in_flight: FuturesOrdered<RequestFuture>,
    input_exhausted: bool,
    done: bool,
    calibration: CalibrationState,
    /// Calibration probe plans only: per-request outcome reporting, plus
    /// request-error tolerance so one failed probe costs one disagreement
    /// instead of killing the rung. `None` in production.
    request_sink: Option<RequestOutcomeSink>,
    /// Submission ordinal of the next request to complete. Meaningful only
    /// with a sink installed; `FuturesOrdered` guarantees completion order
    /// matches submission order, so this maps 1:1 onto probe batches.
    request_ordinal: usize,
}

impl ExtendStream {
    /// Creates a stream over `input`. `schema` is the output schema (input
    /// columns followed by `output_schema`); `output_batch_size` is the target
    /// size of emitted batches.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: SendableRecordBatchStream,
        user_instruction: String,
        input_columns: Vec<String>,
        output_schema: SchemaRef,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        model_card: ModelCard,
        schema: SchemaRef,
        metrics: SemanticOperatorMetrics,
        inference_timer: Arc<WallClockTimer>,
        optimal_batch_size: Option<usize>,
        request_window: usize,
        output_batch_size: usize,
        reasoning: ReasoningMode,
        certification: Option<StreamCertification>,
    ) -> Self {
        let output_columns: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let renderer =
            ExtendBatchRenderer::new(user_instruction.clone(), input_columns, output_columns);

        let prompt = Prompt::new(
            user_instruction.clone(),
            Some(reasoning.system_template(OperatorKind::Extend)),
            reasoning.user_template(OperatorKind::Extend),
            output_schema.clone(),
        )
        .with_reasoning(reasoning);

        let admission = model_provider.admission_control();

        let calibration = CalibrationState::new(certification, || CalibrationParts {
            scorer: extend_probe_scorer(output_schema.clone()),
            inference_timer: inference_timer.clone(),
        });

        let output_coalescer = BatchCoalescer::new(schema.clone(), output_batch_size);

        Self {
            input,
            renderer,
            prompt,
            output_schema,
            model_id,
            model_provider,
            model_card,
            schema,
            metrics,
            inference_timer,
            optimal_batch_size,
            pending_slices: VecDeque::new(),
            coalesce_buffer: Vec::new(),
            coalesce_rows: 0,
            output_coalescer,
            request_window: request_window.max(1),
            admission,
            in_flight: FuturesOrdered::new(),
            input_exhausted: false,
            done: false,
            calibration,
            request_sink: None,
            request_ordinal: 0,
        }
    }

    /// Turn this stream into a calibration probe stream: report every
    /// request's outcome and tolerate request errors.
    pub fn with_request_sink(mut self, sink: RequestOutcomeSink) -> Self {
        self.request_sink = Some(sink);
        self
    }

    /// Report a completed request to the calibration sink, if installed.
    fn report_request_outcome(&mut self, outcome: RequestOutcome) {
        if let Some(sink) = &self.request_sink {
            sink.report(self.request_ordinal, outcome);
            self.request_ordinal += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_batch(
        renderer: ExtendBatchRenderer,
        prompt: Prompt,
        output_schema: SchemaRef,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        _model_card: ModelCard,
        schema: SchemaRef,
        input_batch: RecordBatch,
        admission: Option<AdmissionControl>,
        probe: bool,
    ) -> Result<ProcessingResult> {
        let num_rows = input_batch.num_rows();

        if num_rows == 0 {
            return Ok(ProcessingResult {
                batch: RecordBatch::new_empty(schema),
                metrics: InferenceMetrics::zero(),
                rows_processed: 0,
                failed_rows: None,
                truncated: false,
                outcome: RequestOutcome::Answered,
            });
        }

        // One permit per in-flight request; held until the response is
        // processed so the endpoint-global cap bounds actual load.
        let _permit = match admission {
            Some(ac) => Some(ac.acquire().await),
            None => None,
        };

        // A provider `Err` ends the stream in production. A probe stream reports it as a
        // request that got no answer: one disagreement for this calibration,
        // but no evidence about the model (see `RequestOutcome`).
        let predict_result = match model_provider
            .predict(&model_id, &prompt, &input_batch, 0, num_rows, &renderer)
            .await
        {
            Ok(result) => result,
            Err(e) if probe => {
                log::debug!("[certify] probe request got no answer, scored as disagreement: {e}");
                return Ok(ProcessingResult {
                    batch: RecordBatch::new_empty(schema),
                    metrics: InferenceMetrics::zero(),
                    rows_processed: num_rows,
                    failed_rows: Some(num_rows),
                    truncated: false,
                    outcome: RequestOutcome::NoAnswer,
                });
            }
            Err(e) => return Err(e),
        };

        let rejection = predict_result.rejection(num_rows);
        let outcome = RequestOutcome::from_rejection(rejection.as_ref());
        // Read before `metrics` is moved out of `predict_result` below: the
        // partial move ends any whole-struct use of it.
        let truncated = predict_result.truncated;
        let accumulated_metrics = predict_result.metrics;

        // `PredictResult::rejection` checks `row_count_mismatch` as well as the
        // length: on a parse failure the provider null-fills to `num_rows`, so
        // the length alone always matches. It also tells a request that never
        // returned apart from one that returned an unusable response.
        if let Some(rejection) = rejection {
            log::warn!(
                "ExtendRecovery: {rejection}, emitting NULL derived columns for all {num_rows} rows",
            );

            let mut all_columns = input_batch.columns().to_vec();
            for field in output_schema.fields() {
                all_columns.push(new_null_array(field.data_type(), num_rows));
            }

            let output_batch = RecordBatch::try_new(schema, all_columns)?;

            return Ok(ProcessingResult {
                batch: output_batch,
                metrics: accumulated_metrics,
                rows_processed: num_rows,
                failed_rows: Some(num_rows),
                truncated,
                outcome,
            });
        }

        let json_response = outputs_to_json(&predict_result.outputs, &output_schema)?;
        let output_arrays = prompt.parse_response(&json_response, num_rows)?;

        if output_arrays.len() != output_schema.fields().len() {
            return Err(DataFusionError::Execution(format!(
                "Expected {} output columns but got {}",
                output_schema.fields().len(),
                output_arrays.len()
            )));
        }

        let mut all_columns = input_batch.columns().to_vec();
        all_columns.extend(output_arrays);

        let output_batch = RecordBatch::try_new(schema, all_columns)?;

        debug_assert_eq!(
            output_batch.num_rows(),
            input_batch.num_rows(),
            "Extend operator must preserve cardinality"
        );

        Ok(ProcessingResult {
            batch: output_batch,
            metrics: accumulated_metrics,
            rows_processed: num_rows,
            failed_rows: None,
            truncated,
            outcome,
        })
    }

    /// Concatenate the coalesce buffer into a single RecordBatch and reset.
    fn take_coalesced_batch(&mut self) -> Result<RecordBatch> {
        let batches = std::mem::take(&mut self.coalesce_buffer);
        self.coalesce_rows = 0;
        let input_schema = self.input.schema();
        concat_batches(&input_schema, &batches).map_err(|e| {
            DataFusionError::Execution(format!("Failed to concat coalesced batches: {}", e))
        })
    }

    /// Split a coalesced batch into request-sized slices (zero-copy).
    ///
    /// When `keep_remainder` is true, a trailing slice smaller than
    /// `optimal_batch_size` goes back into the coalesce buffer instead of
    /// being dispatched: upstream operators that emit variable-size batches
    /// (e.g. `TokenRepartitionExec`) would otherwise fragment requests into
    /// under-filled batches. The remainder is flushed as a request only once
    /// the input is exhausted.
    fn enqueue_request_slices(&mut self, combined: RecordBatch, keep_remainder: bool) {
        match self.optimal_batch_size {
            Some(max_size) if combined.num_rows() > max_size => {
                let total = combined.num_rows();
                let mut offset = 0;
                while offset + max_size <= total {
                    self.pending_slices
                        .push_back(combined.slice(offset, max_size));
                    offset += max_size;
                }
                if offset < total {
                    let tail = combined.slice(offset, total - offset);
                    if keep_remainder {
                        self.coalesce_rows += tail.num_rows();
                        self.coalesce_buffer.push(tail);
                    } else {
                        self.pending_slices.push_back(tail);
                    }
                }
            }
            _ => self.pending_slices.push_back(combined),
        }
    }

    /// Assemble the next request-sized input batch: drain buffered slices,
    /// then coalesce upstream batches up to `optimal_batch_size`.
    fn next_request_batch(&mut self, cx: &mut Context<'_>) -> Result<NextRequest> {
        loop {
            if let Some(batch) = self.pending_slices.pop_front() {
                return Ok(NextRequest::Ready(batch));
            }

            if self.input_exhausted {
                if self.coalesce_rows > 0 {
                    let combined = self.take_coalesced_batch()?;
                    self.enqueue_request_slices(combined, false);
                    continue;
                }
                return Ok(NextRequest::Exhausted);
            }

            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let Some(max_size) = self.optimal_batch_size else {
                        return Ok(NextRequest::Ready(batch));
                    };
                    self.coalesce_rows += batch.num_rows();
                    self.coalesce_buffer.push(batch);
                    if self.coalesce_rows >= max_size {
                        let combined = self.take_coalesced_batch()?;
                        self.enqueue_request_slices(combined, true);
                    }
                }
                Poll::Ready(Some(Err(e))) => return Err(e),
                Poll::Ready(None) => self.input_exhausted = true,
                Poll::Pending => return Ok(NextRequest::Pending),
            }
        }
    }

    /// Dispatch one request batch into the in-flight window.
    fn spawn_request(&mut self, batch: RecordBatch) {
        self.inference_timer.start();
        let fut = Self::process_batch(
            self.renderer.clone(),
            self.prompt.clone(),
            self.output_schema.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.model_card.clone(),
            self.schema.clone(),
            batch,
            self.admission.clone(),
            self.request_sink.is_some(),
        );
        self.in_flight.push_back(Box::pin(fut));
    }
}

/// Serializes a request's model outputs as the JSON array
/// `Prompt::parse_response` decodes. Shared with the generate stream.
pub(crate) fn outputs_to_json(
    outputs: &[ModelOutput],
    output_schema: &SchemaRef,
) -> Result<String> {
    let field_names: Vec<String> = output_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let mut json_array = Vec::with_capacity(outputs.len());

    for output in outputs {
        json_array.push(output_to_row_json(output, &field_names)?);
    }

    serde_json::to_string(&json_array).map_err(|e| {
        DataFusionError::Execution(format!("Failed to serialize outputs to JSON: {}", e))
    })
}

/// Maps one model output slot to its JSON row object ahead of
/// `Prompt::parse_response`. Shared by `outputs_to_json` and calibration so
/// both apply the same per-variant wrapping rules.
pub(crate) fn output_to_row_json(
    output: &ModelOutput,
    field_names: &[String],
) -> Result<serde_json::Value> {
    use serde_json::{Map, Number, Value};

    match output {
        ModelOutput::Structured(v) => {
            if v.is_object() {
                Ok(v.clone())
            } else if field_names.len() == 1 {
                let mut map = Map::new();
                map.insert(field_names[0].clone(), v.clone());
                Ok(Value::Object(map))
            } else {
                Err(DataFusionError::Execution(format!(
                    "Structured ModelOutput is not an object and cannot be mapped to multiple output fields: {:?}",
                    v
                )))
            }
        }
        ModelOutput::Text(s) => {
            if field_names.len() != 1 {
                return Err(DataFusionError::Execution(
                    "Text ModelOutput can only map to single-field output schema".to_string(),
                ));
            }
            let mut map = Map::new();
            map.insert(field_names[0].clone(), Value::String(s.clone()));
            Ok(Value::Object(map))
        }
        ModelOutput::Boolean(b) => {
            if field_names.len() != 1 {
                return Err(DataFusionError::Execution(
                    "Boolean ModelOutput can only map to single-field output schema".to_string(),
                ));
            }
            let mut map = Map::new();
            map.insert(field_names[0].clone(), Value::Bool(*b));
            Ok(Value::Object(map))
        }
        ModelOutput::Float(f) => {
            if field_names.len() != 1 {
                return Err(DataFusionError::Execution(
                    "Float ModelOutput can only map to single-field output schema".to_string(),
                ));
            }
            let mut map = Map::new();
            let num = Number::from_f64(*f)
                .ok_or_else(|| DataFusionError::Execution(format!("Invalid float value: {}", f)))?;
            map.insert(field_names[0].clone(), Value::Number(num));
            Ok(Value::Object(map))
        }
        // An abstention reaches Arrow as a NULL: `build_string_array` turns
        // `Value::Null` into `append_null`.
        ModelOutput::Null => {
            if field_names.len() != 1 {
                return Err(DataFusionError::Execution(
                    "Null ModelOutput can only map to single-field output schema".to_string(),
                ));
            }
            let mut map = Map::new();
            map.insert(field_names[0].clone(), Value::Null);
            Ok(Value::Object(map))
        }
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported ModelOutput type: {:?}",
            output
        ))),
    }
}

/// Canonical text normalization (trim / case / whitespace) for agreement
/// scoring only. It never changes what the user receives, only whether two
/// runs count as agreeing.
fn canonical_text(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Do two derived values, already built by the production path, agree?
///
/// Both sides came out of `Prompt::parse_response`, so coercion is the
/// operator's; the only certification-specific rule is the text normalization
/// above. NaN agrees with NaN.
fn derived_values_agree(a: &dyn Array, ai: usize, b: &dyn Array, bi: usize) -> bool {
    use datafusion::arrow::array::{BooleanArray, Float64Array, Int64Array};
    use datafusion::arrow::datatypes::DataType;
    use nousdb_core::StringColumn;

    match (a.is_null(ai), b.is_null(bi)) {
        (true, true) => return true,
        (false, false) => {}
        _ => return false,
    }

    // Strings first, independent of encoding: a `(Utf8, Utf8)`-only arm would
    // fall through to `_ => false` for `Utf8View`, scoring every probe as a
    // disagreement.
    if let (Some(a), Some(b)) = (StringColumn::try_new(a), StringColumn::try_new(b)) {
        return canonical_text(a.value(ai)) == canonical_text(b.value(bi));
    }

    macro_rules! both {
        ($ty:ty) => {
            match (
                a.as_any().downcast_ref::<$ty>(),
                b.as_any().downcast_ref::<$ty>(),
            ) {
                (Some(a), Some(b)) => (a.value(ai), b.value(bi)),
                _ => return false,
            }
        };
    }

    match (a.data_type(), b.data_type()) {
        (DataType::Boolean, DataType::Boolean) => {
            let (a, b) = both!(BooleanArray);
            a == b
        }
        (DataType::Int64, DataType::Int64) => {
            let (a, b) = both!(Int64Array);
            a == b
        }
        (DataType::Float64, DataType::Float64) => {
            let (a, b) = both!(Float64Array);
            a == b || (a.is_nan() && b.is_nan())
        }
        _ => false,
    }
}

/// Agreement predicate for extend calibration: a probe agrees only when every
/// derived field agrees. A missing probe row (extend preserves cardinality, so
/// this means rows were dropped) or a missing column is a disagreement.
pub(crate) fn extend_probe_scorer(output_schema: SchemaRef) -> ProbeScorer {
    Arc::new(
        move |rung: &ProbeObservation, reference: &ProbeObservation| {
            let (Some(rung_row), Some(reference_row)) = (rung.row, reference.row) else {
                return false;
            };
            output_schema.fields().iter().all(|field| {
                match (
                    rung.batch.column_by_name(field.name()),
                    reference.batch.column_by_name(field.name()),
                ) {
                    (Some(a), Some(b)) => {
                        derived_values_agree(a.as_ref(), rung_row, b.as_ref(), reference_row)
                    }
                    _ => false,
                }
            })
        },
    )
}

impl Stream for ExtendStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }

        // Calibration prologue (quality certification). Free when disabled;
        // otherwise buffers the prefix and awaits the shared calibration
        // before any production request is dispatched.
        match poll_calibration(
            &mut this.calibration,
            cx,
            CalibrationHost {
                input: &mut this.input,
                optimal_batch_size: &mut this.optimal_batch_size,
                pending_slices: &mut this.pending_slices,
                coalesce_buffer: &mut this.coalesce_buffer,
                coalesce_rows: &mut this.coalesce_rows,
                input_exhausted: &mut this.input_exhausted,
            },
        ) {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(e)) => {
                this.done = true;
                return Poll::Ready(Some(Err(e)));
            }
            Poll::Pending => return Poll::Pending,
        }

        loop {
            // Emit any batch the output coalescer has already completed
            // before doing more work.
            if let Some(batch) = this.output_coalescer.next_completed_batch() {
                return Poll::Ready(Some(Ok(batch)));
            }

            // Phase 1: keep the request window full while input can supply
            // request-sized batches. Requests dispatch while earlier ones are
            // still in flight; there is no per-request barrier.
            let mut input_pending = false;
            while this.in_flight.len() < this.request_window {
                match this.next_request_batch(cx) {
                    Ok(NextRequest::Ready(batch)) => this.spawn_request(batch),
                    Ok(NextRequest::Pending) => {
                        input_pending = true;
                        break;
                    }
                    Ok(NextRequest::Exhausted) => break,
                    Err(e) => {
                        this.done = true;
                        return Poll::Ready(Some(Err(e)));
                    }
                }
            }

            // Phase 2: surface the next completed request, in submission
            // order (FuturesOrdered), so output order matches input order.
            match this.in_flight.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(processing_result))) => {
                    this.metrics.record_inference(&processing_result.metrics);
                    this.inference_timer.stop();
                    // Rows count as produced at request completion, before
                    // the output coalescer emits them downstream.
                    this.metrics
                        .record_rows_processed(processing_result.rows_processed);
                    this.metrics
                        .record_rows_produced(processing_result.batch.num_rows());
                    if let Some(n) = processing_result.failed_rows {
                        this.metrics.record_batch_failure(n);
                    }
                    if processing_result.truncated {
                        this.metrics.record_truncation();
                    }
                    this.report_request_outcome(processing_result.outcome);

                    // Probe streams keep one batch per request, as the
                    // calibration scorer expects.
                    if this.request_sink.is_some() {
                        return Poll::Ready(Some(Ok(processing_result.batch)));
                    }
                    if processing_result.batch.num_rows() > 0 {
                        this.output_coalescer.push_batch(processing_result.batch)?;
                    }
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    // A probe stream survives a request error: it is reported
                    // as a failure and stands in as an empty batch, keeping
                    // one batch per request. Provider errors never get here
                    // in probe mode (`process_batch` reports them as
                    // `NoAnswer`), so this is an undecodable answer.
                    if this.request_sink.is_some() {
                        log::debug!("[certify] probe request failed, scored as disagreement: {e}");
                        this.report_request_outcome(RequestOutcome::Unusable);
                        return Poll::Ready(Some(Ok(RecordBatch::new_empty(this.schema.clone()))));
                    }
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // Nothing in flight. Phase 1 stopped on Exhausted or
                    // Pending (a full window would have left futures here).
                    if this.input_exhausted {
                        // Flush the coalescer's partial tail before finishing.
                        // `done` must not be set while rows are still buffered:
                        // it short-circuits the next poll.
                        this.output_coalescer.finish_buffered_batch()?;
                        if let Some(batch) = this.output_coalescer.next_completed_batch() {
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        this.done = true;
                        return Poll::Ready(None);
                    }
                    if input_pending {
                        return Poll::Pending;
                    }
                    continue;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl RecordBatchStream for ExtendStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use serde_json::json;

    fn utf8_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("out", DataType::Utf8, true)]))
    }

    fn multi_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]))
    }

    #[test]
    fn row_json_wraps_scalars_for_single_field_schemas() {
        let names = vec!["out".to_string()];
        assert_eq!(
            output_to_row_json(&ModelOutput::Text("x".into()), &names).unwrap(),
            json!({"out": "x"})
        );
        assert_eq!(
            output_to_row_json(&ModelOutput::Boolean(true), &names).unwrap(),
            json!({"out": true})
        );
        assert_eq!(
            output_to_row_json(&ModelOutput::Float(1.5), &names).unwrap(),
            json!({"out": 1.5})
        );
        assert_eq!(
            output_to_row_json(&ModelOutput::Structured(json!(7)), &names).unwrap(),
            json!({"out": 7})
        );
        // Objects pass through untouched, regardless of schema arity.
        assert_eq!(
            output_to_row_json(&ModelOutput::Structured(json!({"a": 1, "b": 2})), &names).unwrap(),
            json!({"a": 1, "b": 2})
        );
    }

    #[test]
    fn row_json_rejects_scalars_for_multi_field_schemas() {
        let names = vec!["a".to_string(), "b".to_string()];
        assert!(output_to_row_json(&ModelOutput::Text("x".into()), &names).is_err());
        assert!(output_to_row_json(&ModelOutput::Structured(json!(7)), &names).is_err());
        assert!(output_to_row_json(&ModelOutput::Embedding(vec![0.0]), &names).is_err());
    }

    /// Build a probe observation whose derived columns hold `values` at row 0.
    fn observed(schema: &SchemaRef, values: Vec<ArrayRef>) -> ProbeObservation {
        ProbeObservation {
            batch: RecordBatch::try_new(schema.clone(), values).unwrap(),
            row: Some(0),
            outcome: RequestOutcome::Answered,
        }
    }

    fn utf8_obs(schema: &SchemaRef, v: Option<&str>) -> ProbeObservation {
        observed(schema, vec![Arc::new(StringArray::from(vec![v]))])
    }

    /// Canonical normalization (trim / case / whitespace) is the only
    /// certification-specific rule; everything else is the value production
    /// already built. Outputs that fail to coerce never reach the scorer: they
    /// fail the request, which is scored as disagreement via the outcome sink.
    #[test]
    fn utf8_agreement_is_canonicalized() {
        let schema = utf8_schema();
        let scorer = extend_probe_scorer(schema.clone());
        assert!(scorer(
            &utf8_obs(&schema, Some("  POSITIVE   review ")),
            &utf8_obs(&schema, Some("positive review"))
        ));
        assert!(!scorer(
            &utf8_obs(&schema, Some("positive")),
            &utf8_obs(&schema, Some("negative"))
        ));
    }

    #[test]
    fn typed_fields_compare_exactly() {
        let int_schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("out", DataType::Int64, true)]));
        let scorer = extend_probe_scorer(int_schema.clone());
        let obs = |v: i64| observed(&int_schema, vec![Arc::new(Int64Array::from(vec![v]))]);
        assert!(scorer(&obs(42), &obs(42)));
        assert!(!scorer(&obs(42), &obs(43)));

        let float_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "out",
            DataType::Float64,
            true,
        )]));
        let scorer = extend_probe_scorer(float_schema.clone());
        let obs = |v: f64| observed(&float_schema, vec![Arc::new(Float64Array::from(vec![v]))]);
        assert!(scorer(&obs(1.0), &obs(1.0)));
        assert!(!scorer(&obs(1.0), &obs(1.5)));
        // NaN == NaN: production would show the user the same value twice.
        assert!(scorer(&obs(f64::NAN), &obs(f64::NAN)));
    }

    #[test]
    fn multi_field_agreement_requires_every_field() {
        let schema = multi_schema();
        let scorer = extend_probe_scorer(schema.clone());
        let obs = |label: &str, score: f64| {
            observed(
                &schema,
                vec![
                    Arc::new(StringArray::from(vec![label])),
                    Arc::new(Float64Array::from(vec![score])),
                ],
            )
        };
        assert!(scorer(&obs(" comedy ", 0.9), &obs("Comedy", 0.9)));
        assert!(!scorer(&obs("comedy", 0.8), &obs("Comedy", 0.9)));
    }

    #[test]
    fn nulls_agree_with_nulls_and_a_missing_row_disagrees() {
        let schema = utf8_schema();
        let scorer = extend_probe_scorer(schema.clone());
        assert!(scorer(&utf8_obs(&schema, None), &utf8_obs(&schema, None)));
        assert!(!scorer(
            &utf8_obs(&schema, None),
            &utf8_obs(&schema, Some("something"))
        ));

        // Extend is cardinality-preserving, so an absent probe row means rows
        // were dropped, which is never agreement.
        let mut absent = utf8_obs(&schema, Some("x"));
        absent.row = None;
        assert!(!scorer(&absent, &utf8_obs(&schema, Some("x"))));
    }

    /// The scorer is built from the derived schema, so a free-text `rationale`
    /// column in the batch is never compared.
    #[test]
    fn probe_scoring_ignores_a_rationale_column() {
        let derived = utf8_schema();
        let scorer = extend_probe_scorer(derived.clone());

        let with_rationale: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("out", DataType::Utf8, true),
            Field::new("rationale", DataType::Utf8, true),
        ]));
        let obs = |out: &str, why: &str| {
            observed(
                &with_rationale,
                vec![
                    Arc::new(StringArray::from(vec![out])),
                    Arc::new(StringArray::from(vec![why])),
                ],
            )
        };

        assert!(
            scorer(
                &obs("positive", "the reviewer praises the pacing"),
                &obs("positive", "warm about the score, so positive"),
            ),
            "wildly different rationales must not count as disagreement"
        );
        assert!(!scorer(
            &obs("positive", "same words"),
            &obs("negative", "same words"),
        ));
    }

    /// The abstention convention end to end: a `null` answer must reach Arrow
    /// as a NULL, not as the empty string.
    #[test]
    fn a_null_answer_becomes_an_arrow_null() {
        let names = vec!["out".to_string()];
        assert_eq!(
            output_to_row_json(&ModelOutput::Null, &names).unwrap(),
            json!({ "out": null })
        );

        let schema = utf8_schema();
        let json = outputs_to_json(
            &[ModelOutput::Null, ModelOutput::Text("baking".into())],
            &schema,
        )
        .unwrap();
        let prompt = Prompt::new("topic", None::<String>, "extend/user.md", schema);
        let arrays = prompt.parse_response(&json, 2).unwrap();
        let col = arrays[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8 column");
        assert!(col.is_null(0), "an abstention must be NULL, not \"\"");
        assert_eq!(col.value(1), "baking");
    }
}
