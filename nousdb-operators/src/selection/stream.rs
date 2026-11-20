//! Execution stream for semantic selection: request packing, in-flight window and filtering.

use datafusion::arrow::array::BooleanArray;
use datafusion::arrow::compute::{BatchCoalescer, concat_batches, filter_record_batch};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};

use crate::certification::{
    CalibrationHost, CalibrationParts, CalibrationState, ProbeObservation, ProbeScorer,
    RequestOutcome, RequestOutcomeSink, StreamCertification, poll_calibration,
};
use crate::metrics::{SemanticOperatorMetrics, WallClockTimer};
use crate::selection::renderer::SemanticSelectionRenderer;
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
/// flight, and emits only the rows the model accepts.
///
/// With a fetch ([`Self::with_fetch`]) it ends after that many accepted rows,
/// and it dispatches a request only while the rows already in flight could
/// not fill the fetch even if the model accepted every one of them. Requests
/// are packed exactly as without a fetch, so the output is a prefix of the
/// unlimited output and no request is sent that the answers so far did not
/// make necessary.
pub struct SemanticSelectionStream {
    input: SendableRecordBatchStream,
    renderer: SemanticSelectionRenderer,
    prompt: Prompt,
    model_provider: Arc<dyn ModelProvider>,
    model_id: String,
    model_card: ModelCard,
    schema: SchemaRef,
    metrics: SemanticOperatorMetrics,
    inference_timer: Arc<WallClockTimer>,
    optimal_batch_size: Option<usize>,
    pending_slices: VecDeque<RecordBatch>,
    coalesce_buffer: Vec<RecordBatch>,
    coalesce_rows: usize,
    /// Re-coalesces per-request result batches (at most `optimal_batch_size`
    /// rows each, often fewer after filtering) up to the session's
    /// `batch_size` before emitting downstream. Bypassed in probe mode, which
    /// needs one batch per request.
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
    /// Accepted rows to emit before finishing (`None`: no limit).
    fetch: Option<usize>,
    /// Accepted rows handed to the output coalescer so far; at most `fetch`.
    accepted_rows: usize,
    /// Row count of each in-flight request, in submission order (the order
    /// `in_flight` completes in), and their sum.
    in_flight_sizes: VecDeque<usize>,
    in_flight_rows: usize,
    /// The fetch is filled: flush the output coalescer, then finish.
    fetch_reached: bool,
}

impl SemanticSelectionStream {
    /// Creates a stream over `input`; `schema` is the input schema and
    /// `output_batch_size` the target size of emitted batches.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: SendableRecordBatchStream,
        filter_instruction: String,
        input_columns: Vec<String>,
        model_provider: Arc<dyn ModelProvider>,
        model_id: String,
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
        let renderer = SemanticSelectionRenderer::new(filter_instruction.clone(), input_columns);

        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )]));

        let prompt = Prompt::new(
            filter_instruction.clone(),
            Some(reasoning.system_template(OperatorKind::Selection)),
            reasoning.user_template(OperatorKind::Selection),
            output_schema,
        )
        .with_reasoning(reasoning);

        let admission = model_provider.admission_control();

        let calibration = CalibrationState::new(certification, || CalibrationParts {
            scorer: selection_probe_scorer(),
            inference_timer: inference_timer.clone(),
        });

        let output_coalescer = BatchCoalescer::new(schema.clone(), output_batch_size);

        Self {
            input,
            renderer,
            prompt,
            model_provider,
            model_id,
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
            fetch: None,
            accepted_rows: 0,
            in_flight_sizes: VecDeque::new(),
            in_flight_rows: 0,
            fetch_reached: false,
        }
    }

    /// Finish after `fetch` accepted rows (`None`: read the whole input).
    /// A fetch of zero makes no model call.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        if fetch == Some(0) {
            self.done = true;
        }
        self
    }

    /// Whether to dispatch one more request. Without a fetch, always. With
    /// one, only while the rows in flight are fewer than the rows still
    /// missing: those requests cannot fill the fetch, so another is needed
    /// whatever the model answers.
    fn wants_another_request(&self) -> bool {
        match self.fetch {
            None => true,
            Some(fetch) => self.in_flight_rows < fetch.saturating_sub(self.accepted_rows),
        }
    }

    /// Turn this stream into a calibration probe stream: report every
    /// request's outcome and tolerate request errors.
    pub fn with_request_sink(mut self, sink: RequestOutcomeSink) -> Self {
        self.request_sink = Some(sink);
        self
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_batch(
        renderer: SemanticSelectionRenderer,
        prompt: Prompt,
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
        // length: on a parse failure the provider null-fills to `num_rows`, and
        // for a boolean predicate those nulls become `false`, so without the
        // flag a failed batch would filter out every row instead of passing
        // through. It also tells a request that never returned apart from one
        // that returned an unusable response.
        if let Some(rejection) = rejection {
            log::warn!(
                "SelectionRecovery: {rejection} — passing all {num_rows} rows through UNFILTERED; \
                 the predicate was not applied to them",
            );

            return Ok(ProcessingResult {
                batch: input_batch,
                metrics: accumulated_metrics,
                rows_processed: num_rows,
                failed_rows: Some(num_rows),
                truncated,
                outcome,
            });
        }

        let boolean_array = Self::outputs_to_boolean_array(&predict_result.outputs, num_rows)?;

        let filtered_batch = filter_record_batch(&input_batch, &boolean_array)?;

        Ok(ProcessingResult {
            batch: filtered_batch,
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
        self.in_flight_sizes.push_back(batch.num_rows());
        self.in_flight_rows += batch.num_rows();
        let fut = Self::process_batch(
            self.renderer.clone(),
            self.prompt.clone(),
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

    /// Report a completed request to the calibration sink, if installed.
    fn report_request_outcome(&mut self, outcome: RequestOutcome) {
        if let Some(sink) = &self.request_sink {
            sink.report(self.request_ordinal, outcome);
            self.request_ordinal += 1;
        }
    }

    fn outputs_to_boolean_array(
        outputs: &[ModelOutput],
        expected_count: usize,
    ) -> Result<BooleanArray> {
        if outputs.len() != expected_count {
            return Err(DataFusionError::Execution(format!(
                "Expected {} outputs but got {}",
                expected_count,
                outputs.len()
            )));
        }

        let mut booleans = Vec::with_capacity(expected_count);

        for (idx, output) in outputs.iter().enumerate() {
            let boolean_value = coerce_output_to_bool(output)
                .map_err(|e| DataFusionError::Execution(format!("{e} at row {idx}")))?;
            booleans.push(boolean_value);
        }

        Ok(BooleanArray::from(booleans))
    }
}

/// Coerce one model output slot to the selection predicate's boolean.
/// Shared by production filtering and calibration slot agreement.
pub(crate) fn coerce_output_to_bool(output: &ModelOutput) -> Result<bool> {
    match output {
        ModelOutput::Boolean(b) => Ok(*b),
        ModelOutput::Text(s) => {
            let s_lower = s.trim().to_lowercase();
            Ok(match s_lower.as_str() {
                "yes" | "true" | "1" => true,
                "no" | "false" | "0" => false,
                _ => false, // Default to false for unrecognized strings
            })
        }
        ModelOutput::Structured(v) => {
            if let Some(obj) = v.as_object() {
                // Schema field is "matches_filter"; fall back to a lone
                // boolean entry for models that ignore the field name.
                let field_value = obj.get("matches_filter").or_else(|| {
                    if obj.len() == 1 {
                        obj.values().next()
                    } else {
                        None
                    }
                });

                if let Some(field_value) = field_value {
                    if let Some(b) = field_value.as_bool() {
                        Ok(b)
                    } else {
                        Err(DataFusionError::Execution(format!(
                            "Boolean field in structured output is not a boolean: {:?}",
                            field_value
                        )))
                    }
                } else {
                    Err(DataFusionError::Execution(format!(
                        "Could not find boolean field in structured output: {:?}",
                        obj
                    )))
                }
            } else {
                Err(DataFusionError::Execution(format!(
                    "Structured output is not an object: {:?}",
                    v
                )))
            }
        }
        ModelOutput::Float(f) => Ok(*f >= 0.5),
        // A row the model declines to judge does not pass the filter.
        ModelOutput::Null => Ok(false),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported ModelOutput type for boolean conversion: {:?}",
            output
        ))),
    }
}

/// Agreement predicate for selection calibration.
///
/// The operator's answer for a probe row is whether that row survived, i.e.
/// whether its id is present in the output batch. Two arms agree when they made
/// the same keep/drop decision; `ModelOutput` coercion has already been applied.
pub(crate) fn selection_probe_scorer() -> ProbeScorer {
    Arc::new(|rung: &ProbeObservation, reference: &ProbeObservation| {
        rung.row.is_some() == reference.row.is_some()
    })
}

impl Stream for SemanticSelectionStream {
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
            if this.fetch_reached {
                this.done = true;
                return Poll::Ready(None);
            }

            // Phase 1: keep the request window full while input can supply
            // request-sized batches. Requests dispatch while earlier ones are
            // still in flight; there is no per-request barrier. A fetch holds
            // back requests the ones in flight may make unnecessary.
            let mut input_pending = false;
            while this.in_flight.len() < this.request_window && this.wants_another_request() {
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
            let completed = this.in_flight.poll_next_unpin(cx);
            if matches!(completed, Poll::Ready(Some(_))) {
                let rows = this.in_flight_sizes.pop_front().unwrap_or(0);
                this.in_flight_rows -= rows;
            }
            match completed {
                Poll::Ready(Some(Ok(mut processing_result))) => {
                    // Under a fetch, keep only the rows still missing.
                    if let Some(fetch) = this.fetch {
                        let missing = fetch - this.accepted_rows;
                        if processing_result.batch.num_rows() > missing {
                            processing_result.batch = processing_result.batch.slice(0, missing);
                        }
                        this.accepted_rows += processing_result.batch.num_rows();
                    }
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
                    // calibration scorer expects, including batches filtered
                    // to empty.
                    if this.request_sink.is_some() {
                        return Poll::Ready(Some(Ok(processing_result.batch)));
                    }
                    if processing_result.batch.num_rows() > 0 {
                        this.output_coalescer.push_batch(processing_result.batch)?;
                    }
                    if this.fetch.is_some_and(|fetch| this.accepted_rows >= fetch) {
                        // Enough rows passed: the loop head emits the buffered
                        // rows and finishes. Nothing is in flight here, since
                        // `wants_another_request` never dispatches past the
                        // request that could fill the fetch; the reset only
                        // keeps that from depending on the dispatch rule.
                        this.in_flight = FuturesOrdered::new();
                        this.in_flight_sizes.clear();
                        this.in_flight_rows = 0;
                        this.output_coalescer.finish_buffered_batch()?;
                        this.fetch_reached = true;
                    }
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    // A probe stream survives a request error: it is reported
                    // as a failure and stands in as an empty batch, keeping
                    // one batch per request. Provider errors never get here
                    // in probe mode (`process_batch` reports them as
                    // `NoAnswer`), so this is an uncoercible answer.
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

impl RecordBatchStream for SemanticSelectionStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
