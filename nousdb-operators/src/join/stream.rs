//! Block-nested-loop stream for the semantic join, including its calibration
//! states and the shared once-initializer used for the right-side build.

use datafusion::arrow::array::{ArrayRef, StringBuilder, UInt64Array};
use datafusion::arrow::compute::{concat_batches, take};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::internal_datafusion_err;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};

use crate::certification::{
    CalibrationOutcome, ItemSpace, ProbeCalibrationContext, ProbeInputBuilder, ProbeScorer,
    RequestOutcome, RequestOutcomeSink, StreamCertification, attach_probe_ids,
    run_probe_calibration,
};
use crate::join::renderer::SemanticJoinRenderer;
use crate::metrics::{SemanticJoinMetrics, WallClockTimer};
use nousdb_core::certification::OperatorKind;
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::utils::{extract_value_from_array, join_static_prompt_tokens};
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::InferenceMetrics;
use nousdb_core::providers::provider::{AdmissionControl, ModelProvider};
use nousdb_core::providers::schemas::ModelOutput;
use nousdb_core::token_estimator::{
    TokenEstimator, compute_optimal_pair_batch_size, row_format_tokens,
};

use datafusion::arrow::compute::BatchCoalescer;
use futures::Future;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use futures::stream::FuturesOrdered;
use futures::stream::Stream;
use futures::stream::StreamExt;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Left rows to buffer before runtime block sizing, so the left operand
/// contributes its own token histogram instead of being proxied from the
/// right. Enough for a stable mean/quantile, small enough that the buffered
/// prefix recycles into a handful of blocks.
const LEFT_SIZING_SAMPLE_ROWS: usize = 256;

/// Everything the pre-flight overflow guard needs to price a concrete pair
/// set before dispatching it. Built once per stream after the right relation
/// materializes; `None` disables the guard (probe execs with no right
/// columns, or a failed estimator/template load; both are logged).
///
/// Block sizes are probabilistic: with probability at most α a real batch
/// still exceeds the envelope, and without a check that request degrades into
/// NULL-filled answers. Splitting is fidelity-safe under certification: it only
/// reduces the realized pair count below the certified k̂, and φ(k) is
/// non-increasing in k.
struct PreflightGuard {
    estimator: TokenEstimator,
    /// Effective input budget ([`ModelCard::effective_input_budget`]).
    available: usize,
    /// Measured static prompt tokens (templates + join condition).
    static_overhead: usize,
    left_fmt: usize,
    right_fmt: usize,
    pair_ref: usize,
    /// Join-column names of the left relation (for per-block tokenization).
    left_col_names: Vec<String>,
    /// Per-row token counts over the right relation's join columns,
    /// precomputed once because the right side is rescanned per left block.
    right_row_tokens: Vec<usize>,
}

/// Type alias for shared results across partitions.
type SharedResult<T> = std::result::Result<T, Arc<DataFusionError>>;

/// Shared future type for right-side data built once and shared across partitions.
type OnceFutPending<T> = Shared<BoxFuture<'static, SharedResult<Arc<T>>>>;

/// The materialized right (build-side) data, built once and shared across all
/// left (probe) partitions via `Arc`.
pub struct JoinRightData {
    /// Merged right relation (a single batch).
    pub batch: RecordBatch,
    /// Memory reservation tracking the right data's memory usage.
    /// Freed on drop.
    pub reservation: MemoryReservation,
}

impl std::fmt::Debug for JoinRightData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinRightData")
            .field("rows", &self.batch.num_rows())
            .field("mem_bytes", &self.reservation.size())
            .finish()
    }
}

/// Thread-safe one-shot async initializer.
///
/// The first call to `try_once()` executes the closure to produce a future.
/// All subsequent calls (from other partitions) return a clone of the same
/// shared future, so the right side is built once and shared across all left
/// partitions.
///
/// Mirrors DataFusion's `OnceAsync<T>`, which is not exported.
pub struct OnceAsync<T> {
    fut: std::sync::Mutex<Option<SharedResult<OnceFut<T>>>>,
}

impl<T> Default for OnceAsync<T> {
    fn default() -> Self {
        Self {
            fut: std::sync::Mutex::new(None),
        }
    }
}

impl<T> std::fmt::Debug for OnceAsync<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OnceAsync")
    }
}

impl<T: 'static> OnceAsync<T> {
    /// If this is the first call, invoke `f` to obtain a future and return
    /// a `OnceFut` wrapping it. Subsequent calls return a clone of the same
    /// shared future.
    pub fn try_once<F, Fut>(&self, f: F) -> Result<OnceFut<T>>
    where
        F: FnOnce() -> Result<Fut>,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.fut
            .lock()
            .expect("OnceAsync mutex poisoned")
            .get_or_insert_with(|| f().map(OnceFut::new).map_err(Arc::new))
            .clone()
            .map_err(DataFusionError::Shared)
    }
}

/// A shared future that resolves to `Arc<T>`. Cloning is cheap (reference count).
pub struct OnceFut<T> {
    state: OnceFutState<T>,
}

impl<T> Clone for OnceFut<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

enum OnceFutState<T> {
    Pending(OnceFutPending<T>),
    Ready(SharedResult<Arc<T>>),
}

impl<T> Clone for OnceFutState<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending(p) => Self::Pending(p.clone()),
            Self::Ready(r) => Self::Ready(r.clone()),
        }
    }
}

impl<T: 'static> OnceFut<T> {
    fn new<Fut>(fut: Fut) -> Self
    where
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        use futures::FutureExt;
        Self {
            state: OnceFutState::Pending(
                fut.map(|res| res.map(Arc::new).map_err(Arc::new))
                    .boxed()
                    .shared(),
            ),
        }
    }

    /// Poll the underlying future. Returns `Poll::Ready(&T)` when done.
    pub fn get(&mut self, cx: &mut Context<'_>) -> Poll<Result<&T>> {
        if let OnceFutState::Pending(fut) = &mut self.state {
            let r = futures::ready!(fut.poll_unpin(cx));
            self.state = OnceFutState::Ready(r);
        }

        match &self.state {
            OnceFutState::Pending(_) => unreachable!(),
            OnceFutState::Ready(r) => Poll::Ready(
                r.as_ref()
                    .map(|r| r.as_ref())
                    .map_err(|e| DataFusionError::Shared(e.clone())),
            ),
        }
    }
}

/// Result of processing a batch of pairs: the output batch and its metrics.
struct ProcessingResult {
    batch: RecordBatch,
    metrics: InferenceMetrics,
    pairs_evaluated: usize,
    failed_rows: Option<usize>,
    /// The model ran out of output budget mid-response. Independent of
    /// `failed_rows`: it says why a response was unusable, and is counted even
    /// when the response still decoded.
    truncated: bool,
    /// What a probe stream reports to its sink; production ignores it.
    outcome: RequestOutcome,
}

type RequestFuture = Pin<Box<dyn Future<Output = Result<ProcessingResult>> + Send>>;

/// Outcome of trying to generate the next capped pair set.
enum NextPairs {
    /// A pair set plus a clone of the left block it indexes into.
    Ready(Vec<(usize, usize)>, RecordBatch),
    /// Left input returned `Poll::Pending` (its waker is registered).
    Pending,
    /// Left input is exhausted and no buffered rows remain.
    Exhausted,
}

/// Block Nested Loop Join state machine.
///
/// ```text
/// WaitingForBuildData → ComputeBlockSizes
///     → [CalibrationBuffering → CalibrationAwaiting]   (certification only)
///     → Pumping → Done
/// ```
enum StreamState {
    /// Wait for the shared right-side data. The first partition triggers
    /// materialization; others await the same future.
    WaitingForBuildData,

    /// After right is materialized, adopt plan-time block sizes when the
    /// optimizer provided them; otherwise fall through to runtime sizing.
    ComputeBlockSizes,

    /// Runtime sizing only (no plan-time block sizes): buffer a prefix of the
    /// left input so both operands contribute a real token histogram, size
    /// the blocks, then recycle the buffered rows through normal pumping.
    /// Certification never takes this path: a spec requires plan-time sizes.
    SamplingLeftRows,

    /// Quality certification only: buffer left rows until the production
    /// pair enumeration can cover the calibration prefix (or the left side
    /// is exhausted), then race to install the shared calibration future.
    CalibrationBuffering,

    /// Quality certification only: await the shared calibration outcome,
    /// re-shape the block geometry to the certified pair budget, and recycle
    /// the buffered left rows through normal pumping.
    CalibrationAwaiting,

    /// Collect left blocks, generate L×R pair sets, keep up to
    /// `request_window` LLM requests in flight, and push completed results (in
    /// submission order) into the output coalescer, without barriers.
    Pumping,

    /// All pairs processed.
    Done,
}

/// Stream that evaluates a semantic join for one left partition against the
/// shared right relation, using block nested loops over model-evaluated pairs.
pub struct SemanticJoinStream {
    /// Left input stream (outer relation, streamed in blocks)
    left_input: SendableRecordBatchStream,

    /// Shared future for the right (build) side data. Resolves to
    /// `Arc<JoinRightData>` once materialization completes.
    right_fut: OnceFut<JoinRightData>,

    /// Materialized right relation (single RecordBatch). Set once
    /// `right_fut` resolves.
    right_data: Option<Arc<JoinRightData>>,

    /// Total rows in right relation (cached after materialization)
    right_total_rows: usize,

    /// Column indices from left relation for join evaluation
    left_join_indices: Vec<usize>,

    /// Column indices from right relation for join evaluation
    right_join_indices: Vec<usize>,

    /// Left-side column count of the pair batch: the naming split for
    /// `build_pair_batch` and the reference-table split the renderer applies.
    /// Taken from the resolved column-name lists, so a probe exec (which
    /// gathers every pair column off its left child) names and renders the two
    /// sides as production does.
    pair_left_cols: usize,

    /// The natural language join condition
    join_condition: String,

    /// Renderer for join-specific data preparation
    renderer: SemanticJoinRenderer,

    /// Prompt for template rendering and response parsing
    prompt: Prompt,

    /// The model provider to use for inference
    model_provider: Arc<dyn ModelProvider>,

    /// Which model to use
    model_id: String,

    /// Model configuration for token-aware batch sizing
    model_card: ModelCard,

    /// Output schema (concatenation of left and right schemas)
    schema: SchemaRef,

    /// Maximum pairs to evaluate in one LLM call (computed from token budget)
    max_pairs_per_call: usize,

    /// Rows per left block
    left_block_size: usize,

    /// Rows per right block
    right_block_size: usize,

    /// Current accumulated left block (concatenated into single batch)
    current_left_block: Option<RecordBatch>,

    /// Row count in current left block
    current_left_block_rows: usize,

    /// Leftover rows from a batch split that exceeded `left_block_size`.
    /// Consumed first when the next left block is collected.
    leftover_batch: Option<RecordBatch>,

    /// Accumulation buffer for the current left block. A field rather than a
    /// local so state survives `Poll::Pending` re-entry.
    pending_left_batches: Vec<RecordBatch>,

    /// Current scan position within the right relation for this left block
    right_scan_offset: usize,

    /// Whether the left input stream is exhausted
    left_exhausted: bool,

    /// Metrics for tracking LLM calls and token usage
    metrics: SemanticJoinMetrics,

    /// Shared wall-clock timer for inference time
    inference_timer: Arc<WallClockTimer>,

    /// Output batch coalescer, accumulates small output batches and emits
    /// when reaching target_batch_size rows.
    output_coalescer: BatchCoalescer,

    /// Pre-computed left block size from optimizer (None = compute at runtime)
    precomputed_left_block_size: Option<usize>,

    /// Pre-computed right block size from optimizer (None = compute at runtime)
    precomputed_right_block_size: Option<usize>,

    /// Left batches buffered by `SamplingLeftRows` for runtime sizing;
    /// recycled through the normal pipeline (via `leftover_batch`) once the
    /// block sizes are set.
    sizing_sample: Vec<RecordBatch>,

    /// Total rows across `sizing_sample`.
    sizing_sample_rows: usize,

    /// Pre-flight overflow guard (`None` disables it). Built in
    /// `ComputeBlockSizes`, once the right relation is materialized.
    preflight: Option<PreflightGuard>,

    /// Per-row token counts over the current left block's join columns,
    /// filled when the block is finalized. Empty when the guard is disabled.
    current_left_block_tokens: Vec<usize>,

    /// Max concurrent in-flight pair-batch requests (1 = serial dispatch).
    /// Results surface in submission order regardless of the window.
    request_window: usize,

    /// Endpoint-global permit pool shared across partitions and operators.
    admission: Option<AdmissionControl>,

    in_flight: FuturesOrdered<RequestFuture>,

    /// All pair sets have been generated (left side fully consumed).
    generation_done: bool,

    /// Quality-certification handle (spec, cache, shared once, metrics).
    /// `None` skips the calibration states entirely.
    certification: Option<StreamCertification>,

    /// Left batches buffered for the calibration prefix; recycled through
    /// the normal pipeline (via `leftover_batch`) once the outcome lands.
    calib_buffered: Vec<RecordBatch>,

    /// Total rows across `calib_buffered`.
    calib_rows: usize,

    /// Shared calibration future (present in CalibrationAwaiting).
    calib_fut: Option<OnceFut<CalibrationOutcome>>,

    /// Whether this partition's closure won the install race (diagnostic).
    calib_installed: bool,

    /// Probe mode, set only on the execs the calibration prologue builds.
    /// Its presence changes three things so that agreement is measured rather
    /// than masked:
    ///
    /// - each completed request reports its outcome by submission ordinal, so
    ///   the prologue can score a failed request as a disagreement instead of
    ///   reading the "emit no matches" recovery as agreement;
    /// - a per-request `Err` is reported and replaced by an empty batch rather
    ///   than ending the stream, so one bad probe costs one disagreement;
    /// - output is emitted one batch per request, bypassing the coalescer,
    ///   because the prologue maps request ordinal to probe by position and an
    ///   empty result is a meaningful answer (the pair did not match).
    request_sink: Option<RequestOutcomeSink>,

    /// Submission ordinal of the next request to complete (probe mode only).
    request_ordinal: usize,

    /// Current state
    state: StreamState,
}

impl SemanticJoinStream {
    /// Creates the stream for one left partition.
    ///
    /// `precomputed_left_block_size` and `precomputed_right_block_size` are the
    /// optimizer's block sizes; when either is `None` the sizes are computed at
    /// runtime. `certification` enables the calibration states.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input: SendableRecordBatchStream,
        right_fut: OnceFut<JoinRightData>,
        left_join_indices: Vec<usize>,
        right_join_indices: Vec<usize>,
        join_condition: String,
        model_provider: Arc<dyn ModelProvider>,
        model_id: String,
        model_card: ModelCard,
        schema: SchemaRef,
        metrics: SemanticJoinMetrics,
        inference_timer: Arc<WallClockTimer>,
        output_batch_size: usize,
        left_col_names: Vec<String>,
        right_col_names: Vec<String>,
        precomputed_left_block_size: Option<usize>,
        precomputed_right_block_size: Option<usize>,
        request_window: usize,
        reasoning: ReasoningMode,
        certification: Option<StreamCertification>,
    ) -> Self {
        // The renderer splits the pair batch by these name lists' lengths, not
        // by `left_join_indices`, so a probe exec can gather all pair columns
        // off one side and still render the two reference tables.
        let pair_left_cols = left_col_names.len();
        let renderer =
            SemanticJoinRenderer::new(join_condition.clone(), left_col_names, right_col_names);

        // The model returns a boolean per pair.
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "join_match",
            DataType::Boolean,
            false,
        )]));

        let prompt = Prompt::new(
            join_condition.clone(),
            Some(reasoning.system_template(OperatorKind::Join)),
            reasoning.user_template(OperatorKind::Join),
            output_schema,
        )
        .with_reasoning(reasoning);

        let output_coalescer = BatchCoalescer::new(schema.clone(), output_batch_size);

        let admission = model_provider.admission_control();

        Self {
            left_input,
            right_fut,
            right_data: None,
            right_total_rows: 0,
            left_join_indices,
            right_join_indices,
            pair_left_cols,
            join_condition,
            renderer,
            prompt,
            model_provider,
            model_id,
            model_card,
            schema,
            max_pairs_per_call: 0,
            left_block_size: 0,
            right_block_size: 0,
            current_left_block: None,
            current_left_block_rows: 0,
            leftover_batch: None,
            pending_left_batches: Vec::new(),
            right_scan_offset: 0,
            left_exhausted: false,
            metrics,
            inference_timer,
            output_coalescer,
            precomputed_left_block_size,
            precomputed_right_block_size,
            sizing_sample: Vec::new(),
            sizing_sample_rows: 0,
            preflight: None,
            current_left_block_tokens: Vec::new(),
            request_window: request_window.max(1),
            admission,
            in_flight: FuturesOrdered::new(),
            generation_done: false,
            certification,
            calib_buffered: Vec::new(),
            calib_rows: 0,
            calib_fut: None,
            calib_installed: false,
            request_sink: None,
            request_ordinal: 0,
            state: StreamState::WaitingForBuildData,
        }
    }

    /// Report each request's outcome by submission ordinal (probe execs only).
    pub fn with_request_sink(mut self, sink: RequestOutcomeSink) -> Self {
        self.request_sink = Some(sink);
        self
    }

    /// Record one completed request's outcome and advance the ordinal.
    /// `FuturesOrdered` surfaces results in submission order, so the ordinal
    /// identifies the request, and therefore the probe, exactly.
    fn report_request_outcome(&mut self, outcome: RequestOutcome) {
        if let Some(sink) = &self.request_sink {
            sink.report(self.request_ordinal, outcome);
        }
        self.request_ordinal += 1;
    }

    /// Converts an Arrow array value to a string for prompt construction.
    ///
    /// Delegates to core's `extract_value_from_array`, the rendering rule
    /// extend and selection also use, so a join column reads the same bytes in
    /// a pair batch as in a selection or extend prompt.
    fn array_value_to_string(array: &ArrayRef, index: usize) -> Result<String> {
        extract_value_from_array(array, index)
    }

    /// Build the synthetic pair-row batch sent to the model: one row per
    /// (left, right) pair, join columns only, stringified as `left_0..`,
    /// `right_0..` (all Utf8). Shared by production dispatch and the
    /// calibration prologue, so probe requests render byte-identically to
    /// production requests.
    ///
    /// `num_left_cols` names the columns rather than counting them: gathered
    /// columns beyond the first `num_left_cols` become `right_*`, whichever
    /// relation they were taken from. In production that equals
    /// `left_join_indices.len()`. A probe exec gathers every pair column off its
    /// left child (its right child is a placeholder), and passing the real split
    /// keeps its batch identical, names included, to the production batch for
    /// the same pairs.
    pub(crate) fn build_pair_batch(
        left_batch: &RecordBatch,
        right_batch: &RecordBatch,
        pairs: &[(usize, usize)],
        left_join_indices: &[usize],
        right_join_indices: &[usize],
        num_left_cols: usize,
    ) -> Result<RecordBatch> {
        let left_indices: UInt64Array = pairs.iter().map(|(l, _)| Some(*l as u64)).collect();
        let right_indices: UInt64Array = pairs.iter().map(|(_, r)| Some(*r as u64)).collect();

        let mut pair_batch_columns: Vec<ArrayRef> = Vec::new();
        let mut pair_schema_fields = Vec::new();
        let name = |i: usize| {
            if i < num_left_cols {
                Field::new(format!("left_{}", i), DataType::Utf8, false)
            } else {
                Field::new(
                    format!("right_{}", i - num_left_cols),
                    DataType::Utf8,
                    false,
                )
            }
        };

        for (i, &col_idx) in left_join_indices.iter().enumerate() {
            let taken = take(left_batch.column(col_idx), &left_indices, None)?;
            let string_col = Self::array_to_string_array(&taken)?;
            pair_batch_columns.push(string_col);
            pair_schema_fields.push(name(i));
        }

        let offset = left_join_indices.len();
        for (i, &col_idx) in right_join_indices.iter().enumerate() {
            let taken = take(right_batch.column(col_idx), &right_indices, None)?;
            let string_col = Self::array_to_string_array(&taken)?;
            pair_batch_columns.push(string_col);
            pair_schema_fields.push(name(offset + i));
        }

        let pair_schema = Arc::new(Schema::new(pair_schema_fields));
        Ok(RecordBatch::try_new(pair_schema, pair_batch_columns)?)
    }

    /// Processes a batch of pairs through the model, using Arrow `take()` for
    /// both prompt construction and output assembly.
    #[allow(clippy::too_many_arguments)]
    async fn process_pair_batch(
        left_batch: RecordBatch,
        right_batch: RecordBatch,
        pairs: Vec<(usize, usize)>,
        left_join_indices: Vec<usize>,
        right_join_indices: Vec<usize>,
        num_left_cols: usize,
        _join_condition: String,
        renderer: SemanticJoinRenderer,
        prompt: Prompt,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        schema: SchemaRef,
        admission: Option<AdmissionControl>,
        probe: bool,
    ) -> Result<ProcessingResult> {
        let pairs_count = pairs.len();

        if pairs.is_empty() {
            return Ok(ProcessingResult {
                batch: RecordBatch::new_empty(schema),
                metrics: InferenceMetrics::zero(),
                pairs_evaluated: 0,
                failed_rows: None,
                truncated: false,
                outcome: RequestOutcome::Answered,
            });
        }

        // One permit per in-flight request, held until the response is
        // processed so the endpoint-global cap bounds actual load.
        let _permit = match admission {
            Some(ac) => Some(ac.acquire().await),
            None => None,
        };

        let pair_batch = Self::build_pair_batch(
            &left_batch,
            &right_batch,
            &pairs,
            &left_join_indices,
            &right_join_indices,
            num_left_cols,
        )?;

        log::debug!(
            "[SemanticJoin] Calling model '{}' with {} pairs (pair_batch: {} rows x {} cols)",
            model_id,
            pairs.len(),
            pair_batch.num_rows(),
            pair_batch.num_columns(),
        );

        // A provider `Err` ends the stream in production. A probe stream
        // reports it as a request that got no answer: one disagreement for this
        // calibration, but no evidence about the model (see `RequestOutcome`).
        let predict_result = match model_provider
            .predict(&model_id, &prompt, &pair_batch, 0, pairs.len(), &renderer)
            .await
        {
            Ok(result) => result,
            Err(e) if probe => {
                log::debug!(
                    "[SemanticJoin] probe request got no answer, scored as disagreement: {e}"
                );
                return Ok(ProcessingResult {
                    batch: RecordBatch::new_empty(schema),
                    metrics: InferenceMetrics::zero(),
                    pairs_evaluated: pairs_count,
                    failed_rows: Some(pairs_count),
                    truncated: false,
                    outcome: RequestOutcome::NoAnswer,
                });
            }
            Err(e) => return Err(e),
        };

        // Read before `metrics` is moved out of `predict_result` below.
        let truncated = predict_result.truncated;

        // `PredictResult::rejection` checks `row_count_mismatch` as well as the
        // length: on a parse failure the provider null-fills to `pairs.len()`,
        // so the length alone always matches. It also distinguishes a request
        // that never returned from one that returned unusable output; keep the
        // two messages distinct.
        if let Some(rejection) = predict_result.rejection(pairs.len()) {
            log::warn!(
                "JoinRecovery: {rejection} — emitting no matches for all {pairs_count} pairs",
            );

            return Ok(ProcessingResult {
                batch: RecordBatch::new_empty(schema),
                metrics: predict_result.metrics,
                pairs_evaluated: pairs_count,
                failed_rows: Some(pairs_count),
                truncated,
                outcome: RequestOutcome::from_rejection(Some(&rejection)),
            });
        }

        let mut matched_pairs = Vec::new();
        for (i, output) in predict_result.outputs.iter().enumerate() {
            if Self::model_output_to_boolean(output)? {
                matched_pairs.push(pairs[i]);
            }
        }

        log::debug!(
            "[SemanticJoin] Model returned {} outputs, {} matched | tokens: in={}, out={}, total={}",
            predict_result.outputs.len(),
            matched_pairs.len(),
            predict_result.metrics.input_tokens,
            predict_result.metrics.output_tokens,
            predict_result.metrics.total_tokens
        );

        if matched_pairs.is_empty() {
            return Ok(ProcessingResult {
                batch: RecordBatch::new_empty(schema),
                metrics: predict_result.metrics,
                pairs_evaluated: pairs_count,
                failed_rows: None,
                truncated,
                outcome: RequestOutcome::Answered,
            });
        }

        let batch =
            Self::construct_output_batch(&left_batch, &right_batch, &matched_pairs, schema)?;

        Ok(ProcessingResult {
            batch,
            metrics: predict_result.metrics,
            pairs_evaluated: pairs_count,
            failed_rows: None,
            truncated,
            outcome: RequestOutcome::Answered,
        })
    }

    /// Converts an Arrow array to a string array; the prompt needs a text
    /// representation of every data type. Nulls become `"NULL"`.
    fn array_to_string_array(array: &ArrayRef) -> Result<ArrayRef> {
        let len = array.len();
        let mut builder = StringBuilder::with_capacity(len, len * 16);

        for i in 0..len {
            if array.is_null(i) {
                builder.append_value("NULL");
            } else {
                let s = Self::array_value_to_string(array, i)?;
                builder.append_value(&s);
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Interprets a model output as a boolean match decision.
    fn model_output_to_boolean(output: &ModelOutput) -> Result<bool> {
        match output {
            ModelOutput::Boolean(b) => Ok(*b),
            ModelOutput::Text(s) => {
                let normalized = s.trim().to_lowercase();
                match normalized.as_str() {
                    "true" | "yes" | "1" => Ok(true),
                    "false" | "no" | "0" => Ok(false),
                    _ => Err(DataFusionError::Execution(format!(
                        "Could not interpret '{}' as boolean",
                        s
                    ))),
                }
            }
            ModelOutput::Structured(v) => {
                if let Some(obj) = v.as_object() {
                    let field_value = obj.get("join_match");
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
            _ => Err(DataFusionError::Execution(format!(
                "Unsupported model output type: {:?}",
                output
            ))),
        }
    }

    /// Builds the output batch (left columns, then right columns) for the
    /// matched pairs with one `take()` per column.
    fn construct_output_batch(
        left_data: &RecordBatch,
        right_data: &RecordBatch,
        matched_pairs: &[(usize, usize)],
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        let left_indices: UInt64Array =
            matched_pairs.iter().map(|(l, _)| Some(*l as u64)).collect();
        let right_indices: UInt64Array =
            matched_pairs.iter().map(|(_, r)| Some(*r as u64)).collect();

        let mut columns = Vec::with_capacity(schema.fields().len());
        for col in left_data.columns() {
            columns.push(take(col, &left_indices, None)?);
        }
        for col in right_data.columns() {
            columns.push(take(col, &right_indices, None)?);
        }

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    /// Builds the pre-flight overflow guard. Returns `None` (with a log line)
    /// when any input is unavailable, and the stream then dispatches unguarded.
    fn build_preflight(&self, right_batch: &RecordBatch) -> Option<PreflightGuard> {
        let estimator = match TokenEstimator::default_estimator() {
            Ok(est) => est,
            Err(e) => {
                log::warn!("[SemanticJoin] pre-flight guard disabled: no token estimator ({e})");
                return None;
            }
        };
        let left_schema = self.left_input.schema();
        let left_col_names: Vec<String> = self
            .left_join_indices
            .iter()
            .map(|&i| left_schema.field(i).name().clone())
            .collect();
        let right_col_names: Vec<String> = self
            .right_join_indices
            .iter()
            .map(|&i| right_batch.schema().field(i).name().clone())
            .collect();
        if right_col_names.is_empty() {
            // Probe execs: the right side is a placeholder and the certified
            // geometry already bounds the request.
            return None;
        }
        let right_row_tokens = match estimator.estimate_batch_tokens(right_batch, &right_col_names)
        {
            Ok(tokens) => tokens,
            Err(e) => {
                log::warn!(
                    "[SemanticJoin] pre-flight guard disabled: right tokenization failed ({e})"
                );
                return None;
            }
        };
        let reasoning = self.prompt.reasoning_mode();
        let static_overhead = match join_static_prompt_tokens(
            &reasoning.system_template(OperatorKind::Join),
            reasoning.user_template(OperatorKind::Join),
            &self.join_condition,
            &estimator,
        ) {
            Ok(tokens) => tokens,
            Err(e) => {
                log::warn!(
                    "[SemanticJoin] pre-flight guard disabled: template render failed ({e})"
                );
                return None;
            }
        };
        let left_fmt = row_format_tokens(&left_col_names, &estimator);
        let right_fmt = row_format_tokens(&right_col_names, &estimator);
        Some(PreflightGuard {
            available: self.model_card.effective_input_budget(),
            static_overhead,
            left_fmt,
            right_fmt,
            pair_ref: 15,
            left_col_names,
            right_row_tokens,
            estimator,
        })
    }

    /// Prices a concrete pair set against the effective input budget. Returns
    /// `None` when the guard is disabled or the left block was not tokenized.
    fn pair_set_fits(&self, pairs: &[(usize, usize)]) -> Option<bool> {
        let guard = self.preflight.as_ref()?;
        if self.current_left_block_tokens.is_empty() {
            return None;
        }
        let mut left_seen: HashSet<usize> = HashSet::new();
        let mut right_seen: HashSet<usize> = HashSet::new();
        let mut total = guard.static_overhead + pairs.len() * guard.pair_ref;
        for &(l, r) in pairs {
            if left_seen.insert(l) {
                total +=
                    self.current_left_block_tokens.get(l).copied().unwrap_or(0) + guard.left_fmt;
            }
            if right_seen.insert(r) {
                total += guard.right_row_tokens.get(r).copied().unwrap_or(0) + guard.right_fmt;
            }
        }
        Some(total <= guard.available)
    }

    /// Generates the next capped pair set, collecting left blocks as needed.
    ///
    /// Partial accumulation survives `Poll::Pending` re-entry via the stream's
    /// fields.
    fn next_pair_set(&mut self, cx: &mut Context<'_>) -> Result<NextPairs> {
        loop {
            // A. The current left block still has right rows to scan: emit one pair set.
            if self.current_left_block.is_some() {
                if self.right_scan_offset < self.right_total_rows {
                    let left_total = self.current_left_block_rows;
                    let scan_start = self.right_scan_offset;
                    let max_pairs = self.max_pairs_per_call;
                    let mut right_end =
                        (scan_start + self.right_block_size).min(self.right_total_rows);

                    let build_pairs = |right_end: usize| {
                        let mut pairs = Vec::new();
                        'block: for left_idx in 0..left_total {
                            for right_idx in scan_start..right_end {
                                pairs.push((left_idx, right_idx));
                                if pairs.len() >= max_pairs {
                                    break 'block;
                                }
                            }
                        }
                        pairs
                    };
                    let mut pairs = build_pairs(right_end);

                    // Pre-flight: block sizes are probabilistic, so a real batch
                    // can still exceed the input budget. Halve the right tile
                    // until the priced request fits rather than dispatching a
                    // request that would degrade into NULL-fills. Only the tile
                    // shrinks; the enumeration still visits every pair exactly
                    // once via `right_scan_offset`. Under certification a split
                    // only reduces the realized pair count below the certified
                    // k̂, and φ(k) is non-increasing in k, so it is
                    // fidelity-safe.
                    while pairs.len() > 1 && self.pair_set_fits(&pairs) == Some(false) {
                        self.metrics.base.record_overflow_error();
                        let width = right_end - scan_start;
                        if width <= 1 {
                            // The left block alone exceeds the budget and the
                            // tile cannot shrink further without dropping
                            // pairs. Dispatch and let per-request recovery
                            // handle a failure.
                            log::warn!(
                                "[SemanticJoin] pair set of {} pairs exceeds the input budget \
                                 even at right tile width 1; dispatching anyway",
                                pairs.len()
                            );
                            break;
                        }
                        right_end = scan_start + (width / 2).max(1);
                        pairs = build_pairs(right_end);
                    }

                    self.right_scan_offset = right_end;

                    log::debug!(
                        "[SemanticJoin] Generated {} pairs (right_scan_offset={}/{})",
                        pairs.len(),
                        self.right_scan_offset,
                        self.right_total_rows
                    );

                    if !pairs.is_empty() {
                        let left_block = self
                            .current_left_block
                            .clone()
                            .expect("left block checked above");
                        return Ok(NextPairs::Ready(pairs, left_block));
                    }
                }

                // Right scan complete for this block: drop it and collect the next.
                self.metrics.record_right_scan_completed();
                self.current_left_block = None;
                self.current_left_block_rows = 0;
                self.current_left_block_tokens.clear();
                self.right_scan_offset = 0;
            }

            // B. Consume leftover rows from a previous split.
            if let Some(leftover) = self.leftover_batch.take() {
                let leftover_rows = leftover.num_rows();
                if leftover_rows <= self.left_block_size - self.current_left_block_rows {
                    self.current_left_block_rows += leftover_rows;
                    self.pending_left_batches.push(leftover);
                } else {
                    let remaining_capacity = self.left_block_size - self.current_left_block_rows;
                    let take_batch = leftover.slice(0, remaining_capacity);
                    let rest_batch =
                        leftover.slice(remaining_capacity, leftover_rows - remaining_capacity);
                    self.current_left_block_rows += remaining_capacity;
                    self.pending_left_batches.push(take_batch);
                    self.leftover_batch = Some(rest_batch);
                }
            }

            // C. Poll the left input until the block is full or input is exhausted.
            while self.current_left_block_rows < self.left_block_size && !self.left_exhausted {
                match self.left_input.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(batch))) => {
                        if batch.num_rows() == 0 {
                            continue;
                        }
                        let remaining_capacity =
                            self.left_block_size - self.current_left_block_rows;
                        if batch.num_rows() <= remaining_capacity {
                            self.current_left_block_rows += batch.num_rows();
                            self.pending_left_batches.push(batch);
                        } else {
                            let take_batch = batch.slice(0, remaining_capacity);
                            let rest_batch = batch
                                .slice(remaining_capacity, batch.num_rows() - remaining_capacity);
                            self.current_left_block_rows += remaining_capacity;
                            self.pending_left_batches.push(take_batch);
                            self.leftover_batch = Some(rest_batch);
                        }
                    }
                    Poll::Ready(Some(Err(e))) => return Err(e),
                    Poll::Ready(None) => self.left_exhausted = true,
                    Poll::Pending => return Ok(NextPairs::Pending),
                }
            }

            // D. Finalize the block, or finish.
            if self.current_left_block_rows == 0 {
                return Ok(NextPairs::Exhausted);
            }

            let left_schema = self.pending_left_batches[0].schema();
            self.current_left_block =
                Some(concat_batches(&left_schema, &self.pending_left_batches)?);
            self.pending_left_batches.clear();
            self.right_scan_offset = 0;

            // Tokenize the block's join columns once for the pre-flight guard
            // (the block is priced once per right tile).
            if let (Some(guard), Some(block)) = (&self.preflight, &self.current_left_block) {
                self.current_left_block_tokens = guard
                    .estimator
                    .estimate_batch_tokens(block, &guard.left_col_names)
                    .unwrap_or_default();
            }

            log::debug!(
                "[SemanticJoin] Left block collected: {} rows, left_exhausted={}",
                self.current_left_block_rows,
                self.left_exhausted
            );
        }
    }

    /// Left rows to buffer before installing calibration.
    ///
    /// The design draws pairs from the `left × right` grid spanned by the
    /// pooled left rows and the materialized right relation, so it needs enough
    /// left rows that (a) the grid holds the design's item target and (b) the
    /// tallest tile (`k_max / R` left rows) can be assembled from distinct
    /// rows. Block alignment does not matter: a calibration tile is sampled,
    /// not enumerated in production order.
    fn calibration_left_rows_target(&self) -> usize {
        let Some(cert) = &self.certification else {
            return 0;
        };
        let spec = &cert.spec;
        let k_max = spec.ladder.first().copied().unwrap_or(1);
        let right_block = self.right_block_size.max(1);
        let tallest_tile = k_max.div_ceil(right_block);
        if self.right_total_rows == 0 {
            return tallest_tile;
        }
        spec.sample_target()
            .div_ceil(self.right_total_rows)
            .max(tallest_tile)
    }

    /// Race to install the shared calibration future.
    ///
    /// The winner's future hands the runner the `left × right` grid spanned by
    /// the pooled left rows (contributed by every partition) and the shared
    /// right relation. Only the pairs the design picks are materialized, so the
    /// join shares the runner without building its whole cross product. Every
    /// partition keeps and recycles its own buffered left rows.
    ///
    /// The closure only builds the future: `try_once` runs it synchronously
    /// under the `OnceAsync` mutex from inside `poll_next`, and waiting for the
    /// other partitions there would park a tokio worker thread on partitions
    /// that need one to make progress.
    fn install_calibration(&mut self) -> Result<()> {
        let cert = self.certification.take().ok_or_else(|| {
            internal_datafusion_err!("join calibration started without certification")
        })?;

        let left_schema = self.left_input.schema();
        let right_batch = self
            .right_data
            .as_ref()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "join calibration started before the right side was buffered"
                )
            })?
            .batch
            .clone();

        let ctx = ProbeCalibrationContext {
            factory: cert.factory.clone(),
            task_ctx: cert.task_ctx.clone(),
            window_budget: cert.window_budget,
            scorer: join_probe_scorer(),
            metrics: cert.metrics.clone(),
            inference_timer: self.inference_timer.clone(),
        };
        let spec = cert.spec.clone();
        let cache = cert.cache.clone();
        let left_join_indices = self.left_join_indices.clone();
        let right_join_indices = self.right_join_indices.clone();
        let num_left_cols = self.pair_left_cols;

        let pool = cert.pool.clone();
        let fut = cert.once.try_once(move || {
            Ok(async move {
                let left_batches = pool.assemble().await;
                let left_all = concat_batches(&left_schema, &left_batches)?;
                let right_rows = right_batch.num_rows();
                let space = ItemSpace::Grid {
                    left: left_all.num_rows(),
                    right: right_rows,
                };
                // Item id `p` is the pair `(p / right, p % right)`; the design
                // emits each tile left-major, so the materialized batch has the
                // shape production dispatches.
                let stride = right_rows.max(1);
                let build: ProbeInputBuilder = Arc::new(move |ids: &[usize]| {
                    let pairs: Vec<(usize, usize)> =
                        ids.iter().map(|&p| (p / stride, p % stride)).collect();
                    let batch = Self::build_pair_batch(
                        &left_all,
                        &right_batch,
                        &pairs,
                        &left_join_indices,
                        &right_join_indices,
                        num_left_cols,
                    )?;
                    attach_probe_ids(batch)
                });
                run_probe_calibration(spec, space, build, ctx, cache).await
            })
        })?;
        self.calib_fut = Some(fut);
        Ok(())
    }

    /// Dispatches one pair-batch request into the in-flight window.
    fn spawn_pair_request(&mut self, pairs: Vec<(usize, usize)>, left_batch: RecordBatch) {
        self.inference_timer.start();
        let right_batch = self
            .right_data
            .as_ref()
            .expect("right data must be set")
            .batch
            .clone();
        let fut = Self::process_pair_batch(
            left_batch,
            right_batch,
            pairs,
            self.left_join_indices.clone(),
            self.right_join_indices.clone(),
            self.pair_left_cols,
            self.join_condition.clone(),
            self.renderer.clone(),
            self.prompt.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.schema.clone(),
            self.admission.clone(),
            self.request_sink.is_some(),
        );
        self.in_flight.push_back(Box::pin(fut));
    }
}

impl Stream for SemanticJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Emit a completed batch if the coalescer has accumulated enough rows.
            if let Some(batch) = self.output_coalescer.next_completed_batch() {
                return Poll::Ready(Some(Ok(batch)));
            }

            match &mut self.state {
                // Wait for the shared right-side data.
                StreamState::WaitingForBuildData => {
                    let right_data = match self.right_fut.get(cx) {
                        Poll::Ready(Ok(data)) => data,
                        Poll::Ready(Err(e)) => {
                            self.state = StreamState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Pending => return Poll::Pending,
                    };

                    let right_rows = right_data.batch.num_rows();
                    let right_bytes = right_data.batch.get_array_memory_size();

                    self.right_total_rows = right_rows;
                    log::debug!(
                        "[SemanticJoin] Right data materialized: {} rows, {} bytes",
                        right_rows,
                        right_bytes
                    );

                    if self.right_total_rows == 0 {
                        self.state = StreamState::Done;
                        return Poll::Ready(None);
                    }

                    // Keep the shared `Arc<JoinRightData>` from the future's state.
                    if self.right_data.is_none()
                        && let OnceFutState::Ready(Ok(arc_data)) = &self.right_fut.state
                    {
                        self.right_data = Some(arc_data.clone());
                    }

                    self.state = StreamState::ComputeBlockSizes;
                    continue;
                }

                // Compute token-aware block sizes.
                StreamState::ComputeBlockSizes => {
                    // Arm the pre-flight overflow guard now that the right
                    // relation is materialized (both sizing branches pass
                    // through here once).
                    if self.preflight.is_none() {
                        let right_batch = self
                            .right_data
                            .as_ref()
                            .expect("right data must be set in ComputeBlockSizes")
                            .batch
                            .clone();
                        self.preflight = self.build_preflight(&right_batch);
                    }

                    // Use pre-computed block sizes if available, otherwise
                    // sample the left input and size at runtime.
                    if let (Some(left_bs), Some(right_bs)) = (
                        self.precomputed_left_block_size,
                        self.precomputed_right_block_size,
                    ) {
                        let max_pairs = left_bs * right_bs;
                        log::debug!(
                            "[SemanticJoin] Using pre-computed block sizes: left_block={}, right_block={}, max_pairs={}",
                            left_bs,
                            right_bs,
                            max_pairs
                        );

                        self.max_pairs_per_call = max_pairs;
                        self.left_block_size = left_bs;
                        self.right_block_size = right_bs;

                        self.metrics.record_chosen_batch_size(max_pairs);
                        self.metrics.record_left_block_size(left_bs);
                        self.metrics.record_right_block_size(right_bs);
                    } else {
                        // Runtime sizing needs a left sample first.
                        // `CertifyBatchSizeRule` only attaches a spec when both
                        // plan-time block sizes exist, so certification never
                        // reaches this branch.
                        debug_assert!(
                            self.certification.is_none(),
                            "certification requires plan-time block sizes"
                        );
                        self.state = StreamState::SamplingLeftRows;
                        continue;
                    }

                    if let Some(cert) = &self.certification {
                        // The rule only attaches a spec when both plan-time block
                        // sizes are `Some`, so the precomputed branch ran and
                        // the top rung equals the realized geometry.
                        debug_assert_eq!(
                            self.max_pairs_per_call,
                            cert.spec.ladder.first().copied().unwrap_or(0),
                            "certification expects plan-time geometry with product k*_alpha"
                        );
                        self.state = StreamState::CalibrationBuffering;
                    } else {
                        self.state = StreamState::Pumping;
                    }
                    continue;
                }

                // Sample the left input, then size at runtime.
                StreamState::SamplingLeftRows => {
                    while self.sizing_sample_rows < LEFT_SIZING_SAMPLE_ROWS && !self.left_exhausted
                    {
                        match self.left_input.poll_next_unpin(cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                if batch.num_rows() == 0 {
                                    continue;
                                }
                                self.sizing_sample_rows += batch.num_rows();
                                self.sizing_sample.push(batch);
                            }
                            Poll::Ready(Some(Err(e))) => {
                                self.state = StreamState::Done;
                                return Poll::Ready(Some(Err(e)));
                            }
                            Poll::Ready(None) => self.left_exhausted = true,
                            Poll::Pending => return Poll::Pending,
                        }
                    }

                    let right_batch = self
                        .right_data
                        .as_ref()
                        .expect("right data must be set in SamplingLeftRows")
                        .batch
                        .clone();
                    let right_batches = [right_batch];
                    let sizing = compute_optimal_pair_batch_size(
                        &self.sizing_sample,
                        &right_batches,
                        &self.join_condition,
                        &self.left_join_indices,
                        &self.right_join_indices,
                        &self.model_card,
                        self.prompt.reasoning_mode(),
                    )?;

                    log::debug!(
                        "[SemanticJoin] Runtime block sizes: max_pairs_per_call={}, left_block={}, \
                         right_block={}, histogram_rows={}, left_histogram_rows={}, bound={}",
                        sizing.max_pairs_per_call,
                        sizing.left_block_size,
                        sizing.right_block_size,
                        sizing.histogram_rows,
                        sizing.left_histogram_rows,
                        sizing.bound,
                    );

                    self.max_pairs_per_call = sizing.max_pairs_per_call;
                    self.left_block_size = sizing.left_block_size;
                    self.right_block_size = sizing.right_block_size;

                    self.metrics
                        .record_chosen_batch_size(sizing.max_pairs_per_call);
                    self.metrics.record_histogram_rows(sizing.histogram_rows);
                    self.metrics.record_left_block_size(sizing.left_block_size);
                    self.metrics
                        .record_right_block_size(sizing.right_block_size);

                    // Recycle the sampled prefix through the normal pipeline. It
                    // must flow through `leftover_batch`: section B of
                    // `next_pair_set` slices it against block capacity, while
                    // pre-loading `pending_left_batches` would bypass
                    // `current_left_block_rows` accounting. The sample is a
                    // prefix consumed first, so left-row order is preserved.
                    debug_assert!(self.leftover_batch.is_none());
                    debug_assert!(self.pending_left_batches.is_empty());
                    if !self.sizing_sample.is_empty() {
                        let left_schema = self.left_input.schema();
                        let sample = std::mem::take(&mut self.sizing_sample);
                        match concat_batches(&left_schema, &sample) {
                            Ok(batch) => self.leftover_batch = Some(batch),
                            Err(e) => {
                                self.state = StreamState::Done;
                                return Poll::Ready(Some(Err(e.into())));
                            }
                        }
                    }
                    self.sizing_sample_rows = 0;
                    self.state = StreamState::Pumping;
                    continue;
                }

                // Buffer left rows for the calibration prefix.
                StreamState::CalibrationBuffering => {
                    let target = self.calibration_left_rows_target();
                    // Restate the shared pool's target in left rows: the plan
                    // sized it in pairs, and the right relation is only now
                    // materialized. Idempotent: every partition computes the
                    // same numbers from the same shared right relation.
                    let pool = self
                        .certification
                        .as_ref()
                        .map(|cert| (cert.pool.clone(), cert.partition));
                    if let Some((pool, _)) = &pool {
                        pool.retarget(target);
                    }

                    while !self.left_exhausted {
                        if let Some((pool, partition)) = &pool
                            && pool.quota_met(*partition)
                        {
                            break;
                        }
                        match self.left_input.poll_next_unpin(cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                if batch.num_rows() == 0 {
                                    continue;
                                }
                                // Donate a copy to the shared sample and keep the
                                // rows (whole batches only): the recycle path
                                // re-processes everything this partition read.
                                if let Some((pool, partition)) = &pool {
                                    pool.donate(*partition, &batch);
                                }
                                self.calib_rows += batch.num_rows();
                                self.calib_buffered.push(batch);
                            }
                            Poll::Ready(Some(Err(e))) => {
                                self.state = StreamState::Done;
                                return Poll::Ready(Some(Err(e)));
                            }
                            Poll::Ready(None) => self.left_exhausted = true,
                            Poll::Pending => return Poll::Pending,
                        }
                    }

                    if let Some((pool, partition)) = &pool
                        && self.left_exhausted
                    {
                        // Nothing more is coming from this partition; releasing
                        // it keeps the assembling partition from waiting out the
                        // whole deadline.
                        pool.seal_partition(*partition);
                    }

                    if self.calib_rows == 0 {
                        // Empty partition: never touch the shared OnceAsync, so it
                        // cannot poison live partitions with k̂=1 or wait for an
                        // install that never comes.
                        self.state = StreamState::Pumping;
                        continue;
                    }

                    if let Err(e) = self.install_calibration() {
                        self.state = StreamState::Done;
                        return Poll::Ready(Some(Err(e)));
                    }
                    self.state = StreamState::CalibrationAwaiting;
                    continue;
                }

                // Await the shared calibration outcome.
                StreamState::CalibrationAwaiting => {
                    let k_hat = match self
                        .calib_fut
                        .as_mut()
                        .expect("CalibrationAwaiting requires an installed future")
                        .get(cx)
                    {
                        Poll::Ready(Ok(outcome)) => outcome.k_hat,
                        Poll::Ready(Err(e)) => {
                            self.state = StreamState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Pending => return Poll::Pending,
                    };

                    // Re-shape the geometry to the certified pair budget,
                    // keeping the tile width the calibration windows were
                    // drawn with. `max_pairs == L̂·R̂` exactly; anything smaller
                    // silently drops pairs in `next_pair_set`.
                    let (l_hat, r_hat) = decompose_certified_pairs(k_hat, self.right_block_size);
                    log::info!(
                        "[SemanticJoin] calibration resolved: k_hat={} => left_block={}, right_block={} (installed_here={})",
                        k_hat,
                        l_hat,
                        r_hat,
                        self.calib_installed
                    );
                    self.left_block_size = l_hat;
                    self.right_block_size = r_hat;
                    self.max_pairs_per_call = l_hat * r_hat;
                    // The certified size lands in the `certified_batch_size`
                    // gauge inside the calibration run; the block-size counts
                    // are additive and already hold the plan geometry, so
                    // re-recording would double-count.

                    // Recycle the buffered prefix through the normal pipeline at
                    // the certified geometry (calibration only scored copies).
                    // It must flow through `leftover_batch`: section B of
                    // `next_pair_set` slices it against block capacity, while
                    // pre-loading `pending_left_batches` would bypass
                    // `current_left_block_rows` accounting.
                    debug_assert!(self.leftover_batch.is_none());
                    debug_assert!(self.pending_left_batches.is_empty());
                    if !self.calib_buffered.is_empty() {
                        let left_schema = self.left_input.schema();
                        let buffered = std::mem::take(&mut self.calib_buffered);
                        match concat_batches(&left_schema, &buffered) {
                            Ok(batch) => self.leftover_batch = Some(batch),
                            Err(e) => {
                                self.state = StreamState::Done;
                                return Poll::Ready(Some(Err(e.into())));
                            }
                        }
                    }
                    self.calib_rows = 0;
                    self.calib_fut = None;
                    self.state = StreamState::Pumping;
                    continue;
                }

                // Collect, generate, dispatch and drain.
                StreamState::Pumping => {
                    // Phase 1: keep the request window full. Pair generation
                    // (including collecting further left blocks) proceeds
                    // while earlier requests are still in flight.
                    let mut input_pending = false;
                    while !self.generation_done && self.in_flight.len() < self.request_window {
                        match self.next_pair_set(cx) {
                            Ok(NextPairs::Ready(pairs, left_batch)) => {
                                self.spawn_pair_request(pairs, left_batch);
                            }
                            Ok(NextPairs::Pending) => {
                                input_pending = true;
                                break;
                            }
                            Ok(NextPairs::Exhausted) => {
                                self.generation_done = true;
                                break;
                            }
                            Err(e) => {
                                self.state = StreamState::Done;
                                return Poll::Ready(Some(Err(e)));
                            }
                        }
                    }

                    // Phase 2: surface the next completed request, in
                    // submission order, into the output coalescer.
                    match self.in_flight.poll_next_unpin(cx) {
                        Poll::Ready(Some(Ok(processing_result))) => {
                            self.metrics
                                .base
                                .record_inference(&processing_result.metrics);
                            self.inference_timer.stop();

                            self.metrics
                                .base
                                .record_rows_processed(processing_result.pairs_evaluated);
                            self.metrics
                                .base
                                .record_rows_produced(processing_result.batch.num_rows());
                            if let Some(n) = processing_result.failed_rows {
                                self.metrics.base.record_batch_failure(n);
                            }
                            if processing_result.truncated {
                                self.metrics.base.record_truncation();
                            }

                            if self.request_sink.is_some() {
                                // Probe mode: one batch per request, in
                                // submission order, empties included. The
                                // prologue reads request ordinal as probe
                                // identity, and "no matches" is an answer.
                                self.report_request_outcome(processing_result.outcome);
                                return Poll::Ready(Some(Ok(processing_result.batch)));
                            }

                            if processing_result.batch.num_rows() > 0 {
                                let batch_rows = processing_result.batch.num_rows();
                                self.output_coalescer.push_batch(processing_result.batch)?;
                                log::debug!(
                                    "[SemanticJoin] Coalescer: pushed {} rows, ready={}",
                                    batch_rows,
                                    self.output_coalescer.has_completed_batch()
                                );
                            }

                            // The loop top emits completed coalescer batches.
                            continue;
                        }
                        Poll::Ready(Some(Err(e))) => {
                            if self.request_sink.is_some() {
                                // A failed probe is one disagreement, not a
                                // dead rung: report it and keep the arm alive.
                                // Provider errors never get here in probe mode
                                // (`process_pair_batch` reports them as
                                // `NoAnswer`), so this is an answer the
                                // operator could not coerce.
                                log::debug!(
                                    "[SemanticJoin] probe request failed, scored as disagreement: {e}"
                                );
                                self.report_request_outcome(RequestOutcome::Unusable);
                                let empty = RecordBatch::new_empty(self.schema.clone());
                                return Poll::Ready(Some(Ok(empty)));
                            }
                            self.state = StreamState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(None) => {
                            // Nothing in flight. Either fully drained, or the
                            // left input can still produce pair sets.
                            if self.generation_done {
                                self.state = StreamState::Done;
                                continue;
                            }
                            if input_pending {
                                return Poll::Pending;
                            }
                            continue;
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }

                StreamState::Done => {
                    log::debug!("[SemanticJoin] Stream done, flushing remaining coalescer buffer");
                    // Flush remaining buffered output, one batch per poll.
                    self.output_coalescer.finish_buffered_batch()?;
                    if let Some(batch) = self.output_coalescer.next_completed_batch() {
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl RecordBatchStream for SemanticJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Decomposes a certified pairs-per-request budget into join block geometry,
/// keeping the tile width the calibration windows were drawn with:
/// `k >= R` gives `(k / R, R)`, otherwise `(1, k)`. The product is at most
/// `k_hat` (equal iff `R` divides `k_hat`, which holds for a power-of-two
/// ladder when `R` is a power of two). Shared by the stream's calibration
/// outcome and the rule's plan-time cache-hit path so the two cannot diverge.
/// The floor-sqrt shape of `decompose_pairs_to_block_sizes` is not used here
/// because it changes the tile width, and with it the packing distribution the
/// certificate was calibrated under.
pub(crate) fn decompose_certified_pairs(k_hat: usize, right_block: usize) -> (usize, usize) {
    let k = k_hat.max(1);
    let r = right_block.max(1);
    if k >= r { (k / r, r) } else { (1, k) }
}

/// Agreement between a batched pair and its solo reference: did the pair
/// survive the join on both arms?
///
/// The join emits a pair iff `model_output_to_boolean` says true, so survival
/// is the predicate's answer, as in the selection scorer: agreement is decided
/// on the rows the operator produced, not on a re-derivation of its output
/// coercion.
///
/// `process_pair_batch` errors when any slot in a request is uninterpretable,
/// so a filler's malformed output fails the request and its probe scores as a
/// disagreement. This can forfeit certificates but never mint one, and matches
/// production, where a bad slot is a hard error. On a model that misformats
/// join output it holds `k̂` down.
pub(crate) fn join_probe_scorer() -> ProbeScorer {
    Arc::new(|rung, reference| rung.row.is_some() == reference.row.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification::ProbeObservation;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::array::StringArray;

    #[test]
    fn decomposition_keeps_tile_width() {
        assert_eq!(decompose_certified_pairs(16, 4), (4, 4));
        assert_eq!(decompose_certified_pairs(8, 4), (2, 4));
        assert_eq!(decompose_certified_pairs(4, 4), (1, 4));
        assert_eq!(decompose_certified_pairs(2, 4), (1, 2));
        assert_eq!(decompose_certified_pairs(1, 4), (1, 1));
        assert_eq!(decompose_certified_pairs(16, 5), (3, 5));
        assert_eq!(decompose_certified_pairs(0, 4), (1, 1));
        assert_eq!(decompose_certified_pairs(8, 0), (8, 1));
    }

    /// A pair agrees when it survived on both arms, and also when it was
    /// rejected on both.
    #[test]
    fn scorer_compares_survival_on_both_arms() {
        let scorer = join_probe_scorer();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "left_0",
            DataType::Utf8,
            false,
        )]));
        let obs = |row: Option<usize>| ProbeObservation {
            batch: RecordBatch::new_empty(schema.clone()),
            row,
            outcome: RequestOutcome::Answered,
        };

        assert!(scorer(&obs(Some(0)), &obs(Some(0))), "matched on both arms");
        assert!(
            scorer(&obs(None), &obs(None)),
            "rejected on both arms is agreement, not a missing observation"
        );
        assert!(!scorer(&obs(Some(0)), &obs(None)));
        assert!(!scorer(&obs(None), &obs(Some(0))));
        // The row index carries no meaning: the batched arm emits only the
        // pairs that matched, so positions differ from the solo arm's.
        assert!(scorer(&obs(Some(7)), &obs(Some(0))));
    }

    #[test]
    fn build_pair_batch_stringifies_join_columns_in_pair_order() {
        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("ignored", DataType::Int64, false),
                Field::new("lval", DataType::Int64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![9, 9, 9])),
                Arc::new(Int64Array::from(vec![Some(10), None, Some(30)])),
            ],
        )
        .unwrap();
        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("rval", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();

        let pairs = vec![(0, 1), (1, 0), (2, 1)];
        let batch =
            SemanticJoinStream::build_pair_batch(&left, &right, &pairs, &[1], &[0], 1).unwrap();

        assert_eq!(batch.num_rows(), 3);
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, vec!["left_0", "right_0"]);
        let l = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let r = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            (l.value(0), l.value(1), l.value(2)),
            ("10", "NULL", "30"),
            "left join column stringified per pair, NULLs spelled out"
        );
        assert_eq!((r.value(0), r.value(1), r.value(2)), ("b", "a", "b"));
    }

    /// The pair batch is the join prompt's only source of row text. Join columns
    /// in any string encoding or scalar type (dates, timestamps, `Float32`,
    /// small integers) must come out as the same text selection and extend
    /// would render for them.
    #[test]
    fn build_pair_batch_renders_every_string_encoding_and_scalar_type() {
        use datafusion::arrow::array::{
            Date32Array, Float32Array, Int8Array, LargeStringArray, StringViewArray,
            TimestampSecondArray,
        };

        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8View, true),
                Field::new("born", DataType::Date32, true),
            ])),
            vec![
                Arc::new(StringViewArray::from(vec![
                    Some("Ada Lovelace, born long before the first computer"),
                    None,
                ])),
                Arc::new(Date32Array::from(vec![Some(19_737), None])),
            ],
        )
        .unwrap();
        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, true),
                Field::new("score", DataType::Float32, true),
                Field::new(
                    "seen",
                    DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Second, None),
                    true,
                ),
                Field::new("rank", DataType::Int8, true),
            ])),
            vec![
                Arc::new(LargeStringArray::from(vec![Some("Notes"), Some("Engine")])),
                Arc::new(Float32Array::from(vec![Some(1.5), None])),
                Arc::new(TimestampSecondArray::from(vec![Some(1_705_314_600), None])),
                Arc::new(Int8Array::from(vec![Some(-3), Some(4)])),
            ],
        )
        .unwrap();

        let pairs = vec![(0, 0), (1, 1)];
        let batch =
            SemanticJoinStream::build_pair_batch(&left, &right, &pairs, &[0, 1], &[0, 1, 2, 3], 2)
                .unwrap();

        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            vec![
                "left_0", "left_1", "right_0", "right_1", "right_2", "right_3"
            ]
        );
        let cell = |col: usize, row: usize| {
            batch
                .column(col)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("pair batch columns are always Utf8")
                .value(row)
                .to_string()
        };
        let row0: Vec<_> = (0..6).map(|c| cell(c, 0)).collect();
        assert_eq!(
            row0,
            vec![
                "Ada Lovelace, born long before the first computer",
                "2024-01-15",
                "Notes",
                "1.5",
                "2024-01-15T10:30:00",
                "-3",
            ]
        );
        let row1: Vec<_> = (0..6).map(|c| cell(c, 1)).collect();
        assert_eq!(row1, vec!["NULL", "NULL", "Engine", "NULL", "NULL", "4"]);
    }
}
