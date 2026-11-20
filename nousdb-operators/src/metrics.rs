use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, Time,
};
use nousdb_core::providers::metrics::InferenceMetrics;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Wall-clock timer shared across all partitions of a physical plan.
///
/// Unlike DataFusion's per-partition `Time` (which sums across partitions),
/// this tracks the real elapsed time from when the first partition starts
/// to when the last partition finishes. With N partitions running in parallel,
/// `subset_time` reports N×T while `WallClockTimer` reports T.
#[derive(Debug)]
pub struct WallClockTimer {
    /// The instant when the first partition started (set once).
    start: Mutex<Option<Instant>>,
    /// Maximum elapsed nanoseconds observed across all partitions.
    elapsed_ns: AtomicU64,
}

impl Default for WallClockTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl WallClockTimer {
    /// Creates a timer that has not started.
    pub fn new() -> Self {
        Self {
            start: Mutex::new(None),
            elapsed_ns: AtomicU64::new(0),
        }
    }

    /// Mark the start of timed work. Only the first call sets the start time;
    /// subsequent calls from other partitions are no-ops.
    pub fn start(&self) {
        let mut guard = self.start.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Instant::now());
        }
    }

    /// Record the current elapsed time since start. Uses `fetch_max` so the
    /// metric always reflects the latest completion across all partitions.
    pub fn stop(&self) {
        let guard = self.start.lock().unwrap();
        if let Some(start) = *guard {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.elapsed_ns.fetch_max(elapsed, Ordering::Relaxed);
        }
    }

    /// Get the wall-clock duration recorded so far.
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_ns.load(Ordering::Relaxed))
    }
}

/// Metrics for semantic operator execution.
///
/// Tracks LLM calls, token usage, and row counts per partition.
/// Time metrics (inference_time, histogram_estimation_time) are tracked
/// separately via shared `WallClockTimer` instances on the physical plan
/// to avoid the per-partition summing problem in `EXPLAIN ANALYZE`.
pub struct SemanticOperatorMetrics {
    /// Number of LLM calls made
    pub llm_calls: Count,
    /// Total input/prompt tokens processed
    pub input_tokens: Count,
    /// Total output/completion tokens generated
    pub output_tokens: Count,
    /// Total tokens (input + output)
    pub total_tokens: Count,
    /// Number of rows processed (input cardinality)
    pub rows_processed: Count,
    /// Number of rows produced (output cardinality)
    pub rows_produced: Count,
    /// Optimizer-predicted batch size (from BatchSizeOptimizationRule)
    pub predicted_batch_size: Gauge,
    /// Number of context window overflow errors encountered
    pub overflow_errors: Count,
    /// Number of batches where the LLM returned a row-count mismatch
    pub failed_batches: Count,
    /// Total rows affected by row-count mismatch failures
    pub failed_rows: Count,
    /// Requests the model ended by exhausting its output token budget
    /// (`finish_reason == "length"`).
    ///
    /// Counted for every truncated response, whether or not it was then
    /// rejected. A truncated response is usually undecodable and then also
    /// counts in `failed_batches`; this counter says why, and the fix is a
    /// smaller batch or a larger `ModelCard.max_output_tokens`.
    pub truncated_batches: Count,
    /// Tokens the model spent inside `<think>` blocks, when the endpoint
    /// reports them.
    ///
    /// A subset of [`Self::output_tokens`], not an addition: reasoning is
    /// billed out of the completion budget, so adding it to `total_tokens`
    /// would double-count. Zero for providers that do not report it.
    pub reasoning_tokens: Count,
}

impl SemanticOperatorMetrics {
    /// Create a new metrics instance for a partition.
    ///
    /// # Arguments
    ///
    /// * `metrics_set` - The ExecutionPlanMetricsSet to register metrics with
    /// * `partition` - The partition number for this metrics instance
    pub fn new(metrics_set: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            llm_calls: MetricBuilder::new(metrics_set).counter("llm_calls", partition),
            input_tokens: MetricBuilder::new(metrics_set).counter("input_tokens", partition),
            output_tokens: MetricBuilder::new(metrics_set).counter("output_tokens", partition),
            total_tokens: MetricBuilder::new(metrics_set).counter("total_tokens", partition),
            rows_processed: MetricBuilder::new(metrics_set).counter("rows_processed", partition),
            rows_produced: MetricBuilder::new(metrics_set).counter("rows_produced", partition),
            predicted_batch_size: MetricBuilder::new(metrics_set)
                .gauge("predicted_batch_size", partition),
            overflow_errors: MetricBuilder::new(metrics_set).counter("overflow_errors", partition),
            failed_batches: MetricBuilder::new(metrics_set).counter("failed_batches", partition),
            failed_rows: MetricBuilder::new(metrics_set).counter("failed_rows", partition),
            truncated_batches: MetricBuilder::new(metrics_set)
                .counter("truncated_batches", partition),
            reasoning_tokens: MetricBuilder::new(metrics_set)
                .counter("reasoning_tokens", partition),
        }
    }

    /// Record metrics from a single inference call.
    pub fn record_inference(&self, metrics: &InferenceMetrics) {
        self.llm_calls.add(1);
        self.input_tokens.add(metrics.input_tokens as usize);
        self.output_tokens.add(metrics.output_tokens as usize);
        self.total_tokens.add(metrics.total_tokens as usize);
        // Not added to `total_tokens`: it is already inside `output_tokens`.
        self.reasoning_tokens.add(metrics.reasoning_tokens as usize);
    }

    /// Record the number of rows processed (input cardinality).
    pub fn record_rows_processed(&self, count: usize) {
        self.rows_processed.add(count);
    }

    /// Record the number of rows produced (output cardinality).
    pub fn record_rows_produced(&self, count: usize) {
        self.rows_produced.add(count);
    }

    /// Record the optimizer-predicted batch size.
    pub fn record_predicted_batch_size(&self, size: usize) {
        self.predicted_batch_size.set(size);
    }

    /// Record a context window overflow error.
    pub fn record_overflow_error(&self) {
        self.overflow_errors.add(1);
    }

    /// Record a request whose response could not be used, and the rows that
    /// took the operator's recovery path as a result.
    ///
    /// Rows counted in `failed_rows` were not processed by a model answer. For
    /// selection they are emitted unfiltered (the predicate never applied), for
    /// extend they carry NULL derived columns, and for a join they produce no
    /// matches. A large `failed_rows` means the result differs from what the
    /// query asked for.
    pub fn record_batch_failure(&self, rows: usize) {
        self.failed_batches.add(1);
        self.failed_rows.add(rows);
    }

    /// Record a request the model ended by running out of output budget.
    ///
    /// Separate from [`Self::record_batch_failure`]: a failure is scoped to the
    /// rows that took a recovery path, while truncation is a property of the
    /// response and is counted even when the response still decoded.
    pub fn record_truncation(&self) {
        self.truncated_batches.add(1);
    }
}

/// Metrics for the quality-certification calibration prologue.
///
/// Recorded by the partition that runs calibration (probe counts are
/// per-operator-instance, not per-partition, since calibration runs once).
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Singleton (`k = 1`) reference-arm calls issued: exactly `n` per
    /// calibration, shared across all ladder rungs.
    pub reference_calls: Count,
    /// Batched probe calls issued across all rungs.
    pub rung_calls: Count,
    /// Ladder rungs attempted (including curtailed ones).
    pub rungs_tried: Count,
    /// The certified batch size `k̂` the stream settled on.
    pub certified_batch_size: Gauge,
    /// Rows buffered into the calibration prefix.
    pub calibration_rows: Count,
    /// Wall-clock time spent in the calibration prologue. Only the partition
    /// that runs calibration records it, so this is a real elapsed time, not a
    /// per-partition sum. Separate from the operator's `inference_timer` so the
    /// prologue's cost is visible on its own in `EXPLAIN ANALYZE`.
    pub calibration_time: Time,
}

impl CalibrationMetrics {
    /// Registers the calibration metrics for `partition` in `metrics_set`.
    pub fn new(metrics_set: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            reference_calls: MetricBuilder::new(metrics_set)
                .counter("calibration_reference_calls", partition),
            rung_calls: MetricBuilder::new(metrics_set)
                .counter("calibration_rung_calls", partition),
            rungs_tried: MetricBuilder::new(metrics_set)
                .counter("calibration_rungs_tried", partition),
            certified_batch_size: MetricBuilder::new(metrics_set)
                .gauge("certified_batch_size", partition),
            calibration_rows: MetricBuilder::new(metrics_set)
                .counter("calibration_rows", partition),
            calibration_time: MetricBuilder::new(metrics_set)
                .subset_time("calibration_time", partition),
        }
    }
}

/// Metrics for semantic join execution.
///
/// Extends `SemanticOperatorMetrics` with join-specific fields for
/// block nested loop sizing and right-side scan tracking.
pub struct SemanticJoinMetrics {
    /// Base metrics shared by all semantic operators
    pub base: SemanticOperatorMetrics,
    /// Chosen dynamic batch size (max pairs per LLM call)
    pub chosen_batch_size: Gauge,
    /// Number of rows used to build per-batch histograms
    pub histogram_rows: Count,
    /// Iterations spent in safe batch size convergence
    pub safe_batch_iterations: Count,
    /// Left block size chosen for join BNL
    pub left_block_size: Gauge,
    /// Right block size chosen for join BNL
    pub right_block_size: Gauge,
    /// Number of complete right-side scans (= left blocks processed)
    pub right_scans_completed: Count,
}

impl SemanticJoinMetrics {
    /// Create a new join metrics instance for a partition.
    pub fn new(metrics_set: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            base: SemanticOperatorMetrics::new(metrics_set, partition),
            chosen_batch_size: MetricBuilder::new(metrics_set)
                .gauge("chosen_batch_size", partition),
            histogram_rows: MetricBuilder::new(metrics_set).counter("histogram_rows", partition),
            safe_batch_iterations: MetricBuilder::new(metrics_set)
                .counter("safe_batch_iterations", partition),
            left_block_size: MetricBuilder::new(metrics_set).gauge("left_block_size", partition),
            right_block_size: MetricBuilder::new(metrics_set).gauge("right_block_size", partition),
            right_scans_completed: MetricBuilder::new(metrics_set)
                .counter("right_scans_completed", partition),
        }
    }

    /// Record the chosen dynamic batch size used for chunking.
    pub fn record_chosen_batch_size(&self, size: usize) {
        self.chosen_batch_size.set(size);
    }

    /// Record rows used to estimate histogram statistics.
    pub fn record_histogram_rows(&self, count: usize) {
        self.histogram_rows.add(count);
    }

    /// Record iterations needed by safe-batch convergence.
    pub fn record_safe_batch_iterations(&self, iterations: usize) {
        self.safe_batch_iterations.add(iterations);
    }

    /// Record the left block size chosen for join BNL.
    pub fn record_left_block_size(&self, size: usize) {
        self.left_block_size.set(size);
    }

    /// Record the right block size chosen for join BNL.
    pub fn record_right_block_size(&self, size: usize) {
        self.right_block_size.set(size);
    }

    /// Record completion of one full right-side scan.
    pub fn record_right_scan_completed(&self) {
        self.right_scans_completed.add(1);
    }
}

/// Metrics for token-based repartitioning.
///
/// Tracks token estimation, row distribution, and partition balance.
/// Integrates with DataFusion's metrics system for visibility via `EXPLAIN ANALYZE`.
pub struct TokenRepartitionMetrics {
    /// Total tokens estimated across all rows
    pub tokens_estimated: Count,
    /// Number of rows repartitioned
    pub rows_repartitioned: Count,
    /// Tokens assigned to this partition
    pub partition_tokens: Count,
    /// Time spent estimating tokens
    pub estimation_time: Time,
    /// Imbalance ratio (max/min partition tokens)
    pub imbalance_ratio: Gauge,
    /// Number of synchronized waves completed
    pub waves_completed: Count,
}

impl TokenRepartitionMetrics {
    /// Create a new metrics instance for a partition.
    ///
    /// # Arguments
    ///
    /// * `metrics_set` - The ExecutionPlanMetricsSet to register metrics with
    /// * `partition` - The partition number for this metrics instance
    pub fn new(metrics_set: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            tokens_estimated: MetricBuilder::new(metrics_set)
                .counter("tokens_estimated", partition),
            rows_repartitioned: MetricBuilder::new(metrics_set)
                .counter("rows_repartitioned", partition),
            partition_tokens: MetricBuilder::new(metrics_set)
                .counter("partition_tokens", partition),
            estimation_time: MetricBuilder::new(metrics_set)
                .subset_time("estimation_time", partition),
            imbalance_ratio: MetricBuilder::new(metrics_set).gauge("imbalance_ratio", partition),
            waves_completed: MetricBuilder::new(metrics_set).counter("waves_completed", partition),
        }
    }

    /// Record the total tokens estimated.
    pub fn record_tokens_estimated(&self, count: usize) {
        self.tokens_estimated.add(count);
    }

    /// Record the number of rows repartitioned.
    pub fn record_rows_repartitioned(&self, count: usize) {
        self.rows_repartitioned.add(count);
    }

    /// Record the tokens assigned to a specific partition.
    pub fn record_partition_tokens(&self, tokens: usize) {
        self.partition_tokens.add(tokens);
    }

    /// Record the time spent estimating tokens.
    pub fn record_estimation_time(&self, duration: Duration) {
        self.estimation_time.add_duration(duration);
    }

    /// Record the imbalance ratio. The last write wins, so the final wave's
    /// value reflects the cumulative balance.
    pub fn record_imbalance_ratio(&self, ratio: f64) {
        // Gauge holds a usize: store the ratio times 100 (two decimal places).
        self.imbalance_ratio.set((ratio * 100.0) as usize);
    }

    /// Record a completed synchronized wave.
    pub fn record_wave_completed(&self) {
        self.waves_completed.add(1);
    }
}
