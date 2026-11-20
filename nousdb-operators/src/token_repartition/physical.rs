//! Physical (`ExecutionPlan`) operator for token repartitioning, driven by a wave-barrier coordinator task.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, UInt32Array};
use datafusion::arrow::compute::{self, concat_batches};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_physical_plan::ExecutionPlanProperties;

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use nousdb_core::token_estimator::{TokenEstimator, TokenEstimatorConfig};

use crate::token_repartition::config::PartitionStrategy;

use crate::metrics::TokenRepartitionMetrics;
use crate::token_repartition::partitioner::TokenPartitioner;
use crate::token_repartition::stream::TokenRepartitionStream;

/// Channel buffer size per output partition.
///
/// Deep enough that a downstream partition busy with a model call does not
/// stall the coordinator. Wave batches are small, so the buffering is cheap,
/// while a shallow buffer would gate every partition on the slowest consumer.
const CHANNEL_BUFFER: usize = 16;

/// Shared state for the channel-based wave coordinator.
///
/// Uses `std::sync::Mutex` (not tokio) because `execute()` is synchronous.
/// On the first `execute()` call, the coordinator is spawned and all receivers
/// are created. Each subsequent `execute(partition)` takes its receiver.
struct SharedState {
    /// Pre-allocated receivers, one per output partition.
    /// `execute(partition)` takes (Option::take) its receiver.
    receivers: Vec<Option<mpsc::Receiver<Result<RecordBatch>>>>,

    /// Handle to the coordinator task (kept alive for the operator's lifetime).
    _coordinator_handle: Option<JoinHandle<()>>,
}

/// Physical execution plan for token-based repartitioning.
///
/// A single coordinator task collects one batch from every live input
/// partition (a "wave"), applies the partitioning strategy to the whole wave,
/// and scatters the results to the output channels. The strategy therefore sees
/// complete waves rather than batches arriving in nondeterministic order, and
/// downstream operators start as soon as a wave's results are scattered.
#[derive(Debug)]
pub struct TokenRepartitionExec {
    /// The input execution plan
    input: Arc<dyn ExecutionPlan>,

    /// Columns to estimate tokens from
    token_columns: Vec<String>,

    /// Target number of partitions
    num_partitions: usize,

    /// Maximum tokens per partition (optional)
    token_budget: Option<usize>,

    /// Partitioning strategy
    strategy: PartitionStrategy,

    /// Output schema (same as input)
    schema: SchemaRef,

    /// Plan properties
    properties: Arc<PlanProperties>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,

    /// Shared state for channel-based coordination.
    /// Uses std::sync::Mutex because execute() is not async.
    shared_state: std::sync::Mutex<Option<SharedState>>,
}

impl TokenRepartitionExec {
    /// Creates a repartitioning operator producing `num_partitions` output
    /// partitions; the output schema equals the input schema.
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        token_columns: Vec<String>,
        num_partitions: usize,
        token_budget: Option<usize>,
        strategy: PartitionStrategy,
    ) -> Result<Self> {
        let schema = input.schema();
        let properties = Self::compute_properties(&input, schema.clone(), num_partitions);

        Ok(Self {
            input,
            token_columns,
            num_partitions,
            token_budget,
            strategy,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            shared_state: std::sync::Mutex::new(None),
        })
    }

    /// Get the partitioning strategy.
    pub fn strategy(&self) -> &PartitionStrategy {
        &self.strategy
    }

    /// Compute plan properties for this operator.
    fn compute_properties(
        _input: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        num_partitions: usize,
    ) -> Arc<PlanProperties> {
        let partitioning = Partitioning::UnknownPartitioning(num_partitions);
        let boundedness = Boundedness::Bounded;
        // Batches are emitted incrementally as waves complete.
        let emission_type = EmissionType::Incremental;

        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            partitioning,
            emission_type,
            boundedness,
        ))
    }

    /// Get the token columns.
    pub fn token_columns(&self) -> &[String] {
        &self.token_columns
    }

    /// Get the target number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Get the token budget per partition.
    pub fn token_budget(&self) -> Option<usize> {
        self.token_budget
    }

    /// Initialize channels and spawn the coordinator task.
    ///
    /// Called once (under lock) on the first `execute()` invocation.
    fn initialize_coordinator(&self, context: Arc<TaskContext>) -> Result<SharedState> {
        let input_partition_count = self.input.output_partitioning().partition_count();
        let mut input_streams = Vec::with_capacity(input_partition_count);
        for i in 0..input_partition_count {
            input_streams.push(self.input.execute(i, context.clone())?);
        }

        // One output channel per output partition.
        let mut senders = Vec::with_capacity(self.num_partitions);
        let mut receivers = Vec::with_capacity(self.num_partitions);
        for _ in 0..self.num_partitions {
            let (tx, rx) = mpsc::channel(CHANNEL_BUFFER);
            senders.push(tx);
            receivers.push(Some(rx));
        }

        let token_columns = self.token_columns.clone();
        let num_partitions = self.num_partitions;
        let token_budget = self.token_budget;
        let strategy = self.strategy.clone();
        let schema = self.schema.clone();

        // Coordinator metrics use partition slot `num_partitions`, one beyond
        // the output range, so they do not collide with per-partition metrics.
        let coordinator_metrics = TokenRepartitionMetrics::new(&self.metrics, num_partitions);

        let handle = tokio::spawn(async move {
            if let Err(e) = wave_coordinator(
                input_streams,
                senders,
                token_columns,
                num_partitions,
                token_budget,
                strategy,
                schema,
                coordinator_metrics,
            )
            .await
            {
                // The error has already been sent to the channels by
                // `wave_coordinator`; log it and do not panic.
                log::error!("TokenRepartition coordinator error: {e}");
            }
        });

        Ok(SharedState {
            receivers,
            _coordinator_handle: Some(handle),
        })
    }
}

impl ExecutionPlan for TokenRepartitionExec {
    fn name(&self) -> &str {
        "TokenRepartitionExec"
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Holds no physical expressions (rows are routed by token counts, not by a hash expression).
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "TokenRepartitionExec expects exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(TokenRepartitionExec::new(
            children[0].clone(),
            self.token_columns.clone(),
            self.num_partitions,
            self.token_budget,
            self.strategy.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let partition_metrics = TokenRepartitionMetrics::new(&self.metrics, partition);

        // Initialize the coordinator on the first call; every call takes its
        // partition's receiver.
        let mut guard = self.shared_state.lock().map_err(|e| {
            DataFusionError::Internal(format!("TokenRepartitionExec lock poisoned: {e}"))
        })?;

        if guard.is_none() {
            *guard = Some(self.initialize_coordinator(context)?);
        }

        let state = guard.as_mut().unwrap();

        let receiver = state
            .receivers
            .get_mut(partition)
            .and_then(|slot| slot.take())
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "TokenRepartitionExec: partition {partition} receiver already taken or out of range"
                ))
            })?;

        let stream = TokenRepartitionStream::new(receiver, self.schema.clone(), partition_metrics);

        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

impl DisplayAs for TokenRepartitionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let strategy_str = match &self.strategy {
            PartitionStrategy::OptimalTokenBalance => "OptimalTokenBalance".to_string(),
            PartitionStrategy::EqualRowCounts => "EqualRowCounts".to_string(),
        };

        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "TokenRepartitionExec: columns={:?}, partitions={}, budget={:?}, strategy={}",
                    self.token_columns, self.num_partitions, self.token_budget, strategy_str
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "columns={:?}\n partitions={}\n budget={:?}\n strategy={}",
                    self.token_columns, self.num_partitions, self.token_budget, strategy_str
                )
            }
        }
    }
}

// Reports only how many receivers remain, for the `#[derive(Debug)]` on the exec.
impl fmt::Debug for SharedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedState")
            .field(
                "receivers_remaining",
                &self.receivers.iter().filter(|r| r.is_some()).count(),
            )
            .finish()
    }
}

/// Filter a batch to keep only the specified row indices using Arrow's take kernel.
fn take_rows(batch: &RecordBatch, indices: &[u32]) -> Result<RecordBatch> {
    let index_array = UInt32Array::from(indices.to_vec());
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| compute::take(col.as_ref(), &index_array, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Wave-synchronized coordinator: the core repartitioning loop.
///
/// Each iteration (a wave barrier) collects one batch from every live input
/// partition concurrently, estimates tokens per batch without merging, applies
/// the partitioning strategy to the combined token counts, then scatters rows
/// directly from the original batches to avoid a full-wave copy.
///
/// Live input streams are polled with `futures::future::join_all`, so gather
/// latency is the maximum, not the sum, of the partitions' times. Scatter uses
/// `take_rows` on the source batches, avoiding a `concat_batches` of the whole
/// wave.
///
/// For example, 100 rows over 4 input partitions with batch size 5 gives 25
/// rows per partition and 5 waves of 20 rows each.
///
/// When all receivers are dropped (e.g. LIMIT), the coordinator detects closed
/// channels and exits early.
#[allow(clippy::too_many_arguments)]
async fn wave_coordinator(
    input_streams: Vec<SendableRecordBatchStream>,
    output_channels: Vec<mpsc::Sender<Result<RecordBatch>>>,
    token_columns: Vec<String>,
    num_partitions: usize,
    token_budget: Option<usize>,
    strategy: PartitionStrategy,
    schema: SchemaRef,
    metrics: TokenRepartitionMetrics,
) -> Result<()> {
    let estimator = TokenEstimator::new(&TokenEstimatorConfig::default())?;
    let mut partitioner = TokenPartitioner::new(num_partitions, token_budget);
    // Cumulative partition-token total after the previous wave, so only each
    // wave's delta is recorded (the partitioner's counters are cumulative
    // under OptimalTokenBalance).
    let mut prev_partition_token_total: usize = 0;

    // None marks an exhausted stream.
    let mut input_streams: Vec<Option<SendableRecordBatchStream>> =
        input_streams.into_iter().map(Some).collect();

    loop {
        // Gather: take the live streams out of the vec so each future owns its
        // stream, poll them concurrently, then put the survivors back.
        let mut live: Vec<(usize, SendableRecordBatchStream)> = input_streams
            .iter_mut()
            .enumerate()
            .filter_map(|(i, opt)| opt.take().map(|s| (i, s)))
            .collect();

        if live.is_empty() {
            break; // all inputs exhausted
        }

        // Poll all streams concurrently; each `Next` future borrows a distinct stream.
        let futs: Vec<_> = live.iter_mut().map(|(_, s)| s.next()).collect();
        let results = futures::future::join_all(futs).await;

        // Process results: collect batches, put surviving streams back
        let mut wave_batches: Vec<RecordBatch> = Vec::new();
        let mut had_error: Option<DataFusionError> = None;

        for ((idx, stream), result) in live.into_iter().zip(results) {
            match result {
                Some(Ok(batch)) if batch.num_rows() > 0 => {
                    wave_batches.push(batch);
                    input_streams[idx] = Some(stream); // still alive
                }
                Some(Ok(_empty)) => {
                    // Empty batch: keep the stream alive, skip it in this wave.
                    input_streams[idx] = Some(stream);
                }
                // Keep the first error; the stream is not put back.
                Some(Err(e)) if had_error.is_none() => had_error = Some(e),
                Some(Err(_)) => {}
                None => {
                    // Stream exhausted: leave the slot as None.
                }
            }
        }

        // If any stream produced an error, propagate to all output channels
        if let Some(e) = had_error {
            let msg = e.to_string();
            for tx in &output_channels {
                let _ = tx.send(Err(DataFusionError::Internal(msg.clone()))).await;
            }
            return Err(e);
        }

        // If no batches were collected, all inputs are done
        if wave_batches.is_empty() {
            break;
        }

        // Token counts are estimated per source batch and concatenated; the
        // strategy only needs the counts, so column data is never merged.
        let estimation_start = std::time::Instant::now();
        let mut all_token_counts: Vec<usize> = Vec::new();
        let mut batch_offsets: Vec<usize> = Vec::new(); // cumulative row offset per batch
        let mut offset: usize = 0;

        for batch in &wave_batches {
            let counts: Vec<usize> = match estimator.estimate_batch_tokens(batch, &token_columns) {
                Ok(c) => c,
                Err(e) => {
                    let err: DataFusionError = e;
                    let msg = err.to_string();
                    for tx in &output_channels {
                        let _ = tx.send(Err(DataFusionError::Internal(msg.clone()))).await;
                    }
                    return Err(err);
                }
            };
            all_token_counts.extend_from_slice(&counts);
            batch_offsets.push(offset);
            offset += batch.num_rows();
        }
        metrics.record_estimation_time(estimation_start.elapsed());
        metrics.record_tokens_estimated(all_token_counts.iter().sum());

        let assignments = match &strategy {
            PartitionStrategy::OptimalTokenBalance => {
                // Cumulative state: greedy compensates for past imbalances.
                partitioner.assign_rows(&all_token_counts)
            }
            PartitionStrategy::EqualRowCounts => {
                // Reset per wave: zigzag sorts the current wave independently.
                // The delta baseline resets with it.
                partitioner.reset();
                prev_partition_token_total = 0;
                partitioner.assign_rows_zigzag(&all_token_counts)
            }
        };

        metrics.record_imbalance_ratio(partitioner.imbalance_ratio());
        let partition_token_total: usize = partitioner.partition_tokens().iter().sum();
        metrics.record_partition_tokens(partition_token_total - prev_partition_token_total);
        prev_partition_token_total = partition_token_total;
        metrics.record_wave_completed();

        // Scatter: build per-output-partition, per-source-batch index lists,
        // then `take_rows` from the original batches. Concatenate only when an
        // output partition receives rows from multiple source batches.
        let n_source = wave_batches.len();
        let mut partition_batch_indices: Vec<Vec<Vec<u32>>> =
            vec![vec![Vec::new(); n_source]; num_partitions];

        for (global_row, &target) in assignments.iter().enumerate() {
            // Map global row index back to (source_batch, row_within_batch)
            let batch_idx = batch_offsets.partition_point(|&off| off <= global_row) - 1;
            let row_in_batch = (global_row - batch_offsets[batch_idx]) as u32;
            partition_batch_indices[target][batch_idx].push(row_in_batch);
        }

        // Build all output batches first, then send them concurrently: a
        // sequential awaited send loop would park the coordinator on the first
        // busy partition and starve the later ones, advancing the whole
        // operator at the pace of the slowest consumer.
        let mut sends = Vec::new();
        for (part_id, per_batch) in partition_batch_indices.iter().enumerate() {
            let mut part_batches: Vec<RecordBatch> = Vec::new();
            for (batch_idx, indices) in per_batch.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                part_batches.push(take_rows(&wave_batches[batch_idx], indices)?);
            }
            if part_batches.is_empty() {
                continue;
            }
            // If rows come from a single source batch, no concat needed
            let output_batch = if part_batches.len() == 1 {
                part_batches.into_iter().next().unwrap()
            } else {
                concat_batches(&schema, &part_batches)?
            };
            // If a send fails, the receiver was dropped (e.g. LIMIT); skip
            sends.push(output_channels[part_id].send(Ok(output_batch)));
        }
        let _ = futures::future::join_all(sends).await;

        // Exit early if all receivers have been dropped
        if output_channels.iter().all(|tx| tx.is_closed()) {
            break;
        }
    }

    // Channels are dropped here, signaling end-of-stream to all receivers
    Ok(())
}
