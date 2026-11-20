//! Physical plan node for the semantic join.

use crate::certification::{
    CertifyRuntime, ProbeExecFactory, RequestOutcomeSink, StreamCertification,
};
use crate::join::stream::{
    JoinRightData, OnceAsync, SemanticJoinStream, decompose_certified_pairs,
};
use crate::metrics::{CalibrationMetrics, SemanticJoinMetrics, WallClockTimer};
use datafusion::arrow::array::{ArrayRef, UInt8Array};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
    metrics::{ExecutionPlanMetricsSet, Metric, MetricValue, MetricsSet, Time},
};
use datafusion_physical_plan::ExecutionPlanProperties;
use futures::StreamExt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::provider::ModelProvider;
use std::fmt;
use std::sync::Arc;

/// Physical Execution Plan for Semantic Join.
///
/// Uses a **build-once, probe-many** pattern (adopted from DataFusion's
/// `NestedLoopJoinExec`):
///
/// - **Right (build) side**: Materialized once via `OnceAsync` into a single
///   `RecordBatch` shared across all left partitions as `Arc<JoinRightData>`.
/// - **Left (probe) side**: Each partition streams its left data independently
///   and probes the shared right data using block nested loops with LLM-based
///   join predicate evaluation.
///
/// `required_input_distribution()` returns `[Unspecified, SinglePartition]`,
/// telling DataFusion's `EnforceDistribution` rule to coalesce the right input
/// to a single partition if needed, while leaving the left side's partitioning
/// unchanged.
#[derive(Debug)]
pub struct SemanticJoinExec {
    /// Left input execution plan (probe side, streamed in blocks)
    left: Arc<dyn ExecutionPlan>,

    /// Right input execution plan (build side, materialized once)
    right: Arc<dyn ExecutionPlan>,

    /// Indices of join columns in left schema
    left_join_indices: Vec<usize>,

    /// Indices of join columns in right schema
    right_join_indices: Vec<usize>,

    /// The model provider for evaluating join predicates
    model_provider: Arc<dyn ModelProvider>,

    /// Which model to use (e.g., "llama3.1")
    model_id: String,

    /// Model configuration resolved at planning time
    model_card: ModelCard,

    /// Natural language join condition
    join_condition: String,

    /// The full output schema (left ∪ right columns)
    output_schema: SchemaRef,

    /// Plan properties (cached)
    properties: Arc<PlanProperties>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,

    /// Wall-clock timer for inference, shared across all partitions
    inference_timer: Arc<WallClockTimer>,

    /// Shared right-side data, built once and shared across left partitions.
    /// `OnceAsync` ensures the first `execute()` call triggers materialization;
    /// subsequent calls from other partitions await the same future.
    right_data: OnceAsync<JoinRightData>,

    /// Optimizer-predicted batch size (left_block_size × right_block_size)
    predicted_batch_size: Option<usize>,

    /// Pre-computed left block size from the optimizer/pre-computation.
    left_block_size: Option<usize>,

    /// Pre-computed right block size from the optimizer/pre-computation.
    right_block_size: Option<usize>,

    /// Max concurrent in-flight pair-batch requests per partition
    /// (1 = serial dispatch)
    request_window: usize,

    /// Reasoning arm resolved by the context builder; selects the system
    /// template and the response contract. `CertifyBatchSizeRule` reads it for
    /// the certificate key, so every clone path must preserve it.
    reasoning: ReasoningMode,

    /// Runtime calibration attachment from `CertifyBatchSizeRule` (spec +
    /// session cache + shared once-per-instance calibration future).
    certification: Option<CertifyRuntime>,

    /// Provenance for a plan-time-resolved certified pair budget (cache
    /// hit); the block sizes already reflect the decomposed geometry.
    certified_fixed: Option<usize>,

    /// Probe mode. When set, the left child already carries materialized pair
    /// rows and these are the real reference-table column names (the
    /// schema-derived ones would read `left_0` / `right_0`). See [`probe_exec`].
    ///
    /// [`probe_exec`]: SemanticJoinExec::probe_exec
    probe_col_names: Option<(Vec<String>, Vec<String>)>,

    /// Per-request outcome reporting, installed on probe execs only.
    request_sink: Option<RequestOutcomeSink>,
}

impl SemanticJoinExec {
    /// Creates the exec, resolving the join column names against each input's
    /// schema. Fails if a column is missing or a name has more than one dot.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        left_join_columns: Vec<String>,
        right_join_columns: Vec<String>,
        output_schema: SchemaRef,
        model_provider: Arc<dyn ModelProvider>,
        model_id: String,
        model_card: ModelCard,
        join_condition: String,
        predicted_batch_size: Option<usize>,
        left_block_size: Option<usize>,
        right_block_size: Option<usize>,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        let left_join_indices =
            resolve_column_indices(&left_join_columns, &left_schema, "left input")?;

        let right_join_indices =
            resolve_column_indices(&right_join_columns, &right_schema, "right input")?;

        let properties = Self::compute_properties(&left, output_schema.clone());

        Ok(Self {
            left,
            right,
            left_join_indices,
            right_join_indices,
            model_provider,
            model_id,
            model_card,
            join_condition,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            right_data: OnceAsync::default(),
            predicted_batch_size,
            left_block_size,
            right_block_size,
            request_window: 1,
            reasoning: ReasoningMode::default(),
            certification: None,
            certified_fixed: None,
            probe_col_names: None,
            request_sink: None,
        })
    }

    /// Set the per-partition in-flight request window (resolved by the
    /// planner from the model card; clamped to at least 1).
    pub fn with_request_window(mut self, request_window: usize) -> Self {
        self.request_window = request_window.max(1);
        self
    }

    /// Set the reasoning arm (resolved by `NousDBContextBuilder` and passed
    /// down by `NousDBPhysicalPlanner`).
    pub fn with_reasoning_mode(mut self, reasoning: ReasoningMode) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// The reasoning arm this join runs with.
    pub fn reasoning_mode(&self) -> ReasoningMode {
        self.reasoning
    }

    /// Attach a quality-certification runtime (plan time, from
    /// `CertifyBatchSizeRule`). Clears any fixed-size provenance.
    pub fn with_certification(&self, runtime: CertifyRuntime) -> Self {
        let mut exec = self.shallow_clone();
        exec.certification = Some(runtime);
        exec.certified_fixed = None;
        exec
    }

    /// Resolve a certificate-cache hit at plan time: decompose the certified
    /// pair budget into block geometry (keeping this plan's tile width, the
    /// same rule the stream applies after a live calibration) and record the
    /// provenance. Clears any runtime attachment.
    pub fn with_fixed_certified_size(&self, k: usize) -> Self {
        let k = k.max(1);
        let (l, r) = decompose_certified_pairs(k, self.right_block_size.unwrap_or(1));
        let mut exec = self.shallow_clone();
        exec.left_block_size = Some(l);
        exec.right_block_size = Some(r);
        exec.predicted_batch_size = Some(l * r);
        exec.certified_fixed = Some(k);
        exec.certification = None;
        exec
    }

    /// Build a probe copy of this operator over an already-materialized pair
    /// relation, fixed at `k` pairs per request.
    ///
    /// Block nested loop crosses every left block with every right tile, so one
    /// execution cannot emit `n` mutually disjoint designed batches. Instead
    /// the designed pairs are fed back as rows (as `build_pair_batch` already
    /// materializes them) and a degenerate geometry cuts them on request
    /// boundaries:
    ///
    /// - `pair_input` becomes the left child (`left_0.., right_0..`, plus
    ///   the prologue's row-identity column) and the right child is a one-row
    ///   placeholder, so `next_pair_set` emits `[(0,0)..(k-1,0)]` per block,
    ///   exactly `k` consecutive pair rows per request, on exact boundaries.
    /// - `left_join_indices` covers all pair columns and `right_join_indices`
    ///   is empty, so `build_pair_batch` re-gathers the pair values verbatim
    ///   and the placeholder contributes nothing to the prompt.
    /// - `probe_col_names` carries the real reference-table headers, which the
    ///   renderer also uses to split the batch back into its two sides. The
    ///   identity column sits past that split and never reaches the prompt.
    ///
    /// Certification is off (no recursion) and the sink is installed so a
    /// failed request scores as a disagreement instead of emitting no matches.
    pub fn probe_exec(
        &self,
        pair_input: Arc<dyn ExecutionPlan>,
        k: usize,
        window: usize,
        sink: RequestOutcomeSink,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let pair_schema = pair_input.schema();
        let (left_names, right_names) = self.join_column_names();
        let pair_columns = left_names.len() + right_names.len();
        if pair_schema.fields().len() != pair_columns + 1 {
            return Err(DataFusionError::Internal(format!(
                "calibration: probe input has {} columns, expected {pair_columns} pair columns \
                 plus a row-identity column",
                pair_schema.fields().len()
            )));
        }
        let placeholder_schema = Arc::new(Schema::new(vec![Field::new(
            "__nousdb_probe_unit",
            DataType::UInt8,
            false,
        )]));
        let placeholder = RecordBatch::try_new(
            placeholder_schema.clone(),
            vec![Arc::new(UInt8Array::from(vec![0u8])) as ArrayRef],
        )?;
        let right: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(
            &[vec![placeholder]],
            placeholder_schema.clone(),
            None,
        )?;

        let output_fields: Vec<Field> = pair_schema
            .fields()
            .iter()
            .chain(placeholder_schema.fields().iter())
            .map(|f| f.as_ref().clone())
            .collect();

        let output_schema: SchemaRef = Arc::new(Schema::new(output_fields));
        Ok(Arc::new(Self {
            properties: Self::compute_properties(&pair_input, output_schema.clone()),
            left: pair_input,
            right,
            left_join_indices: (0..pair_columns).collect(),
            right_join_indices: Vec::new(),
            model_provider: self.model_provider.clone(),
            model_id: self.model_id.clone(),
            model_card: self.model_card.clone(),
            join_condition: self.join_condition.clone(),
            output_schema,
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            right_data: OnceAsync::default(),
            predicted_batch_size: Some(k.max(1)),
            left_block_size: Some(k.max(1)),
            right_block_size: Some(1),
            request_window: window.max(1),
            reasoning: self.reasoning,
            certification: None,
            certified_fixed: None,
            probe_col_names: Some((left_names, right_names)),
            request_sink: Some(sink),
        }))
    }

    /// Probe-plan builder handed to the calibration prologue. Captures a
    /// pre-execution copy so the closure never borrows the live exec.
    fn probe_factory(&self) -> ProbeExecFactory {
        let template = Arc::new(self.shallow_clone());
        Arc::new(move |input, k, window, sink| template.probe_exec(input, k, window, sink))
    }

    /// Field-preserving copy with a fresh metrics set, timer and right-side
    /// once-state; only valid pre-execution, which is when optimizer rules
    /// run.
    fn shallow_clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            left_join_indices: self.left_join_indices.clone(),
            right_join_indices: self.right_join_indices.clone(),
            model_provider: self.model_provider.clone(),
            model_id: self.model_id.clone(),
            model_card: self.model_card.clone(),
            join_condition: self.join_condition.clone(),
            output_schema: self.output_schema.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            right_data: OnceAsync::default(),
            predicted_batch_size: self.predicted_batch_size,
            left_block_size: self.left_block_size,
            right_block_size: self.right_block_size,
            request_window: self.request_window,
            reasoning: self.reasoning,
            certification: self.certification.clone(),
            certified_fixed: self.certified_fixed,
            probe_col_names: self.probe_col_names.clone(),
            request_sink: self.request_sink.clone(),
        }
    }

    /// Plan-time block geometry, present only when both sizes were resolved
    /// by the optimizer. The certify rule gates on this rather than on
    /// `predicted_batch_size`, which the planner always fills (with a fallback
    /// estimate) even for geometries that never execute.
    pub fn planned_block_sizes(&self) -> Option<(usize, usize)> {
        match (self.left_block_size, self.right_block_size) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }

    /// Runtime certification attachment, if any.
    pub fn certification(&self) -> Option<&CertifyRuntime> {
        self.certification.as_ref()
    }

    /// Plan-time-resolved certified pair budget, if any.
    pub fn certified_fixed(&self) -> Option<usize> {
        self.certified_fixed
    }

    /// Model identifier resolved at planning time.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Name of the provider serving `model_id`.
    pub fn provider_name(&self) -> &str {
        self.model_provider.name()
    }

    /// The natural-language join condition (the prompt's bound predicate).
    pub fn join_condition(&self) -> &str {
        &self.join_condition
    }

    /// Schema-derived join column names per side: what the renderer receives
    /// and the prompt displays. The certificate cache key hashes these rather
    /// than the possibly qualified names the user wrote. `execute()` uses the
    /// same derivation.
    pub fn join_column_names(&self) -> (Vec<String>, Vec<String>) {
        let left_schema = self.left.schema();
        let left = self
            .left_join_indices
            .iter()
            .map(|&i| left_schema.field(i).name().clone())
            .collect();
        let right_schema = self.right.schema();
        let right = self
            .right_join_indices
            .iter()
            .map(|&i| right_schema.field(i).name().clone())
            .collect();
        (left, right)
    }

    /// Compute plan properties.
    ///
    /// Output partitioning matches the left (probe) input's partition count,
    /// since each left partition runs its own stream against the shared right data.
    fn compute_properties(left: &Arc<dyn ExecutionPlan>, schema: SchemaRef) -> Arc<PlanProperties> {
        let partitioning =
            Partitioning::UnknownPartitioning(left.output_partitioning().partition_count());

        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for SemanticJoinExec {
    fn name(&self) -> &str {
        "SemanticJoinExec"
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    /// Holds no physical expressions (the join condition is a natural-language string).
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    /// Required input distribution.
    ///
    /// - Left (probe): any partitioning, each partition runs independently
    /// - Right (build): must be a single partition so we can materialize once
    ///
    /// If the right input has multiple partitions, DataFusion's
    /// `EnforceDistribution` rule will automatically insert a
    /// `CoalescePartitionsExec` to merge them into one.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![
            Distribution::UnspecifiedDistribution, // left (probe)
            Distribution::SinglePartition,         // right (build)
        ]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Plan(
                "SemanticJoinExec requires exactly 2 children".to_string(),
            ));
        }

        Ok(Arc::new(Self {
            left: children[0].clone(),
            right: children[1].clone(),
            left_join_indices: self.left_join_indices.clone(),
            right_join_indices: self.right_join_indices.clone(),
            model_provider: self.model_provider.clone(),
            model_id: self.model_id.clone(),
            model_card: self.model_card.clone(),
            join_condition: self.join_condition.clone(),
            output_schema: self.output_schema.clone(),
            properties: Self::compute_properties(&children[0], self.output_schema.clone()),
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            right_data: OnceAsync::default(),
            predicted_batch_size: self.predicted_batch_size,
            left_block_size: self.left_block_size,
            right_block_size: self.right_block_size,
            request_window: self.request_window,
            reasoning: self.reasoning,
            // The shared `OnceAsync` keeps the once-per-operator-instance
            // calibration identity across rebuilds.
            certification: self.certification.clone(),
            certified_fixed: self.certified_fixed,
            probe_col_names: self.probe_col_names.clone(),
            request_sink: self.request_sink.clone(),
        }))
    }

    /// Execute the semantic join for a given partition.
    ///
    /// The first call triggers materialization of the right side via `OnceAsync`.
    /// Subsequent calls from other partitions await the same shared future.
    /// Each left partition gets its own `SemanticJoinStream` that probes the
    /// shared `Arc<JoinRightData>`.
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Build the right side once (the first partition triggers it, others wait).
        let reservation = MemoryConsumer::new(format!("SemanticJoinBuild[{partition}]"))
            .with_can_spill(true)
            .register(context.memory_pool());

        let right_plan = self.right.clone();
        let right_fut = self.right_data.try_once(|| {
            let stream = right_plan.execute(0, Arc::clone(&context))?;
            Ok(collect_right_input(stream, reservation))
        })?;

        let left_stream = self.left.execute(partition, context.clone())?;

        let output_batch_size = context.session_config().batch_size();

        log::debug!(
            "[SemanticJoin] execute(partition={}) | output_batch_size={} | right_plan_partitions={}",
            partition,
            output_batch_size,
            self.right.output_partitioning().partition_count()
        );

        let partition_metrics = SemanticJoinMetrics::new(&self.metrics, partition);
        if let Some(bs) = self.predicted_batch_size {
            partition_metrics.base.record_predicted_batch_size(bs);
        }

        // Column names for the renderer (the same derivation the certify rule
        // hashes into the cache key). A probe exec carries them explicitly: its
        // left child is a pair batch, so the schema-derived names would read
        // `left_0` / `right_0`.
        let (left_col_names, right_col_names) = self
            .probe_col_names
            .clone()
            .unwrap_or_else(|| self.join_column_names());

        // Hand every partition the same Arc'd OnceAsync so calibration runs
        // once per operator instance; metrics are per-partition as usual.
        let partitions = self.properties.output_partitioning().partition_count();
        let certification = self.certification.as_ref().map(|rt| StreamCertification {
            spec: rt.spec.clone(),
            cache: rt.cache.clone(),
            once: rt.once.clone(),
            pool: rt.pool.clone(),
            partition,
            metrics: CalibrationMetrics::new(&self.metrics, partition),
            factory: self.probe_factory(),
            task_ctx: context.clone(),
            window_budget: self.request_window.max(1) * partitions.max(1),
        });

        let mut stream = SemanticJoinStream::new(
            left_stream,
            right_fut,
            self.left_join_indices.clone(),
            self.right_join_indices.clone(),
            self.join_condition.clone(),
            self.model_provider.clone(),
            self.model_id.clone(),
            self.model_card.clone(),
            self.output_schema.clone(),
            partition_metrics,
            self.inference_timer.clone(),
            output_batch_size,
            left_col_names,
            right_col_names,
            self.left_block_size,
            self.right_block_size,
            self.request_window,
            self.reasoning,
            certification,
        );
        if let Some(sink) = &self.request_sink {
            stream = stream.with_request_sink(sink.clone());
        }

        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        let mut metrics = self.metrics.clone_inner();

        let inference_time = Time::new();
        inference_time.add_duration(self.inference_timer.elapsed());
        metrics.push(Arc::new(Metric::new(
            MetricValue::Time {
                name: "inference_time".into(),
                time: inference_time,
            },
            None,
        )));

        Some(metrics)
    }
}

/// Asynchronously collect right input into a single RecordBatch with memory tracking.
///
/// Each incoming batch's memory is reserved via `try_grow()`. If the memory pool
/// is exhausted, returns `Err(ResourcesExhausted)` instead of OOM.
/// After all batches are collected, they are merged via `concat_batches()`.
async fn collect_right_input(
    mut stream: SendableRecordBatchStream,
    reservation: MemoryReservation,
) -> Result<JoinRightData> {
    let schema = stream.schema();
    let mut batches: Vec<RecordBatch> = Vec::new();

    while let Some(result) = stream.next().await {
        let batch = result?;
        if batch.num_rows() == 0 {
            continue;
        }
        let batch_mem = batch.get_array_memory_size();
        reservation.try_grow(batch_mem)?;
        batches.push(batch);
    }

    let merged = if batches.is_empty() {
        RecordBatch::new_empty(schema)
    } else {
        concat_batches(&schema, &batches)?
    };

    // Re-measure after the concat: `get_array_memory_size` on a sliced batch
    // reports the whole parent buffer, so the summed per-batch grows can
    // overstate the retained footprint. The merged batch owns compact buffers
    // and is the only thing kept.
    reservation.try_resize(merged.get_array_memory_size())?;

    Ok(JoinRightData {
        batch: merged,
        reservation,
    })
}

fn resolve_column_indices(
    column_names: &[String],
    schema: &SchemaRef,
    relation_description: &str,
) -> Result<Vec<usize>> {
    let mut indices = Vec::new();

    for col_name in column_names {
        let (_qualifier, unqualified_name) = if col_name.contains('.') {
            let parts: Vec<&str> = col_name.split('.').collect();
            if parts.len() == 2 {
                (Some(parts[0]), parts[1])
            } else {
                return Err(DataFusionError::Execution(format!(
                    "Invalid column name '{}': expected format 'table.column' or 'column'",
                    col_name
                )));
            }
        } else {
            (None, col_name.as_str())
        };

        let field_index = schema
            .fields()
            .iter()
            .position(|field| field.name() == unqualified_name)
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Column '{}' not found in {} schema. \
                     Available columns: [{}]",
                    col_name,
                    relation_description,
                    schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;

        indices.push(field_index);
    }

    Ok(indices)
}

impl DisplayAs for SemanticJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "SemanticJoinExec: left_cols={:?}, right_cols={:?}, model_id={}, provider={}, \
                     context_window={}, max_batch_size={}, join_condition='{}', \
                     predicted_batch_size={:?}, left_block_size={:?}, right_block_size={:?}, \
                     request_window={}",
                    self.left_join_indices,
                    self.right_join_indices,
                    self.model_id,
                    self.model_provider.name(),
                    self.model_card.context_window,
                    self.model_card.max_batch_size,
                    self.join_condition,
                    self.predicted_batch_size,
                    self.left_block_size,
                    self.right_block_size,
                    self.request_window,
                )?;
                if self.reasoning != ReasoningMode::Off {
                    write!(f, ", reasoning={}", self.reasoning.label())?;
                }
                if let Some(rt) = &self.certification {
                    write!(
                        f,
                        ", certify=[ladder={:?}, n={}, s_star={}, delta_prime={:.4}]",
                        rt.spec.ladder, rt.spec.n, rt.spec.s_star, rt.spec.delta_prime,
                    )?;
                }
                if let Some(k) = self.certified_fixed {
                    write!(f, ", certified_batch_size={k}(cache)")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "left_cols={:?}\nright_cols={:?}\nmodel={}\n\
                     context_window={}\nmax_batch_size={}\njoin_condition='{}'",
                    self.left_join_indices,
                    self.right_join_indices,
                    self.model_provider.name(),
                    self.model_card.context_window,
                    self.model_card.max_batch_size,
                    self.join_condition
                )
            }
        }
    }
}
