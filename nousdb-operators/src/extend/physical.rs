//! Physical (`ExecutionPlan`) operator for Extend.

use crate::certification::{
    CertifyRuntime, ProbeExecFactory, RequestOutcomeSink, StreamCertification,
};
use crate::extend::stream::ExtendStream;
use crate::metrics::{CalibrationMetrics, SemanticOperatorMetrics, WallClockTimer};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    metrics::{ExecutionPlanMetricsSet, Metric, MetricValue, MetricsSet, Time},
};
use datafusion_physical_plan::ExecutionPlanProperties;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::provider::ModelProvider;
use std::fmt;
use std::sync::Arc;

/// Physical operator for Extend: calls the model on packed batches of input
/// rows and appends the derived columns.
#[derive(Debug)]
pub struct ExtendExec {
    input: Arc<dyn ExecutionPlan>,
    user_instruction: String,
    input_columns: Vec<String>,
    output_schema: Arc<datafusion::arrow::datatypes::Schema>,
    model_id: String,
    model_provider: Arc<dyn ModelProvider>,
    /// Model configuration resolved at planning time
    model_card: ModelCard,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    /// Wall-clock timer for inference, shared across all partitions
    inference_timer: Arc<WallClockTimer>,
    /// Optimizer-predicted batch size (from BatchSizeOptimizationRule)
    optimal_batch_size: Option<usize>,
    /// Max concurrent in-flight requests per partition (1 = serial dispatch)
    request_window: usize,
    /// Reasoning mode; selects the system template and the response contract.
    /// Read by `CertifyBatchSizeRule` for the certificate key, so it must
    /// survive every clone path.
    reasoning: ReasoningMode,
    /// Quality-certification spec attached by `CertifyBatchSizeRule`; when
    /// present the streams run a calibration prologue before steady state.
    certification: Option<CertifyRuntime>,
    /// Provenance for a plan-time-resolved certified size (certificate-cache
    /// hit or cardinality-gate skip); `optimal_batch_size` already reflects it.
    certified_fixed: Option<usize>,
    /// Set only on calibration probe plans (see [`Self::probe_exec`]): reports
    /// per-request outcomes and makes the stream tolerate request errors.
    request_sink: Option<RequestOutcomeSink>,
}

impl ExtendExec {
    /// Creates an Extend operator over `input`. `schema` is the full output
    /// schema (input columns followed by the derived fields of `output_schema`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        user_instruction: String,
        input_columns: Vec<String>,
        output_schema: Arc<datafusion::arrow::datatypes::Schema>,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        model_card: ModelCard,
        schema: SchemaRef,
        optimal_batch_size: Option<usize>,
    ) -> Self {
        let properties = Self::compute_properties(&input, schema.clone());

        Self {
            input,
            user_instruction,
            input_columns,
            output_schema,
            model_id,
            model_provider,
            model_card,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            optimal_batch_size,
            request_window: 1,
            reasoning: ReasoningMode::default(),
            certification: None,
            certified_fixed: None,
            request_sink: None,
        }
    }

    /// Build a single-purpose copy of this operator for one calibration arm:
    /// `k` rows per request over `input`, `window` requests in flight, no
    /// certification (probe plans never recurse) and `sink` installed.
    ///
    /// Deliberately not `with_new_children`: that reuses `self.schema`, but a
    /// probe input carries an extra row-identity column, so the probe exec's
    /// schema is the probe input's schema plus the derived fields.
    pub fn probe_exec(
        &self,
        input: Arc<dyn ExecutionPlan>,
        k: usize,
        window: usize,
        sink: RequestOutcomeSink,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Self::build_schema(&input.schema(), &self.output_schema);
        let mut exec = ExtendExec::new(
            input,
            self.user_instruction.clone(),
            self.input_columns.clone(),
            self.output_schema.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.model_card.clone(),
            schema,
            Some(k.max(1)),
        )
        .with_request_window(window)
        .with_reasoning_mode(self.reasoning);
        exec.request_sink = Some(sink);
        Ok(Arc::new(exec))
    }

    /// Extend's output schema: input columns followed by the derived fields.
    fn build_schema(
        input_schema: &SchemaRef,
        output_schema: &Arc<datafusion::arrow::datatypes::Schema>,
    ) -> SchemaRef {
        let mut fields = input_schema.fields().to_vec();
        fields.extend(output_schema.fields().iter().cloned());
        Arc::new(datafusion::arrow::datatypes::Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ))
    }

    /// A [`ProbeExecFactory`] bound to this operator's configuration, handed
    /// to the calibration prologue so it dispatches through the engine rather
    /// than through a dispatcher of its own.
    fn probe_factory(&self) -> ProbeExecFactory {
        let template = Arc::new(self.shallow_clone());
        Arc::new(move |input, k, window, sink| template.probe_exec(input, k, window, sink))
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

    /// The reasoning mode this operator runs with.
    pub fn reasoning_mode(&self) -> ReasoningMode {
        self.reasoning
    }

    /// Attach a quality-certification runtime (spec + cache + the shared
    /// once-per-instance calibration cell). Set by `CertifyBatchSizeRule`.
    pub fn with_certification(&self, runtime: CertifyRuntime) -> Self {
        let mut exec = self.shallow_clone();
        exec.certification = Some(runtime);
        exec.certified_fixed = None;
        exec
    }

    /// Resolve the batch size to a plan-time-certified fixed value
    /// (certificate-cache hit or cardinality-gate skip): no prologue runs.
    pub fn with_fixed_certified_size(&self, k: usize) -> Self {
        let mut exec = self.shallow_clone();
        exec.optimal_batch_size = Some(k.max(1));
        exec.certified_fixed = Some(k.max(1));
        exec.certification = None;
        exec
    }

    /// Field-preserving copy with a fresh metrics set; only valid before
    /// execution, which is when optimizer rules run.
    fn shallow_clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            user_instruction: self.user_instruction.clone(),
            input_columns: self.input_columns.clone(),
            output_schema: self.output_schema.clone(),
            model_id: self.model_id.clone(),
            model_provider: self.model_provider.clone(),
            model_card: self.model_card.clone(),
            schema: self.schema.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
            optimal_batch_size: self.optimal_batch_size,
            request_window: self.request_window,
            reasoning: self.reasoning,
            certification: self.certification.clone(),
            certified_fixed: self.certified_fixed,
            request_sink: self.request_sink.clone(),
        }
    }

    /// Optimizer-chosen number of rows per request, if set.
    pub fn optimal_batch_size(&self) -> Option<usize> {
        self.optimal_batch_size
    }

    /// The attached certification runtime, if any.
    pub fn certification(&self) -> Option<&CertifyRuntime> {
        self.certification.as_ref()
    }

    /// The plan-time certified batch size, if one was resolved.
    pub fn certified_fixed(&self) -> Option<usize> {
        self.certified_fixed
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Name of the model provider.
    pub fn provider_name(&self) -> &str {
        self.model_provider.name()
    }

    /// Natural-language instruction for the model.
    pub fn user_instruction(&self) -> &str {
        &self.user_instruction
    }

    /// Names of the input columns shown to the model.
    pub fn input_columns(&self) -> &[String] {
        &self.input_columns
    }

    /// The derived-only output schema (the columns this operator adds).
    pub fn output_schema(&self) -> &Arc<datafusion::arrow::datatypes::Schema> {
        &self.output_schema
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Arc<PlanProperties> {
        let partitioning = input.output_partitioning().clone();
        let boundedness = Boundedness::Bounded;
        let emission_type = EmissionType::Incremental;
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            partitioning,
            emission_type,
            boundedness,
        ))
    }
}

impl ExecutionPlan for ExtendExec {
    fn name(&self) -> &str {
        "ExtendExec"
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

    /// Holds no physical expressions (the instruction and columns are plain strings).
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
        let mut exec = ExtendExec::new(
            children[0].clone(),
            self.user_instruction.clone(),
            self.input_columns.clone(),
            self.output_schema.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.model_card.clone(),
            self.schema.clone(),
            self.optimal_batch_size,
        )
        .with_request_window(self.request_window)
        .with_reasoning_mode(self.reasoning);
        // Preserve certification state; the same Arc'd OnceAsync keeps the
        // once-per-operator-instance calibration identity across rebuilds.
        exec.certification = self.certification.clone();
        exec.certified_fixed = self.certified_fixed;
        exec.request_sink = self.request_sink.clone();
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let partition_metrics = SemanticOperatorMetrics::new(&self.metrics, partition);
        if partition == 0
            && let Some(bs) = self.optimal_batch_size
        {
            partition_metrics.record_predicted_batch_size(bs);
        }

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

        let mut stream = ExtendStream::new(
            input_stream,
            self.user_instruction.clone(),
            self.input_columns.clone(),
            self.output_schema.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.model_card.clone(),
            self.schema.clone(),
            partition_metrics,
            self.inference_timer.clone(),
            self.optimal_batch_size,
            self.request_window,
            context.session_config().batch_size(),
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

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    /// DataFusion's physical limit hook: a limit above the extend may be
    /// applied to its input. Each partition maps rows one to one and in
    /// order, so limiting the input limits the output to the same rows.
    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

impl DisplayAs for ExtendExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ExtendExec: model={}, instruction='{}', input_cols={:?}, provider={}, optimal_batch_size={:?}, request_window={}",
                    self.model_id,
                    self.user_instruction,
                    self.input_columns,
                    self.model_provider.name(),
                    self.optimal_batch_size,
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
                    "model={}\n instruction='{}'\n input_cols={:?}\n provider={}",
                    self.model_id,
                    self.user_instruction,
                    self.input_columns,
                    self.model_provider.name()
                )?;
                Ok(())
            }
        }
    }
}
