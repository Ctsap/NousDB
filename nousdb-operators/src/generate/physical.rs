//! Physical (`ExecutionPlan`) operator for Generate.

use crate::generate::stream::GenerateStream;
use crate::metrics::{SemanticOperatorMetrics, WallClockTimer};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, internal_err};
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    metrics::{ExecutionPlanMetricsSet, Metric, MetricBuilder, MetricValue, MetricsSet, Time},
};
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::token_estimator::generate_chunk_size;
use std::fmt;
use std::sync::Arc;

/// Physical Generate: a leaf that asks the model for `num_rows` rows of
/// `schema`, `chunk_size` rows per request, in one partition.
///
/// Like a scan, it takes a fetch ([`ExecutionPlan::with_fetch`]): DataFusion's
/// `LimitPushdown` hands it a `LIMIT` placed above it, and it then generates
/// only the first `fetch` rows of the same `num_rows`-row table.
///
/// The chunk size is resolved here from the model card rather than by the
/// planner, so anything that builds a `GenerateExec` directly (the planned SQL
/// table function) sizes requests the same way.
#[derive(Debug)]
pub struct GenerateExec {
    schema: SchemaRef,
    instruction: String,
    examples: Vec<String>,
    num_rows: usize,
    model_id: String,
    model_provider: Arc<dyn ModelProvider>,
    /// Model configuration resolved at planning time
    model_card: ModelCard,
    /// Rows asked for per request (output-token-bound, see
    /// [`generate_chunk_size`]).
    chunk_size: usize,
    /// Max concurrent in-flight requests (1 = serial dispatch)
    request_window: usize,
    /// Rows a limit above asked for; only that prefix of the table is generated.
    fetch: Option<usize>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    /// Wall-clock timer for inference
    inference_timer: Arc<WallClockTimer>,
}

impl GenerateExec {
    /// Creates a Generate operator. The rows-per-request chunk size is derived
    /// from `model_card` and the schema; the request window defaults to 1.
    pub fn new(
        schema: SchemaRef,
        instruction: String,
        examples: Vec<String>,
        num_rows: usize,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        model_card: ModelCard,
    ) -> Self {
        let chunk_size = generate_chunk_size(&model_card, schema.fields(), num_rows);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Self {
            schema,
            instruction,
            examples,
            num_rows,
            model_id,
            model_provider,
            model_card,
            chunk_size,
            request_window: 1,
            fetch: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
        }
    }

    /// Set the in-flight request window (resolved by the planner from the
    /// model card; clamped to at least 1).
    pub fn with_request_window(mut self, request_window: usize) -> Self {
        self.request_window = request_window.max(1);
        self
    }

    /// Override the card-derived rows per request (clamped to at least 1).
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Number of rows of the table being generated.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Rows the model is asked for: `num_rows`, or the fetch when it is smaller.
    pub fn rows_to_generate(&self) -> usize {
        self.fetch.map_or(self.num_rows, |f| f.min(self.num_rows))
    }

    /// Rows asked for per request.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Max concurrent in-flight requests.
    pub fn request_window(&self) -> usize {
        self.request_window
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The model card resolved at planning time.
    pub fn model_card(&self) -> &ModelCard {
        &self.model_card
    }

    /// What to generate.
    pub fn instruction(&self) -> &str {
        &self.instruction
    }
}

impl ExecutionPlan for GenerateExec {
    fn name(&self) -> &str {
        "GenerateExec"
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    /// Holds no physical expressions (the instruction and examples are plain strings).
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    /// DataFusion calls this with an empty vec on every leaf it rebuilds.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("GenerateExec is a leaf and takes no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("GenerateExec has one partition, asked for {partition}");
        }
        MetricBuilder::new(&self.metrics)
            .gauge("chunk_size", partition)
            .set(self.chunk_size.min(self.rows_to_generate()));

        let stream = GenerateStream::new(
            self.schema.clone(),
            self.instruction.clone(),
            self.examples.clone(),
            self.num_rows,
            self.chunk_size,
            self.model_id.clone(),
            self.model_provider.clone(),
            self.request_window,
            SemanticOperatorMetrics::new(&self.metrics, partition),
            self.inference_timer.clone(),
        )?
        .with_fetch(self.fetch);
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

    /// At most `rows_to_generate` rows: requests the model fails are dropped.
    /// Telling the planner so keeps it from repartitioning a small generated
    /// table.
    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(
            Statistics::new_unknown(&self.schema)
                .with_num_rows(Precision::Inexact(self.rows_to_generate())),
        ))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// A copy that generates only the first `fetch` rows, as a scan reads
    /// only the rows a limit asks for. Fresh metrics: only valid before
    /// execution, which is when the optimizer calls it.
    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            schema: self.schema.clone(),
            instruction: self.instruction.clone(),
            examples: self.examples.clone(),
            num_rows: self.num_rows,
            model_id: self.model_id.clone(),
            model_provider: self.model_provider.clone(),
            model_card: self.model_card.clone(),
            chunk_size: self.chunk_size,
            request_window: self.request_window,
            fetch,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            inference_timer: Arc::new(WallClockTimer::new()),
        }))
    }
}

impl DisplayAs for GenerateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "GenerateExec: model={}, instruction='{}', num_rows={}, examples={}, provider={}, chunk_size={}, request_window={}",
                    self.model_id,
                    self.instruction,
                    self.num_rows,
                    self.examples.len(),
                    self.model_provider.name(),
                    self.chunk_size,
                    self.request_window,
                )?;
                if let Some(fetch) = self.fetch {
                    write!(f, ", fetch={fetch}")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "model={}\n instruction='{}'\n num_rows={}\n chunk_size={}\n provider={}",
                    self.model_id,
                    self.instruction,
                    self.num_rows,
                    self.chunk_size,
                    self.model_provider.name()
                )?;
                if let Some(fetch) = self.fetch {
                    write!(f, "\n fetch={fetch}")?;
                }
                Ok(())
            }
        }
    }
}
