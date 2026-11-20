//! The two counters a reasoning mode is measured by: `reasoning_tokens` and
//! `truncated_batches`.
//!
//! A counter that is registered but never incremented reads as "reasoning cost
//! nothing" or "nothing was cut off", so these tests pin the wiring for all
//! three operators.
//!
//! `total_tokens` must not grow with reasoning: the endpoint bills reasoning
//! out of the completion budget, so `reasoning_tokens` is a subset of
//! `output_tokens` and adding it again would double-count.

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::{AdmissionControl, ModelProvider};
use nousdb_core::providers::schemas::ModelOutput;

use nousdb_operators::extend::stream::ExtendStream;
use nousdb_operators::join::stream::{JoinRightData, OnceAsync, SemanticJoinStream};
use nousdb_operators::metrics::{SemanticJoinMetrics, SemanticOperatorMetrics, WallClockTimer};
use nousdb_operators::selection::stream::SemanticSelectionStream;

use futures::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// How a mock response ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    /// A complete, decodable answer.
    Complete,
    /// `finish_reason == "length"` with a decodable answer anyway. The batch
    /// is used, so `failed_batches` does not record that it was cut short.
    TruncatedButUsable,
    /// `finish_reason == "length"` and the structured array never terminated:
    /// the ordinary truncation, which also takes the operator's recovery path.
    TruncatedAndUnusable,
}

#[derive(Debug)]
struct MockProvider {
    calls: AtomicUsize,
    /// Reasoning tokens reported per call.
    reasoning_tokens: u64,
    /// Applied to the call at this 0-based index; every other call is
    /// `Complete`.
    ending_at: Option<(usize, Ending)>,
    boolean_mode: bool,
}

impl MockProvider {
    fn new(reasoning_tokens: u64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            reasoning_tokens,
            ending_at: None,
            boolean_mode: false,
        }
    }

    fn ending(mut self, call: usize, ending: Ending) -> Self {
        self.ending_at = Some((call, ending));
        self
    }

    fn boolean(mut self) -> Self {
        self.boolean_mode = true;
        self
    }

    fn ending_for(&self, call_idx: usize) -> Ending {
        match self.ending_at {
            Some((idx, ending)) if idx == call_idx => ending,
            _ => Ending::Complete,
        }
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn models(&self) -> Vec<ModelCard> {
        vec![]
    }

    fn model(&self, _model_id: &str) -> Option<ModelCard> {
        None
    }

    fn admission_control(&self) -> Option<AdmissionControl> {
        None
    }

    async fn predict(
        &self,
        _model_id: &str,
        _prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        _renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let call_idx = self.calls.fetch_add(1, Ordering::SeqCst);
        let ending = self.ending_for(call_idx);

        // A `<think>` block is billed inside the completion, so output_tokens
        // is >= reasoning_tokens here, the way a real endpoint reports it.
        let metrics = InferenceMetrics::new(10, self.reasoning_tokens + 5, 1_000)
            .with_reasoning_tokens(self.reasoning_tokens);

        if ending == Ending::TruncatedAndUnusable {
            // The provider null-fills and flags the mismatch; `truncated` is
            // what tells the operator the cause was the budget.
            return Ok(PredictResult::new(Vec::new(), metrics, true).with_truncation(true));
        }

        let outputs: Vec<ModelOutput> = if let Some(col) = batch.column_by_name("val") {
            let vals = col.as_any().downcast_ref::<Int64Array>().unwrap();
            (start_row..start_row + batch_size)
                .map(|i| {
                    let v = vals.value(i);
                    if self.boolean_mode {
                        ModelOutput::Boolean(v % 2 == 0)
                    } else {
                        ModelOutput::Text(format!("r{v}"))
                    }
                })
                .collect()
        } else {
            let left0 = batch
                .column_by_name("left_0")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (start_row..start_row + batch_size)
                .map(|i| ModelOutput::Boolean(left0.value(i).parse::<i64>().unwrap() % 2 == 0))
                .collect()
        };

        Ok(PredictResult::new(outputs, metrics, false)
            .with_truncation(ending == Ending::TruncatedButUsable))
    }
}

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
}

fn extend_output_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("out", DataType::Utf8, true)]))
}

fn extend_full_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("val", DataType::Int64, false),
        Field::new("out", DataType::Utf8, true),
    ]))
}

fn make_batch(vals: std::ops::Range<i64>) -> RecordBatch {
    let arr: ArrayRef = Arc::new(Int64Array::from(vals.collect::<Vec<_>>()));
    RecordBatch::try_new(input_schema(), vec![arr]).unwrap()
}

fn input_stream(batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        input_schema(),
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

/// Sum of a named counter across the whole metrics set.
fn counter(set: &ExecutionPlanMetricsSet, name: &str) -> usize {
    set.clone_inner()
        .sum_by_name(name)
        .map(|m| m.as_usize())
        .unwrap_or(0)
}

#[tokio::test]
async fn extend_records_reasoning_tokens_and_truncations() {
    let provider = Arc::new(MockProvider::new(300).ending(1, Ending::TruncatedAndUnusable));
    let set = ExecutionPlanMetricsSet::new();
    let stream = ExtendStream::new(
        input_stream(vec![make_batch(0..4)]),
        "double the value".to_string(),
        vec!["val".to_string()],
        extend_output_schema(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        extend_full_schema(),
        SemanticOperatorMetrics::new(&set, 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Think,
        None,
    );
    let _out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(counter(&set, "llm_calls"), 2);
    assert_eq!(
        counter(&set, "reasoning_tokens"),
        600,
        "300 per call, both calls"
    );
    assert_eq!(counter(&set, "truncated_batches"), 1);
    // The ordinary truncation is also a failure, and is attributable to truncation.
    assert_eq!(counter(&set, "failed_batches"), 1);
    assert_eq!(counter(&set, "failed_rows"), 2);
    assert_eq!(
        counter(&set, "total_tokens"),
        2 * (10 + 305),
        "reasoning tokens are inside output_tokens, never added on top"
    );
}

#[tokio::test]
async fn selection_records_reasoning_tokens_and_truncations() {
    let provider = Arc::new(
        MockProvider::new(120)
            .boolean()
            .ending(0, Ending::TruncatedAndUnusable),
    );
    let set = ExecutionPlanMetricsSet::new();
    let stream = SemanticSelectionStream::new(
        input_stream(vec![make_batch(0..4)]),
        "keep even values".to_string(),
        vec!["val".to_string()],
        provider,
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        input_schema(),
        SemanticOperatorMetrics::new(&set, 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Rationale,
        None,
    );
    let _out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(counter(&set, "llm_calls"), 2);
    assert_eq!(counter(&set, "reasoning_tokens"), 240);
    assert_eq!(counter(&set, "truncated_batches"), 1);
    assert_eq!(counter(&set, "failed_batches"), 1);
}

/// A response that was cut off but still decoded is used, so `failed_batches`
/// does not see it; only `truncated_batches` records that it ran out of budget.
#[tokio::test]
async fn a_truncated_but_usable_response_is_counted_without_a_failure() {
    let provider = Arc::new(MockProvider::new(0).ending(0, Ending::TruncatedButUsable));
    let set = ExecutionPlanMetricsSet::new();
    let stream = ExtendStream::new(
        input_stream(vec![make_batch(0..2)]),
        "double the value".to_string(),
        vec!["val".to_string()],
        extend_output_schema(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        extend_full_schema(),
        SemanticOperatorMetrics::new(&set, 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Off,
        None,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    assert_eq!(counter(&set, "truncated_batches"), 1);
    assert_eq!(
        counter(&set, "failed_batches"),
        0,
        "the answer decoded, so nothing took a recovery path"
    );
}

/// With reasoning off, both counters stay at zero.
#[tokio::test]
async fn a_clean_non_reasoning_run_leaves_both_counters_at_zero() {
    let provider = Arc::new(MockProvider::new(0));
    let set = ExecutionPlanMetricsSet::new();
    let stream = ExtendStream::new(
        input_stream(vec![make_batch(0..4)]),
        "double the value".to_string(),
        vec!["val".to_string()],
        extend_output_schema(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        extend_full_schema(),
        SemanticOperatorMetrics::new(&set, 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Off,
        None,
    );
    let _out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(counter(&set, "llm_calls"), 2);
    assert_eq!(counter(&set, "reasoning_tokens"), 0);
    assert_eq!(counter(&set, "truncated_batches"), 0);
}

#[tokio::test]
async fn join_records_reasoning_tokens_and_truncations() {
    let provider = Arc::new(MockProvider::new(50).ending(1, Ending::TruncatedAndUnusable));
    let set = ExecutionPlanMetricsSet::new();

    let left_schema = Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![Arc::new(Int64Array::from(vec![0, 1, 2, 3])) as ArrayRef],
    )
    .unwrap();
    let right_schema = Arc::new(Schema::new(vec![Field::new(
        "rval",
        DataType::Int64,
        false,
    )]));
    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(Int64Array::from(vec![10, 11])) as ArrayRef],
    )
    .unwrap();
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("lval", DataType::Int64, false),
        Field::new("rval", DataType::Int64, false),
    ]));

    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let reservation = MemoryConsumer::new("test-join").register(&pool);
    let once: OnceAsync<JoinRightData> = OnceAsync::default();
    let right_fut = once
        .try_once(move || {
            Ok(async move {
                Ok(JoinRightData {
                    batch: right_batch,
                    reservation,
                })
            })
        })
        .unwrap();

    let left_input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        left_schema,
        futures::stream::iter(vec![Ok(left_batch)]),
    ));

    // 2x2 blocks over 4 left rows x 2 right rows give two 4-pair requests.
    let stream = SemanticJoinStream::new(
        left_input,
        right_fut,
        vec![0],
        vec![0],
        "left value is even".to_string(),
        provider,
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        out_schema,
        SemanticJoinMetrics::new(&set, 0),
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        Some(2),
        Some(2),
        1,
        ReasoningMode::Scaffold,
        None,
    );
    let _out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(counter(&set, "llm_calls"), 2);
    assert_eq!(counter(&set, "reasoning_tokens"), 100);
    assert_eq!(counter(&set, "truncated_batches"), 1);
    // A join counts pairs, so the failed request forfeits all 4 of its pairs.
    assert_eq!(counter(&set, "failed_batches"), 1);
    assert_eq!(counter(&set, "failed_rows"), 4);
}
