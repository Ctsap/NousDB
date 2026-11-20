//! The fetch on semantic selection, modelled on DataFusion's `FilterExec`.
//!
//! A limit above a selection is never pushed to its input (a filter is not
//! one to one). DataFusion's physical `LimitPushdown` hands the row count to
//! the operator instead (`ExecutionPlan::with_fetch`), and each partition
//! then stops after that many accepted rows. Because a model call is the
//! cost, the stream also holds back requests the ones in flight may make
//! unnecessary.

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::config::ConfigOptions;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::limit_pushdown::LimitPushdown;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricValue};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, SendableRecordBatchStream,
    collect, displayable,
};

use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::schemas::ModelOutput;

use nousdb_operators::certification::RequestOutcomeSink;
use nousdb_operators::metrics::{SemanticOperatorMetrics, WallClockTimer};
use nousdb_operators::selection::SemanticSelectionExec;
use nousdb_operators::selection::stream::SemanticSelectionStream;

use futures::StreamExt;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Accepts the even values.
#[derive(Debug, Default)]
struct KeepEven {
    /// Rows per call, in dispatch order.
    calls: Mutex<Vec<usize>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    /// 0-based calls that come back unusable (row-count mismatch).
    fail_calls: HashSet<usize>,
}

impl KeepEven {
    fn calls(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl ModelProvider for KeepEven {
    fn name(&self) -> &str {
        "mock"
    }

    fn models(&self) -> Vec<ModelCard> {
        vec![]
    }

    fn model(&self, _model_id: &str) -> Option<ModelCard> {
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
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(batch_size);
            calls.len() - 1
        };
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(10)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        if self.fail_calls.contains(&call) {
            return Ok(PredictResult::new(vec![], InferenceMetrics::zero(), true));
        }
        let vals = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let outputs = (start_row..start_row + batch_size)
            .map(|i| ModelOutput::Boolean(vals.value(i) % 2 == 0))
            .collect();
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
}

fn batch(vals: impl IntoIterator<Item = i64>) -> RecordBatch {
    let arr: ArrayRef = Arc::new(Int64Array::from(vals.into_iter().collect::<Vec<_>>()));
    RecordBatch::try_new(schema(), vec![arr]).unwrap()
}

fn vals(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let v = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            v.values().to_vec()
        })
        .collect()
}

/// A selection stream over `input`: `k` rows per request, `window` in flight.
async fn run_stream(
    provider: &Arc<KeepEven>,
    input: RecordBatch,
    k: usize,
    window: usize,
    fetch: Option<usize>,
) -> Vec<i64> {
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::iter(vec![Ok(input)]),
    ));
    let stream = SemanticSelectionStream::new(
        input,
        "keep even values".to_string(),
        vec!["val".to_string()],
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        schema(),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        Some(k),
        window,
        8192,
        ReasoningMode::Off,
        None,
    )
    .with_fetch(fetch);
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    vals(&out)
}

// ---------------------------------------------------------------------------
// The stream
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_fetch_ends_the_stream_after_that_many_accepted_rows() {
    // Every request of two rows holds one even value. Without a fetch the
    // stream reads all 40 rows (20 requests) before it emits anything.
    let provider = Arc::new(KeepEven::default());
    let out = run_stream(&provider, batch(0..40), 2, 4, Some(3)).await;
    assert_eq!(out, vec![0, 2, 4]);
    assert_eq!(
        provider.calls(),
        vec![2, 2, 2],
        "three rows need three requests"
    );
}

#[tokio::test(start_paused = true)]
async fn the_last_request_is_truncated_to_the_missing_rows() {
    // Every row passes: six rows are one request of four and half of another.
    let provider = Arc::new(KeepEven::default());
    let out = run_stream(&provider, batch((0..40).map(|v| v * 2)), 4, 8, Some(6)).await;
    assert_eq!(out, vec![0, 2, 4, 6, 8, 10]);
    assert_eq!(provider.calls(), vec![4, 4]);
    assert_eq!(provider.max_in_flight.load(Ordering::SeqCst), 2);
}

/// A request goes out only while the rows in flight could not fill the fetch
/// even if all of them passed. Over a prefix the model rejects, that is one
/// request at a time, however wide the window.
#[tokio::test(start_paused = true)]
async fn no_request_is_sent_that_the_answers_so_far_did_not_make_necessary() {
    let provider = Arc::new(KeepEven::default());
    let rejected_then_accepted = (0..16).map(|v| v * 2 + 1).chain((0..40).map(|v| v * 2));
    let out = run_stream(&provider, batch(rejected_then_accepted), 4, 8, Some(2)).await;
    assert_eq!(out, vec![0, 2]);
    assert_eq!(
        provider.calls(),
        vec![4; 5],
        "four requests of rejected rows, then the one that fills the fetch"
    );
    assert_eq!(provider.max_in_flight.load(Ordering::SeqCst), 1);

    // A fetch the window cannot fill at once uses the whole window.
    let provider = Arc::new(KeepEven::default());
    let out = run_stream(&provider, batch(0..400), 4, 8, Some(100)).await;
    assert_eq!(out.len(), 100);
    assert_eq!(provider.max_in_flight.load(Ordering::SeqCst), 8);
    assert_eq!(
        provider.calls().len(),
        50,
        "100 even values are 50 requests"
    );
}

/// Requests are packed as without a fetch, so the fetched rows are the first
/// rows of the unlimited output, whatever the window.
#[tokio::test(start_paused = true)]
async fn the_fetched_rows_are_a_prefix_of_the_unlimited_output() {
    let input = || batch((0..60).map(|v| (v * 7) % 23));
    let provider = Arc::new(KeepEven::default());
    let all = run_stream(&provider, input(), 3, 4, None).await;
    let all_calls = provider.calls().len();
    assert!(all.len() > 9);

    for window in [1, 4, 16] {
        for fetch in [1, 2, 5, 9, all.len(), all.len() + 50] {
            let provider = Arc::new(KeepEven::default());
            let out = run_stream(&provider, input(), 3, window, Some(fetch)).await;
            assert_eq!(
                out,
                all[..fetch.min(all.len())],
                "window {window}, fetch {fetch}"
            );
            if fetch > all.len() {
                assert_eq!(provider.calls().len(), all_calls, "the whole input is read");
            }
        }
    }
}

#[tokio::test(start_paused = true)]
async fn a_fetch_of_zero_makes_no_call() {
    let provider = Arc::new(KeepEven::default());
    let out = run_stream(&provider, batch(0..40), 2, 4, Some(0)).await;
    assert!(out.is_empty());
    assert!(provider.calls().is_empty());
}

/// Recovery passes a failed request's rows through unfiltered; they are
/// output rows, so they count toward the fetch.
#[tokio::test(start_paused = true)]
async fn rows_passed_through_by_recovery_count_toward_the_fetch() {
    let provider = Arc::new(KeepEven {
        fail_calls: HashSet::from([0]),
        ..KeepEven::default()
    });
    let out = run_stream(&provider, batch(0..40), 4, 4, Some(3)).await;
    assert_eq!(out, vec![0, 1, 2]);
    assert_eq!(provider.calls(), vec![4]);
}

// ---------------------------------------------------------------------------
// The exec, under DataFusion's physical `LimitPushdown`
// ---------------------------------------------------------------------------

fn selection(provider: &Arc<KeepEven>, partitions: &[Vec<RecordBatch>]) -> SemanticSelectionExec {
    let input = MemorySourceConfig::try_new_exec(partitions, schema(), None).unwrap();
    SemanticSelectionExec::new(
        input,
        "keep even values".to_string(),
        vec!["val".to_string()],
        "mock-model".to_string(),
        provider.clone(),
        ModelCard::new("mock", 8192),
        schema(),
        Some(2),
    )
    .with_request_window(4)
}

fn push_limits(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    LimitPushdown::new()
        .optimize(plan, &ConfigOptions::default())
        .unwrap()
}

fn plan_lines(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    displayable(plan.as_ref())
        .indent(false)
        .to_string()
        .lines()
        .map(|l| l.trim().to_string())
        .collect()
}

fn counter(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .unwrap()
        .sum_by_name(name)
        .map(|v| match v {
            MetricValue::Count { count, .. } => count.value(),
            other => other.as_usize(),
        })
        .unwrap_or(0)
}

#[tokio::test(start_paused = true)]
async fn datafusion_hands_a_limit_above_the_selection_to_it_as_a_fetch() {
    let provider = Arc::new(KeepEven::default());
    let exec = Arc::new(selection(&provider, &[vec![batch(0..40)]]));
    assert_eq!(exec.fetch(), None);
    assert!(
        !exec.supports_limit_pushdown(),
        "a filter passes no limit down"
    );

    let plan = push_limits(Arc::new(GlobalLimitExec::new(exec, 0, Some(3))));
    let lines = plan_lines(&plan);
    assert_eq!(lines.len(), 2, "no limit operator is left: {lines:?}");
    assert!(lines[0].starts_with("SemanticSelectionExec:"), "{lines:?}");
    assert!(lines[0].ends_with("request_window=4, fetch=3"), "{lines:?}");
    assert_eq!(plan.fetch(), Some(3));
    assert_eq!(plan.children()[0].fetch(), None, "the input is not limited");

    let out = collect(plan.clone(), Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(vals(&out), vec![0, 2, 4]);
    assert_eq!(counter(plan.as_ref(), "llm_calls"), 3);
    assert_eq!(
        counter(plan.as_ref(), "rows_processed"),
        6,
        "rows the model judged"
    );
    assert_eq!(counter(plan.as_ref(), "rows_produced"), 3, "rows emitted");
}

/// DataFusion applies an OFFSET itself; the selection keeps skip + fetch rows.
#[tokio::test(start_paused = true)]
async fn an_offset_stays_above_the_selection_and_the_fetch_covers_it() {
    let provider = Arc::new(KeepEven::default());
    let exec = Arc::new(selection(&provider, &[vec![batch(0..40)]]));
    let plan = push_limits(Arc::new(GlobalLimitExec::new(exec, 2, Some(3))));

    let lines = plan_lines(&plan);
    assert_eq!(lines[0], "GlobalLimitExec: skip=2, fetch=3");
    assert!(lines[1].ends_with("fetch=5"), "{lines:?}");

    let out = collect(plan, Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(vals(&out), vec![4, 6, 8]);
    assert_eq!(provider.calls().len(), 5);
}

/// Over several partitions the merge takes the fetch and so does every
/// partition of the selection, as with `FilterExec`.
#[tokio::test(flavor = "multi_thread")]
async fn every_partition_gets_the_fetch_under_a_merge() {
    let provider = Arc::new(KeepEven::default());
    let exec = Arc::new(selection(
        &provider,
        &[vec![batch(0..40)], vec![batch(100..140)]],
    ));
    let plan = push_limits(Arc::new(GlobalLimitExec::new(
        Arc::new(CoalescePartitionsExec::new(exec)),
        0,
        Some(3),
    )));

    let lines = plan_lines(&plan);
    assert_eq!(lines[0], "CoalescePartitionsExec: fetch=3", "{lines:?}");
    assert!(lines[1].ends_with("fetch=3"), "{lines:?}");

    let out = vals(
        &collect(plan, Arc::new(TaskContext::default()))
            .await
            .unwrap(),
    );
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|v| v % 2 == 0));
    assert!(
        provider.calls().len() <= 6,
        "at most three requests per partition, not twenty: {:?}",
        provider.calls()
    );
}

/// The fetch is execution state: it survives every rebuild the optimizer
/// does, and a calibration probe never has one.
#[tokio::test]
async fn the_fetch_survives_plan_rebuilds_and_probes_have_none() {
    let provider = Arc::new(KeepEven::default());
    let exec = selection(&provider, &[vec![batch(0..40)]])
        .with_fetch(Some(7))
        .unwrap();
    let child = exec.children()[0].clone();

    let rebuilt = exec
        .clone()
        .replace_children(
            vec![child.clone()],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    assert_eq!(rebuilt.fetch(), Some(7));

    let selection_exec = exec.downcast_ref::<SemanticSelectionExec>().unwrap();
    assert_eq!(selection_exec.with_fixed_certified_size(4).fetch(), Some(7));
    assert_eq!(exec.with_fetch(None).unwrap().fetch(), None);

    let probe = selection_exec
        .probe_exec(child, 2, 1, RequestOutcomeSink::new(|_, _| {}))
        .unwrap();
    assert_eq!(probe.fetch(), None);
}
