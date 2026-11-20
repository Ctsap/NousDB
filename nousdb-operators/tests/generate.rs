//! Generate operator invariants, driven through `GenerateExec` with a mock
//! provider that renders the real prompt: requests cover every row with one
//! short tail, responses decode into the declared types, unusable responses
//! are retried (halved, or unsplit when they never reached the model) within a
//! bounded budget, the prompt prefix is shared across requests, the window
//! overlaps requests without reordering output, and dropping the stream
//! releases admission permits.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::limit_pushdown::LimitPushdown;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, StatisticsArgs,
    StatisticsContext, collect, displayable,
};

use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::{AdmissionControl, ModelProvider};
use nousdb_core::providers::schemas::ModelOutput;

use nousdb_operators::generate::{GenerateExec, GenerateNode, create_generate};

use futures::StreamExt;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the mock does on a given call (0-based, in dispatch order).
#[derive(Debug, Clone, Copy)]
enum Behaviour {
    /// Parse failure, null-filled by the provider.
    Unusable,
    /// Ran out of output budget mid-array.
    Truncated,
    /// The request never reached the model.
    Transport,
    /// A well-formed array whose `founded` is a string.
    WrongType,
    /// `predict` itself fails.
    Error,
}

#[derive(Debug)]
struct GenerateMock {
    /// `(start_row, batch_size)` per call, in dispatch order.
    calls: Mutex<Vec<(usize, usize)>>,
    /// The rendered user prompt per call.
    prompts: Mutex<Vec<String>>,
    behaviour: HashMap<usize, Behaviour>,
    /// Applied to every call not in `behaviour`.
    default_failure: Option<Behaviour>,
    delay: Duration,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    admission: Option<AdmissionControl>,
}

impl GenerateMock {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            prompts: Mutex::new(Vec::new()),
            behaviour: HashMap::new(),
            default_failure: None,
            delay: Duration::from_millis(10),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            admission: None,
        }
    }

    fn failing(mut self, call: usize, behaviour: Behaviour) -> Self {
        self.behaviour.insert(call, behaviour);
        self
    }

    fn calls(&self) -> Vec<(usize, usize)> {
        self.calls.lock().unwrap().clone()
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

/// Row `i` of the generated table, as the model would answer it. `founded`
/// is spelled with a decimal point on purpose, as vLLM does.
fn row(i: usize) -> serde_json::Value {
    json!({
        "name": format!("row{i}"),
        "founded": 1990.0 + i as f64,
        "score": i as f64 / 2.0,
        "public": i.is_multiple_of(2),
    })
}

#[async_trait]
impl ModelProvider for GenerateMock {
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
        self.admission.clone()
    }

    async fn predict(
        &self,
        _model_id: &str,
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((start_row, batch_size));
            calls.len() - 1
        };
        // The same render the real providers do; a template/context mismatch
        // fails here under `UndefinedBehavior::Strict`.
        let rendered = prompt.render_batch(renderer, batch, start_row, batch_size)?;
        self.prompts.lock().unwrap().push(rendered);

        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        let metrics = InferenceMetrics::new(100, 10 * batch_size as u64, 0);
        let rows = |f: fn(usize) -> serde_json::Value| {
            (start_row..start_row + batch_size)
                .map(|i| ModelOutput::Structured(f(i)))
                .collect::<Vec<_>>()
        };

        match self.behaviour.get(&call).copied().or(self.default_failure) {
            None => Ok(PredictResult::new(rows(row), metrics, false)),
            Some(Behaviour::Unusable) => Ok(PredictResult::new(vec![], metrics, true)),
            Some(Behaviour::Truncated) => {
                Ok(PredictResult::new(vec![], metrics, true).with_truncation(true))
            }
            Some(Behaviour::Transport) => Ok(PredictResult::transport_failure(5)),
            Some(Behaviour::WrongType) => Ok(PredictResult::new(
                rows(|i| {
                    let mut r = row(i);
                    r["founded"] = json!("nineteen ninety");
                    r
                }),
                metrics,
                false,
            )),
            Some(Behaviour::Error) => Err(DataFusionError::Execution("endpoint down".into())),
        }
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("founded", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("public", DataType::Boolean, true),
    ]))
}

fn exec(mock: &Arc<GenerateMock>, num_rows: usize, chunk_size: usize) -> GenerateExec {
    GenerateExec::new(
        schema(),
        "fictional startups".to_string(),
        vec![],
        num_rows,
        "mock-model".to_string(),
        mock.clone(),
        ModelCard::new("mock", 8192),
    )
    .with_chunk_size(chunk_size)
}

async fn run(exec: &GenerateExec) -> Result<Vec<RecordBatch>> {
    let stream = exec.execute(0, Arc::new(TaskContext::default()))?;
    stream.collect::<Vec<_>>().await.into_iter().collect()
}

fn names(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone();
            (0..col.len()).map(move |i| col.value(i).to_string())
        })
        .collect()
}

fn counter(exec: &GenerateExec, name: &str) -> usize {
    exec.metrics()
        .unwrap()
        .sum_by_name(name)
        .map(|v| match v {
            MetricValue::Count { count, .. } => count.value(),
            MetricValue::Gauge { gauge, .. } => gauge.value(),
            other => other.as_usize(),
        })
        .unwrap_or(0)
}

#[tokio::test(start_paused = true)]
async fn requests_cover_every_row_with_one_short_tail() {
    let mock = Arc::new(GenerateMock::new());
    let exec = exec(&mock, 23, 10);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.calls(), vec![(0, 10), (10, 10), (20, 3)]);
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        vec![10, 10, 3],
        "one batch per request, no coalescing"
    );
    assert_eq!(
        names(&batches),
        (0..23).map(|i| format!("row{i}")).collect::<Vec<_>>()
    );
    assert_eq!(counter(&exec, "llm_calls"), 3);
    assert_eq!(counter(&exec, "rows_produced"), 23);
    assert_eq!(counter(&exec, "rows_processed"), 0, "a source has no input");
    assert_eq!(counter(&exec, "failed_batches"), 0);
    assert_eq!(counter(&exec, "chunk_size"), 10);
}

#[tokio::test(start_paused = true)]
async fn responses_decode_into_the_declared_types() {
    let mock = Arc::new(GenerateMock::new());
    let batches = run(&exec(&mock, 3, 3)).await.unwrap();
    let batch = &batches[0];
    assert_eq!(batch.schema(), schema());

    let founded = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let score = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let public = batch
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(founded.values().to_vec(), vec![1990, 1991, 1992]);
    assert_eq!(score.values().to_vec(), vec![0.0, 0.5, 1.0]);
    assert_eq!(
        (0..3).map(|i| public.value(i)).collect::<Vec<_>>(),
        vec![true, false, true]
    );
}

#[tokio::test(start_paused = true)]
async fn an_unusable_response_is_retried_as_two_halves() {
    let mock = Arc::new(GenerateMock::new().failing(0, Behaviour::Unusable));
    let exec = exec(&mock, 10, 10);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.calls(), vec![(0, 10), (0, 5), (5, 5)]);
    assert_eq!(
        names(&batches),
        (0..10).map(|i| format!("row{i}")).collect::<Vec<_>>()
    );
    assert_eq!(counter(&exec, "failed_batches"), 1);
    assert_eq!(counter(&exec, "failed_rows"), 0, "every row was recovered");
    assert_eq!(counter(&exec, "llm_calls"), 3);
}

#[tokio::test(start_paused = true)]
async fn a_truncated_response_is_retried_as_two_halves() {
    let mock = Arc::new(GenerateMock::new().failing(0, Behaviour::Truncated));
    let exec = exec(&mock, 7, 7);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.calls(), vec![(0, 7), (0, 4), (4, 3)]);
    assert_eq!(names(&batches).len(), 7);
    assert_eq!(counter(&exec, "truncated_batches"), 1);
    assert_eq!(counter(&exec, "failed_rows"), 0);
}

#[tokio::test(start_paused = true)]
async fn a_transport_failure_is_retried_unsplit() {
    let mock = Arc::new(GenerateMock::new().failing(0, Behaviour::Transport));
    let exec = exec(&mock, 10, 10);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.calls(), vec![(0, 10), (0, 10)]);
    assert_eq!(names(&batches).len(), 10);
    assert_eq!(counter(&exec, "failed_batches"), 1);
}

#[tokio::test(start_paused = true)]
async fn a_type_mismatch_is_retried_rather_than_fatal() {
    let mock = Arc::new(GenerateMock::new().failing(0, Behaviour::WrongType));
    let exec = exec(&mock, 6, 6);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.calls(), vec![(0, 6), (0, 3), (3, 3)]);
    assert_eq!(names(&batches).len(), 6);
    assert_eq!(counter(&exec, "failed_batches"), 1);
}

#[tokio::test(start_paused = true)]
async fn a_failing_single_row_is_dropped_without_a_retry() {
    let mut mock = GenerateMock::new();
    mock.default_failure = Some(Behaviour::Unusable);
    let mock = Arc::new(mock);
    let exec = exec(&mock, 1, 1);
    let batches = run(&exec).await.unwrap();

    assert_eq!(
        mock.calls(),
        vec![(0, 1)],
        "the same prompt would fail again"
    );
    assert!(batches.is_empty());
    assert_eq!(counter(&exec, "failed_rows"), 1);
}

#[tokio::test(start_paused = true)]
async fn an_exhausted_retry_budget_finishes_short() {
    // One initial request ⇒ a budget of two retries.
    let mut mock = GenerateMock::new();
    mock.default_failure = Some(Behaviour::Unusable);
    let mock = Arc::new(mock);
    let exec = exec(&mock, 4, 4);
    let batches = run(&exec).await.unwrap();

    // (0,4) fails → halves [budget 1]; (0,2) fails → halves [budget 0];
    // (0,1) and (1,1) fail as singletons → dropped; (2,2) fails with no budget
    // left → dropped.
    assert_eq!(mock.calls(), vec![(0, 4), (0, 2), (0, 1), (1, 1), (2, 2)]);
    assert!(batches.is_empty(), "the stream ends short, not in error");
    assert_eq!(counter(&exec, "failed_rows"), 4);
    assert_eq!(counter(&exec, "failed_batches"), 5);
}

#[tokio::test(start_paused = true)]
async fn a_predict_error_fails_the_stream() {
    let mock = Arc::new(GenerateMock::new().failing(0, Behaviour::Error));
    let err = run(&exec(&mock, 5, 5)).await.unwrap_err();
    assert!(err.to_string().contains("endpoint down"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn the_prompt_prefix_is_shared_across_requests() {
    let mock = Arc::new(GenerateMock::new());
    run(&exec(&mock, 23, 10)).await.unwrap();
    let prompts = mock.prompts();
    assert_eq!(prompts.len(), 3);

    const TAIL: &str = "## Rows to Generate";
    let split = |p: &str| {
        let at = p.find(TAIL).expect("the size-dependent section is present");
        (p[..at].to_string(), p[at..].to_string())
    };
    let (prefix0, tail0) = split(&prompts[0]);
    for prompt in &prompts[1..] {
        let (prefix, _) = split(prompt);
        assert_eq!(
            prefix, prefix0,
            "everything above the last section is shared"
        );
    }
    assert!(tail0.contains("**23** rows"), "{tail0}");
    assert!(tail0.contains("rows **1–10**"), "{tail0}");
    let (_, tail2) = split(&prompts[2]);
    assert!(tail2.contains("rows **21–23**"), "{tail2}");
    assert!(tail2.contains("exactly **3** objects"), "{tail2}");

    // Nothing size-dependent above the last section.
    assert!(!prefix0.contains("**23**"), "{prefix0}");
    assert!(prefix0.contains("`founded` (integer)"), "{prefix0}");
    assert!(prefix0.contains("**Generate**: fictional startups"));
}

#[tokio::test(start_paused = true)]
async fn example_rows_render_above_the_last_section() {
    let example = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(vec!["Acme"])),
            Arc::new(Int64Array::from(vec![2019])),
            Arc::new(Float64Array::from(vec![0.5])),
            Arc::new(BooleanArray::from(vec![false])),
        ],
    )
    .unwrap();
    let plan = create_generate(schema(), "mock-model", "startups", 2, &[example]).unwrap();
    let LogicalPlan::Extension(ext) = &plan else {
        panic!("expected an extension")
    };
    let node = ext.node.as_any().downcast_ref::<GenerateNode>().unwrap();

    let mock = Arc::new(GenerateMock::new());
    let with_examples = GenerateExec::new(
        node.output_schema(),
        node.instruction.clone(),
        node.examples.clone(),
        node.num_rows,
        node.model.clone(),
        mock.clone(),
        ModelCard::new("mock", 8192),
    );
    run(&with_examples).await.unwrap();

    let prompt = &mock.prompts()[0];
    let examples_at = prompt.find("## Example Rows").expect("examples section");
    let tail_at = prompt.find("## Rows to Generate").unwrap();
    assert!(examples_at < tail_at);
    assert!(
        prompt.contains(r#"{"name":"Acme","founded":2019,"score":0.5,"public":false}"#),
        "{prompt}"
    );

    // Without examples the section is absent and the render still succeeds.
    let plain = Arc::new(GenerateMock::new());
    run(&exec(&plain, 2, 2)).await.unwrap();
    assert!(!plain.prompts()[0].contains("## Example Rows"));
}

#[tokio::test(start_paused = true)]
async fn the_window_overlaps_requests_and_keeps_output_order() {
    let mock = Arc::new(GenerateMock::new());
    let exec = exec(&mock, 30, 5).with_request_window(3);
    let batches = run(&exec).await.unwrap();

    assert_eq!(mock.max_in_flight.load(Ordering::SeqCst), 3);
    assert_eq!(
        names(&batches),
        (0..30).map(|i| format!("row{i}")).collect::<Vec<_>>()
    );
}

#[tokio::test(start_paused = true)]
async fn dropping_the_stream_releases_admission_permits() {
    let mut mock = GenerateMock::new();
    mock.delay = Duration::from_secs(3600);
    let admission = AdmissionControl::new(8);
    mock.admission = Some(admission.clone());
    let mock = Arc::new(mock);
    let exec = exec(&mock, 40, 5).with_request_window(4);
    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();

    let poll = futures::poll!(stream.next());
    assert!(poll.is_pending(), "requests should be in flight");
    assert_eq!(admission.available(), 4);

    drop(stream);
    assert_eq!(
        admission.available(),
        8,
        "dropping the stream must cancel requests and release permits"
    );
}

#[test]
fn the_exec_is_a_single_partition_leaf_sized_from_the_card() {
    let mock = Arc::new(GenerateMock::new());
    let card = ModelCard::new("mock", 8192).with_max_batch_size(4096);
    let exec = Arc::new(GenerateExec::new(
        schema(),
        "x".into(),
        vec![],
        100,
        "mock-model".into(),
        mock,
        card,
    ));
    assert!(exec.children().is_empty());
    assert_eq!(exec.properties().partitioning.partition_count(), 1);
    // (1638 − 20) / (6 + 1+1+1+1 + 48+4+6+2) = 1618 / 70 = 23
    assert_eq!(exec.chunk_size(), 23);
    assert_eq!(
        StatisticsContext::new()
            .compute(exec.as_ref(), &StatisticsArgs::new())
            .unwrap()
            .num_rows
            .get_value(),
        Some(&100)
    );
    let rebuilt = exec
        .clone()
        .replace_children(
            vec![],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    assert!(rebuilt.is::<GenerateExec>());
    assert!(exec.execute(1, Arc::new(TaskContext::default())).is_err());
}

// ---------------------------------------------------------------------------
// Fetch: a limit above Generate, as DataFusion hands it to a scan
// ---------------------------------------------------------------------------

/// `LIMIT fetch OFFSET skip` above `generate`, after DataFusion's physical
/// `LimitPushdown` (the rule the session runs).
fn limited(generate: GenerateExec, skip: usize, fetch: usize) -> Arc<dyn ExecutionPlan> {
    let plan = Arc::new(GlobalLimitExec::new(Arc::new(generate), skip, Some(fetch)));
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

#[tokio::test(start_paused = true)]
async fn a_limit_above_generate_becomes_its_fetch() {
    let mock = Arc::new(GenerateMock::new());
    let plan = limited(exec(&mock, 100, 20), 0, 5);

    // No limit operator is left: the leaf took the fetch, as a scan does.
    let generate = plan
        .downcast_ref::<GenerateExec>()
        .unwrap_or_else(|| panic!("the limit was not absorbed:\n{:?}", plan_lines(&plan)));
    assert_eq!(generate.fetch(), Some(5));
    assert_eq!(generate.num_rows(), 100, "the table is still 100 rows");
    assert_eq!(generate.rows_to_generate(), 5);
    assert!(plan_lines(&plan)[0].ends_with(
        "num_rows=100, examples=0, provider=mock, chunk_size=20, request_window=1, fetch=5"
    ));
    assert_eq!(
        StatisticsContext::new()
            .compute(plan.as_ref(), &StatisticsArgs::new())
            .unwrap()
            .num_rows
            .get_value(),
        Some(&5)
    );

    let batches = collect(plan.clone(), Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(mock.calls(), vec![(0, 5)], "one request for five rows");
    assert_eq!(
        names(&batches),
        (0..5).map(|i| format!("row{i}")).collect::<Vec<_>>()
    );
    assert_eq!(counter(generate, "llm_calls"), 1);
    assert_eq!(counter(generate, "rows_produced"), 5);
    assert_eq!(counter(generate, "chunk_size"), 5);

    // The prompt still describes the whole table: a fetch reads a prefix of
    // it and does not define a smaller one.
    let prompt = &mock.prompts()[0];
    assert!(prompt.contains("**100** rows"), "{prompt}");
    assert!(prompt.contains("rows **1–5**"), "{prompt}");
    assert!(prompt.contains("exactly **5** objects"), "{prompt}");
}

/// DataFusion applies an OFFSET itself; the leaf is asked for skip + fetch rows.
#[tokio::test(start_paused = true)]
async fn an_offset_stays_above_generate_and_the_fetch_covers_it() {
    let mock = Arc::new(GenerateMock::new());
    let plan = limited(exec(&mock, 100, 20), 2, 5);

    let lines = plan_lines(&plan);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0], "GlobalLimitExec: skip=2, fetch=5");
    assert!(lines[1].ends_with("fetch=7"), "{lines:?}");

    let batches = collect(plan, Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(mock.calls(), vec![(0, 7)]);
    assert_eq!(
        names(&batches),
        (2..7).map(|i| format!("row{i}")).collect::<Vec<_>>()
    );
}

#[tokio::test(start_paused = true)]
async fn a_fetch_cuts_the_request_plan_and_never_extends_the_table() {
    // Across requests: 15 of 23 rows is one full request and a short one.
    let mock = Arc::new(GenerateMock::new());
    let plan = limited(exec(&mock, 23, 10), 0, 15);
    let batches = collect(plan, Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(mock.calls(), vec![(0, 10), (10, 5)]);
    assert_eq!(names(&batches).len(), 15);

    // A fetch beyond the table generates the table, no more.
    let mock = Arc::new(GenerateMock::new());
    let plan = limited(exec(&mock, 23, 10), 0, 50);
    let batches = collect(plan, Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert_eq!(mock.calls(), vec![(0, 10), (10, 10), (20, 3)]);
    assert_eq!(names(&batches).len(), 23);
}

/// The retry budget follows the fetched request plan: a dropped row inside
/// the fetch leaves the result short, as without a fetch.
#[tokio::test(start_paused = true)]
async fn a_dropped_row_inside_the_fetch_leaves_the_result_short() {
    let mut mock = GenerateMock::new();
    mock.default_failure = Some(Behaviour::Unusable);
    let mock = Arc::new(mock);
    let plan = limited(exec(&mock, 100, 20), 0, 2);
    let batches = collect(plan.clone(), Arc::new(TaskContext::default()))
        .await
        .unwrap();
    assert!(names(&batches).is_empty());
    // One request of two rows, retried as two single rows (budget 2), dropped.
    assert_eq!(mock.calls(), vec![(0, 2), (0, 1), (1, 1)]);
    let generate = plan.downcast_ref::<GenerateExec>().unwrap();
    assert_eq!(counter(generate, "failed_rows"), 2);
}
