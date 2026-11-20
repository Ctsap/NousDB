//! Windowed-dispatch invariants for the extend/selection streams:
//! requests overlap up to `request_window`, output order and cardinality are
//! preserved, W=1 gives serial dispatch, per-request
//! failures degrade gracefully, and dropping the stream releases admission
//! permits.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
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
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug)]
struct MockProvider {
    /// rows-per-call, in dispatch order
    calls: Mutex<Vec<usize>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    delay: Duration,
    /// 0-based call indices that return a row-count mismatch
    fail_calls: HashSet<usize>,
    admission: Option<AdmissionControl>,
    /// produce Boolean outputs (selection) instead of Text (extend)
    boolean_mode: bool,
}

impl MockProvider {
    fn new(delay: Duration) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            delay,
            fail_calls: HashSet::new(),
            admission: None,
            boolean_mode: false,
        }
    }

    fn call_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
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
        self.admission.clone()
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
        let call_idx = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(batch_size);
            calls.len() - 1
        };

        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        if self.fail_calls.contains(&call_idx) {
            return Ok(PredictResult::new(
                Vec::new(),
                InferenceMetrics::zero(),
                true,
            ));
        }

        let outputs = if let Some(col) = batch.column_by_name("val") {
            // extend/selection input batch
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
            // join synthetic pair batch: left values stringified in "left_0";
            // a pair matches when the left value is even
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
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
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

fn make_extend_stream(
    provider: Arc<MockProvider>,
    input: SendableRecordBatchStream,
    optimal_batch_size: Option<usize>,
    request_window: usize,
) -> ExtendStream {
    make_extend_stream_with_target(provider, input, optimal_batch_size, request_window, 8192)
}

fn make_extend_stream_with_target(
    provider: Arc<MockProvider>,
    input: SendableRecordBatchStream,
    optimal_batch_size: Option<usize>,
    request_window: usize,
    output_batch_size: usize,
) -> ExtendStream {
    ExtendStream::new(
        input,
        "double the value".to_string(),
        vec!["val".to_string()],
        extend_output_schema(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        extend_full_schema(),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        optimal_batch_size,
        request_window,
        output_batch_size,
        ReasoningMode::Off,
        None,
    )
}

fn make_selection_stream(
    provider: Arc<MockProvider>,
    input: SendableRecordBatchStream,
    optimal_batch_size: Option<usize>,
    request_window: usize,
) -> SemanticSelectionStream {
    SemanticSelectionStream::new(
        input,
        "keep even values".to_string(),
        vec!["val".to_string()],
        provider,
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        input_schema(),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        optimal_batch_size,
        request_window,
        8192,
        ReasoningMode::Off,
        None,
    )
}

/// Flatten output batches into (val, out) rows for order/cardinality checks.
fn collect_extend_rows(batches: &[RecordBatch]) -> (Vec<i64>, Vec<Option<String>>) {
    let mut vals = Vec::new();
    let mut outs = Vec::new();
    for b in batches {
        let v = b
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let o = b
            .column_by_name("out")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            vals.push(v.value(i));
            outs.push(if o.is_null(i) {
                None
            } else {
                Some(o.value(i).to_string())
            });
        }
    }
    (vals, outs)
}

fn collect_vals(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let v = b
                .column_by_name("val")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..b.num_rows()).map(|i| v.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test(start_paused = true)]
async fn windowed_dispatch_overlaps_requests() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(50)));
    let batches = vec![
        make_batch(0..2),
        make_batch(2..4),
        make_batch(4..6),
        make_batch(6..8),
    ];
    let stream = make_extend_stream(provider.clone(), input_stream(batches), Some(2), 4);
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(
        provider.max_in_flight.load(Ordering::SeqCst),
        4,
        "all 4 requests must be in flight simultaneously at W=4"
    );
    let (vals, outs) = collect_extend_rows(&out);
    assert_eq!(vals, (0..8).collect::<Vec<_>>(), "input order preserved");
    for (v, o) in vals.iter().zip(&outs) {
        assert_eq!(o.as_deref(), Some(format!("r{v}").as_str()));
    }
}

#[tokio::test(start_paused = true)]
async fn w1_reproduces_legacy_serial_call_sequence() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    // [3,3,3] rows coalesced to 4 → request sizes [4, 4, 1]. A coalesce
    // overflow emits only whole k-sized requests; the short tail returns to
    // the buffer and is flushed as one final request at exhaustion.
    let batches = vec![make_batch(0..3), make_batch(3..6), make_batch(6..9)];
    let stream = make_extend_stream(provider.clone(), input_stream(batches), Some(4), 1);
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes(), vec![4, 4, 1]);
    assert_eq!(
        provider.max_in_flight.load(Ordering::SeqCst),
        1,
        "W=1 must never overlap requests"
    );
    let (vals, _) = collect_extend_rows(&out);
    assert_eq!(vals, (0..9).collect::<Vec<_>>());
}

#[tokio::test(start_paused = true)]
async fn windowed_dispatch_keeps_legacy_slicing() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let batches = vec![make_batch(0..3), make_batch(3..6), make_batch(6..9)];
    let stream = make_extend_stream(provider.clone(), input_stream(batches), Some(4), 3);
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    // Same request shapes as W=1; only the overlap changes.
    assert_eq!(provider.call_sizes(), vec![4, 4, 1]);
    let (vals, _) = collect_extend_rows(&out);
    assert_eq!(vals, (0..9).collect::<Vec<_>>());
}

#[tokio::test(start_paused = true)]
async fn failed_request_null_fills_only_its_rows() {
    let mut mock = MockProvider::new(Duration::from_millis(10));
    mock.fail_calls = HashSet::from([1]);
    let provider = Arc::new(mock);
    let stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..6)]),
        Some(2),
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    let (vals, outs) = collect_extend_rows(&out);
    assert_eq!(vals, (0..6).collect::<Vec<_>>(), "cardinality preserved");
    for (v, o) in vals.iter().zip(&outs) {
        if *v == 2 || *v == 3 {
            assert!(o.is_none(), "failed request rows must be NULL-filled");
        } else {
            assert_eq!(o.as_deref(), Some(format!("r{v}").as_str()));
        }
    }
}

#[tokio::test(start_paused = true)]
async fn failed_selection_request_passes_rows_through() {
    let mut mock = MockProvider::new(Duration::from_millis(10));
    mock.boolean_mode = true;
    mock.fail_calls = HashSet::from([1]);
    let provider = Arc::new(mock);
    let stream = make_selection_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..6)]),
        Some(2),
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    // Evens kept ({0,1}→0, {4,5}→4); failed request rows {2,3} pass through.
    assert_eq!(collect_vals(&out), vec![0, 2, 3, 4]);
}

#[tokio::test(start_paused = true)]
async fn admission_caps_concurrency_below_window() {
    let mut mock = MockProvider::new(Duration::from_millis(50));
    mock.admission = Some(AdmissionControl::new(2));
    let provider = Arc::new(mock);
    let stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..8)]),
        Some(2),
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(
        provider.max_in_flight.load(Ordering::SeqCst),
        2,
        "global admission (2 permits) must cap a W=4 window"
    );
    let (vals, _) = collect_extend_rows(&out);
    assert_eq!(vals, (0..8).collect::<Vec<_>>());
}

#[tokio::test(start_paused = true)]
async fn input_stall_does_not_lose_wakeups() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(1);
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        input_schema(),
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }),
    ));

    tokio::spawn(async move {
        tx.send(Ok(make_batch(0..2))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        tx.send(Ok(make_batch(2..4))).await.unwrap();
        // tx drops here, ending the input.
    });

    let stream = make_extend_stream(provider.clone(), input, None, 2);
    let out = tokio::time::timeout(
        Duration::from_secs(30),
        stream.map(|r| r.unwrap()).collect::<Vec<_>>(),
    )
    .await
    .expect("stream stalled: lost wakeup across input gap");

    let (vals, _) = collect_extend_rows(&out);
    assert_eq!(vals, vec![0, 1, 2, 3]);
}

#[tokio::test(start_paused = true)]
async fn drop_mid_flight_releases_admission_permits() {
    let mut mock = MockProvider::new(Duration::from_secs(3600));
    let admission = AdmissionControl::new(8);
    mock.admission = Some(admission.clone());
    let provider = Arc::new(mock);
    let mut stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..8)]),
        Some(2),
        4,
    );

    let poll = futures::poll!(stream.next());
    assert!(poll.is_pending(), "requests should be in flight");
    assert_eq!(
        admission.available(),
        4,
        "4 of 8 permits held by in-flight requests"
    );

    drop(stream);
    assert_eq!(
        admission.available(),
        8,
        "dropping the stream must cancel requests and release permits"
    );
}

#[tokio::test(start_paused = true)]
async fn join_windowed_dispatch_overlaps_pair_requests() {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};

    let provider = Arc::new(MockProvider::new(Duration::from_millis(50)));

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
    let right_data_batch = right_batch.clone();
    let right_fut = once
        .try_once(move || {
            Ok(async move {
                Ok(JoinRightData {
                    batch: right_data_batch,
                    reservation,
                })
            })
        })
        .unwrap();

    let left_input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        left_schema,
        futures::stream::iter(vec![Ok(left_batch)]),
    ));

    // Blocks of 2×2 → two pair-batch calls of 4 pairs each; W=2 overlaps them.
    let stream = SemanticJoinStream::new(
        left_input,
        right_fut,
        vec![0],
        vec![0],
        "left value is even".to_string(),
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        out_schema,
        SemanticJoinMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        Some(2),
        Some(2),
        2,
        ReasoningMode::Off,
        None,
    );

    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(
        provider.call_sizes(),
        vec![4, 4],
        "two pair batches of 4 pairs"
    );
    assert_eq!(
        provider.max_in_flight.load(Ordering::SeqCst),
        2,
        "both pair-batch requests must overlap at W=2"
    );

    // Even left values (0, 2) match both right rows → 4 output rows,
    // in submission order: block [0,1] first, then block [2,3].
    let mut lvals = Vec::new();
    let mut rvals = Vec::new();
    for b in &out {
        let l = b
            .column_by_name("lval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let r = b
            .column_by_name("rval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            lvals.push(l.value(i));
            rvals.push(r.value(i));
        }
    }
    assert_eq!(lvals, vec![0, 0, 2, 2]);
    assert_eq!(rvals, vec![10, 11, 10, 11]);
}

/// Build the shared right-side future for a join stream test.
fn make_right_fut(
    right_batch: RecordBatch,
) -> nousdb_operators::join::stream::OnceFut<JoinRightData> {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let reservation = MemoryConsumer::new("test-join").register(&pool);
    let once: OnceAsync<JoinRightData> = OnceAsync::default();
    once.try_once(move || {
        Ok(async move {
            Ok(JoinRightData {
                batch: right_batch,
                reservation,
            })
        })
    })
    .unwrap()
}

/// With no plan-time block sizes the stream samples the left prefix, sizes with
/// a real left histogram, and must still emit every matched pair exactly
/// once, in order: the sampled prefix is recycled, not consumed.
#[tokio::test(start_paused = true)]
async fn join_runtime_sizing_samples_left_and_loses_no_rows() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));

    let left_schema = Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]));
    let left_batches: Vec<RecordBatch> = vec![
        RecordBatch::try_new(
            left_schema.clone(),
            vec![Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef],
        )
        .unwrap(),
        RecordBatch::try_new(
            left_schema.clone(),
            vec![Arc::new(Int64Array::from(vec![2, 3])) as ArrayRef],
        )
        .unwrap(),
    ];
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

    let left_input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        left_schema,
        futures::stream::iter(left_batches.into_iter().map(Ok)),
    ));

    let stream = SemanticJoinStream::new(
        left_input,
        make_right_fut(right_batch),
        vec![0],
        vec![0],
        "left value is even".to_string(),
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        out_schema,
        SemanticJoinMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        None, // ← runtime sizing
        None,
        2,
        ReasoningMode::Off,
        None,
    );

    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    // Every even left value matches both right rows, in left order; no row
    // from the sampled prefix may be dropped or duplicated.
    let mut lvals = Vec::new();
    let mut rvals = Vec::new();
    for b in &out {
        let l = b
            .column_by_name("lval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let r = b
            .column_by_name("rval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            lvals.push(l.value(i));
            rvals.push(r.value(i));
        }
    }
    assert_eq!(lvals, vec![0, 0, 2, 2]);
    assert_eq!(rvals, vec![10, 11, 10, 11]);
    // Every pair was evaluated exactly once.
    assert_eq!(provider.call_sizes().iter().sum::<usize>(), 8);
}

/// Long right rows make the planned 1×4 tile exceed the input budget: the
/// pre-flight guard must split the tile (recording overflow_errors) while
/// still evaluating every pair exactly once.
#[tokio::test(start_paused = true)]
async fn join_preflight_guard_splits_oversized_tiles() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));

    let left_schema = Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef],
    )
    .unwrap();
    // ~500 tokens per right row: two of them plus the static prompt overhead
    // overflow the ~1850-token input budget below; one fits.
    let long = "specification ".repeat(500);
    let right_schema = Arc::new(Schema::new(vec![Field::new("rval", DataType::Utf8, false)]));
    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(StringArray::from(vec![
            long.clone(),
            long.clone(),
            long.clone(),
            long.clone(),
        ])) as ArrayRef],
    )
    .unwrap();
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("lval", DataType::Int64, false),
        Field::new("rval", DataType::Utf8, false),
    ]));

    let left_input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        left_schema,
        futures::stream::iter(vec![Ok(left_batch)]),
    ));

    let card = ModelCard::new("mock", 2048).with_max_output_tokens(200);

    let metrics_set = ExecutionPlanMetricsSet::new();
    let join_metrics = SemanticJoinMetrics::new(&metrics_set, 0);
    let overflow_counter = join_metrics.base.overflow_errors.clone();

    // Plan-time geometry 1×4: one tile of 4 long right rows per left row.
    let stream = SemanticJoinStream::new(
        left_input,
        make_right_fut(right_batch),
        vec![0],
        vec![0],
        "left value is even".to_string(),
        provider.clone(),
        "mock-model".to_string(),
        card,
        out_schema,
        join_metrics,
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        Some(1),
        Some(4),
        1,
        ReasoningMode::Off,
        None,
    );

    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert!(
        overflow_counter.value() >= 1,
        "the guard must record at least one overflow split"
    );
    // Every pair still evaluated exactly once (2 left rows × 4 right rows),
    // just across more, smaller requests than the planned 1×4 tiles.
    assert_eq!(provider.call_sizes().iter().sum::<usize>(), 8);
    assert!(
        provider.call_sizes().iter().all(|&s| s < 4),
        "no request may carry the full oversized tile, got {:?}",
        provider.call_sizes()
    );
    // Left row 0 (even) matches all four right rows; row 1 none.
    let total_out: usize = out.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_out, 4);
}

#[tokio::test(start_paused = true)]
async fn empty_input_completes_without_calls() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let stream = make_extend_stream(provider.clone(), input_stream(vec![]), Some(2), 4);
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    assert!(out.is_empty());
    assert!(provider.call_sizes().is_empty());
}

/// Production extend re-coalesces per-request results up to the session
/// batch size: many small LLM requests must not fragment the output.
#[tokio::test(start_paused = true)]
async fn extend_production_output_stays_coalesced() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..60)]),
        Some(5),
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes().len(), 12, "60 rows at k=5");
    assert_eq!(
        out.len(),
        1,
        "12 request results must merge into one 8192-capacity output batch"
    );
    let (vals, outs) = collect_extend_rows(&out);
    assert_eq!(vals, (0..60).collect::<Vec<_>>(), "order preserved");
    for (v, o) in vals.iter().zip(&outs) {
        assert_eq!(o.as_deref(), Some(format!("r{v}").as_str()));
    }
}

/// Selection's per-request results are often short (or empty) after
/// filtering; production output must still arrive re-coalesced with no
/// zero-row batches surfacing downstream.
#[tokio::test(start_paused = true)]
async fn selection_production_output_stays_coalesced() {
    let mut mock = MockProvider::new(Duration::from_millis(10));
    mock.boolean_mode = true;
    let provider = Arc::new(mock);
    let stream = make_selection_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..60)]),
        Some(5),
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes().len(), 12, "60 rows at k=5");
    assert_eq!(out.len(), 1, "filtered results must merge into one batch");
    assert_eq!(
        collect_vals(&out),
        (0..60).filter(|v| v % 2 == 0).collect::<Vec<_>>(),
        "evens kept, in order"
    );
}

/// Probe (calibration) streams bypass the output coalescer: the scorer maps
/// request ordinal to probe by batch position, so every request must surface
/// exactly one batch.
#[tokio::test(start_paused = true)]
async fn extend_probe_mode_emits_one_batch_per_request() {
    use nousdb_operators::certification::RequestOutcomeSink;

    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..6)]),
        Some(2),
        1,
    )
    .with_request_sink(RequestOutcomeSink::new(|_, _| {}));
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes(), vec![2, 2, 2]);
    assert_eq!(out.len(), 3, "one output batch per request in probe mode");
    assert!(out.iter().all(|b| b.num_rows() == 2));
}

/// Probe-mode selection must emit filtered-to-empty batches too; dropping them
/// would shift the ordinal-to-probe mapping.
#[tokio::test(start_paused = true)]
async fn selection_probe_mode_emits_one_batch_per_request() {
    use nousdb_operators::certification::RequestOutcomeSink;

    let mut mock = MockProvider::new(Duration::from_millis(10));
    mock.boolean_mode = true;
    let provider = Arc::new(mock);
    let stream = make_selection_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..4)]),
        Some(1),
        1,
    )
    .with_request_sink(RequestOutcomeSink::new(|_, _| {}));
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes(), vec![1, 1, 1, 1]);
    let row_counts: Vec<usize> = out.iter().map(|b| b.num_rows()).collect();
    assert_eq!(
        row_counts,
        vec![1, 0, 1, 0],
        "probe mode surfaces empty request results instead of merging"
    );
}

/// The output coalescer honors its target mid-stream and flushes the partial
/// tail at end-of-input.
#[tokio::test(start_paused = true)]
async fn extend_output_respects_target_batch_size() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
    let stream = make_extend_stream_with_target(
        provider.clone(),
        input_stream(vec![make_batch(0..10)]),
        Some(2),
        1,
        4,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    let row_counts: Vec<usize> = out.iter().map(|b| b.num_rows()).collect();
    assert_eq!(row_counts, vec![4, 4, 2], "target mid-stream, tail flushed");
    let (vals, _) = collect_extend_rows(&out);
    assert_eq!(vals, (0..10).collect::<Vec<_>>());
}

/// A DataFusion-default-sized (8192-row) input batch is sliced into k-row
/// requests and re-coalesced into a single 8192-row output batch.
#[tokio::test(start_paused = true)]
async fn extend_splits_a_datafusion_sized_batch() {
    let provider = Arc::new(MockProvider::new(Duration::from_millis(1)));
    let stream = make_extend_stream(
        provider.clone(),
        input_stream(vec![make_batch(0..8192)]),
        Some(64),
        16,
    );
    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(provider.call_sizes().len(), 128, "8192 rows at k=64");
    assert!(provider.call_sizes().iter().all(|&s| s == 64));
    assert_eq!(out.len(), 1, "one full 8192-row output batch");
    assert_eq!(out[0].num_rows(), 8192);
    let (vals, outs) = collect_extend_rows(&out);
    assert_eq!(vals, (0..8192).collect::<Vec<_>>(), "order preserved");
    assert_eq!(outs[8191].as_deref(), Some("r8191"));
}
