//! Probe-exec dispatch: the calibration prologue dispatches through the
//! operator's own engine path, so it inherits the request window and its
//! requests are indistinguishable from production requests.
//!
//! Two invariants are pinned here:
//! 1. Concurrency is the operator's. A probe arm keeps up to
//!    `min(window_budget, n)` requests in flight. The prologue is a
//!    whole-operator barrier (every other partition waits for the shared
//!    outcome), so `window_budget` is `request_window × partitions`, the
//!    concurrency production would have spread across all of them.
//! 2. Probe requests are byte-identical to production requests. The
//!    row-identity column the prologue appends is not in the renderer's
//!    `input_columns`, so it never reaches the prompt and cannot perturb the
//!    shared prefix.

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

use nousdb_core::certification::{
    CertKey, CertificateCache, CertificationSpec, OperatorKind, PackingPolicy, QualityCertConfig,
    budget_for, build_ladder,
};
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::schemas::ModelOutput;

use nousdb_operators::certification::{
    PROBE_ID_COLUMN, PrefixPool, ProbeExecFactory, StreamCertification,
};
use nousdb_operators::join::stream::OnceAsync;
use nousdb_operators::metrics::{CalibrationMetrics, SemanticOperatorMetrics, WallClockTimer};
use nousdb_operators::selection::physical::SemanticSelectionExec;
use nousdb_operators::selection::stream::SemanticSelectionStream;

use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Records peak concurrent in-flight requests and every rendered prompt.
/// Each call holds its slot for a tick, so overlapping requests are visible.
#[derive(Debug, Default)]
struct ConcurrencyProbeMock {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    prompts: Mutex<Vec<String>>,
}

impl ConcurrencyProbeMock {
    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait]
impl ModelProvider for ConcurrencyProbeMock {
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
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);

        // Capture what the model would see: the same call real providers make.
        if let Ok(rendered) = prompt.render_batch(renderer, batch, start_row, batch_size) {
            self.prompts.lock().unwrap().push(rendered);
        }

        // Hold the slot so genuinely concurrent requests overlap.
        tokio::time::sleep(Duration::from_millis(50)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        let vals = batch
            .column_by_name("val")
            .unwrap()
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

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
}

fn input_stream(total: i64) -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut start = 0i64;
    while start < total {
        let end = (start + 100).min(total);
        let arr: ArrayRef = Arc::new(Int64Array::from((start..end).collect::<Vec<_>>()));
        batches.push(RecordBatch::try_new(input_schema(), vec![arr]).unwrap());
        start = end;
    }
    Box::pin(RecordBatchStreamAdapter::new(
        input_schema(),
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

fn cert_key() -> CertKey {
    CertKey {
        provider: "mock".into(),
        model_id: "mock-model".into(),
        operator_kind: OperatorKind::Selection,
        template_hash: 0xC0FFEE,
        packing_policy: PackingPolicy::Flat,
    }
}

fn demo_spec() -> Arc<CertificationSpec> {
    let cfg = QualityCertConfig {
        eta: 0.10,
        delta: 0.10,
        // No probe growth: `n` stays at the requested budget so these tests
        // account for every model call exactly.
        max_probes: 0,
        ..Default::default()
    };
    let ladder = build_ladder(16);
    let (delta_prime, n, s_star) = budget_for(&cfg, ladder.len(), 1).unwrap();
    Arc::new(CertificationSpec {
        ladder,
        eta: cfg.eta,
        delta_prime,
        n,
        s_star,
        seed: 42,
        cache_key: cert_key(),
        delta_share: cfg.delta,
        tolerated: cfg.tolerated_disagreements,
        max_probes: cfg.max_probes,
    })
}

const N: usize = 36;

fn probe_factory(provider: Arc<ConcurrencyProbeMock>) -> ProbeExecFactory {
    let placeholder = MemorySourceConfig::try_new_exec(&[vec![]], input_schema(), None).unwrap();
    let template = Arc::new(SemanticSelectionExec::new(
        placeholder,
        "keep even values".to_string(),
        vec!["val".to_string()],
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        input_schema(),
        Some(16),
    ));
    Arc::new(move |input, k, window, sink| template.probe_exec(input, k, window, sink))
}

fn certified_stream(
    provider: Arc<ConcurrencyProbeMock>,
    cache: Arc<CertificateCache>,
    window_budget: usize,
) -> SemanticSelectionStream {
    let metrics_set = ExecutionPlanMetricsSet::new();
    SemanticSelectionStream::new(
        input_stream(576),
        "keep even values".to_string(),
        vec!["val".to_string()],
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        input_schema(),
        SemanticOperatorMetrics::new(&metrics_set, 0),
        Arc::new(WallClockTimer::new()),
        Some(16),
        window_budget,
        8192,
        ReasoningMode::Off,
        Some(StreamCertification {
            // Single-partition test: the pool's quota covers the whole run.
            pool: Arc::new(PrefixPool::new(demo_spec().sample_target(), 1)),
            partition: 0,
            spec: demo_spec(),
            cache,
            once: Arc::new(OnceAsync::default()),
            metrics: CalibrationMetrics::new(&metrics_set, 0),
            factory: probe_factory(provider),
            task_ctx: Arc::new(TaskContext::default()),
            window_budget,
        }),
    )
}

/// A budget of 1 (a card with `max_concurrent_requests` unset, one partition)
/// still overlaps the reference arm with the top rung, which are independent
/// until scoring, but never exceeds two in flight.
#[tokio::test]
async fn serial_budget_only_overlaps_the_two_concurrent_arms() {
    let provider = Arc::new(ConcurrencyProbeMock::default());
    let cache = Arc::new(CertificateCache::default());
    let stream = certified_stream(provider.clone(), cache.clone(), 1);
    let _: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(
        cache.get(&cert_key()).unwrap().certified_k,
        Some(16),
        "perfect mock fidelity must certify the top rung"
    );
    assert!(
        provider.peak() <= 2,
        "budget=1 means one request per arm; saw {} in flight",
        provider.peak()
    );
}

/// With a large concurrency budget the prologue drives many requests at once.
/// The operator's `FuturesOrdered` does this; the prologue owns no dispatcher.
#[tokio::test]
async fn probe_arms_inherit_the_request_window() {
    let provider = Arc::new(ConcurrencyProbeMock::default());
    let cache = Arc::new(CertificateCache::default());
    let stream = certified_stream(provider.clone(), cache.clone(), 16);
    let _: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(cache.get(&cert_key()).unwrap().certified_k, Some(16));
    assert!(
        provider.peak() > 2,
        "budget=16 must put more than the two arm heads in flight; saw {}",
        provider.peak()
    );
}

/// The window is capped at `n`: beyond one request per probe there is nothing
/// left to dispatch, so an oversized budget cannot oversubscribe the endpoint.
#[tokio::test]
async fn window_is_capped_at_the_probe_budget() {
    let provider = Arc::new(ConcurrencyProbeMock::default());
    let cache = Arc::new(CertificateCache::default());
    let stream = certified_stream(provider.clone(), cache.clone(), 4096);
    let _: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(cache.get(&cert_key()).unwrap().certified_k, Some(16));
    // Two arms of at most n requests each run concurrently.
    assert!(
        provider.peak() <= 2 * N,
        "window must clamp to n per arm; saw {}",
        provider.peak()
    );
}

/// The row-identity column must not reach the model: renderers project by
/// their explicit `input_columns`, so probe prompts stay byte-identical to
/// production prompts and the shared prefix is unperturbed.
#[tokio::test]
async fn probe_id_never_reaches_the_prompt() {
    let provider = Arc::new(ConcurrencyProbeMock::default());
    let cache = Arc::new(CertificateCache::default());
    let stream = certified_stream(provider.clone(), cache.clone(), 8);
    let _: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    let prompts = provider.prompts();
    assert!(!prompts.is_empty(), "the mock must have seen requests");
    assert!(
        prompts.iter().all(|p| !p.contains(PROBE_ID_COLUMN)),
        "the probe id column leaked into a rendered prompt"
    );
}
