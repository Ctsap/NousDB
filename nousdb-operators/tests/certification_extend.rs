//! Calibration-prologue invariants for the extend stream: the same contract
//! `certification_prologue.rs` pins for selection, driven through the shared
//! `certification::driver` machinery with extend's canonical exact-match
//! comparator: descending ladder exit, the reference arm issued exactly once
//! (n singleton calls total), failures scored as disagreements (never the
//! NULL-fill recovery), once-per-instance calibration across partitions,
//! short-input fallback, and seed determinism.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
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
    CalibrationOutcome, PrefixPool, ProbeExecFactory, StreamCertification,
};
use nousdb_operators::extend::physical::ExtendExec;
use nousdb_operators::extend::stream::ExtendStream;
use nousdb_operators::join::stream::OnceAsync;
use nousdb_operators::metrics::{CalibrationMetrics, SemanticOperatorMetrics, WallClockTimer};

use futures::StreamExt;
use serde_json::json;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

fn parity_label(v: i64) -> &'static str {
    if v % 2 == 0 { "even" } else { "odd" }
}

/// Mock with controllable per-batch-size fidelity for a derived Utf8 label:
/// the ground truth is the parity of `val`; calls whose `batch_size` is in
/// `flip_at` invert every label (batched output then disagrees with the
/// singleton reference), `corrupt_at` returns a row-count mismatch,
/// `error_at` returns a transport error (`Err` from predict).
/// `transport_first` / `error_first` fail the first that many calls, whatever
/// their size, with a transport failure / an `Err`, then recover.
#[derive(Debug, Default)]
struct ExtendFidelityMock {
    calls: Mutex<Vec<usize>>,
    flip_at: HashSet<usize>,
    corrupt_at: HashSet<usize>,
    error_at: HashSet<usize>,
    transport_first: usize,
    error_first: usize,
}

impl ExtendFidelityMock {
    fn call_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }

    fn count_of_size(&self, size: usize) -> usize {
        self.call_sizes().iter().filter(|&&s| s == size).count()
    }
}

#[async_trait]
impl ModelProvider for ExtendFidelityMock {
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
        let ordinal = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(batch_size);
            calls.len() - 1
        };
        if ordinal < self.transport_first {
            // The request never came back.
            return Ok(PredictResult::transport_failure(1));
        }
        if ordinal < self.error_first {
            // `Err` from predict.
            return Err(DataFusionError::Execution("mock endpoint down".to_string()));
        }

        if self.error_at.contains(&batch_size) {
            return Err(DataFusionError::Execution(
                "mock transport failure".to_string(),
            ));
        }
        if self.corrupt_at.contains(&batch_size) {
            return Ok(PredictResult::new(
                Vec::new(),
                InferenceMetrics::zero(),
                true,
            ));
        }

        let flip = self.flip_at.contains(&batch_size);
        let vals = batch
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let outputs = (start_row..start_row + batch_size)
            .map(|i| {
                let even = vals.value(i) % 2 == 0;
                let label = if even != flip { "even" } else { "odd" };
                ModelOutput::Structured(json!({ "out": label }))
            })
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

fn derived_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("out", DataType::Utf8, true)]))
}

fn full_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("val", DataType::Int64, false),
        Field::new("out", DataType::Utf8, true),
    ]))
}

fn make_batch(vals: std::ops::Range<i64>) -> RecordBatch {
    let arr: ArrayRef = Arc::new(Int64Array::from(vals.collect::<Vec<_>>()));
    RecordBatch::try_new(input_schema(), vec![arr]).unwrap()
}

/// Rows 0..n split into batches of 100 (mirrors a realistic upstream).
fn input_stream(total: i64) -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut start = 0i64;
    while start < total {
        let end = (start + 100).min(total);
        batches.push(make_batch(start..end));
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
        operator_kind: OperatorKind::Extend,
        template_hash: 0xE71E,
        packing_policy: PackingPolicy::Flat,
    }
}

/// Demo operating point: eta = delta = 0.10, full ladder [16, 8, 4, 2] =>
/// delta' = 0.025, n = 36, s* = 36 (zero-miss floor => curtail on first miss).
fn demo_spec(seed: u64) -> Arc<CertificationSpec> {
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
        seed,
        cache_key: cert_key(),
        delta_share: cfg.delta,
        tolerated: cfg.tolerated_disagreements,
        max_probes: cfg.max_probes,
    })
}

const N: usize = 36; // demo probe budget; prefix = N * 16 = 576 rows

/// The prologue dispatches by executing probe copies of the operator, so the
/// test supplies the same factory the exec builds in `execute()`. The template
/// exec's own input is irrelevant: `probe_exec` replaces it with the designed
/// probe rows.
fn probe_factory(provider: Arc<ExtendFidelityMock>) -> ProbeExecFactory {
    let placeholder = MemorySourceConfig::try_new_exec(&[vec![]], input_schema(), None).unwrap();
    let template = Arc::new(ExtendExec::new(
        placeholder,
        "label the parity of val".to_string(),
        vec!["val".to_string()],
        derived_schema(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        full_schema(),
        Some(16),
    ));
    Arc::new(move |input, k, window, sink| template.probe_exec(input, k, window, sink))
}

fn make_certified_stream(
    provider: Arc<ExtendFidelityMock>,
    input: SendableRecordBatchStream,
    spec: Arc<CertificationSpec>,
    cache: Arc<CertificateCache>,
    once: Arc<OnceAsync<CalibrationOutcome>>,
) -> ExtendStream {
    let metrics_set = ExecutionPlanMetricsSet::new();
    ExtendStream::new(
        input,
        "label the parity of val".to_string(),
        vec!["val".to_string()],
        derived_schema(),
        "mock-model".to_string(),
        provider.clone(),
        ModelCard::new("mock", 8192),
        full_schema(),
        SemanticOperatorMetrics::new(&metrics_set, 0),
        Arc::new(WallClockTimer::new()),
        Some(16), // k*_alpha from the (simulated) sizing pass
        1,        // request window W = 1: deterministic serial dispatch
        8192,
        ReasoningMode::Off,
        Some(StreamCertification {
            // Single-partition tests: the pool's quota is the whole design.
            pool: Arc::new(PrefixPool::new(spec.sample_target(), 1)),
            partition: 0,
            spec,
            cache,
            once,
            metrics: CalibrationMetrics::new(&metrics_set, 0),
            factory: probe_factory(provider),
            task_ctx: Arc::new(TaskContext::default()),
            // Matches request_window x 1 partition: probe arms stay serial,
            // so per-rung call counts remain exactly predictable.
            window_budget: 1,
        }),
    )
}

async fn collect_pairs(stream: ExtendStream) -> Vec<(i64, Option<String>)> {
    let batches: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    let mut pairs = Vec::new();
    for b in &batches {
        let vals = b
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let labels = b
            .column_by_name("out")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..vals.len() {
            let label = labels.is_valid(i).then(|| labels.value(i).to_string());
            pairs.push((vals.value(i), label));
        }
    }
    pairs
}

/// Every input row extended with its correct parity label, in input order.
fn expected(upto: i64) -> Vec<(i64, Option<String>)> {
    (0..upto)
        .map(|v| (v, Some(parity_label(v).to_string())))
        .collect()
}

/// Bad fidelity at k=16, perfect at k=8 gives a descending exit at 8; the
/// reference arm runs exactly n singleton calls in total (shared across rungs);
/// curtailment stops rung 16 after one call; the prefix is recycled through
/// the pipeline at k̂ with every derived label correct and cardinality
/// preserved (extend emits all rows, unlike selection).
#[tokio::test(start_paused = true)]
async fn descending_exit_with_reference_arm_once() {
    let provider = Arc::new(ExtendFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;

    // Output: all 600 rows, in order, each with the correct label.
    assert_eq!(pairs, expected(600));

    // Singleton calls: the reference arm (n, shared across every rung and
    // never recomputed) plus the k=1 diagnostic rung (n). k̂ > 1 means no
    // steady-state singletons on top.
    assert_eq!(provider.count_of_size(1), 2 * N);

    // Rung 16: s* = n makes one miss futile => curtailed after 1 call (W=1).
    assert_eq!(provider.count_of_size(16), 1);

    // Rung 8 is the next candidate down; every power of two below k*_alpha is
    // measured. Calibration (N) + all 600 rows recycled at 8 (75 slices).
    assert_eq!(provider.count_of_size(8), N + 75);

    // Certificate cached under the EXTEND operator kind.
    let cert = cache.get(&cert_key()).expect("certificate must be cached");
    assert_eq!(cert.certified_k, Some(8));
    let gating: Vec<usize> = cert.gating_rungs().map(|r| r.k).collect();
    assert_eq!(gating, vec![16, 8], "descending exit stops at the passer");
    assert!(cert.per_rung[0].curtailed && cert.per_rung[0].k == 16);
    let second = &cert.per_rung[1];
    assert!(!second.curtailed && second.k == 8);
    assert_eq!(second.x, N as u64);
    assert!(second.lower_bound >= 0.90);

    // The diagnostic rung runs even on the happy path.
    let diagnostic = cert.self_agreement_rung().expect("k=1 rung must be run");
    assert_eq!(diagnostic.k, 1);
    assert_eq!(diagnostic.x, N as u64, "the mock is solo-deterministic");
    assert_eq!(cert.per_rung.len(), 3, "two gating rungs + the diagnostic");
}

/// Every rung has broken fidelity => ladder exhausted, k̂ = 1, each rung
/// curtailed within `n - s* + 1` calls, and the failed ladder is cached.
#[tokio::test(start_paused = true)]
async fn ladder_exhausted_falls_back_to_k1() {
    // Every gating rung of the full ladder [16, 8, 4, 2] must be broken for
    // the ladder to be exhausted; leaving the smallest intact certifies it.
    let provider = Arc::new(ExtendFidelityMock {
        flip_at: HashSet::from([16, 8, 4, 2]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;
    assert_eq!(pairs, expected(600));

    // Each gating rung curtails after exactly n - s* + 1 = 1 call at W = 1.
    for k in [16, 8, 4, 2] {
        assert_eq!(provider.count_of_size(k), 1, "rung {k} must curtail at 1");
    }

    // k̂ = 1: n references + n diagnostic + all 600 rows as singletons.
    assert_eq!(provider.count_of_size(1), 2 * N + 600);

    let cert = cache.get(&cert_key()).expect("failed ladder is cached too");
    assert_eq!(cert.certified_k, None);
    assert!(cert.gating_rungs().all(|r| r.curtailed));
    assert_eq!(
        cert.gating_rungs().count(),
        4,
        "every candidate was measured"
    );
    // The model's answers exhausted it, not request errors: cacheable.
    assert!(cert.per_rung.iter().all(|r| r.errors == 0));

    // The mock reproduces its own solo answers perfectly, so the exhausted
    // ladder here means packing hurts, not that the model is inconsistent.
    let diagnostic = cert.self_agreement_rung().expect("k=1 rung must be run");
    assert!(!diagnostic.gating, "k=1 must never gate");
    assert_eq!(diagnostic.x, N as u64);
}

/// A row-count mismatch during calibration is a disagreement: the extend
/// stream's NULL-fill recovery must not masquerade as agreement, or the
/// corrupted rung 16 would certify.
#[tokio::test(start_paused = true)]
async fn row_count_mismatch_scores_as_disagreement() {
    let provider = Arc::new(ExtendFidelityMock {
        corrupt_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;

    assert_eq!(pairs, expected(600));
    let cert = cache.get(&cert_key()).unwrap();
    assert_eq!(
        cert.certified_k,
        Some(8),
        "the corrupted rung must NOT certify; the next one down does"
    );
    assert!(cert.per_rung[0].curtailed);
}

/// Transport-error variant: `Err` from predict during a calibration rung is
/// scored as a disagreement and the stream never fails. (In steady state the
/// extend stream does propagate transport errors; only calibration calls
/// are shielded.)
#[tokio::test(start_paused = true)]
async fn transport_error_scores_as_disagreement() {
    let provider = Arc::new(ExtendFidelityMock {
        error_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;

    assert_eq!(
        pairs,
        expected(600),
        "stream must not error on calibration Err"
    );
    let cert = cache.get(&cert_key()).unwrap();
    assert_eq!(cert.certified_k, Some(8));
    assert_eq!(cert.per_rung[0].errors, 1, "the Err is a missing answer");
}

/// The endpoint fails the first 40 requests (`Err` from predict), then
/// recovers: run 1 falls back to k = 1 and caches
/// nothing; the next query's instance recalibrates and certifies.
#[tokio::test(start_paused = true)]
async fn transient_failures_are_not_cached_and_the_next_query_recalibrates() {
    let provider = Arc::new(ExtendFidelityMock {
        error_first: 40,
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(collect_pairs(stream).await, expected(600));
    assert_eq!(
        provider.count_of_size(1),
        2 * N + 600,
        "run 1 runs at k = 1"
    );
    assert!(cache.is_empty(), "request errors must not be cached");

    let run1_calls = provider.call_sizes().len();
    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(collect_pairs(stream).await, expected(600));
    let cert = cache.get(&cert_key()).expect("run 2 recalibrates");
    assert_eq!(cert.certified_k, Some(16));
    let run2 = &provider.call_sizes()[run1_calls..];
    assert_eq!(run2.iter().filter(|&&s| s == 1).count(), 2 * N);
    assert_eq!(run2.iter().filter(|&&s| s == 16).count(), N + 37);
}

/// Two partitions share one OnceAsync => exactly one calibration; the loser
/// keeps its buffered rows and processes them at the certified k̂.
#[tokio::test(start_paused = true)]
async fn two_partitions_one_calibration() {
    let provider = Arc::new(ExtendFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());
    let spec = demo_spec(42);

    // Partition A (600 rows) installs and runs the calibration.
    let stream_a = make_certified_stream(
        provider.clone(),
        input_stream(600),
        spec.clone(),
        cache.clone(),
        once.clone(),
    );
    let pairs_a = collect_pairs(stream_a).await;
    assert_eq!(pairs_a, expected(600));
    // Reference arm + k=1 diagnostic rung.
    assert_eq!(provider.count_of_size(1), 2 * N);

    // Partition B (200 rows < prefix) reaches exhaustion, calls try_once,
    // gets the existing outcome (no second calibration, no reference arm)
    // and processes its whole buffer at k̂ = 8.
    let stream_b = make_certified_stream(
        provider.clone(),
        input_stream(200),
        spec,
        cache.clone(),
        once,
    );
    let pairs_b = collect_pairs(stream_b).await;
    assert_eq!(pairs_b, expected(200));

    assert_eq!(provider.count_of_size(1), 2 * N, "no second reference arm");
    assert_eq!(provider.count_of_size(16), 1, "no second rung-16 trial");
    // A's 600 rows + B's 200 all ride the normal pipeline at k̂ = 8, on top of
    // the N calibration calls for that rung.
    assert_eq!(provider.count_of_size(8), N + 75 + 25);
}

/// Input too small to seat a probe design at any rung gives k̂ = 1 with zero
/// model calls during calibration (no reference arm) and no cache write: a
/// sample that could not certify under any outcome is not evidence about
/// fidelity. 20 rows seat 10 probes at k = 2, below the `n_min` of 22.
#[tokio::test(start_paused = true)]
async fn an_unseatable_sample_falls_back_without_calls_or_cache() {
    let provider = Arc::new(ExtendFidelityMock::default());
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(20),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;

    assert_eq!(pairs, expected(20));
    // Only steady-state singletons: 20 calls of size 1, nothing else.
    assert_eq!(provider.call_sizes(), vec![1; 20]);
    assert_eq!(cache.len(), 0, "ShortInput must not write a certificate");
}

/// A sample short of `n × k_max` drops to a rung it can seat and probes there,
/// rather than spending a full pass over the input to conclude `k = 1` without
/// a model call (a large `k*_alpha` over a few hundred rows).
#[tokio::test(start_paused = true)]
async fn a_sample_short_of_k_max_certifies_a_lower_rung() {
    let provider = Arc::new(ExtendFidelityMock::default());
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(45), // cannot seat n = 36 probes plus a k=16 pool
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let pairs = collect_pairs(stream).await;
    assert_eq!(pairs, expected(45), "every row is still derived");

    let cert = cache
        .get(&cert_key())
        .expect("a seatable sample must yield a certificate");
    let k_hat = cert.certified_k.expect("the mock is faithful at every k");
    assert!(
        (2..16).contains(&k_hat),
        "k̂={k_hat} is not a rung below the planned top"
    );
    assert!(provider.count_of_size(k_hat) > 0, "certified rung unprobed");
    assert!(
        cert.n + k_hat - 1 <= 45,
        "design overruns the 45-row sample"
    );
}

/// Identical seed => identical designation => identical call trace
/// (W = 1 keeps dispatch order deterministic).
#[tokio::test(start_paused = true)]
async fn same_seed_same_call_trace() {
    let mut traces = Vec::new();
    for _ in 0..2 {
        let provider = Arc::new(ExtendFidelityMock {
            flip_at: HashSet::from([16]),
            ..Default::default()
        });
        let stream = make_certified_stream(
            provider.clone(),
            input_stream(600),
            demo_spec(7),
            Arc::new(CertificateCache::default()),
            Arc::new(OnceAsync::default()),
        );
        let _ = collect_pairs(stream).await;
        traces.push(provider.call_sizes());
    }
    assert_eq!(traces[0], traces[1]);
}
