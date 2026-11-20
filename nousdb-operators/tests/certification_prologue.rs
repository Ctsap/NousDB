//! Calibration-prologue invariants for the selection stream, end to end
//! against a mock provider: descending ladder exit, futility curtailment, the
//! reference arm issued exactly once (n singleton calls total), failures
//! scored as disagreements (never the pass-through recovery), once-per-
//! instance calibration across partitions, short-input and empty-partition
//! fallbacks, and seed determinism.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array};
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
use nousdb_operators::join::stream::OnceAsync;
use nousdb_operators::metrics::{CalibrationMetrics, SemanticOperatorMetrics, WallClockTimer};
use nousdb_operators::selection::physical::SemanticSelectionExec;
use nousdb_operators::selection::stream::SemanticSelectionStream;

use futures::StreamExt;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Mock with controllable per-batch-size fidelity: the ground truth is
/// `val % 2 == 0`; calls whose `batch_size` is listed in `flip_at` invert
/// every output (batched output then disagrees with the singleton
/// reference), `corrupt_at` returns a row-count mismatch, `error_at`
/// returns a transport error (`Err` from predict).
/// `transport_first` / `error_first` fail the first that many calls, whatever
/// their size, with a transport failure / an `Err`, then recover.
#[derive(Debug, Default)]
struct FidelityMockProvider {
    calls: Mutex<Vec<usize>>,
    flip_at: HashSet<usize>,
    corrupt_at: HashSet<usize>,
    error_at: HashSet<usize>,
    transport_first: usize,
    error_first: usize,
}

impl FidelityMockProvider {
    fn call_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }

    fn count_of_size(&self, size: usize) -> usize {
        self.call_sizes().iter().filter(|&&s| s == size).count()
    }
}

#[async_trait]
impl ModelProvider for FidelityMockProvider {
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
                let truth = vals.value(i) % 2 == 0;
                ModelOutput::Boolean(truth != flip)
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

fn make_batch(vals: std::ops::Range<i64>) -> RecordBatch {
    let arr: ArrayRef = Arc::new(Int64Array::from(vals.collect::<Vec<_>>()));
    RecordBatch::try_new(input_schema(), vec![arr]).unwrap()
}

/// Rows 0..n split into batches of 100 (mirrors a realistic upstream).
/// A partition's input: `count` consecutive values starting at `start`.
///
/// Distinct ranges per partition so a test can assert that every row came back
/// exactly once, from the partition that read it.
fn input_stream_range(start: usize, count: usize) -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut at = start as i64;
    let end_all = (start + count) as i64;
    while at < end_all {
        let end = (at + 100).min(end_all);
        batches.push(make_batch(at..end));
        at = end;
    }
    Box::pin(RecordBatchStreamAdapter::new(
        input_schema(),
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

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
        operator_kind: OperatorKind::Selection,
        template_hash: 0xF1DE,
        packing_policy: PackingPolicy::Flat,
    }
}

/// Demo operating point: eta = delta = 0.10, full ladder [16, 8, 4, 2] =>
/// delta' = 0.025, n = 36, s* = 36 (zero-miss floor => curtail on first miss).
fn demo_spec(seed: u64) -> Arc<CertificationSpec> {
    spec_with_tolerance(seed, None)
}

/// A spec whose planned budget sits above `n_min`, so a shortfall can still be
/// certified at a reduced `n`. With no tolerance the plan *is* the floor, and
/// any shortfall is arithmetically uncertifiable.
fn spec_with_tolerance(seed: u64, tolerated: Option<usize>) -> Arc<CertificationSpec> {
    let cfg = QualityCertConfig {
        eta: 0.10,
        delta: 0.10,
        tolerated_disagreements: tolerated,
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
fn probe_factory(provider: Arc<FidelityMockProvider>) -> ProbeExecFactory {
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

fn make_certified_stream(
    provider: Arc<FidelityMockProvider>,
    input: SendableRecordBatchStream,
    spec: Arc<CertificationSpec>,
    cache: Arc<CertificateCache>,
    once: Arc<OnceAsync<CalibrationOutcome>>,
) -> SemanticSelectionStream {
    let pool = Arc::new(PrefixPool::new(spec.sample_target(), 1));
    make_pooled_stream(provider, input, spec, cache, once, pool, 0)
}

/// Build one partition's stream against a **shared** pool, so a test can model
/// several partitions of the same operator instance contributing to one sample.
#[allow(clippy::too_many_arguments)]
fn make_pooled_stream(
    provider: Arc<FidelityMockProvider>,
    input: SendableRecordBatchStream,
    spec: Arc<CertificationSpec>,
    cache: Arc<CertificateCache>,
    once: Arc<OnceAsync<CalibrationOutcome>>,
    pool: Arc<PrefixPool>,
    partition: usize,
) -> SemanticSelectionStream {
    let metrics_set = ExecutionPlanMetricsSet::new();
    SemanticSelectionStream::new(
        input,
        "keep even values".to_string(),
        vec!["val".to_string()],
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        input_schema(),
        SemanticOperatorMetrics::new(&metrics_set, 0),
        Arc::new(WallClockTimer::new()),
        Some(16), // k*_alpha from the (simulated) sizing pass
        1,        // request window W = 1: deterministic serial dispatch
        8192,
        ReasoningMode::Off,
        Some(StreamCertification {
            pool,
            partition,
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

async fn collect_vals(stream: SemanticSelectionStream) -> Vec<i64> {
    let batches: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    let mut vals = Vec::new();
    for b in &batches {
        let col = b
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        for i in 0..col.len() {
            vals.push(col.value(i));
        }
    }
    vals
}

fn evens(upto: i64) -> Vec<i64> {
    (0..upto).filter(|v| v % 2 == 0).collect()
}

/// (a)+(b): bad fidelity at k=16, perfect at k=8 gives a descending exit at 8;
/// the reference arm runs exactly n singleton calls in total (shared across
/// rungs); curtailment stops rung 16 after one call;
/// prefix is recycled through the pipeline at k̂ with correct output.
/// The k=1 diagnostic rung runs too, even though a rung passed.
#[tokio::test(start_paused = true)]
async fn descending_exit_with_reference_arm_once() {
    let provider = Arc::new(FidelityMockProvider {
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
    let vals = collect_vals(stream).await;

    // Output: every even value, in input order (prefix first, then rest).
    assert_eq!(vals, evens(600));

    // Singleton calls: the reference arm (n, shared across every rung and
    // never recomputed) plus the k=1 diagnostic rung (n). k̂ = 8 means no
    // steady-state singletons on top.
    assert_eq!(provider.count_of_size(1), 2 * N);

    // Rung 16: s* = n makes one miss futile => curtailed after 1 call (W=1).
    assert_eq!(provider.count_of_size(16), 1);

    // Rung 8 is the next candidate down; every power of two below k*_alpha is
    // measured. Calibration (N) + all 600 rows recycled at 8 (75).
    assert_eq!(provider.count_of_size(8), N + 75);

    // Certificate cached: rung 16 curtailed, rung 8 passed, certified_k = 8.
    let cert = cache.get(&cert_key()).expect("certificate must be cached");
    assert_eq!(cert.certified_k, Some(8));
    let gating: Vec<usize> = cert.gating_rungs().map(|r| r.k).collect();
    assert_eq!(gating, vec![16, 8], "descending exit stops at the passer");
    assert!(cert.per_rung[0].curtailed && cert.per_rung[0].k == 16);
    let second = &cert.per_rung[1];
    assert!(!second.curtailed && second.k == 8);
    assert_eq!(second.x, N as u64);
    assert!(second.lower_bound >= 0.90);

    // The diagnostic rung runs even on the happy path: a k̂ of 8 under a
    // self-agreement of 0.86 is a different claim than one under 0.99.
    let diagnostic = cert.self_agreement_rung().expect("k=1 rung must be run");
    assert_eq!(diagnostic.k, 1);
    assert!(!diagnostic.curtailed);
    assert_eq!(diagnostic.x, N as u64, "the mock is solo-deterministic");
    assert_eq!(cert.per_rung.len(), 3, "two gating rungs + the diagnostic");
}

/// (c)+(d): every rung has broken fidelity, so the ladder is exhausted, k̂ = 1,
/// each rung curtailed within `n - s* + 1` calls, and the failed ladder is
/// cached as information.
#[tokio::test(start_paused = true)]
async fn ladder_exhausted_falls_back_to_k1() {
    // Every gating rung of the full ladder [16, 8, 4, 2] must be broken;
    // leaving the smallest intact would certify it.
    let provider = Arc::new(FidelityMockProvider {
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
    let vals = collect_vals(stream).await;
    assert_eq!(vals, evens(600));

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
    assert!(cert.is_cacheable());

    // The diagnostic rung separates the two causes: the mock reproduces its
    // own solo answers perfectly, so an exhausted ladder here means packing
    // hurts, not that the model is inconsistent.
    let diagnostic = cert.self_agreement_rung().expect("k=1 rung must be run");
    assert!(!diagnostic.gating, "k=1 must never gate");
    assert_eq!(diagnostic.x, N as u64);
    assert!(diagnostic.lower_bound >= 0.90);
}

/// (e)/(j): a row-count mismatch during calibration is a disagreement: the
/// selection stream's pass-through recovery must not masquerade as 100%
/// agreement, or the corrupted rung 16 would certify.
#[tokio::test(start_paused = true)]
async fn row_count_mismatch_scores_as_disagreement() {
    let provider = Arc::new(FidelityMockProvider {
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
    let vals = collect_vals(stream).await;

    assert_eq!(vals, evens(600));
    let cert = cache.get(&cert_key()).unwrap();
    assert_eq!(
        cert.certified_k,
        Some(8),
        "the corrupted rung must NOT certify; the next one down does"
    );
    assert!(cert.per_rung[0].curtailed);
    // An unusable answer is an answer: evidence about the model, not an error.
    assert_eq!(cert.per_rung[0].errors, 0);
}

/// (e) transport-error variant: `Err` from predict is scored by the
/// calibration as a disagreement and the stream never fails.
#[tokio::test(start_paused = true)]
async fn transport_error_scores_as_disagreement() {
    let provider = Arc::new(FidelityMockProvider {
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
    let vals = collect_vals(stream).await;

    assert_eq!(vals, evens(600), "stream must not error on calibration Err");
    let cert = cache.get(&cert_key()).unwrap();
    assert_eq!(cert.certified_k, Some(8));
    // The `Err` is a missing answer, tallied apart from answered disagreements.
    // A certified size is cached anyway: rung 8 passed on its own answers, so
    // the error above it only made k̂ conservative.
    assert_eq!(cert.per_rung[0].errors, 1);
    let failed_by_errors: Vec<usize> = cert.rungs_failed_by_errors().map(|r| r.k).collect();
    assert_eq!(failed_by_errors, vec![16]);
}

/// Unusable answers at every packed size (truncated, undecodable, wrong
/// length) are evidence about the model, so the exhausted ladder is cached;
/// only missing answers are kept out of the cache.
#[tokio::test(start_paused = true)]
async fn unusable_answers_exhaust_the_ladder_and_are_cached() {
    let provider = Arc::new(FidelityMockProvider {
        corrupt_at: HashSet::from([16, 8, 4, 2]),
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
    assert_eq!(collect_vals(stream).await, evens(600));

    let cert = cache
        .get(&cert_key())
        .expect("an answered failure is cached");
    assert_eq!(cert.certified_k, None);
    assert!(cert.per_rung.iter().all(|r| r.errors == 0));
}

/// The endpoint fails the first 40 requests, then recovers. Every reference
/// probe goes unanswered, so every rung is curtailed on a request error. A
/// missing answer is a disagreement, so this run falls back to k = 1, but the
/// exhausted ladder is not cached, and the next query's operator instance
/// (fresh `OnceAsync`, same session cache) recalibrates and certifies.
async fn transient_failures_are_not_cached(provider: Arc<FidelityMockProvider>) {
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(collect_vals(stream).await, evens(600));
    for k in [16, 8, 4, 2] {
        assert_eq!(provider.count_of_size(k), 1, "rung {k} must curtail at 1");
    }
    // Run 1 still runs at k̂ = 1: n references + n diagnostic + 600 solo rows.
    assert_eq!(provider.count_of_size(1), 2 * N + 600);
    assert!(
        cache.is_empty(),
        "a ladder exhausted by request errors must not be cached"
    );

    let run1_calls = provider.call_sizes().len();
    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(collect_vals(stream).await, evens(600));

    let cert = cache
        .get(&cert_key())
        .expect("the recovered endpoint calibrates again and certifies");
    assert_eq!(cert.certified_k, Some(16));
    assert!(cert.per_rung.iter().all(|r| r.errors == 0));
    // Run 2: a full calibration (n references, N probes at 16, n diagnostic)
    // and the 600 rows at 16: 37 full requests plus the one tail of 8.
    let run2 = &provider.call_sizes()[run1_calls..];
    let count = |k: usize| run2.iter().filter(|&&s| s == k).count();
    assert_eq!(count(1), 2 * N);
    assert_eq!(count(16), N + 37);
    assert_eq!(count(8), 1);
}

#[tokio::test(start_paused = true)]
async fn transient_transport_failures_are_not_cached_and_the_next_query_recalibrates() {
    transient_failures_are_not_cached(Arc::new(FidelityMockProvider {
        transport_first: 40,
        ..Default::default()
    }))
    .await;
}

#[tokio::test(start_paused = true)]
async fn transient_provider_errors_are_not_cached_and_the_next_query_recalibrates() {
    transient_failures_are_not_cached(Arc::new(FidelityMockProvider {
        error_first: 40,
        ..Default::default()
    }))
    .await;
}

/// (f): two partitions share one OnceAsync => exactly one calibration; the
/// loser keeps its buffered rows and processes them at the certified k̂.
#[tokio::test(start_paused = true)]
async fn two_partitions_one_calibration() {
    let provider = Arc::new(FidelityMockProvider {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());
    let spec = demo_spec(42);

    // Partition A (600 rows) is driven to completion first: it installs and
    // runs the calibration.
    let stream_a = make_certified_stream(
        provider.clone(),
        input_stream(600),
        spec.clone(),
        cache.clone(),
        once.clone(),
    );
    let vals_a = collect_vals(stream_a).await;
    assert_eq!(vals_a, evens(600));
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
    let vals_b = collect_vals(stream_b).await;
    assert_eq!(vals_b, evens(200));

    assert_eq!(provider.count_of_size(1), 2 * N, "no second reference arm");
    assert_eq!(provider.count_of_size(16), 1, "no second rung-16 trial");
    // A's 600 rows + B's 200 all ride the normal pipeline at k̂ = 8, on top of
    // the N calibration calls for that rung.
    assert_eq!(provider.count_of_size(8), N + 75 + 25);
}

/// (g): an empty partition never touches the shared OnceAsync (no k=1
/// poisoning, no deadlock); a later partition still calibrates normally.
#[tokio::test(start_paused = true)]
async fn empty_partition_does_not_poison() {
    let provider = Arc::new(FidelityMockProvider {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());
    let spec = demo_spec(42);

    // Empty partition first.
    let empty = make_certified_stream(
        provider.clone(),
        input_stream(0),
        spec.clone(),
        cache.clone(),
        once.clone(),
    );
    let vals = collect_vals(empty).await;
    assert!(vals.is_empty());
    assert!(
        provider.call_sizes().is_empty(),
        "no calls from empty partition"
    );
    assert!(cache.get(&cert_key()).is_none());

    // Real partition afterwards: calibration proceeds untouched.
    let stream = make_certified_stream(
        provider.clone(),
        input_stream(600),
        spec,
        cache.clone(),
        once,
    );
    let vals = collect_vals(stream).await;
    assert_eq!(vals, evens(600));
    assert_eq!(cache.get(&cert_key()).unwrap().certified_k, Some(8));
}

/// (h): input too small to seat a probe design at any rung gives k̂ = 1 with
/// zero model calls during calibration (no reference arm) and no cache
/// write: a sample that could not certify under any outcome is not evidence
/// about fidelity.
///
/// 20 rows seat 10 probes at the smallest rung (k = 2), below the `n_min` of
/// 22 at which even a perfect run clears `1 - eta`.
#[tokio::test(start_paused = true)]
async fn an_unseatable_sample_falls_back_without_calls_or_cache() {
    let provider = Arc::new(FidelityMockProvider::default());
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        input_stream(20),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let vals = collect_vals(stream).await;

    assert_eq!(vals, evens(20));
    // Only steady-state singletons: 20 calls of size 1, nothing else.
    assert_eq!(provider.call_sizes(), vec![1; 20]);
    assert_eq!(cache.len(), 0, "ShortInput must not write a certificate");
}

/// A sample short of `n × k_max` does not forfeit certification: the design
/// drops to a rung the rows can seat and probes there, rather than spending a
/// full pass over the input to conclude `k = 1` without a model call (for
/// example `k*_alpha = 86` planned over 1000 rows).
#[tokio::test(start_paused = true)]
async fn a_sample_short_of_k_max_certifies_a_lower_rung() {
    let provider = Arc::new(FidelityMockProvider::default());
    let cache = Arc::new(CertificateCache::default());

    // 45 rows cannot seat n = 36 probes plus a 16-row batch's worth of pool
    // (51 rows), but they comfortably seat a lower rung.
    let stream = make_certified_stream(
        provider.clone(),
        input_stream(45),
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let vals = collect_vals(stream).await;
    assert_eq!(vals, evens(45), "every row is still processed");

    let cert = cache
        .get(&cert_key())
        .expect("a seatable sample must yield a certificate");
    let k_hat = cert.certified_k.expect("the mock is faithful at every k");
    assert!(
        (2..16).contains(&k_hat),
        "k̂={k_hat} is not a rung below the planned top"
    );
    assert!(
        provider.count_of_size(k_hat) > 0,
        "the certified rung was never probed"
    );
    assert_eq!(
        provider.count_of_size(1),
        2 * cert.n,
        "reference arm plus the k=1 diagnostic, both at the realized budget"
    );
    assert!(
        cert.n + k_hat - 1 <= 45,
        "design of n={} at k={k_hat} overruns the 45-row sample",
        cert.n
    );
}

/// (i): identical seed => identical designation => identical call trace
/// (W = 1 keeps dispatch order deterministic).
#[tokio::test(start_paused = true)]
async fn same_seed_same_call_trace() {
    let mut traces = Vec::new();
    for _ in 0..2 {
        let provider = Arc::new(FidelityMockProvider {
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
        let _ = collect_vals(stream).await;
        traces.push(provider.call_sizes());
    }
    assert_eq!(traces[0], traces[1]);
}

/// No single partition holds enough rows for the design, and calibration runs
/// anyway: the prologue pools its `n × k_max` prefix across partitions rather
/// than buffering it from one partition's stream, so a high partition count
/// does not push the runner onto the `ShortInput` path.
///
/// Here 15 partitions hold 5 rows each (75 total). The pool wants more than any
/// one of them holds, so every partition contributes everything it has and
/// seals; pooled, they cover the design.
#[tokio::test(start_paused = true)]
async fn many_small_partitions_still_calibrate() {
    const PARTITIONS: usize = 15;
    const ROWS_EACH: usize = 5;

    let provider = Arc::new(FidelityMockProvider {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());
    let spec = demo_spec(42);
    let pool = Arc::new(PrefixPool::new(spec.sample_target(), PARTITIONS));

    // Quota per partition: ceil(sample_target / 15). Each partition holds
    // fewer rows than that, so every one of them contributes everything it has
    // and seals; the pooled sample is the whole 75 rows.
    assert!(
        pool.quota_rows() > ROWS_EACH,
        "the test is meaningless unless no partition can fill the design alone"
    );

    let streams: Vec<_> = (0..PARTITIONS)
        .map(|p| {
            make_pooled_stream(
                provider.clone(),
                input_stream_range(p * ROWS_EACH, ROWS_EACH),
                spec.clone(),
                cache.clone(),
                once.clone(),
                pool.clone(),
                p,
            )
        })
        .collect();

    // Drive all partitions concurrently, as `CoalescePartitionsExec` does.
    let outputs = futures::future::join_all(streams.into_iter().map(collect_vals)).await;

    // Every row is accounted for exactly once, in its own partition.
    let mut all: Vec<i64> = outputs.into_iter().flatten().collect();
    all.sort_unstable();
    assert_eq!(
        all,
        evens((PARTITIONS * ROWS_EACH) as i64),
        "no rows lost or duplicated"
    );

    let cert = cache
        .get(&cert_key())
        .expect("a certificate must be written");

    // Exactly one calibration: one reference arm, one trial per rung.
    assert_eq!(
        provider.count_of_size(1),
        2 * cert.n,
        "one reference arm and one diagnostic rung across all partitions"
    );
    assert_eq!(provider.count_of_size(16), 1, "one rung-16 trial");

    assert_eq!(cert.certified_k, Some(8));
    assert!(
        cert.n >= N,
        "the pooled sample covers the planned budget (n={}, planned {N})",
        cert.n
    );
    assert_eq!(cert.n_planned, N);
}

/// A sample that cannot afford the requested tolerance at `k_max` buys it at a
/// lower rung instead of keeping the top and running a zero-tolerance ladder.
///
/// At the bare floor a rung dies on its first disagreement, so every rung below
/// it dies too and the whole descent concludes `k = 1` on evidence that says
/// nothing about fidelity.
#[tokio::test(start_paused = true)]
async fn a_short_sample_keeps_the_tolerance_and_lowers_the_top() {
    let provider = Arc::new(FidelityMockProvider {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());

    // Tolerating 2 disagreements plans an n well above the n_min of 36, and
    // 640 rows seat only 40 probes at k = 16: enough to clear the floor, not
    // enough to keep the tolerance.
    let spec = spec_with_tolerance(42, Some(2));
    let planned = spec.n;
    assert!(planned > 640 / 16, "the sample must be short of the plan");
    let stream = make_certified_stream(
        provider.clone(),
        input_stream(640),
        spec,
        cache.clone(),
        once,
    );
    let vals = collect_vals(stream).await;
    assert_eq!(vals, evens(640), "every row is still processed");

    let cert = cache
        .get(&cert_key())
        .expect("a short-but-usable sample must still yield a certificate");
    let k_hat = cert.certified_k.expect("k = 8 is faithful in this mock");
    assert!(k_hat < 16, "the top must have dropped, got k̂={k_hat}");
    assert!(
        cert.n * k_hat <= 640,
        "design of n={} at k={k_hat} overruns the sample",
        cert.n
    );
    assert_eq!(cert.n_planned, planned);
    assert_eq!(
        provider.count_of_size(1),
        2 * cert.n,
        "reference arm and diagnostic rung are both sized to the realized \
         budget, not the planned one"
    );
}
