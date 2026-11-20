//! Calibration-prologue invariants for the join stream: the same contract
//! `certification_prologue.rs` / `certification_extend.rs` pin for
//! selection/extend, but over pairs: the calibration prefix is the first
//! `n·k_max` pairs of the production block-nested-loop enumeration (buffered
//! right relation × the first left rows), the reference arm is n solo-pair
//! calls issued exactly once, a certified pair budget is decomposed back to
//! block geometry keeping the tile width, and the buffered left rows are
//! recycled through the normal pipeline with no pair dropped, including
//! ragged right tails.

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
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
use nousdb_operators::join::physical::SemanticJoinExec;
use nousdb_operators::join::stream::{JoinRightData, OnceAsync, SemanticJoinStream};
use nousdb_operators::metrics::{CalibrationMetrics, SemanticJoinMetrics, WallClockTimer};

use futures::StreamExt;
use serde_json::json;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Mock with controllable per-pair-count fidelity for the boolean
/// `join_match` slot: ground truth is parity equality between the pair's
/// stringified `left_0`/`right_0` values; calls whose pair count is in
/// `flip_at` invert every verdict, `corrupt_at` returns a row-count
/// mismatch, `error_at` a transport error. `transport_first` / `error_first`
/// fail the first that many calls, whatever their size, with a transport
/// failure / an `Err`, then recover.
#[derive(Debug, Default)]
struct JoinFidelityMock {
    calls: Mutex<Vec<usize>>,
    flip_at: HashSet<usize>,
    corrupt_at: HashSet<usize>,
    error_at: HashSet<usize>,
    transport_first: usize,
    error_first: usize,
}

impl JoinFidelityMock {
    fn call_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }

    fn count_of_size(&self, size: usize) -> usize {
        self.call_sizes().iter().filter(|&&s| s == size).count()
    }
}

#[async_trait]
impl ModelProvider for JoinFidelityMock {
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
        // The join stream (and the calibration prologue) always send the
        // synthetic pair batch: stringified join columns `left_0`, `right_0`.
        let lvals = batch
            .column_by_name("left_0")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let rvals = batch
            .column_by_name("right_0")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let outputs = (start_row..start_row + batch_size)
            .map(|i| {
                let l: i64 = lvals.value(i).parse().unwrap();
                let r: i64 = rvals.value(i).parse().unwrap();
                let matched = (l % 2 == r % 2) != flip;
                ModelOutput::Structured(json!({ "join_match": matched }))
            })
            .collect();
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
    }
}

fn left_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]))
}

fn right_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "rval",
        DataType::Int64,
        false,
    )]))
}

fn out_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("lval", DataType::Int64, false),
        Field::new("rval", DataType::Int64, false),
    ]))
}

/// Left rows 0..total split into batches of 8 (buffering pulls whole
/// batches; with T = 24 the calibration buffer is exactly 3 of these).
fn left_stream(total: i64) -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut start = 0i64;
    while start < total {
        let end = (start + 8).min(total);
        let arr: ArrayRef = Arc::new(Int64Array::from((start..end).collect::<Vec<_>>()));
        batches.push(RecordBatch::try_new(left_schema(), vec![arr]).unwrap());
        start = end;
    }
    Box::pin(RecordBatchStreamAdapter::new(
        left_schema(),
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

/// A resolved-right-side future over rows 0..rows (one per stream, as in
/// production where every partition holds a clone of the same OnceFut).
fn right_fut(rows: i64) -> nousdb_operators::join::stream::OnceFut<JoinRightData> {
    let batch = RecordBatch::try_new(
        right_schema(),
        vec![Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>())) as ArrayRef],
    )
    .unwrap();
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let reservation = MemoryConsumer::new("test-join-cert").register(&pool);
    let once: OnceAsync<JoinRightData> = OnceAsync::default();
    once.try_once(move || Ok(async move { Ok(JoinRightData { batch, reservation }) }))
        .unwrap()
}

fn cert_key() -> CertKey {
    CertKey {
        provider: "mock".into(),
        model_id: "mock-model".into(),
        operator_kind: OperatorKind::Join,
        template_hash: 0x101E,
        packing_policy: PackingPolicy::JoinTile { right_block: 4 },
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

const N: usize = 36; // demo probe budget; prefix = N * 16 = 576 pairs

/// Probe-plan builder for the calibration prologue. The template's own
/// children are irrelevant (`probe_exec` replaces the left one with the
/// designed pair rows and the right one with a placeholder), but they must
/// carry the join column names, since the template derives the reference-table
/// headers from their schemas.
fn probe_factory(provider: Arc<JoinFidelityMock>) -> ProbeExecFactory {
    let left = MemorySourceConfig::try_new_exec(&[vec![]], left_schema(), None).unwrap();
    let right = MemorySourceConfig::try_new_exec(&[vec![]], right_schema(), None).unwrap();
    let template = Arc::new(
        SemanticJoinExec::try_new(
            left,
            right,
            vec!["lval".to_string()],
            vec!["rval".to_string()],
            out_schema(),
            provider,
            "mock-model".to_string(),
            ModelCard::new("mock", 8192),
            "left and right values have the same parity".to_string(),
            Some(16),
            Some(4),
            Some(4),
        )
        .unwrap(),
    );
    Arc::new(move |input, k, window, sink| template.probe_exec(input, k, window, sink))
}

/// Certified join stream at plan geometry (4, 4) => k*_alpha = 16, W = 1.
fn make_certified_stream(
    provider: Arc<JoinFidelityMock>,
    left_total: i64,
    right_rows: i64,
    spec: Arc<CertificationSpec>,
    cache: Arc<CertificateCache>,
    once: Arc<OnceAsync<CalibrationOutcome>>,
) -> SemanticJoinStream {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let factory = probe_factory(provider.clone());
    SemanticJoinStream::new(
        left_stream(left_total),
        right_fut(right_rows),
        vec![0],
        vec![0],
        "left and right values have the same parity".to_string(),
        provider,
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        out_schema(),
        SemanticJoinMetrics::new(&metrics_set, 0),
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        Some(4), // plan-time left block
        Some(4), // plan-time right tile
        1,       // request window W = 1: deterministic serial dispatch
        ReasoningMode::Off,
        Some(StreamCertification {
            // Single-partition tests: the pool's quota is the whole design.
            pool: Arc::new(PrefixPool::new(spec.sample_target(), 1)),
            partition: 0,
            spec,
            cache,
            once,
            metrics: CalibrationMetrics::new(&metrics_set, 0),
            factory,
            task_ctx: Arc::new(TaskContext::default()),
            // Matches request_window x 1 partition: probe arms stay serial, so
            // per-rung call counts remain exactly predictable.
            window_budget: 1,
        }),
    )
}

/// The same stream with certification off (the production path).
fn make_plain_stream(
    provider: Arc<JoinFidelityMock>,
    left_total: i64,
    right_rows: i64,
) -> SemanticJoinStream {
    let metrics_set = ExecutionPlanMetricsSet::new();
    SemanticJoinStream::new(
        left_stream(left_total),
        right_fut(right_rows),
        vec![0],
        vec![0],
        "left and right values have the same parity".to_string(),
        provider,
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        out_schema(),
        SemanticJoinMetrics::new(&metrics_set, 0),
        Arc::new(WallClockTimer::new()),
        8192,
        vec!["lval".to_string()],
        vec!["rval".to_string()],
        Some(4),
        Some(4),
        1,
        ReasoningMode::Off,
        None,
    )
}

async fn collect_matches(stream: SemanticJoinStream) -> Vec<(i64, i64)> {
    let batches: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    let mut pairs = Vec::new();
    for b in &batches {
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
        for i in 0..l.len() {
            pairs.push((l.value(i), r.value(i)));
        }
    }
    pairs
}

/// Parity matches of the full cross product, in the production enumeration
/// order at geometry `(l_bs, r_bs)`: left blocks, right tiles per block,
/// left-major within each tile.
fn expected_matches(left_rows: i64, right_rows: i64, l_bs: i64, r_bs: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut block_start = 0;
    while block_start < left_rows {
        let block_end = (block_start + l_bs).min(left_rows);
        let mut tile_start = 0;
        while tile_start < right_rows {
            let tile_end = (tile_start + r_bs).min(right_rows);
            for l in block_start..block_end {
                for r in tile_start..tile_end {
                    if l % 2 == r % 2 {
                        out.push((l, r));
                    }
                }
            }
            tile_start = tile_end;
        }
        block_start = block_end;
    }
    out
}

/// Bad fidelity at 16 pairs, perfect at 8 => descending exit at k̂ = 8,
/// decomposed keep-R to (2, 4); the reference arm runs exactly n solo-pair
/// calls TOTAL; curtailment stops rung 16 after one call; the buffered left
/// rows are recycled through the pipeline at the certified geometry and
/// every parity match of the full 40×24 product comes out, in order.
#[tokio::test(start_paused = true)]
async fn descending_exit_with_reference_arm_once() {
    let provider = Arc::new(JoinFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    // 480 parity matches of 40×24, in the (2, 4) enumeration order: the
    // ladder's second rung is 8 pairs, which decomposes keep-R to (2, 4),
    // keeping the tile width the calibration was drawn under.
    assert_eq!(matches, expected_matches(40, 24, 2, 4));

    // Solo-pair calls: the reference arm (n) plus the k=1 diagnostic rung (n).
    assert_eq!(provider.count_of_size(1), 2 * N);

    // Rung 16: s* = n makes one miss futile => curtailed after 1 call (W=1).
    assert_eq!(provider.count_of_size(16), 1);

    // Rung 8 calibration (N) + all 40 left rows recycled at (2,4): 20 blocks
    // of 2 left rows, each scanning 6 tiles of 4 right rows = 120.
    assert_eq!(provider.count_of_size(8), N + 120);

    // Certificate cached under the JOIN kind / JoinTile packing.
    let cert = cache.get(&cert_key()).expect("certificate must be cached");
    assert_eq!(cert.certified_k, Some(8));
    let gating: Vec<usize> = cert.gating_rungs().map(|r| r.k).collect();
    assert_eq!(gating, vec![16, 8], "descending exit stops at the passer");
    assert!(cert.per_rung[0].curtailed && cert.per_rung[0].k == 16);
    let rung8 = &cert.per_rung[1];
    assert!(!rung8.curtailed && rung8.k == 8);
    assert_eq!(rung8.x, N as u64);
    assert!(rung8.lower_bound >= 0.90);

    // The diagnostic rung runs even on the happy path.
    let diagnostic = cert.self_agreement_rung().expect("k=1 rung must be run");
    assert_eq!(diagnostic.k, 1);
    assert_eq!(diagnostic.x, N as u64, "the mock is solo-deterministic");
}

/// Every rung broken => ladder exhausted, k̂ = 1 decomposes to (1, 1):
/// per-pair calls for everything, each rung curtailed within one call, and
/// the failed ladder is cached.
#[tokio::test(start_paused = true)]
async fn ladder_exhausted_falls_back_to_k1() {
    // Every gating rung of the full ladder [16, 8, 4, 2] must be broken;
    // leaving the smallest intact would certify it.
    let provider = Arc::new(JoinFidelityMock {
        flip_at: HashSet::from([16, 8, 4, 2]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;
    assert_eq!(matches, expected_matches(40, 24, 1, 1));

    // Each gating rung curtails after exactly n - s* + 1 = 1 call at W = 1.
    for k in [16, 8, 4, 2] {
        assert_eq!(provider.count_of_size(k), 1, "rung {k} must curtail at 1");
    }

    // k̂ = 1: n references + n diagnostic + all 40×24 = 960 production pairs
    // as singletons.
    assert_eq!(provider.count_of_size(1), 2 * N + 960);

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

/// A row-count mismatch during calibration is a disagreement: the join
/// stream's emit-no-matches recovery must not masquerade as agreement.
#[tokio::test(start_paused = true)]
async fn row_count_mismatch_scores_as_disagreement() {
    let provider = Arc::new(JoinFidelityMock {
        corrupt_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    assert_eq!(matches, expected_matches(40, 24, 2, 4));
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
/// join stream does propagate transport errors; only calibration calls are
/// shielded.)
#[tokio::test(start_paused = true)]
async fn transport_error_scores_as_disagreement() {
    let provider = Arc::new(JoinFidelityMock {
        error_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    assert_eq!(
        matches,
        expected_matches(40, 24, 2, 4),
        "stream must not error on calibration Err"
    );
    let cert = cache.get(&cert_key()).unwrap();
    assert_eq!(cert.certified_k, Some(8));
    assert_eq!(cert.per_rung[0].errors, 1, "the Err is a missing answer");
}

/// The endpoint never answers the first 40 pair requests (transport
/// failures), then recovers: run 1 falls back to (1, 1) and caches nothing;
/// the next query's instance recalibrates over pairs and certifies 16 = (4, 4).
#[tokio::test(start_paused = true)]
async fn transient_failures_are_not_cached_and_the_next_query_recalibrates() {
    let provider = Arc::new(JoinFidelityMock {
        transport_first: 40,
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(
        collect_matches(stream).await,
        expected_matches(40, 24, 1, 1)
    );
    assert_eq!(
        provider.count_of_size(1),
        2 * N + 960,
        "run 1 runs at k = 1"
    );
    assert!(cache.is_empty(), "request errors must not be cached");

    let run1_calls = provider.call_sizes().len();
    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    assert_eq!(
        collect_matches(stream).await,
        expected_matches(40, 24, 4, 4)
    );
    let cert = cache.get(&cert_key()).expect("run 2 recalibrates");
    assert_eq!(cert.certified_k, Some(16));
    // N probes at 16, then 10 left blocks of 4 × 6 right tiles of 4.
    let run2 = &provider.call_sizes()[run1_calls..];
    assert_eq!(run2.iter().filter(|&&s| s == 1).count(), 2 * N);
    assert_eq!(run2.iter().filter(|&&s| s == 16).count(), N + 60);
}

/// Two partitions share one OnceAsync, so there is exactly one calibration; the loser
/// (whose left side is shorter than the buffering target) recycles its own
/// buffered rows at the certified geometry.
#[tokio::test(start_paused = true)]
async fn two_partitions_one_calibration() {
    let provider = Arc::new(JoinFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());
    let once = Arc::new(OnceAsync::default());
    let spec = demo_spec(42);

    // Partition A (40 left rows) installs and runs the calibration.
    let stream_a = make_certified_stream(
        provider.clone(),
        40,
        24,
        spec.clone(),
        cache.clone(),
        once.clone(),
    );
    let matches_a = collect_matches(stream_a).await;
    assert_eq!(matches_a, expected_matches(40, 24, 2, 4));
    // Reference arm + k=1 diagnostic rung.
    assert_eq!(provider.count_of_size(1), 2 * N);

    // Partition B (16 left rows < T = 24) reaches exhaustion, calls
    // try_once, gets the existing outcome (no second calibration, no
    // reference arm) and processes its whole buffer at (2, 4).
    let stream_b = make_certified_stream(provider.clone(), 16, 24, spec, cache.clone(), once);
    let matches_b = collect_matches(stream_b).await;
    assert_eq!(matches_b, expected_matches(16, 24, 2, 4));

    assert_eq!(provider.count_of_size(1), 2 * N, "no second reference arm");
    assert_eq!(provider.count_of_size(16), 1, "no second rung-16 trial");
    // A's 40 rows + B's 16 all ride the pipeline at (2,4): 28 two-row blocks
    // × 6 tiles = 168, on top of rung 8's N calibration calls.
    assert_eq!(provider.count_of_size(8), N + 168);
}

/// Pair space too small to seat a probe design at any rung gives k̂ = 1 with zero
/// calibration calls (no reference arm) and no cache write; production then
/// runs per-pair. 2×10 = 20 pairs seat 19 probes at k = 2, below the `n_min`
/// of 22.
#[tokio::test(start_paused = true)]
async fn an_unseatable_sample_falls_back_without_calls_or_cache() {
    let provider = Arc::new(JoinFidelityMock::default());
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        2,
        10,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    assert_eq!(matches, expected_matches(2, 10, 1, 1));
    // Only production singletons: 20 calls of size 1, nothing else.
    assert_eq!(provider.call_sizes(), vec![1; 20]);
    assert_eq!(cache.len(), 0, "ShortInput must not write a certificate");
}

/// A pair space short of `n × k_max` drops to a rung it can seat, as when the
/// planner's `k*_alpha = L·R = 1960` pairs meet a sample that never comes close
/// to `n · 1960`.
#[tokio::test(start_paused = true)]
async fn a_sample_short_of_k_max_certifies_a_lower_rung() {
    let provider = Arc::new(JoinFidelityMock::default());
    let cache = Arc::new(CertificateCache::default());

    // 3 × 15 = 45 pairs cannot seat n = 36 probes plus a 16-pair batch's
    // worth of pool (51 pairs), but they seat a lower rung.
    let stream = make_certified_stream(
        provider.clone(),
        3,
        15,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    let cert = cache
        .get(&cert_key())
        .expect("a seatable sample must yield a certificate");
    let k_hat = cert.certified_k.expect("the mock is faithful at every k");
    assert!(
        (2..16).contains(&k_hat),
        "k̂={k_hat} is not a rung below the planned top"
    );
    assert!(
        cert.n + k_hat - 1 <= 45,
        "design overruns the 45-pair sample"
    );
    // Every matching pair is still emitted exactly once, at whatever geometry
    // the certified budget decomposes to.
    let mut got = matches;
    got.sort_unstable();
    let mut want = expected_matches(3, 15, 1, 1);
    want.sort_unstable();
    assert_eq!(got, want, "no pair lost or duplicated");
}

/// No pair may be silently dropped after re-shaping the geometry: the set of
/// emitted pairs equals the parity matches of the full cross product.
#[tokio::test(start_paused = true)]
async fn no_pair_dropped_after_certification() {
    let provider = Arc::new(JoinFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let stream = make_certified_stream(
        provider.clone(),
        40,
        24,
        demo_spec(42),
        Arc::new(CertificateCache::default()),
        Arc::new(OnceAsync::default()),
    );
    let mut matches = collect_matches(stream).await;
    matches.sort_unstable();

    let mut truth: Vec<(i64, i64)> = (0..40)
        .flat_map(|l| (0..24).filter(move |r| l % 2 == r % 2).map(move |r| (l, r)))
        .collect();
    truth.sort_unstable();
    assert_eq!(matches, truth);
}

/// Ragged right tail (26 = 6 full tiles of 4 + one of 2): after certifying
/// k̂ = 8 => (2, 4), every block scan ends in a 2×2 = 4-pair request (a size
/// that never ran as a calibration rung) and no pair is lost at the ragged
/// boundary.
#[tokio::test(start_paused = true)]
async fn ragged_tails_unchanged() {
    let provider = Arc::new(JoinFidelityMock {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    let cache = Arc::new(CertificateCache::default());

    let stream = make_certified_stream(
        provider.clone(),
        40,
        26,
        demo_spec(42),
        cache.clone(),
        Arc::new(OnceAsync::default()),
    );
    let matches = collect_matches(stream).await;

    // 520 parity matches of 40×26, in (2, 4) order with ragged tails.
    assert_eq!(matches, expected_matches(40, 26, 2, 4));

    assert_eq!(provider.count_of_size(1), 2 * N);
    assert_eq!(provider.count_of_size(16), 1);
    // 26 right rows at tile width 4: every 2-row left block scans 6 full tiles
    // of 8 pairs plus one ragged tile of 2 right rows (4 pairs). 20 blocks.
    assert_eq!(provider.count_of_size(8), N + 120);
    assert_eq!(provider.count_of_size(4), 20, "one ragged request per scan");

    assert_eq!(cache.get(&cert_key()).unwrap().certified_k, Some(8));
}

/// Identical seed => identical designation => identical call trace
/// (W = 1 keeps dispatch order deterministic).
#[tokio::test(start_paused = true)]
async fn same_seed_same_call_trace() {
    let mut traces = Vec::new();
    for _ in 0..2 {
        let provider = Arc::new(JoinFidelityMock {
            flip_at: HashSet::from([16]),
            ..Default::default()
        });
        let stream = make_certified_stream(
            provider.clone(),
            40,
            24,
            demo_spec(7),
            Arc::new(CertificateCache::default()),
            Arc::new(OnceAsync::default()),
        );
        let _ = collect_matches(stream).await;
        traces.push(provider.call_sizes());
    }
    assert_eq!(traces[0], traces[1]);
}

/// Probe mode emits exactly one output batch per request, empties included.
///
/// The prologue maps request ordinal to probe by position, so a request that
/// matched nothing must still surface a batch; production's output coalescer
/// would swallow it. Both the all-reject and all-match cases are checked.
#[tokio::test]
async fn probe_mode_emits_one_batch_per_request() {
    use datafusion::arrow::array::UInt32Array;
    use nousdb_operators::certification::{PROBE_ID_COLUMN, RequestOutcomeSink};

    // 12 pair rows at k = 4 => 3 requests. Parity decides the verdict, so
    // even/odd rejects every pair and even/even matches every pair.
    async fn run(right: &[&str]) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("left_0", DataType::Utf8, false),
            Field::new("right_0", DataType::Utf8, false),
            Field::new(PROBE_ID_COLUMN, DataType::UInt32, false),
        ]));
        let lefts: Vec<&str> = (0..12).map(|_| "0").collect();
        let rights: Vec<&str> = (0..12).map(|i| right[i % right.len()]).collect();
        let input = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(lefts)) as ArrayRef,
                Arc::new(StringArray::from(rights)) as ArrayRef,
                Arc::new(UInt32Array::from_iter_values(0..12)) as ArrayRef,
            ],
        )
        .unwrap();

        let provider = Arc::new(JoinFidelityMock::default());
        let source = MemorySourceConfig::try_new_exec(&[vec![input]], schema, None).unwrap();
        let sink = RequestOutcomeSink::new(|_, _| {});
        let factory = probe_factory(provider);
        let exec = factory(source, 4, 1, sink).unwrap();
        exec.execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await
    }

    let rejected = run(&["1"]).await;
    assert_eq!(rejected.len(), 3, "one batch per request, empties included");
    assert!(rejected.iter().all(|b| b.num_rows() == 0));

    let matched = run(&["0"]).await;
    assert_eq!(matched.len(), 3);
    assert!(
        matched.iter().all(|b| b.num_rows() == 4),
        "every pair of every request survived"
    );
    // Row identity survives the join's output assembly, which is what
    // `find_probe_row` relies on to locate the designated probe.
    assert!(
        matched[0].column_by_name(PROBE_ID_COLUMN).is_some(),
        "probe id must ride through to the output"
    );
}

/// The probe-mode output path must not leak into production: an uncertified
/// join still coalesces its matches into full output batches.
#[tokio::test(start_paused = true)]
async fn production_output_stays_coalesced() {
    let provider = Arc::new(JoinFidelityMock::default());
    let stream = make_plain_stream(provider.clone(), 40, 24);
    let batches: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;

    // 10 left blocks x 6 right tiles = 60 requests, each matching 8 of its 16
    // pairs, all of which fit one 8192-row output batch.
    assert_eq!(provider.call_sizes().len(), 60);
    assert_eq!(
        batches.len(),
        1,
        "output must not degrade to one batch/request"
    );
    assert_eq!(
        batches[0].num_rows(),
        expected_matches(40, 24, 4, 4).len(),
        "every match still comes out"
    );
}
