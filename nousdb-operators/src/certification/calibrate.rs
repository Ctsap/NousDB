//! The calibration runner, driving the engine's own operators.
//!
//! Each arm of the calibration is an ordinary execution: a designed batch of
//! probe rows wrapped in an in-memory source, a probe copy of the operator
//! fixed at `k` rows per request, and `execute()`. Coalescing, the in-flight
//! request window, admission permits, recovery and metrics all come from the
//! operator itself.
//!
//! Statistical design:
//! - **Designation before observation**: probe items, slots and every filler
//!   are drawn (seeded, deterministic) before the first model call; see
//!   [`crate::certification::probe::designate`].
//! - **One scored tuple per calibration batch**: the designated probe is
//!   compared, and the other `k − 1` items are fillers drawn independently per
//!   probe from the shared pool. `Z_j` is a function of that probe's own draw,
//!   so the `n` indicators are independent and each rung's agreement count is
//!   `Binomial(n, φ(k))`; batches sharing a filler row is not a dependence.
//! - **Reference arm computed once**: the `n` solo (`k = 1`) outputs are
//!   obtained once and shared across every rung (the union bound is
//!   dependence-agnostic).
//! - **Fixed `n`, futility-only curtailment**: a gating rung aborts only when
//!   a pass is arithmetically impossible. There is no early success exit,
//!   which would void Clopper–Pearson coverage.
//! - **`k = 1` is measured but never gates**: a final solo-vs-solo rung
//!   reports the model's self-consistency, the ceiling on every `φ(k)`. It is
//!   excluded from `L`, so it spends no risk budget.
//! - **Failures are disagreements**: transport errors, parse failures and
//!   wrong-length outputs all score as disagreement. The streams' recovery
//!   paths are not inherited: a `RequestOutcomeSink` marks those requests so
//!   selection's pass-through and extend's NULL-fill cannot masquerade as
//!   agreement.
//! - **Missing answers do not decide the cache**: the sink also says whether a
//!   failed request got an answer at all. Probes that disagreed only for want
//!   of one are tallied per rung (`RungResult::errors`), and a ladder
//!   exhausted by those errors rather than by the model's answers is not cached
//!   (`Certificate::is_cacheable`): this run falls back to `k = 1` and the next
//!   query recalibrates.
//!
//! Packing is handled by the engine's coalescer, which cuts the `n·k`-item
//! probe input on exact `k` boundaries so every probe sits on its designated
//! slot at every rung. The probe exec's request window is sized for the phase.
//!
//! The sample is pooled across every partition of the operator (see
//! [`crate::certification::pool`]). Prefix-exchangeability still applies: an
//! input ordered by something correlated with difficulty biases the estimate,
//! because each partition contributes its own leading rows.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use futures::StreamExt;

use nousdb_core::certification::{
    Certificate, CertificateCache, CertificationSpec, RungResult, cp_lower_bound,
};

use crate::certification::probe::{
    ItemSpace, ProbeExecFactory, ProbeInputBuilder, ProbeObservation, ProbeScorer, RequestOutcome,
    RequestOutcomeSink, Rung, designate, find_probe_row,
};
use crate::certification::types::{CalibrationOutcome, Disposition};
use crate::metrics::{CalibrationMetrics, WallClockTimer};

/// Everything the calibration future owns (it must be `'static`).
pub struct ProbeCalibrationContext {
    /// Builds a probe copy of the operator over a designed input.
    pub factory: ProbeExecFactory,
    /// Task context used to execute probe plans.
    pub task_ctx: Arc<TaskContext>,
    /// `request_window × partition_count`: the concurrency production would
    /// spread across all partitions, which this phase may use because every
    /// other partition waits on the outcome.
    pub window_budget: usize,
    /// Operator-specific agreement on output rows.
    pub scorer: ProbeScorer,
    /// Calibration metrics sink.
    pub metrics: CalibrationMetrics,
    /// Timer for wall-clock time spent in inference.
    pub inference_timer: Arc<WallClockTimer>,
}

/// Incremental agreement tally for one rung.
///
/// Observations arrive from two arms that may be running concurrently, so a
/// probe is scored only once both its reference and its batched observation
/// have landed. That keeps futility curtailment live even while the reference
/// arm is still in flight.
struct Scoreboard {
    reference: Vec<Option<ProbeObservation>>,
    rung: Vec<Option<ProbeObservation>>,
    scored: Vec<bool>,
    agreements: u64,
    disagreements: u64,
    /// Disagreements caused by a missing answer alone (see [`RungResult::errors`]).
    errors: u64,
    allowed_misses: u64,
    scorer: ProbeScorer,
}

impl Scoreboard {
    fn new(n: usize, allowed_misses: u64, scorer: ProbeScorer) -> Self {
        Self {
            reference: (0..n).map(|_| None).collect(),
            rung: (0..n).map(|_| None).collect(),
            scored: vec![false; n],
            agreements: 0,
            disagreements: 0,
            errors: 0,
            allowed_misses,
            scorer,
        }
    }

    fn set_reference(&mut self, j: usize, obs: ProbeObservation) {
        if j < self.reference.len() {
            self.reference[j] = Some(obs);
            self.score_ready();
        }
    }

    fn set_rung(&mut self, j: usize, obs: ProbeObservation) {
        if j < self.rung.len() {
            self.rung[j] = Some(obs);
            self.score_ready();
        }
    }

    /// Score every probe whose two observations are now both present.
    fn score_ready(&mut self) {
        for j in 0..self.scored.len() {
            if self.scored[j] {
                continue;
            }
            let (Some(reference), Some(rung)) = (&self.reference[j], &self.rung[j]) else {
                continue;
            };
            // A failed request on either arm is a disagreement, never an
            // agreement; the recovery paths would otherwise hide it.
            let agree = !reference.failed() && !rung.failed() && (self.scorer)(rung, reference);
            self.scored[j] = true;
            if agree {
                self.agreements += 1;
            } else {
                self.disagreements += 1;
                // Blamed on the endpoint only when an answer is missing and no
                // answer that did arrive was unusable: an unusable answer on
                // either arm makes this a disagreement whatever the missing one
                // would have said.
                let outcomes = [reference.outcome, rung.outcome];
                if outcomes.contains(&RequestOutcome::NoAnswer)
                    && !outcomes.contains(&RequestOutcome::Unusable)
                {
                    self.errors += 1;
                }
            }
        }
    }

    /// Even a perfect remainder can no longer reach `s*`.
    fn futile(&self) -> bool {
        self.disagreements > self.allowed_misses
    }

    /// Carry the reference arm into the next rung, clearing rung state.
    fn next_rung(&mut self) {
        self.rung.iter_mut().for_each(|slot| *slot = None);
        self.scored.iter_mut().for_each(|s| *s = false);
        self.agreements = 0;
        self.disagreements = 0;
        self.errors = 0;
    }

    /// Probes whose reference never arrived count against every rung, so a
    /// poisoned reference can only forfeit certificates, never mint one. They
    /// are missing answers, so they never decide the cache.
    fn seal_missing_references(&mut self) {
        for slot in self.reference.iter_mut() {
            if slot.is_none() {
                *slot = Some(ProbeObservation {
                    batch: RecordBatch::new_empty(Arc::new(
                        datafusion::arrow::datatypes::Schema::empty(),
                    )),
                    row: None,
                    outcome: RequestOutcome::NoAnswer,
                });
            }
        }
        self.score_ready();
    }
}

/// Per-request outcomes written by the stream's sink, read back by ordinal
/// once the matching batch arrives. The sink reports before the batch is
/// emitted, so the initial `Answered` is never read for a failed request.
#[derive(Clone)]
struct FailureLog(Arc<Mutex<Vec<RequestOutcome>>>);

impl FailureLog {
    fn new(n: usize) -> Self {
        Self(Arc::new(Mutex::new(vec![RequestOutcome::Answered; n])))
    }

    fn sink(&self) -> RequestOutcomeSink {
        let outcomes = self.0.clone();
        RequestOutcomeSink::new(move |ordinal, outcome| {
            let mut outcomes = outcomes.lock().expect("failure log poisoned");
            if ordinal < outcomes.len() {
                outcomes[ordinal] = outcome;
            }
        })
    }

    /// An out-of-range ordinal is a request the design never issued: no answer.
    fn get(&self, ordinal: usize) -> RequestOutcome {
        self.0
            .lock()
            .expect("failure log poisoned")
            .get(ordinal)
            .copied()
            .unwrap_or(RequestOutcome::NoAnswer)
    }
}

/// Build and start one calibration arm: `indices.len()` designed items, `k`
/// items per request, `window` requests in flight.
fn start_arm(
    ctx: &ProbeCalibrationContext,
    build_input: &ProbeInputBuilder,
    indices: &[usize],
    k: usize,
    window: usize,
    failures: &FailureLog,
) -> Result<SendableRecordBatchStream> {
    let input = build_input(indices)?;
    let schema = input.schema();
    let source: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[vec![input]], schema, None)?;
    let exec = (ctx.factory)(source, k, window, failures.sink())?;
    exec.execute(0, ctx.task_ctx.clone())
}

/// Consume an arm's output, turning each request's batch into the observation
/// for the probe it carried. `on_observation` returns `false` to stop early
/// (futility), which drops the stream and cancels its in-flight requests.
async fn drive_arm(
    mut stream: SendableRecordBatchStream,
    probe_ids: &[u32],
    failures: &FailureLog,
    mut on_observation: impl FnMut(usize, ProbeObservation) -> bool,
) -> usize {
    let mut ordinal = 0usize;
    while let Some(batch) = stream.next().await {
        if ordinal >= probe_ids.len() {
            break;
        }
        // A probe stream turns request errors into reported failures, so an
        // `Err` reaching here is a stream-level fault: score it and stop
        // consuming. Unobserved probes stay unscored, which the caller reads
        // as a short arm (conservative, never a free pass).
        let (observation, stream_faulted) = match batch {
            Ok(batch) => (
                ProbeObservation {
                    row: find_probe_row(&batch, probe_ids[ordinal]),
                    batch,
                    outcome: failures.get(ordinal),
                },
                false,
            ),
            Err(e) => {
                log::warn!("[certify] probe arm faulted at request {ordinal}: {e}");
                (
                    ProbeObservation {
                        batch: RecordBatch::new_empty(stream.schema()),
                        row: None,
                        outcome: RequestOutcome::NoAnswer,
                    },
                    true,
                )
            }
        };
        let keep_going = on_observation(ordinal, observation);
        ordinal += 1;
        if stream_faulted || !keep_going {
            break;
        }
    }
    ordinal
}

/// Score a gating rung once its arm has drained: Clopper–Pearson bound, log
/// line, and the [`RungResult`] the certificate records. Returns whether it
/// passed.
///
/// An arm that returned fewer than `n` results is curtailed: it hit futility
/// or faulted, and the unobserved probes are unscored rather than agreeing.
fn score_rung(
    spec: &CertificationSpec,
    board: &Scoreboard,
    k: usize,
    completed: usize,
    per_rung: &mut Vec<RungResult>,
) -> bool {
    let n = spec.n;
    let curtailed = completed < n || board.futile();
    let x = board.agreements;
    let errors = board.errors;
    let lower_bound = cp_lower_bound(x, n as u64, spec.delta_prime);
    let passed = !curtailed && lower_bound >= 1.0 - spec.eta;
    log::info!(
        "[certify] rung k={k}: X={x}/{completed} (n={n}){}, phi_lb={lower_bound:.6} => {}",
        no_answer_note(errors),
        if passed {
            "PASS"
        } else if curtailed {
            "curtailed"
        } else {
            "fail"
        }
    );
    per_rung.push(RungResult {
        k,
        gating: true,
        x,
        n: n as u64,
        scored: completed as u64,
        errors,
        lower_bound,
        curtailed,
    });
    passed
}

/// `", 3 without an answer"` when some disagreements were request errors.
fn no_answer_note(errors: u64) -> String {
    if errors == 0 {
        String::new()
    } else {
        format!(", {errors} without an answer")
    }
}

/// Run the full calibration and certify the largest safe batch size.
///
/// Writes the certificate to the cache on a pass or an exhausted ladder, but
/// never on the short-input path (a sample below `n_min` is not evidence about
/// fidelity) and never for a ladder exhausted by request errors rather than
/// the model's answers ([`Certificate::is_cacheable`]), which would pin the
/// session to `k = 1` after the endpoint recovers.
pub async fn run_probe_calibration(
    spec: Arc<CertificationSpec>,
    space: ItemSpace,
    build_input: ProbeInputBuilder,
    ctx: ProbeCalibrationContext,
    cache: Arc<CertificateCache>,
) -> Result<CalibrationOutcome> {
    // Tag every request this prologue sends, for the per-request trace
    // (`NOUSDB_TRACE_DIR`); restored on every return path.
    let _trace_phase = nousdb_core::providers::trace::PhaseGuard::enter("calibration");
    let cal_start = Instant::now();
    // Items, not rows: for a join this counts pairs.
    let items = space.len();
    ctx.metrics.calibration_rows.add(items);

    let Some(&planned_k_max) = spec.ladder.first() else {
        return Err(DataFusionError::Internal(
            "calibration: certification spec has an empty ladder".into(),
        ));
    };
    let n_planned = spec.n;
    debug_assert!(
        spec.s_star <= n_planned,
        "budget_for guarantees s_star <= n"
    );

    // Commit the ladder and the probe budget to the sample that arrived.
    //
    // The plan sizes the design from a cardinality estimate (or nothing, when
    // the input has no statistics). A design of `n` probes at ladder top `k`
    // needs `n + k - 1` items, so a small table, a selective upstream filter or
    // a straggling partition can leave the sample short.
    //
    // Retargeting depends only on how much data exists, never on an
    // observation: the draw still happens before the first model call, so each
    // rung's agreement count stays Binomial(n, phi(k)) and Clopper–Pearson
    // coverage is untouched. A lower top certifies a smaller `k`, and a smaller
    // `n` yields a stricter `s_star`: weaker certificates, not invalid ones.
    let Some(retargeted) = spec.retarget(items) else {
        // No cache write on this path: a sample too small to certify is not
        // evidence about fidelity.
        log::warn!(
            "[certify] pooled sample has {items} items, too few to seat a probe design at any \
             rung of k_max={planned_k_max} (planned n={n_planned}, eta={}, delta_share={}); \
             falling back to k=1 without probing",
            spec.eta,
            spec.delta_share,
        );
        log::debug!("[certify] {:?}", Disposition::ShortInput { rows: items });
        ctx.metrics.certified_batch_size.set(1);
        ctx.metrics
            .calibration_time
            .add_duration(cal_start.elapsed());
        return Ok(CalibrationOutcome { k_hat: 1 });
    };

    let spec = if retargeted.ladder == spec.ladder && retargeted.n == n_planned {
        spec
    } else {
        log::info!(
            "[certify] design retargeted to the realized sample of {items} items: ladder \
             {:?} -> {:?}, n={} (planned {n_planned}), s*={}, delta'={:.6}",
            spec.ladder,
            retargeted.ladder,
            retargeted.n,
            retargeted.s_star,
            retargeted.delta_prime,
        );
        Arc::new(retargeted)
    };
    let n = spec.n;

    // Designation: every random draw (probe items, slots, fillers) happens here,
    // before the first model call. The design owns the ladder from here on:
    // under a tiled geometry a rung's realized size can sit below its nominal
    // ladder entry, and the realized size is what executes and gets certified.
    let design = designate(&spec, &space);
    let Some(top) = (design.rung_count() > 0).then(|| design.rung(0)) else {
        return Err(DataFusionError::Internal(
            "calibration: designation produced no rungs".into(),
        ));
    };

    // Concurrency for the phase, capped at the number of requests an arm has:
    // beyond `n` there is nothing left to dispatch.
    let window = ctx.window_budget.clamp(1, n);

    ctx.inference_timer.start();
    let allowed_misses = (n - spec.s_star) as u64;
    let scoreboard = Arc::new(Mutex::new(Scoreboard::new(
        n,
        allowed_misses,
        ctx.scorer.clone(),
    )));

    let rung_sizes: Vec<usize> = (0..design.rung_count())
        .map(|i| design.rung_size(i))
        .collect();
    log::info!(
        "[certify] reference arm: {n} singleton calls, concurrent with rung k={} \
         (rungs {rung_sizes:?} from ladder {:?}, delta'={:.6}, s*={}, window={window})",
        top.k,
        spec.ladder,
        spec.delta_prime,
        spec.s_star
    );

    let mut per_rung: Vec<RungResult> = Vec::with_capacity(spec.ladder.len());
    let mut certified: Option<usize> = None;

    // Reference arm and rung 0 run concurrently: the batched calls do not
    // depend on the references, only the scoring does, so the scoreboard
    // resolves each probe as its pair completes.
    let reference_failures = FailureLog::new(n);
    let reference_stream = start_arm(
        &ctx,
        &build_input,
        &design.reference_indices(),
        1,
        window,
        &reference_failures,
    )?;

    let reference_task = {
        let scoreboard = scoreboard.clone();
        let ids: Vec<u32> = (0..n as u32).collect();
        let failures = reference_failures.clone();
        async move {
            drive_arm(reference_stream, &ids, &failures, |j, obs| {
                scoreboard
                    .lock()
                    .expect("scoreboard poisoned")
                    .set_reference(j, obs);
                true
            })
            .await
        }
    };

    let rung0_failures = FailureLog::new(n);
    let rung0_stream = start_arm(
        &ctx,
        &build_input,
        &top.indices,
        top.k,
        window,
        &rung0_failures,
    )?;

    let rung0_task = {
        let scoreboard = scoreboard.clone();
        let ids = top.probe_ids.clone();
        let failures = rung0_failures.clone();
        async move {
            drive_arm(rung0_stream, &ids, &failures, |j, obs| {
                let mut board = scoreboard.lock().expect("scoreboard poisoned");
                board.set_rung(j, obs);
                // Futility only: keep going while a pass is still reachable.
                !board.futile()
            })
            .await
        }
    };

    let (reference_calls, rung0_calls) = futures::future::join(reference_task, rung0_task).await;
    ctx.metrics.reference_calls.add(reference_calls);
    ctx.metrics.rung_calls.add(rung0_calls);
    ctx.metrics.rungs_tried.add(1);

    {
        // Probes whose reference never landed (short arm, plan fault) are
        // sealed as failures so they count against every rung.
        let mut board = scoreboard.lock().expect("scoreboard poisoned");
        board.seal_missing_references();
        if reference_calls < n {
            log::warn!(
                "[certify] reference arm returned {reference_calls}/{n} results; \
                 the rest count as disagreements on every rung"
            );
        }
        if score_rung(&spec, &board, top.k, rung0_calls, &mut per_rung) {
            certified = Some(top.k);
        }
    }

    // Remaining rungs, descending, against the completed reference arm.
    if certified.is_none() {
        for rung_idx in 1..design.rung_count() {
            scoreboard.lock().expect("scoreboard poisoned").next_rung();
            ctx.metrics.rungs_tried.add(1);

            let Rung {
                k,
                indices,
                probe_ids: ids,
            } = design.rung(rung_idx);
            let failures = FailureLog::new(n);
            let stream = start_arm(&ctx, &build_input, &indices, k, window, &failures)?;

            let completed = {
                let scoreboard = scoreboard.clone();
                drive_arm(stream, &ids, &failures, |j, obs| {
                    let mut board = scoreboard.lock().expect("scoreboard poisoned");
                    board.set_rung(j, obs);
                    !board.futile()
                })
                .await
            };
            ctx.metrics.rung_calls.add(completed);

            let board = scoreboard.lock().expect("scoreboard poisoned");
            if score_rung(&spec, &board, k, completed, &mut per_rung) {
                certified = Some(k);
                break;
            }
        }
    }

    // The k = 1 diagnostic rung: a second pass of the same solo calls over the
    // same rows, scored against the reference arm with the same comparator, so
    // it measures the model's self-consistency.
    //
    // It never gates: an exhausted ladder already falls back to k = 1, so its
    // result cannot change the executed batch size. It is excluded from `L`, so
    // it does not take a share of `delta'` from the rungs that decide `k̂`.
    //
    // It disambiguates an exhausted ladder: a model that does not reproduce its
    // own solo answers caps every phi(k), which batching cannot be blamed for.
    // It runs unconditionally, including after a rung passed, so a reader can
    // tell a `k̂` certified under low self-agreement from one under high.
    {
        scoreboard.lock().expect("scoreboard poisoned").next_rung();
        let failures = FailureLog::new(n);
        let stream = start_arm(
            &ctx,
            &build_input,
            &design.reference_indices(),
            1,
            window,
            &failures,
        )?;
        let ids: Vec<u32> = (0..n as u32).collect();
        let completed = {
            let scoreboard = scoreboard.clone();
            // No futility: a rung that cannot fail has nothing to curtail for.
            drive_arm(stream, &ids, &failures, |j, obs| {
                scoreboard
                    .lock()
                    .expect("scoreboard poisoned")
                    .set_rung(j, obs);
                true
            })
            .await
        };
        ctx.metrics.reference_calls.add(completed);

        let board = scoreboard.lock().expect("scoreboard poisoned");
        let x = board.agreements;
        let errors = board.errors;
        let lower_bound = cp_lower_bound(x, n as u64, spec.delta_prime);
        log::info!(
            "[certify] rung k=1 (diagnostic, solo vs solo): X={x}/{completed} (n={n}){}, \
             phi_lb={lower_bound:.6}; no rung can exceed this, so eta must be at least \
             {:.3} for ANY k to certify",
            no_answer_note(errors),
            1.0 - lower_bound
        );
        per_rung.push(RungResult {
            k: 1,
            gating: false,
            x,
            n: n as u64,
            scored: completed as u64,
            errors,
            lower_bound,
            curtailed: false,
        });
    }

    ctx.inference_timer.stop();

    // Final size: min(k*_alpha, certified). The top rung is at most k*_alpha
    // (retargeting only lowers it), so the min is a formality.
    let k_hat = certified.unwrap_or(1).min(planned_k_max);
    let certificate = Certificate {
        certified_k: certified,
        per_rung,
        eta: spec.eta,
        delta_prime: spec.delta_prime,
        n,
        n_planned,
    };

    // The statistics above decided this run, with missing answers counted as
    // disagreements. Only evidence about the model's answers may outlive it.
    let failed_by_errors: Vec<usize> = certificate.rungs_failed_by_errors().map(|r| r.k).collect();
    let cached = certificate.is_cacheable();
    if !cached {
        let (errors, scored) = certificate
            .gating_rungs()
            .fold((0, 0), |(e, s), r| (e + r.errors, s + r.scored));
        log::warn!(
            "[certify] calibration NOT cached: request errors (transport failures or provider \
             errors), not the model's answers, exhausted the ladder — rung(s) k={failed_by_errors:?} \
             would have passed had their missing answers agreed ({errors} of the {scored} probes \
             scored across the ladder got no answer). This run uses k_hat={k_hat}; the next query \
             of this operator in the session recalibrates."
        );
    } else if !failed_by_errors.is_empty() {
        log::warn!(
            "[certify] rung(s) k={failed_by_errors:?} failed on request errors (transport \
             failures or provider errors), not on the model's answers; the cached k_hat={k_hat} \
             is conservative for this session"
        );
    }
    if cached {
        cache.insert(spec.cache_key.clone(), certificate);
    }
    ctx.metrics.certified_batch_size.set(k_hat);
    ctx.metrics
        .calibration_time
        .add_duration(cal_start.elapsed());

    let disposition = match certified {
        Some(k) => Disposition::Certified(k),
        None => Disposition::LadderExhausted,
    };
    log::info!(
        "[certify] calibration complete: k_hat={k_hat} ({disposition:?}); {}",
        if cached {
            "certificate cached"
        } else {
            "certificate NOT cached (request errors decided it)"
        }
    );

    Ok(CalibrationOutcome { k_hat })
}
