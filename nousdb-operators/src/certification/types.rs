//! Types shared by the plan-time and execution-time halves of quality-certified
//! batch sizing.
//!
//! `CertifyBatchSizeRule` attaches a [`CertifyRuntime`] to an exec; the exec
//! turns it into a per-partition [`StreamCertification`] in `execute()`; the
//! prologue resolves it into a [`CalibrationOutcome`] shared across partitions.
//! The runner lives in [`crate::certification::calibrate`]; certified operators
//! dispatch by executing probe copies of themselves.

use std::sync::Arc;

use datafusion::execution::TaskContext;

use nousdb_core::certification::{CertificateCache, CertificationSpec};

use crate::certification::pool::PrefixPool;
use crate::certification::probe::ProbeExecFactory;
use crate::join::stream::OnceAsync;
use crate::metrics::CalibrationMetrics;

/// How a calibration concluded: the reason behind `k̂`, for the trace.
///
/// The decision reaches the stream as [`CalibrationOutcome::k_hat`] and the
/// evidence reaches readers through the [`CertificateCache`]. This value
/// records why, and is the only place `ShortInput` is distinguishable from an
/// exhausted ladder (`ShortInput` writes no certificate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// A gating rung passed; `k̂` is the certified size (already clamped by
    /// `k*_alpha`, since the ladder tops out there).
    Certified(usize),
    /// Every gating rung failed or was curtailed; `k̂ = 1` (exact semantics).
    /// The `k = 1` diagnostic rung is not a gating rung and cannot prevent
    /// this; read it to tell "packing hurts" from "the model is not
    /// self-consistent". See `Certificate::self_agreement_rung`.
    LadderExhausted,
    /// The pooled sample was too short to support even a minimal probe budget;
    /// `k̂ = 1` with no model calls and no cache write.
    ///
    /// A partially short sample is a different case: the probe budget `n` is
    /// committed from the rows that arrived, before any model call, so a
    /// smaller sample yields a weaker but still valid certificate.
    ShortInput {
        /// Number of rows in the pooled sample.
        rows: usize,
    },
}

/// Result shared with every partition of the operator instance.
///
/// Carries no rows. The sample lives in the [`PrefixPool`] and is only read for
/// scoring; every partition recycles the rows it buffered itself, because the
/// pooled prefix holds copies from every partition and re-injecting it would
/// duplicate rows and emit them from the wrong output partition.
///
/// It carries no certificate either: the runner writes that to the
/// [`CertificateCache`].
#[derive(Debug)]
pub struct CalibrationOutcome {
    /// Batch size the steady state runs at.
    pub k_hat: usize,
}

/// Plan-time attachment carried by a semantic exec node
/// (built by `CertifyBatchSizeRule`).
#[derive(Debug, Clone)]
pub struct CertifyRuntime {
    /// Certification parameters.
    pub spec: Arc<CertificationSpec>,
    /// Cache the certificate is written to and read from.
    pub cache: Arc<CertificateCache>,
    /// Once-per-operator-instance calibration shared across partitions.
    pub once: Arc<OnceAsync<CalibrationOutcome>>,
    /// The calibration sample, contributed to by every partition.
    pub pool: Arc<PrefixPool>,
}

/// Per-partition handle passed from the exec into its stream.
#[derive(Clone)]
pub struct StreamCertification {
    /// Certification parameters.
    pub spec: Arc<CertificationSpec>,
    /// Cache the certificate is written to and read from.
    pub cache: Arc<CertificateCache>,
    /// Once-per-operator-instance calibration shared across partitions.
    pub once: Arc<OnceAsync<CalibrationOutcome>>,
    /// Shared sample; this stream donates its quota and the winner assembles.
    pub pool: Arc<PrefixPool>,
    /// This stream's partition index, its slot in the pool.
    pub partition: usize,
    /// Calibration metrics sink.
    pub metrics: CalibrationMetrics,
    /// Builds probe plans from the owning operator, so calibration dispatches
    /// through the engine.
    pub factory: ProbeExecFactory,
    /// Task context used to execute probe plans.
    pub task_ctx: Arc<TaskContext>,
    /// Concurrency the prologue may use: `request_window × partition_count`.
    /// Calibration is a whole-operator barrier (every other partition waits on
    /// the shared outcome), so it may use the concurrency production would
    /// spread across all partitions. The endpoint-global admission semaphore
    /// remains the real limiter.
    pub window_budget: usize,
}

impl std::fmt::Debug for StreamCertification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamCertification")
            .field("spec", &self.spec)
            .field("window_budget", &self.window_budget)
            .finish_non_exhaustive()
    }
}
