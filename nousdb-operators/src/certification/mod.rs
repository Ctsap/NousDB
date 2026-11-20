//! Execution-time half of quality-certified batch sizing: the calibration
//! prologue that runs inside semantic operator streams.
//!
//! Plan-time artifacts (`CertificationSpec`, budgets, the certificate cache)
//! live in `nousdb_core::certification`. This module owns the async runner
//! that buffers a prefix of the realized input, measures batch fidelity
//! against solo-processing references, and picks the certified size `k̂`.
//!
//! - `probe`: designation, probe-input construction, and the two hooks the
//!   operators provide: a [`ProbeExecFactory`] (build a copy of the operator
//!   at batch size `k`) and a [`RequestOutcomeSink`] (report how each request
//!   ended).
//! - `calibrate`: the runner. It owns no dispatcher: each arm is an
//!   ordinary `execute()` of a probe plan, so coalescing, the request window,
//!   admission gating, recovery and metrics all come from the operator.
//! - `pool`: the calibration sample, pooled across every partition of the
//!   operator instance.
//! - `driver`: the per-stream state machine (donate to the pool, install the
//!   shared once-per-instance calibration, await, recycle at `k̂`), shared by
//!   selection and extend. Join drives the same phases from its own
//!   `StreamState`, because its prefix is a pair batch built by crossing
//!   buffered left rows with the shared right relation and can be neither
//!   buffered nor recycled like an input prefix.
//! - `types`: the plan-time/execution-time handoff types.

pub mod calibrate;
pub(crate) mod driver;
pub mod pool;
pub mod probe;
pub mod types;

pub use calibrate::{ProbeCalibrationContext, run_probe_calibration};
pub use pool::PrefixPool;

pub(crate) use driver::{CalibrationHost, CalibrationParts, CalibrationState, poll_calibration};
pub use probe::{
    Design, ItemSpace, PROBE_ID_COLUMN, ProbeExecFactory, ProbeInputBuilder, ProbeObservation,
    ProbeScorer, RequestOutcome, RequestOutcomeSink, Rung, attach_probe_ids, build_probe_input,
    designate, find_probe_row,
};
pub use types::{CalibrationOutcome, CertifyRuntime, Disposition, StreamCertification};
