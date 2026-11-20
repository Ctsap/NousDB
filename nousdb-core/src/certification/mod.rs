//! Quality-certified batch sizing: statistical core.
//!
//! This module holds the planning-side statistics: exact Clopper–Pearson lower
//! bounds on batch fidelity ([`numerics`]), the candidate-ladder and
//! probe-budget arithmetic ([`spec`]), deterministic designation randomness
//! ([`rng`]), and the session-scoped certificate cache ([`cache`]). Running the
//! calibration probes and scoring agreement is the job of the operator layer.
//!
//! Fidelity `phi(k)` is the probability that a row's answer under a size-`k`
//! packed request equals its solo (`k = 1`) answer, for a uniformly drawn row
//! placed at a uniformly drawn slot among `k - 1` other items drawn uniformly
//! from the relation. It describes batching, not the order in which a table
//! happens to be scanned. The certified size executed by an operator is
//! `k* = min(k*_alpha, k_{eta,delta})`: the context-window feasibility size
//! clamped by the largest batch size whose fidelity lower bound clears
//! `1 - eta` at confidence `1 - delta`.
//!
//! The per-rung risk budget is `delta_prime` (`delta / L`) throughout this
//! module. It is unrelated to `SafeBatchSizingConfig::alpha`, the
//! context-window overflow probability, which has its own budget.

pub mod cache;
pub mod numerics;
pub mod rng;
pub mod spec;

pub use cache::{CertKey, Certificate, CertificateCache, OperatorKind, PackingPolicy, RungResult};
pub use numerics::{cp_lower_bound, n_min, s_star};
pub use rng::{SplitMix64, fnv1a_64};
pub use spec::{
    Budget, CertificationSpec, DEFAULT_MAX_PROBES, QualityCertConfig, budget_for, build_ladder,
    feasible_budget,
};
