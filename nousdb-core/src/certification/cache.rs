//! Session-scoped certificate cache.
//!
//! A certificate is evidence about a specific operator instance
//! (operator kind, model, prompt template with bound parameters, packing
//! policy), so all of those are part of the key. Certificates transfer
//! across queries in the same session; consumers must re-clamp a cached size
//! against the current query's feasibility size `k*_alpha`.
//!
//! There is no expiry: everything a certificate depends on is in its key, so
//! the session lifetime is the invalidation policy.
//!
//! Whether a certificate is stored is decided by the caller with
//! [`Certificate::is_cacheable`]: a ladder exhausted by request errors (not by
//! the model's answers) is not cached, so a transient endpoint outage is not
//! recorded as a fact about the model.

use std::collections::HashMap;
use std::sync::RwLock;

use super::numerics::cp_lower_bound;

/// Which semantic operator produced/consumes a certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    /// Semantic selection (filter).
    Selection,
    /// Semantic extension (per-row generation or extraction).
    Extend,
    /// Semantic join.
    Join,
}

/// The batch geometry calibration was drawn under. A certificate calibrated
/// under one geometry does not transfer to another, because the batch
/// distribution is geometry-dependent.
///
/// This concerns shape only, not which items get packed together: fillers are
/// drawn uniformly from the relation under both variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackingPolicy {
    /// A request is an unordered set of `k` items (extend and selection
    /// streams).
    Flat,
    /// Block-nested-loop join packing: one request is a left block crossed
    /// with a right tile of `right_block` rows, enumerated left-major, so a
    /// size-k batch shares its left rows across the tile. Sub-`k_max` rungs are
    /// drawn as aligned sub-geometries of that packing: `(k/R, R)` for
    /// `k >= R`, `(1, k)` below. Tail tiles stay ragged when `R` does not
    /// divide the right-side row count (an approximation).
    ///
    /// The tile width is part of the key because it shapes every calibration
    /// batch: a certificate drawn at `R = 4` describes a different batch
    /// distribution than one drawn at `R = 5`.
    JoinTile {
        /// Number of right-side rows in a tile (`R`).
        right_block: usize,
    },
}

/// Certificate-cache key identifying one operator instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CertKey {
    /// Provider name.
    pub provider: String,
    /// Model identifier within the provider.
    pub model_id: String,
    /// Operator that produced the certificate.
    pub operator_kind: OperatorKind,
    /// Stable FNV-1a hash over template identity, instruction text and input
    /// columns (see [`fnv1a_64`](super::rng::fnv1a_64)).
    pub template_hash: u64,
    /// Batch geometry the certificate was calibrated under.
    pub packing_policy: PackingPolicy,
}

/// Outcome of one rung.
#[derive(Debug, Clone)]
pub struct RungResult {
    /// Batch size tested by this rung.
    pub k: usize,
    /// Whether this rung can certify. False for the `k = 1` diagnostic rung,
    /// which scores solo answers against solo answers: it reports model
    /// self-consistency, not batching fidelity, so it is excluded from the
    /// ladder's risk budget and never sets `certified_k`. Its lower bound is
    /// the ceiling on every `phi(k)`; a shortfall there is not caused by
    /// batching.
    pub gating: bool,
    /// Agreements observed among the probes that were actually scored.
    pub x: u64,
    /// Committed probe budget for the rung: the denominator of the
    /// Clopper–Pearson bound, whether or not the rung ran to completion.
    pub n: u64,
    /// Probes actually scored. Below `n` when the rung was curtailed on
    /// futility, in which case `x / scored` is the observed fidelity.
    pub scored: u64,
    /// Scored probes that disagreed because a request got no answer
    /// (transport failure, provider error, faulted probe arm or a missing
    /// reference) on either arm, while neither arm returned an unusable
    /// answer. Already part of `scored - x`, so they count against the rung
    /// like any other disagreement and a dead endpoint cannot mint a
    /// certificate. They are tracked separately because they are evidence
    /// about the endpoint, not the model, and must not decide what gets cached
    /// ([`Certificate::is_cacheable`]).
    pub errors: u64,
    /// Clopper–Pearson lower bound `cp_lower_bound(x, n, delta_prime)`.
    pub lower_bound: f64,
    /// Rung aborted on futility (`disagreements > n - s_star`).
    pub curtailed: bool,
}

impl RungResult {
    /// Disagreements produced by the model's answers: scored probes that
    /// disagreed for any reason other than a missing answer (a real mismatch,
    /// or an unusable answer such as a truncated, undecodable or wrong-length
    /// one).
    pub fn answered_disagreements(&self) -> u64 {
        self.scored
            .saturating_sub(self.x)
            .saturating_sub(self.errors)
    }
}

/// Result of a full calibration: the certified size (if any rung passed)
/// plus the per-rung evidence. A failed ladder is cached too (it resolves to
/// `k = 1` and spares the next query the probe cost) unless request errors
/// rather than the model's answers exhausted it ([`Self::is_cacheable`]).
#[derive(Debug, Clone)]
pub struct Certificate {
    /// Largest batch size that passed, if any.
    pub certified_k: Option<usize>,
    /// Evidence for each rung, in the order tried.
    pub per_rung: Vec<RungResult>,
    /// Tolerated fidelity loss: a rung passes when its lower bound is at
    /// least `1 - eta`.
    pub eta: f64,
    /// Per-rung risk budget (`delta / L`).
    pub delta_prime: f64,
    /// Probe budget actually spent, committed from the realized sample size
    /// before the first model call.
    pub n: usize,
    /// Probe budget the plan asked for. A smaller `n` means a smaller sample
    /// was available, so the interval is wider and `s_star` stricter: a weaker
    /// certificate, not an invalid one.
    pub n_planned: usize,
}

impl Certificate {
    /// Gating rungs, in the order they were tried (descending `k`).
    pub fn gating_rungs(&self) -> impl Iterator<Item = &RungResult> {
        self.per_rung.iter().filter(|r| r.gating)
    }

    /// The `k = 1` diagnostic rung: solo answers scored against solo answers.
    ///
    /// Its lower bound is the ceiling on every `phi(k)`: a model that does not
    /// reproduce its own solo answers cannot be certified at any batch size.
    /// Read next to `certified_k == None`, it separates "packing hurts" from
    /// "the model is not self-consistent on this task".
    pub fn self_agreement_rung(&self) -> Option<&RungResult> {
        self.per_rung.iter().find(|r| !r.gating)
    }

    /// Gating rungs that failed, but would have passed had every probe that got
    /// no answer agreed: its request errors, and the probes a futility
    /// curtailment left unscored. Equivalently, the model's own disagreements
    /// stayed within `n - s_star`, so the failure is the endpoint's.
    ///
    /// The counterfactual is the most favourable one: a rung is blamed on the
    /// model only when its answered disagreements alone rule it out.
    pub fn rungs_failed_by_errors(&self) -> impl Iterator<Item = &RungResult> {
        self.gating_rungs().filter(move |r| {
            r.errors > 0
                && r.n > 0
                && Some(r.k) != self.certified_k
                && cp_lower_bound(
                    r.n - r.answered_disagreements().min(r.n),
                    r.n,
                    self.delta_prime,
                ) >= 1.0 - self.eta
        })
    }

    /// Whether this calibration may be stored in the session's
    /// [`CertificateCache`].
    ///
    /// A certified size is always cacheable: its rung passed despite whatever
    /// errors it absorbed, and a higher rung knocked out by errors only makes
    /// the certified size conservative. An exhausted ladder is cacheable only
    /// when the model's answers alone exhausted it, that is, when every gating
    /// rung would still fail with each missing answer counted as an agreement.
    /// Otherwise the cached "certify nothing" would describe a transient
    /// endpoint failure and the session would stay at `k = 1` after the
    /// endpoint recovered.
    pub fn is_cacheable(&self) -> bool {
        self.certified_k.is_some() || self.rungs_failed_by_errors().next().is_none()
    }
}

/// Session-scoped, thread-safe certificate store (an interior
/// `RwLock<HashMap>`).
#[derive(Debug, Default)]
pub struct CertificateCache {
    inner: RwLock<HashMap<CertKey, Certificate>>,
}

impl CertificateCache {
    /// Returns a clone of the certificate stored under `key`, if any.
    pub fn get(&self, key: &CertKey) -> Option<Certificate> {
        self.inner
            .read()
            .expect("CertificateCache lock poisoned")
            .get(key)
            .cloned()
    }

    /// Stores `certificate` under `key`, replacing any existing entry.
    pub fn insert(&self, key: CertKey, certificate: Certificate) {
        self.inner
            .write()
            .expect("CertificateCache lock poisoned")
            .insert(key, certificate);
    }

    /// Number of cached certificates.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("CertificateCache lock poisoned")
            .len()
    }

    /// Whether the cache holds no certificates.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of every cached certificate, for inspection and reporting.
    pub fn entries(&self) -> Vec<(CertKey, Certificate)> {
        self.inner
            .read()
            .expect("CertificateCache lock poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(model: &str) -> CertKey {
        CertKey {
            provider: "mock".into(),
            model_id: model.into(),
            operator_kind: OperatorKind::Selection,
            template_hash: 0xABCD,
            packing_policy: PackingPolicy::Flat,
        }
    }

    fn cert(k: Option<usize>) -> Certificate {
        Certificate {
            certified_k: k,
            per_rung: vec![],
            eta: 0.05,
            delta_prime: 0.01,
            n: 90,
            n_planned: 90,
        }
    }

    fn rung(k: usize, gating: bool, lower_bound: f64) -> RungResult {
        RungResult {
            k,
            gating,
            x: 88,
            n: 90,
            scored: 90,
            errors: 0,
            lower_bound,
            curtailed: false,
        }
    }

    /// The diagnostic rung is reachable without being mistaken for a failed
    /// batch size, and vice versa.
    #[test]
    fn gating_and_diagnostic_rungs_are_separable() {
        let mut c = cert(None);
        c.per_rung = vec![
            rung(16, true, 0.61),
            rung(8, true, 0.72),
            rung(1, false, 0.84),
        ];

        let gating: Vec<usize> = c.gating_rungs().map(|r| r.k).collect();
        assert_eq!(gating, vec![16, 8]);

        let diagnostic = c.self_agreement_rung().expect("k=1 rung");
        assert_eq!(diagnostic.k, 1);
        // The ceiling on every phi(k): no eta below 0.16 is attainable.
        assert!((diagnostic.lower_bound - 0.84).abs() < 1e-12);

        // A certificate without the diagnostic arm still reads cleanly.
        let mut older = cert(Some(8));
        older.per_rung = vec![rung(16, true, 0.61), rung(8, true, 0.91)];
        assert!(older.self_agreement_rung().is_none());
        assert_eq!(older.gating_rungs().count(), 2);
    }

    #[test]
    fn insert_get_roundtrip() {
        let cache = CertificateCache::default();
        assert!(cache.is_empty());
        assert!(cache.get(&key("m1")).is_none());

        cache.insert(key("m1"), cert(Some(16)));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key("m1")).unwrap().certified_k, Some(16));
        // Different model id -> different key.
        assert!(cache.get(&key("m2")).is_none());

        // Failed ladders are cached as information.
        cache.insert(key("m2"), cert(None));
        assert_eq!(cache.get(&key("m2")).unwrap().certified_k, None);
        assert_eq!(cache.len(), 2);
    }

    /// Operating point eta = 0.10, delta' = 0.025, n = 36, whose zero-miss
    /// floor `s* = n` makes the first disagreement futile.
    fn demo_ladder(rungs: Vec<RungResult>, certified_k: Option<usize>) -> Certificate {
        Certificate {
            certified_k,
            per_rung: rungs,
            eta: 0.10,
            delta_prime: 0.025,
            n: 36,
            n_planned: 36,
        }
    }

    /// A gating rung at that operating point: `scored` probes, `x` agreements,
    /// `errors` of the disagreements without an answer.
    fn scored_rung(k: usize, scored: u64, x: u64, errors: u64) -> RungResult {
        let n = 36;
        RungResult {
            k,
            gating: true,
            x,
            n,
            scored,
            errors,
            lower_bound: cp_lower_bound(x, n, 0.025),
            curtailed: scored < n || x < n,
        }
    }

    fn diagnostic(x: u64) -> RungResult {
        RungResult {
            gating: false,
            ..scored_rung(1, 36, x, 0)
        }
    }

    #[test]
    fn the_demo_point_has_a_zero_miss_floor() {
        // Every case below relies on this: one answered disagreement is fatal,
        // a perfect remainder is a pass.
        assert_eq!(super::super::s_star(36, 0.025, 0.90), Some(36));
    }

    /// Transport failures at the start of a run curtail every rung on its first
    /// probe. The ladder is exhausted, but by the endpoint.
    #[test]
    fn a_ladder_exhausted_by_request_errors_is_not_cacheable() {
        let c = demo_ladder(
            [16, 8, 4, 2]
                .into_iter()
                .map(|k| scored_rung(k, 1, 0, 1))
                .chain([diagnostic(36)])
                .collect(),
            None,
        );
        let blamed: Vec<usize> = c.rungs_failed_by_errors().map(|r| r.k).collect();
        assert_eq!(blamed, vec![16, 8, 4, 2]);
        assert!(!c.is_cacheable());
    }

    /// A model whose answers disagree exhausts the ladder on its own, and that
    /// failed ladder is worth caching.
    #[test]
    fn a_ladder_exhausted_by_answers_stays_cacheable() {
        let c = demo_ladder(
            [16, 8, 4, 2]
                .into_iter()
                .map(|k| scored_rung(k, 1, 0, 0))
                .chain([diagnostic(36)])
                .collect(),
            None,
        );
        assert_eq!(c.rungs_failed_by_errors().count(), 0);
        assert!(c.is_cacheable());
    }

    /// Errors only matter when they were decisive: a rung whose answered
    /// disagreements already exceed `n - s*` fails whatever the missing
    /// answers would have said.
    #[test]
    fn errors_next_to_a_fatal_answered_disagreement_do_not_block_caching() {
        // 1 agreement, 1 error, 1 answered disagreement: 35/36 at best.
        let c = demo_ladder(
            vec![scored_rung(16, 3, 1, 1), scored_rung(8, 1, 0, 0)],
            None,
        );
        assert_eq!(c.per_rung[0].answered_disagreements(), 1);
        assert_eq!(c.rungs_failed_by_errors().count(), 0);
        assert!(c.is_cacheable());
    }

    /// One salvageable rung is enough: a healthy endpoint might certify it, so
    /// "certify nothing" is not what the answers say.
    #[test]
    fn one_rung_failed_by_errors_makes_an_exhausted_ladder_uncacheable() {
        let c = demo_ladder(
            vec![
                scored_rung(16, 1, 0, 0),
                scored_rung(8, 1, 0, 0),
                scored_rung(4, 20, 19, 1),
                scored_rung(2, 1, 0, 0),
                diagnostic(36),
            ],
            None,
        );
        let blamed: Vec<usize> = c.rungs_failed_by_errors().map(|r| r.k).collect();
        assert_eq!(blamed, vec![4]);
        assert!(!c.is_cacheable());
    }

    /// A certified size is cached even when errors knocked out the rungs above
    /// it: its own rung passed, so `k̂` is conservative, never unsafe.
    #[test]
    fn a_certified_size_is_cacheable_even_below_rungs_failed_by_errors() {
        let c = demo_ladder(
            vec![
                scored_rung(16, 1, 0, 1),
                scored_rung(8, 36, 36, 0),
                diagnostic(36),
            ],
            Some(8),
        );
        let blamed: Vec<usize> = c.rungs_failed_by_errors().map(|r| r.k).collect();
        assert_eq!(blamed, vec![16], "still reported, so a reader can see it");
        assert!(c.is_cacheable());
    }

    /// Errors on the non-gating diagnostic rung never decide anything.
    #[test]
    fn diagnostic_errors_are_not_blamed() {
        let mut errored_diagnostic = diagnostic(30);
        errored_diagnostic.errors = 6;
        let c = demo_ladder(vec![scored_rung(16, 1, 0, 0), errored_diagnostic], None);
        assert_eq!(c.rungs_failed_by_errors().count(), 0);
        assert!(c.is_cacheable());
    }

    #[test]
    fn overwrite_replaces() {
        let cache = CertificateCache::default();
        cache.insert(key("m"), cert(Some(8)));
        cache.insert(key("m"), cert(Some(4)));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key("m")).unwrap().certified_k, Some(4));
    }
}
