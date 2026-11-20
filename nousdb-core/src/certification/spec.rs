//! Plan-time certification artifacts: configuration, candidate ladder,
//! statistical budget, and the spec attached to physical operator nodes.

use crate::certification::cache::CertKey;
use crate::certification::numerics::{LnFactorial, n_min, s_star, s_star_cached};

/// Default probe ceiling. Probing costs `n * (1 + L)` model calls, so surplus
/// sample is spent on extra probes only up to this many; see
/// [`feasible_budget`].
pub const DEFAULT_MAX_PROBES: usize = 128;

/// How much larger than [`min_items`] a sample the calibration prologue asks
/// for.
///
/// Fillers are drawn from the pooled sample, so the pool is the empirical
/// relation the fidelity is measured against. Its sampling error is
/// `O(1/sqrt(|pool|))` and the Clopper–Pearson width is `O(1/sqrt(n))`, so a
/// small constant factor above the floor puts the pool's contribution below
/// the interval's. Buffering more only delays the operator's first output,
/// because the prologue is a barrier.
const SAMPLE_FACTOR: usize = 2;

/// Items a design of `n` probes at ladder top `k_max` cannot do without: the
/// probes themselves, plus enough non-probe items to fill one batch.
pub fn min_items(n: usize, k_max: usize) -> usize {
    n + k_max.saturating_sub(1)
}

/// Items the prologue asks its pool to gather. See `SAMPLE_FACTOR`.
pub fn sample_target(n: usize, k_max: usize) -> usize {
    min_items(n, k_max).saturating_mul(SAMPLE_FACTOR)
}

/// Session-level configuration for quality-certified batch sizing.
///
/// Defaults are `eta = delta = 0.05`. `delta` is the certification risk
/// budget, unrelated to `SafeBatchSizingConfig::alpha`, the context-window
/// overflow probability.
#[derive(Debug, Clone)]
pub struct QualityCertConfig {
    /// Fidelity tolerance: certify `k` only if fidelity >= `1 - eta`.
    pub eta: f64,
    /// Total risk budget across the ladder (split as `delta_prime = delta/L`,
    /// or `delta/(L * |O_q|)` under [`Self::query_level_budget`]).
    pub delta: f64,
    /// Split `delta` across every certified operator instance in the query,
    /// making the guarantee hold simultaneously for all of them.
    pub query_level_budget: bool,
    /// Disagreements a rung may absorb and still certify.
    ///
    /// `None` budgets the bare floor `n_min`, where a rung fails on its first
    /// disagreement. `Some(t)` solves for the smallest `n` with
    /// `s_star(n) <= n - t`.
    ///
    /// With zero tolerance a single early disagreement exhausts the ladder,
    /// which says little about fidelity, so real models usually need a nonzero
    /// value. Each extra tolerated miss costs additional sample.
    pub tolerated_disagreements: Option<usize>,
    /// Cardinality gate: skip certification (run at `k = 1`) when the
    /// operator's estimated input rows fall below this. `0` disables the gate.
    pub min_rows_for_certification: usize,
    /// Ceiling on the probe budget `n` once [`feasible_budget`] starts spending
    /// surplus sample on extra probes. Probing costs `n * (1 + L)` model calls.
    /// It never pushes `n` below the budget [`Self::tolerated_disagreements`]
    /// asked for, so any value at or below that budget disables growth.
    pub max_probes: usize,
    /// Seed for probe-row and batch-slot designation (deterministic runs).
    pub seed: u64,
}

impl QualityCertConfig {
    /// This operator's share of `delta` before the ladder split, i.e.
    /// `delta / |O_q|` under [`Self::query_level_budget`] and `delta` otherwise.
    pub fn delta_share(&self, num_operators: usize) -> f64 {
        let operators = if self.query_level_budget {
            num_operators.max(1)
        } else {
            1
        };
        self.delta / operators as f64
    }
}

impl Default for QualityCertConfig {
    fn default() -> Self {
        Self {
            eta: 0.05,
            delta: 0.05,
            // A query with several certified operators must split the risk
            // budget across them, or the guarantee does not hold
            // simultaneously for all of them.
            query_level_budget: true,
            tolerated_disagreements: None,
            min_rows_for_certification: 0,
            max_probes: DEFAULT_MAX_PROBES,
            seed: 42,
        }
    }
}

/// Everything the execution-time calibration prologue needs, resolved at
/// plan time by `CertifyBatchSizeRule`.
#[derive(Debug, Clone, PartialEq)]
pub struct CertificationSpec {
    /// Gating batch sizes, strictly descending; `ladder[0] == k*_alpha`. Never
    /// contains `1`: `k = 1` is measured as a diagnostic outside the risk
    /// budget (see [`build_ladder`]).
    pub ladder: Vec<usize>,
    /// Fidelity tolerance (certify iff lower bound >= `1 - eta`).
    pub eta: f64,
    /// Per-rung risk budget after the multiplicity split.
    pub delta_prime: f64,
    /// Probe budget per rung, committed before any probing.
    pub n: usize,
    /// Minimum passing agreement count; a rung is curtailed as soon as
    /// `disagreements > n - s_star`.
    pub s_star: usize,
    /// Designation RNG seed.
    pub seed: u64,
    /// Certificate-cache key for this operator instance.
    pub cache_key: CertKey,
    /// This operator's share of `delta` before the ladder split, i.e.
    /// `delta / |O_q|`, equal to `delta_prime * ladder.len()`.
    ///
    /// Carried so [`Self::retarget`] can re-cost a shorter ladder at execution
    /// time, when the session's [`QualityCertConfig`] is not available and
    /// `delta_prime` alone cannot be un-split.
    pub delta_share: f64,
    /// [`QualityCertConfig::tolerated_disagreements`], carried for the same
    /// reason as [`Self::delta_share`].
    pub tolerated: Option<usize>,
    /// [`QualityCertConfig::max_probes`], carried for the same reason.
    pub max_probes: usize,
}

impl CertificationSpec {
    /// Items the design cannot do without: `n` probes plus enough non-probe
    /// items to fill one top-rung batch. Fillers are shared across probes, so
    /// this is a sum rather than `n * k_max`.
    pub fn min_items(&self) -> usize {
        min_items(self.n, self.ladder.first().copied().unwrap_or(0))
    }

    /// Items the prologue asks the pool to gather: a constant factor above
    /// [`Self::min_items`]. See [`sample_target`].
    pub fn sample_target(&self) -> usize {
        sample_target(self.n, self.ladder.first().copied().unwrap_or(0))
    }

    /// The largest budget `available_rows` can pay for, or `None` when the
    /// sample cannot support even the smallest rung.
    ///
    /// The ladder top never rises above this spec's (candidates are at or below
    /// `ladder[0]`), so a retarget only weakens what is claimed. Seed and cache
    /// key are preserved, so a rerun on the same data re-draws the same design.
    pub fn retarget(&self, available_rows: usize) -> Option<CertificationSpec> {
        let k_max = self.ladder.first().copied()?;
        let budget = feasible_budget(
            k_max,
            self.delta_share,
            self.eta,
            self.tolerated,
            self.max_probes,
            available_rows,
        )?;
        Some(CertificationSpec {
            ladder: budget.ladder,
            eta: self.eta,
            delta_prime: budget.delta_prime,
            n: budget.n,
            s_star: budget.s_star,
            seed: self.seed,
            cache_key: self.cache_key.clone(),
            delta_share: self.delta_share,
            tolerated: self.tolerated,
            max_probes: self.max_probes,
        })
    }
}

/// Candidate ladder `{2^j : 2^j < k*_alpha} ∪ {k*_alpha}`, descending.
///
/// Returns an empty ladder for `k*_alpha <= 1` (no certification possible,
/// since `phi(1) = 1` by definition). A `k*_alpha` adjacent to a power of two
/// (for example 65) produces near-duplicate top rungs like `{65, 64, ...}`.
///
/// Every candidate is measured: the ladder is never narrowed to a subset,
/// because dropping rungs silently changes which batch sizes can be certified.
/// A longer ladder costs rungs rather than budget, since `n_min` is
/// logarithmic in `delta'` (at `eta = delta = 0.10`, going from `L = 3` to
/// `L = 4` moves `n` from 33 to 36).
///
/// # `k = 1` is not included
///
/// `k = 1` is measured separately as a diagnostic: it compares solo answers
/// against solo answers, so it reports model self-consistency rather than
/// batching fidelity. It cannot change the executed batch size (an exhausted
/// ladder already falls back to `k = 1`) and is excluded from `L` so it does
/// not shrink `delta'` for the rungs that gate.
pub fn build_ladder(k_star_alpha: usize) -> Vec<usize> {
    if k_star_alpha <= 1 {
        return Vec::new();
    }
    let mut ladder = vec![k_star_alpha];
    let mut power = 2usize;
    while power < k_star_alpha {
        ladder.push(power);
        power = match power.checked_mul(2) {
            Some(p) => p,
            None => break,
        };
    }
    ladder.sort_unstable_by(|a, b| b.cmp(a));
    ladder
}

/// Statistical budget for one operator instance: `(delta_prime, n, s_star)`.
///
/// `delta_prime = delta / (L * |O_q|)` splits the risk budget Bonferroni-style
/// across the `L` ladder rungs, and across the query's certified operators when
/// [`QualityCertConfig::query_level_budget`] is set. `n` is the probe floor
/// [`n_min`], raised only when [`QualityCertConfig::tolerated_disagreements`]
/// asks for slack, and `s_star` is the resulting pass threshold.
///
/// Returns `None` when the configuration is degenerate (empty ladder,
/// `eta`/`delta` outside `(0, 1)`); callers should then skip certification
/// rather than panic mid-planning.
pub fn budget_for(
    cfg: &QualityCertConfig,
    ladder_len: usize,
    num_operators: usize,
) -> Option<(f64, usize, usize)> {
    if ladder_len == 0 {
        return None;
    }
    if !(cfg.eta > 0.0 && cfg.eta < 1.0 && cfg.delta > 0.0 && cfg.delta < 1.0) {
        return None;
    }
    let delta_prime = cfg.delta_share(num_operators) / ladder_len as f64;
    if delta_prime <= 0.0 || delta_prime >= 1.0 {
        return None;
    }
    let (n, s) = match cfg.tolerated_disagreements {
        Some(tolerated) => smallest_budget_tolerating(tolerated as u64, delta_prime, cfg.eta)?,
        None => {
            let n = n_min(delta_prime, cfg.eta);
            let s = s_star_cached(n, delta_prime, 1.0 - cfg.eta, &mut LnFactorial::new())?;
            (n, s)
        }
    };
    Some((delta_prime, n as usize, s as usize))
}

/// A ladder and the statistical budget that will be spent on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    /// Gating batch sizes, descending; `ladder[0]` is the top rung.
    pub ladder: Vec<usize>,
    /// Per-rung risk budget, `delta_share / ladder.len()`.
    pub delta_prime: f64,
    /// Probe budget per rung.
    pub n: usize,
    /// Minimum passing agreement count at `n`.
    pub s_star: usize,
}

impl Budget {
    /// Disagreements a rung can absorb and still certify.
    pub fn tolerance(&self) -> usize {
        self.n - self.s_star
    }

    /// Items the design cannot do without. See [`min_items`].
    pub fn min_items(&self) -> usize {
        min_items(self.n, self.ladder.first().copied().unwrap_or(0))
    }
}

/// The largest budget a sample of `available_items` can pay for.
///
/// # Why the ladder top is not always `k*_alpha`
///
/// Fillers are shared across probes, so a design of `n` probes at ladder top
/// `k` needs `n + k - 1` items: the probes plus one batch's worth of pool.
/// When the sample is too small for that (for example 40 probes at `k = 86`
/// over 77 rows), a smaller top is chosen. Certifying a smaller `k` is a
/// weaker claim, not an invalid one.
///
/// # Why the requested budget, not the bare floor, picks the top
///
/// Candidates are walked in descending order and the first whose requested `n`
/// (the budget [`QualityCertConfig::tolerated_disagreements`] asks for) fits is
/// taken, not the first that merely clears [`n_min`]. A top that only clears
/// the floor has zero tolerance, so its first disagreement curtails the rung
/// on evidence that says little about fidelity; a smaller top may afford a
/// larger `n` that absorbs several disagreements. If no top affords the
/// requested budget, the largest top that clears the floor is used.
///
/// Surplus sample is then spent on extra probes up to `max_probes`. A larger
/// `n` tightens the Clopper–Pearson interval and lets the rung absorb more
/// disagreements, at the cost of model calls only.
///
/// Every quantity here depends only on cardinality, decided before the first
/// model call, so each rung's agreement count is still `Binomial(n, phi(k))`
/// and the coverage guarantee is unchanged.
///
/// `delta_share` is this operator's share of `delta` before the ladder split
/// (`delta / |O_q|`); `delta_prime` falls out of the ladder length that is
/// actually chosen. Returns `None` when no rung is affordable at all, or when
/// the configuration is degenerate.
pub fn feasible_budget(
    k_star_alpha: usize,
    delta_share: f64,
    eta: f64,
    tolerated: Option<usize>,
    max_probes: usize,
    available_items: usize,
) -> Option<Budget> {
    if !(eta > 0.0 && eta < 1.0 && delta_share > 0.0 && delta_share < 1.0) {
        return None;
    }
    // Candidate tops are the ladder of `k*_alpha` itself: `k*_alpha` and every
    // power of two below it, descending.
    let candidates = build_ladder(k_star_alpha);

    // Largest top that clears the floor but not the requested budget, kept as
    // a fallback so a sample too thin for the tolerance still certifies
    // something.
    let mut fallback: Option<Budget> = None;

    for &top in &candidates {
        let ladder = build_ladder(top);
        let delta_prime = delta_share / ladder.len() as f64;
        if !(delta_prime > 0.0 && delta_prime < 1.0) {
            continue;
        }
        // Probes this top can be given: the sample minus the one batch's worth
        // of pool the fillers are drawn from.
        let capacity = available_items.saturating_sub(top.saturating_sub(1));
        let floor = n_min(delta_prime, eta) as usize;
        if capacity < floor {
            // Below the floor every rung is doomed however the probes land.
            continue;
        }
        let requested = match tolerated {
            Some(t) => match smallest_budget_tolerating(t as u64, delta_prime, eta) {
                Some((n, _)) => n as usize,
                None => continue,
            },
            None => floor,
        };

        // Spend the sample, but never below what was asked for and never above
        // the probe ceiling.
        let spend = |floor_n: usize| -> Option<Budget> {
            let n = capacity.min(max_probes.max(floor_n));
            let s = s_star(n as u64, delta_prime, 1.0 - eta)? as usize;
            Some(Budget {
                ladder: ladder.clone(),
                delta_prime,
                n,
                s_star: s,
            })
        };

        if capacity >= requested {
            if let Some(budget) = spend(requested) {
                return Some(budget);
            }
        } else if fallback.is_none() {
            fallback = spend(floor);
        }
    }

    fallback
}

/// Smallest `n` whose passing threshold leaves room for `tolerated`
/// disagreements, with that threshold.
///
/// `s_star` is monotone enough in `n` for a linear scan from the floor. The
/// binomial tail concentrates as `n` grows, so `n - s_star(n)` increases
/// without bound; the cap only guards against a degenerate `(delta_prime, eta)`
/// making the target unreachable.
///
/// The scan walks `n` upward, so one factorial table serves the whole search.
fn smallest_budget_tolerating(tolerated: u64, delta_prime: f64, eta: f64) -> Option<(u64, u64)> {
    const MAX_PROBES: u64 = 100_000;
    let mut ln_fact = LnFactorial::new();
    let mut n = n_min(delta_prime, eta).max(1);
    while n <= MAX_PROBES {
        if let Some(s) = s_star_cached(n, delta_prime, 1.0 - eta, &mut ln_fact)
            && n - s >= tolerated
        {
            return Some((n, s));
        }
        n += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification::cache::{CertKey, OperatorKind, PackingPolicy};

    fn demo_cfg() -> QualityCertConfig {
        QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            ..Default::default()
        }
    }

    #[test]
    fn ladder_shapes() {
        // A non-power-of-two k*_alpha keeps itself as the top rung.
        assert_eq!(build_ladder(18), vec![18, 16, 8, 4, 2]);
        assert_eq!(build_ladder(12), vec![12, 8, 4, 2]);
        assert_eq!(build_ladder(16), vec![16, 8, 4, 2]);
        assert_eq!(build_ladder(32), vec![32, 16, 8, 4, 2]);
        assert_eq!(build_ladder(4), vec![4, 2]);
        assert_eq!(build_ladder(2), vec![2]);
        // phi(1) = 1 by definition: nothing to certify.
        assert_eq!(build_ladder(1), Vec::<usize>::new());
        assert_eq!(build_ladder(0), Vec::<usize>::new());
    }

    /// Every candidate is measured: no power-of-two size below `k*_alpha` is
    /// dropped.
    #[test]
    fn ladder_is_complete_and_descending() {
        for k_star in [2usize, 4, 12, 16, 18, 32, 65, 128] {
            let ladder = build_ladder(k_star);
            assert_eq!(ladder[0], k_star, "{k_star}: k*_alpha must be the top rung");
            assert!(
                ladder.windows(2).all(|w| w[0] > w[1]),
                "{k_star}: {ladder:?} must stay strictly descending"
            );
            // Every power of two strictly below k*_alpha, and nothing else.
            let expected: Vec<usize> = std::iter::once(k_star)
                .chain((1..).map(|j| 1usize << j).take_while(|p| *p < k_star))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .rev()
                .collect();
            assert_eq!(ladder, expected, "{k_star}: unexpected rungs");
        }
    }

    /// `k = 1` is a diagnostic, never a gating rung: it must not appear in the
    /// ladder, where it would consume a share of `delta'`.
    #[test]
    fn ladder_never_contains_one() {
        for k_star in [2usize, 4, 12, 16, 18, 32, 65, 128] {
            assert!(
                !build_ladder(k_star).contains(&1),
                "{k_star}: k=1 must not be a gating rung"
            );
        }
    }

    #[test]
    fn paper_budget() {
        // eta = delta = 0.05, L = 5 -> delta' = 0.01, n = 90, s* = 90.
        let cfg = QualityCertConfig::default();
        let (dp, n, s) = budget_for(&cfg, 5, 1).unwrap();
        assert!((dp - 0.01).abs() < 1e-12);
        assert_eq!(n, 90);
        assert_eq!(s, 90);
    }

    #[test]
    fn query_level_budget() {
        // Two certified operators share the budget: delta' = 1/120
        // (delta = 0.05, L = 3, |O_q| = 2) -> n = 94.
        let cfg = QualityCertConfig {
            query_level_budget: true,
            ..Default::default()
        };
        let (dp, n, s) = budget_for(&cfg, 3, 2).unwrap();
        assert!((dp - 1.0 / 120.0).abs() < 1e-12);
        assert_eq!(n, 94);
        assert_eq!(s, 94);
    }

    /// Without an explicit tolerance the budget is exactly the floor, and the
    /// floor tolerates nothing: a rung fails on its first disagreement.
    /// Checked over the ladder for `k*_alpha = 16`.
    #[test]
    fn demo_budget_defaults_to_the_floor() {
        let l = build_ladder(16).len();
        assert_eq!(l, 4, "[16, 8, 4, 2]");
        let (dp, n, s) = budget_for(&demo_cfg(), l, 1).unwrap();
        assert!((dp - 0.025).abs() < 1e-12);
        assert_eq!(n, n_min(0.025, 0.10) as usize);
        assert_eq!(n, 36);
        assert_eq!(s, 36);
    }

    /// A longer ladder costs rungs, not budget: `n_min` is logarithmic in
    /// `delta'`, so each added rung adds a handful of probes per rung.
    #[test]
    fn a_longer_ladder_barely_moves_the_probe_floor() {
        let n_of = |l: usize| budget_for(&demo_cfg(), l, 1).unwrap().1;
        assert_eq!(n_of(3), 33);
        assert_eq!(n_of(4), 36);
        assert_eq!(n_of(5), 38);
        // Doubling the rung count costs well under a doubling of n.
        assert!(n_of(6) < 2 * n_of(3));
    }

    /// Asking for tolerance directly gets exactly that, at the smallest budget
    /// that provides it.
    #[test]
    fn tolerated_disagreements_solves_for_n() {
        for want in [1usize, 2, 5, 10] {
            let cfg = QualityCertConfig {
                tolerated_disagreements: Some(want),
                ..demo_cfg()
            };
            let (_, n, s) = budget_for(&cfg, 3, 1).unwrap();
            assert!(
                n - s >= want,
                "n={n} s*={s} tolerates {} but {want} was asked for",
                n - s
            );
            // Minimality: one fewer probe must not suffice.
            if n > 1 {
                let smaller = crate::certification::numerics::s_star(
                    (n - 1) as u64,
                    cfg.delta / 3.0,
                    1.0 - cfg.eta,
                );
                assert!(
                    smaller.is_none_or(|s| (n - 1) - (s as usize) < want),
                    "n={n} is not the smallest budget tolerating {want}"
                );
            }
        }
    }

    /// Tolerance costs sample, monotonically: more slack, more probes.
    #[test]
    fn tolerance_is_monotone_in_probe_cost() {
        let budget = |tolerated: Option<usize>| {
            let cfg = QualityCertConfig {
                tolerated_disagreements: tolerated,
                ..demo_cfg()
            };
            budget_for(&cfg, 3, 1).unwrap().1
        };
        let floor = budget(None);
        assert!(floor < budget(Some(1)));
        assert!(budget(Some(1)) < budget(Some(5)));
    }

    #[test]
    fn degenerate_budgets_are_none() {
        assert!(budget_for(&QualityCertConfig::default(), 0, 1).is_none());
        let bad = QualityCertConfig {
            eta: 0.0,
            ..Default::default()
        };
        assert!(budget_for(&bad, 3, 1).is_none());
    }

    #[test]
    fn sizing_is_probes_plus_one_batch() {
        let spec = CertificationSpec {
            ladder: vec![16, 8, 4],
            eta: 0.10,
            delta_prime: 1.0 / 30.0,
            n: 40,
            s_star: 39,
            seed: 42,
            cache_key: CertKey {
                provider: "mock".into(),
                model_id: "m".into(),
                operator_kind: OperatorKind::Selection,
                template_hash: 1,
                packing_policy: PackingPolicy::Flat,
            },
            delta_share: 0.10,
            tolerated: None,
            max_probes: DEFAULT_MAX_PROBES,
        };
        // 40 probes plus one 16-item batch's worth of pool, doubled for the
        // pool the fillers are actually drawn from.
        assert_eq!(spec.min_items(), 55);
        assert_eq!(spec.sample_target(), 110);
    }

    /// A configuration with `eta = 0.15`, `delta = 0.05` and one tolerated
    /// disagreement.
    fn sembench_cfg() -> QualityCertConfig {
        QualityCertConfig {
            eta: 0.15,
            delta: 0.05,
            tolerated_disagreements: Some(1),
            ..Default::default()
        }
    }

    fn feasible(cfg: &QualityCertConfig, k_star: usize, rows: usize) -> Option<Budget> {
        feasible_budget(
            k_star,
            cfg.delta,
            cfg.eta,
            cfg.tolerated_disagreements,
            cfg.max_probes,
            rows,
        )
    }

    /// When the sample affords the requested budget at `k*_alpha` and the probe
    /// ceiling equals that budget, the result matches `budget_for`: same
    /// ladder, `delta'`, `n` and `s*`.
    #[test]
    fn affordable_at_k_star_reproduces_budget_for() {
        for (cfg, k_star) in [
            (QualityCertConfig::default(), 16usize),
            (demo_cfg(), 12),
            (sembench_cfg(), 12),
        ] {
            let ladder = build_ladder(k_star);
            let (delta_prime, n, s_star) = budget_for(&cfg, ladder.len(), 1).unwrap();
            let got = feasible_budget(
                k_star,
                cfg.delta,
                cfg.eta,
                cfg.tolerated_disagreements,
                n, // probe ceiling == the requested budget: no growth
                min_items(n, k_star),
            )
            .unwrap();
            assert_eq!(got.ladder, ladder);
            assert!((got.delta_prime - delta_prime).abs() < 1e-12);
            assert_eq!(got.n, n);
            assert_eq!(got.s_star, s_star);
        }
    }

    /// `k*_alpha = 86` over 1000 rows certifies the full ladder.
    #[test]
    fn sembench_selection_drops_to_an_affordable_top() {
        let cfg = sembench_cfg();
        let got = feasible(&cfg, 86, 1000).expect("1000 rows must certify something");
        // Fillers come from the shared pool, so the full ladder is affordable:
        // 1000 rows seat `n + 85` items at the planner's own k*_alpha.
        assert_eq!(got.ladder, vec![86, 64, 32, 16, 8, 4, 2]);
        assert!(
            got.min_items() <= 1000,
            "design needs {} of 1000 rows",
            got.min_items()
        );
        // Surplus sample is spent on probes, up to the ceiling.
        assert_eq!(got.n, cfg.max_probes);
        assert!(
            got.tolerance() >= 3,
            "n={} s*={} tolerates only {}",
            got.n,
            got.s_star,
            got.tolerance()
        );
    }

    /// 77 rows are too few for `k*_alpha = 86`, but a shared filler pool still
    /// seats a useful top rung.
    #[test]
    fn a_tiny_relation_still_certifies_a_useful_top() {
        let got = feasible(&sembench_cfg(), 86, 77).expect("77 rows seat a design");
        assert!(
            *got.ladder.first().unwrap() >= 16,
            "77 rows should seat more than k={}",
            got.ladder[0]
        );
        assert!(got.min_items() <= 77);
    }

    /// With `k*_alpha = 1960` pairs the sample is far below `n * 1960`, but the
    /// shared pool still seats the top rung.
    #[test]
    fn a_huge_k_star_still_finds_a_rung() {
        let got = feasible(&sembench_cfg(), 1960, 4002).expect("4002 pairs certify something");
        assert_eq!(*got.ladder.first().unwrap(), 1960, "the top is affordable");
        assert!(got.min_items() <= 4002);
    }

    /// Below the smallest affordable design there is nothing to measure.
    #[test]
    fn a_sample_too_small_for_any_rung_is_none() {
        assert!(feasible(&sembench_cfg(), 86, 10).is_none());
        assert!(feasible(&sembench_cfg(), 86, 0).is_none());
        // phi(1) = 1 by definition: no ladder, nothing to certify.
        assert!(feasible(&sembench_cfg(), 1, 10_000).is_none());
    }

    /// More data never buys a worse certificate: neither a smaller top nor
    /// fewer probes.
    #[test]
    fn more_rows_never_weakens_the_budget() {
        let cfg = sembench_cfg();
        let mut prev: Option<Budget> = None;
        for rows in [80usize, 200, 500, 1000, 2000, 4000, 16_000] {
            let Some(got) = feasible(&cfg, 86, rows) else {
                continue;
            };
            assert!(got.min_items() <= rows, "design overruns the sample");
            if let Some(prev) = &prev {
                assert!(
                    got.ladder[0] >= prev.ladder[0],
                    "{rows} rows chose a smaller top than fewer rows did"
                );
                if got.ladder[0] == prev.ladder[0] {
                    assert!(got.n >= prev.n, "{rows} rows bought fewer probes");
                }
            }
            prev = Some(got);
        }
        assert!(prev.is_some());
    }

    /// The probe ceiling bounds cost without cutting into the requested
    /// tolerance budget.
    #[test]
    fn max_probes_caps_growth_but_never_the_request() {
        let cfg = sembench_cfg();
        let generous = feasible(&cfg, 86, 16_000).unwrap();
        assert!(generous.n <= cfg.max_probes);

        let capped = feasible_budget(86, cfg.delta, cfg.eta, Some(1), 1, 16_000).unwrap();
        let requested = budget_for(&QualityCertConfig { ..cfg.clone() }, capped.ladder.len(), 1)
            .unwrap()
            .1;
        assert_eq!(
            capped.n, requested,
            "a ceiling of 1 must not shrink the requested budget"
        );
    }

    /// Builds a spec from `feasible_budget` against the operator's input
    /// estimate.
    fn planned_spec(
        cfg: &QualityCertConfig,
        k_star: usize,
        input_rows: usize,
    ) -> CertificationSpec {
        let budget = feasible(cfg, k_star, input_rows).expect("a feasible budget");
        CertificationSpec {
            ladder: budget.ladder,
            eta: cfg.eta,
            delta_prime: budget.delta_prime,
            n: budget.n,
            s_star: budget.s_star,
            seed: 42,
            cache_key: CertKey {
                provider: "mock".into(),
                model_id: "m".into(),
                operator_kind: OperatorKind::Selection,
                template_hash: 1,
                packing_policy: PackingPolicy::Flat,
            },
            delta_share: cfg.delta,
            tolerated: cfg.tolerated_disagreements,
            max_probes: cfg.max_probes,
        }
    }

    /// `retarget` only ever weakens: same seed, same key, never a higher top.
    #[test]
    fn retarget_never_raises_the_top() {
        let cfg = sembench_cfg();
        let spec = planned_spec(&cfg, 16, 1000);
        assert_eq!(spec.ladder[0], 16);

        // Far more rows than planned: the top stays where it is.
        let wide = spec.retarget(1_000_000).unwrap();
        assert_eq!(wide.ladder[0], 16);
        assert_eq!(wide.seed, spec.seed);
        assert_eq!(wide.cache_key, spec.cache_key);

        // A thin sample drops the top rather than giving up.
        let thin = spec.retarget(40).unwrap();
        assert!(thin.ladder[0] < 16);
        assert!(thin.min_items() <= 40);
    }

    /// The pool is sized from the plan's own `sample_target`, and the runtime
    /// then retargets against what it gathered. Those two must agree, or a run
    /// would spend a different probe budget than the plan states.
    #[test]
    fn retarget_against_the_planned_sample_is_a_no_op() {
        let cfg = sembench_cfg();
        for (k_star, input_rows) in [(16usize, 1000usize), (86, 1000), (12, 4000), (200, 700)] {
            let spec = planned_spec(&cfg, k_star, input_rows);
            let same = spec
                .retarget(spec.sample_target())
                .expect("the planned sample must remain feasible");
            assert_eq!(same.ladder, spec.ladder, "k*={k_star} rows={input_rows}");
            assert_eq!(same.n, spec.n, "k*={k_star} rows={input_rows}");
            assert_eq!(same.s_star, spec.s_star, "k*={k_star} rows={input_rows}");
        }
    }
}
