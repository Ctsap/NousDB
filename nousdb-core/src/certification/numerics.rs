//! Exact binomial confidence machinery for certification decisions.
//!
//! No external statistics crate: the one-sided Clopper–Pearson lower bound
//! `BetaInv(delta_prime; x, n - x + 1)` is computed by bisecting the binomial
//! survival function `S(p) = Pr_p[X >= x]` (strictly increasing in `p`), with
//! terms evaluated in log space from an iteratively summed ln-factorial
//! table. The result is exact for every `n`, with no normal approximation.
//! The golden vectors in this module's tests are checked against scipy's
//! `beta.ppf`.

/// Cached `ln(i!)` table, grown on demand.
///
/// Callers evaluate the bound many times over the same or increasing `n`
/// (binary search in [`s_star`], linear scan in `budget_for`'s tolerance
/// solver), so sharing one table avoids rebuilding it per call.
pub(crate) struct LnFactorial {
    table: Vec<f64>,
}

impl LnFactorial {
    pub(crate) fn new() -> Self {
        Self { table: vec![0.0] }
    }

    /// Extends the table so `ln(i!)` is available for all `i <= n`.
    fn ensure(&mut self, n: usize) {
        while self.table.len() <= n {
            let i = self.table.len();
            self.table.push(self.table[i - 1] + (i as f64).ln());
        }
    }
}

/// One-sided Clopper–Pearson lower confidence bound at level `1 - delta_prime`.
///
/// Guarantees `Pr[result <= true_fidelity] >= 1 - delta_prime` for every true
/// fidelity and every `n` (test inversion; conservative only in the safe
/// direction).
///
/// Special cases: `x == 0` returns `0.0` (Beta(0, ·) is undefined); `x == n`
/// uses the closed form `delta_prime^(1/n)`, which gives the probe-budget
/// floor in [`n_min`].
///
/// # Panics
/// If `n == 0`, `x > n`, or `delta_prime` is outside `(0, 1)`.
pub fn cp_lower_bound(x: u64, n: u64, delta_prime: f64) -> f64 {
    cp_lower_bound_cached(x, n, delta_prime, &mut LnFactorial::new())
}

/// [`cp_lower_bound`] against a caller-owned factorial table.
pub(crate) fn cp_lower_bound_cached(
    x: u64,
    n: u64,
    delta_prime: f64,
    ln_fact: &mut LnFactorial,
) -> f64 {
    assert!(n > 0, "cp_lower_bound: n must be positive");
    assert!(x <= n, "cp_lower_bound: x ({x}) must not exceed n ({n})");
    assert!(
        delta_prime > 0.0 && delta_prime < 1.0,
        "cp_lower_bound: delta_prime must lie in (0, 1), got {delta_prime}"
    );

    if x == 0 {
        return 0.0;
    }
    if x == n {
        return delta_prime.powf(1.0 / n as f64);
    }

    ln_fact.ensure(n as usize);
    let ln_fact = &ln_fact.table;

    // S(p) is 0 at p -> 0 and 1 at p -> 1; bisect for S(p) = delta_prime.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if binomial_survival(x, n, mid, ln_fact) < delta_prime {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Minimum passing agreement count: the smallest `x` with
/// `cp_lower_bound(x, n, delta_prime) >= threshold`, or `None` if even a
/// perfect run (`x == n`) cannot pass.
///
/// Curtailment rule: a rung is futile, and can be aborted, as soon as
/// `disagreements > n - s_star`.
pub fn s_star(n: u64, delta_prime: f64, threshold: f64) -> Option<u64> {
    s_star_cached(n, delta_prime, threshold, &mut LnFactorial::new())
}

/// [`s_star`] against a caller-owned factorial table.
pub(crate) fn s_star_cached(
    n: u64,
    delta_prime: f64,
    threshold: f64,
    ln_fact: &mut LnFactorial,
) -> Option<u64> {
    if cp_lower_bound_cached(n, n, delta_prime, ln_fact) < threshold {
        return None;
    }
    // cp_lower_bound is nondecreasing in x: binary-search the first passer.
    let mut lo = 0u64;
    let mut hi = n; // invariant: cp_lower_bound(hi, ..) >= threshold
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cp_lower_bound_cached(mid, n, delta_prime, ln_fact) >= threshold {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(hi)
}

/// Probe-budget floor: the smallest `n` for which a perfect run can certify,
/// `ceil( ln(1/delta_prime) / ln(1/(1 - eta)) )`.
///
/// At this floor a single disagreement voids the rung. Slack is requested with
/// `QualityCertConfig::tolerated_disagreements`.
///
/// # Panics
/// If `delta_prime` or `eta` is outside `(0, 1)`.
pub fn n_min(delta_prime: f64, eta: f64) -> u64 {
    assert!(
        delta_prime > 0.0 && delta_prime < 1.0,
        "n_min: delta_prime must lie in (0, 1), got {delta_prime}"
    );
    assert!(
        eta > 0.0 && eta < 1.0,
        "n_min: eta must lie in (0, 1), got {eta}"
    );
    ((1.0 / delta_prime).ln() / (1.0 / (1.0 - eta)).ln()).ceil() as u64
}

/// `Pr[X >= x]` for `X ~ Binomial(n, p)`, summed in log space.
/// Requires `1 <= x <= n` and `p` in the open interval `(0, 1)`.
fn binomial_survival(x: u64, n: u64, p: f64, ln_fact: &[f64]) -> f64 {
    let ln_p = p.ln();
    let ln_q = (-p).ln_1p(); // ln(1 - p), accurate for small p
    let mut sum = 0.0_f64;
    for j in x..=n {
        let ln_term = ln_fact[n as usize] - ln_fact[j as usize] - ln_fact[(n - j) as usize]
            + j as f64 * ln_p
            + (n - j) as f64 * ln_q;
        sum += ln_term.exp();
    }
    sum.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification::rng::SplitMix64;
    use std::collections::HashMap;

    /// Golden vectors checked against scipy
    /// `beta.ppf(delta_prime, x, n - x + 1)`, threshold `1 - eta = 0.95`.
    const GOLDEN: &[(u64, u64, f64, f64)] = &[
        (0, 90, 0.01, 0.0),
        (45, 90, 0.01, 0.374227),
        (81, 90, 0.01, 0.802501),
        (89, 90, 0.01, 0.928509),
        (90, 90, 0.01, 0.950119),
        (148, 150, 0.01, 0.945144),
        (149, 150, 0.01, 0.956567),
        (150, 150, 0.01, 0.969765),
        (94, 94, 1.0 / 120.0, 0.950344),
        (119, 120, 1.0 / 120.0, 0.944306),
        (133, 134, 1.0 / 120.0, 0.949998),
        (134, 135, 1.0 / 120.0, 0.950361),
        (168, 170, 1.0 / 120.0, 0.950181),
    ];

    #[test]
    fn golden_vectors_match_reference() {
        for &(x, n, dp, expected) in GOLDEN {
            let got = cp_lower_bound(x, n, dp);
            assert!(
                (got - expected).abs() < 1e-6,
                "cp_lower_bound({x}, {n}, {dp}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn golden_vector_decisions_at_095() {
        // The certify/reject decision at threshold 0.95 must match the golden
        // vectors exactly, including the (133, 134) case that fails by 2e-6.
        let decisions: &[(u64, u64, f64, bool)] = &[
            (89, 90, 0.01, false),
            (90, 90, 0.01, true),
            (148, 150, 0.01, false),
            (149, 150, 0.01, true),
            (94, 94, 1.0 / 120.0, true),
            (119, 120, 1.0 / 120.0, false),
            (133, 134, 1.0 / 120.0, false),
            (134, 135, 1.0 / 120.0, true),
            (168, 170, 1.0 / 120.0, true),
        ];
        for &(x, n, dp, pass) in decisions {
            assert_eq!(
                cp_lower_bound(x, n, dp) >= 0.95,
                pass,
                "decision mismatch for ({x}, {n}, {dp})"
            );
        }
    }

    #[test]
    fn s_star_matches_reference() {
        assert_eq!(s_star(90, 0.01, 0.95), Some(90));
        assert_eq!(s_star(150, 0.01, 0.95), Some(149));
        assert_eq!(s_star(94, 1.0 / 120.0, 0.95), Some(94));
        assert_eq!(s_star(135, 1.0 / 120.0, 0.95), Some(134));
        assert_eq!(s_star(170, 1.0 / 120.0, 0.95), Some(168));
        // Below the probe-budget floor no count can pass.
        assert_eq!(s_star(50, 0.01, 0.95), None);
    }

    #[test]
    fn n_min_matches_reference() {
        assert_eq!(n_min(0.01, 0.05), 90);
        assert_eq!(n_min(1.0 / 120.0, 0.05), 94);
        // eta = delta = 0.10, three-rung ladder.
        assert_eq!(n_min(1.0 / 30.0, 0.10), 33);
    }

    #[test]
    fn s_star_self_consistent_at_demo_budget() {
        // For any n >= n_min the returned s* is the exact decision boundary.
        for n in [33u64, 40, 50] {
            let dp = 1.0 / 30.0;
            let s = s_star(n, dp, 0.90).expect("n >= n_min must admit a passer");
            assert!(cp_lower_bound(s, n, dp) >= 0.90);
            if s > 0 {
                assert!(cp_lower_bound(s - 1, n, dp) < 0.90);
            }
        }
    }

    /// Reusing one table across calls, including a table grown for a large `n`
    /// on a smaller one, gives bit-identical results to a fresh table per call.
    #[test]
    fn cached_table_matches_fresh_table() {
        let mut ln_fact = LnFactorial::new();
        for n in [150u64, 33, 90, 7] {
            for x in [0, 1, n / 2, n - 1, n] {
                assert_eq!(
                    cp_lower_bound_cached(x, n, 0.01, &mut ln_fact),
                    cp_lower_bound(x, n, 0.01),
                    "mismatch at x={x} n={n}"
                );
            }
            assert_eq!(
                s_star_cached(n, 0.01, 0.95, &mut ln_fact),
                s_star(n, 0.01, 0.95),
                "s_star mismatch at n={n}"
            );
        }
    }

    #[test]
    fn bound_is_monotone_in_x() {
        let mut prev = -1.0;
        for x in 0..=90 {
            let lb = cp_lower_bound(x, 90, 0.01);
            assert!(lb >= prev, "cp_lower_bound not monotone at x={x}");
            prev = lb;
        }
    }

    fn run_coverage(sims: usize) {
        // Coverage: for every true fidelity phi,
        // Pr[cp_lower_bound(X, n, dp) > phi] <= dp.
        let dp = 0.01;
        for &phi in &[0.90_f64, 0.949, 0.95, 0.97] {
            for &n in &[90u64, 150] {
                let seed = 0xC0FFEE ^ n ^ phi.to_bits();
                let mut rng = SplitMix64::new(seed);
                let mut memo: HashMap<u64, f64> = HashMap::new();
                let mut violations = 0usize;
                for _ in 0..sims {
                    let mut x = 0u64;
                    for _ in 0..n {
                        if rng.next_f64() < phi {
                            x += 1;
                        }
                    }
                    let lb = *memo.entry(x).or_insert_with(|| cp_lower_bound(x, n, dp));
                    if lb > phi {
                        violations += 1;
                    }
                }
                let rate = violations as f64 / sims as f64;
                let sigma = (dp * (1.0 - dp) / sims as f64).sqrt();
                assert!(
                    rate <= dp + 3.0 * sigma,
                    "coverage violated: phi={phi} n={n} rate={rate}"
                );
            }
        }
    }

    #[test]
    fn coverage_property_10k() {
        run_coverage(10_000);
    }

    #[test]
    #[ignore = "heavyweight statistical check; run explicitly with --ignored"]
    fn coverage_property_100k() {
        run_coverage(100_000);
    }
}
