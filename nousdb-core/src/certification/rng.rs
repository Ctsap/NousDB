//! Deterministic, dependency-free randomness for probe/slot designation.
//!
//! Calibration validity requires all randomness (probe rows, batch slots) to
//! be drawn before any model output is observed, and reproducibility requires
//! the draws to be a pure function of the configured seed, so this module
//! provides a small generator instead of depending on `rand`.

/// SplitMix64 (Steele, Lea & Flood): full-period 64-bit generator, ideal for
/// short deterministic designation sequences.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator whose sequence is fully determined by `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in `0..bound` with rejection sampling (no modulo bias).
    ///
    /// # Panics
    /// If `bound == 0`.
    pub fn uniform(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "SplitMix64::uniform: bound must be positive");
        if bound == 1 {
            return 0;
        }
        let bound64 = bound as u64;
        // Largest multiple of `bound` representable; values >= zone are
        // rejected so the remainder is exactly uniform.
        let zone = (u64::MAX / bound64) * bound64;
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % bound64) as usize;
            }
        }
    }

    /// Uniform f64 in `[0, 1)` from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// FNV-1a 64-bit over a sequence of string parts, with a 0x1F unit separator
/// terminating every part so `["ab", "c"]` / `["a", "bc"]` and `[]` / `[""]`
/// all hash differently.
///
/// Used for certificate-cache keys: the result is stable across processes,
/// unlike `std::collections::hash_map::DefaultHasher`.
pub fn fnv1a_64(parts: &[&str]) -> u64 {
    const OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = OFFSET_BASIS;
    for part in parts {
        for &b in part.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(PRIME);
        }
        hash ^= 0x1F_u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix_reference_values() {
        // First outputs for seed 1234567, cross-checked against the
        // canonical C implementation of splitmix64.
        let mut rng = SplitMix64::new(1234567);
        let first: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(first[0], 6457827717110365317);
        assert_eq!(first[1], 3203168211198807973);
        assert_eq!(first[2], 9817491932198370423);
    }

    #[test]
    fn uniform_stays_in_bounds_and_covers() {
        let mut rng = SplitMix64::new(7);
        let mut seen = [false; 16];
        for _ in 0..10_000 {
            let v = rng.uniform(16);
            assert!(v < 16);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s), "all residues should appear");
        assert_eq!(rng.uniform(1), 0);
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn fnv_separator_disambiguates() {
        assert_ne!(fnv1a_64(&["ab", "c"]), fnv1a_64(&["a", "bc"]));
        assert_eq!(fnv1a_64(&["x", "y"]), fnv1a_64(&["x", "y"]));
        assert_ne!(fnv1a_64(&[]), fnv1a_64(&[""]));
    }
}
