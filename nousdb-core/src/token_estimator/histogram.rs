//! Equi-depth histograms of per-row token counts.

/// One bucket of a [`TokenHistogram`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistogramBucket {
    /// Smallest token count in the bucket.
    pub tau_min: usize,
    /// Largest token count in the bucket.
    pub tau_max: usize,
    /// Number of distinct token counts in the bucket.
    pub distinct_count: usize,
    /// Number of rows in the bucket.
    pub tuple_count: usize,
    /// Mean token count of the rows in the bucket.
    pub average_token_count: f64,
}

/// Summary statistics of a [`TokenHistogram`].
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Mean tokens per row.
    pub mean: f64,
    /// Variance of tokens per row, assuming values are uniform within a
    /// bucket.
    pub variance: f64,
    /// Total tokens across all rows.
    pub total_tokens: f64,
    /// Number of rows summarized.
    pub total_rows: usize,
}

/// An equi-depth histogram of per-row token counts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenHistogram {
    buckets: Vec<HistogramBucket>,
    requested_bucket_count: usize,
    total_rows: usize,
}

impl TokenHistogram {
    /// Builds a histogram with at most `bucket_count` buckets (at least one)
    /// from per-row token counts. Equal values are never split across buckets.
    pub fn from_token_counts(token_counts: &[usize], bucket_count: usize) -> Self {
        if token_counts.is_empty() {
            return Self {
                buckets: Vec::new(),
                requested_bucket_count: bucket_count.max(1),
                total_rows: 0,
            };
        }

        let bucket_count = bucket_count.max(1);

        // Sorted distribution as (token_count, frequency) pairs.
        let mut sorted = token_counts.to_vec();
        sorted.sort_unstable();

        let mut distribution: Vec<(usize, usize)> = Vec::new();
        for token in sorted {
            match distribution.last_mut() {
                Some((last_token, freq)) if *last_token == token => {
                    *freq += 1;
                }
                _ => distribution.push((token, 1)),
            }
        }

        let total_rows = token_counts.len();
        let target_depth = total_rows as f64 / bucket_count as f64;

        let mut buckets = Vec::new();
        let mut current_values: Vec<(usize, usize)> = Vec::new();
        let mut current_freq = 0usize;

        for (token, freq) in distribution {
            let projected_freq = current_freq + freq;

            if !current_values.is_empty()
                && projected_freq as f64 > target_depth
                && buckets.len() + 1 < bucket_count
            {
                buckets.push(Self::build_bucket(&current_values));
                current_values.clear();
                current_freq = 0;
            }

            current_values.push((token, freq));
            current_freq += freq;
        }

        if !current_values.is_empty() {
            buckets.push(Self::build_bucket(&current_values));
        }

        Self {
            buckets,
            requested_bucket_count: bucket_count,
            total_rows,
        }
    }

    fn build_bucket(values: &[(usize, usize)]) -> HistogramBucket {
        let tau_min = values.first().map(|(t, _)| *t).unwrap_or(0);
        let tau_max = values.last().map(|(t, _)| *t).unwrap_or(0);
        let distinct_count = values.len();
        let tuple_count: usize = values.iter().map(|(_, f)| *f).sum();
        let weighted_sum: usize = values.iter().map(|(t, f)| t * f).sum();
        let average_token_count = if tuple_count == 0 {
            0.0
        } else {
            weighted_sum as f64 / tuple_count as f64
        };

        HistogramBucket {
            tau_min,
            tau_max,
            distinct_count,
            tuple_count,
            average_token_count,
        }
    }

    /// Estimates the `p`-quantile (clamped to `[0, 1]`) of the token-count
    /// distribution, interpolating linearly within a bucket. Returns `0.0` for
    /// an empty histogram.
    pub fn quantile(&self, p: f64) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let p = p.clamp(0.0, 1.0);
        if p <= 0.0 {
            return self.buckets[0].tau_min as f64;
        }

        if p >= 1.0 {
            return self.buckets[self.buckets.len() - 1].tau_max as f64;
        }

        let beta = self.buckets.len() as f64;
        let j = (p * beta).ceil().max(1.0) as usize;
        let bucket_idx = j.saturating_sub(1).min(self.buckets.len() - 1);
        let bucket = &self.buckets[bucket_idx];

        let interpolation = (p * beta - (j.saturating_sub(1)) as f64).clamp(0.0, 1.0);

        bucket.tau_min as f64
            + interpolation * (bucket.tau_max.saturating_sub(bucket.tau_min) as f64)
    }

    /// Mean, variance and totals of the histogram.
    pub fn stats(&self) -> HistogramStats {
        if self.total_rows == 0 {
            return HistogramStats {
                mean: 0.0,
                variance: 0.0,
                total_tokens: 0.0,
                total_rows: 0,
            };
        }

        let n = self.total_rows as f64;

        let mean = self
            .buckets
            .iter()
            .map(|b| b.tuple_count as f64 * b.average_token_count)
            .sum::<f64>()
            / n;

        let variance = self
            .buckets
            .iter()
            .map(|b| {
                let within = ((b.tau_max as f64 - b.tau_min as f64).powi(2)) / 12.0;
                let between = (b.average_token_count - mean).powi(2);
                (b.tuple_count as f64 / n) * (within + between)
            })
            .sum::<f64>();

        let total_tokens = n * mean;

        HistogramStats {
            mean,
            variance,
            total_tokens,
            total_rows: self.total_rows,
        }
    }

    /// The buckets, in ascending order of token count.
    pub fn buckets(&self) -> &[HistogramBucket] {
        &self.buckets
    }

    /// Number of rows the histogram was built from.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// The bucket count requested at construction (the actual count may be
    /// lower).
    pub fn requested_bucket_count(&self) -> usize {
        self.requested_bucket_count
    }

    /// Returns a histogram with `new_value = old_value * multiplier + addend`
    /// applied to all bucket boundaries and averages. Used to convert per-row
    /// histograms to per-pair histograms for join batch sizing.
    pub fn scale_and_shift(&self, multiplier: f64, addend: f64) -> Self {
        let buckets = self
            .buckets
            .iter()
            .map(|b| {
                let new_min = (b.tau_min as f64 * multiplier + addend).round() as usize;
                let new_max = (b.tau_max as f64 * multiplier + addend).round() as usize;
                let new_avg = b.average_token_count * multiplier + addend;
                HistogramBucket {
                    tau_min: new_min,
                    tau_max: new_max,
                    distinct_count: b.distinct_count,
                    tuple_count: b.tuple_count,
                    average_token_count: new_avg,
                }
            })
            .collect();
        Self {
            buckets,
            requested_bucket_count: self.requested_bucket_count,
            total_rows: self.total_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_empty_input() {
        let hist = TokenHistogram::from_token_counts(&[], 32);
        assert_eq!(hist.total_rows(), 0);
        assert!(hist.buckets().is_empty());
        assert_eq!(hist.quantile(0.9), 0.0);
    }

    #[test]
    fn test_histogram_quantile_monotonicity() {
        let counts = vec![5, 5, 6, 7, 12, 20, 30, 60, 120, 240];
        let hist = TokenHistogram::from_token_counts(&counts, 4);

        let q50 = hist.quantile(0.5);
        let q90 = hist.quantile(0.9);
        let q99 = hist.quantile(0.99);

        assert!(q50 <= q90);
        assert!(q90 <= q99);
    }

    #[test]
    fn test_histogram_stats_reasonable() {
        let counts = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let hist = TokenHistogram::from_token_counts(&counts, 4);
        let stats = hist.stats();

        assert!(stats.mean > 0.0);
        assert!(stats.variance >= 0.0);
        assert!(stats.total_tokens > 0.0);
        assert_eq!(stats.total_rows, counts.len());
    }

    #[test]
    fn test_scale_and_shift() {
        let counts = vec![10, 20, 30, 40, 50];
        let hist = TokenHistogram::from_token_counts(&counts, 2);
        let original_stats = hist.stats();

        // Apply scale=2.0, shift=30 (simulates per-row → per-pair transform)
        let transformed = hist.scale_and_shift(2.0, 30.0);

        // Bucket count and row count should be preserved
        assert_eq!(transformed.buckets().len(), hist.buckets().len());
        assert_eq!(transformed.total_rows(), hist.total_rows());

        // Each bucket boundary and average should be scaled and shifted
        for (orig, trans) in hist.buckets().iter().zip(transformed.buckets().iter()) {
            let expected_min = (orig.tau_min as f64 * 2.0 + 30.0).round() as usize;
            let expected_max = (orig.tau_max as f64 * 2.0 + 30.0).round() as usize;
            assert_eq!(trans.tau_min, expected_min);
            assert_eq!(trans.tau_max, expected_max);
            assert!(
                (trans.average_token_count - (orig.average_token_count * 2.0 + 30.0)).abs() < 1e-9
            );
            // Counts should be unchanged
            assert_eq!(trans.distinct_count, orig.distinct_count);
            assert_eq!(trans.tuple_count, orig.tuple_count);
        }

        // The transformed mean should be approximately 2*original_mean + 30
        let transformed_stats = transformed.stats();
        let expected_mean = original_stats.mean * 2.0 + 30.0;
        assert!((transformed_stats.mean - expected_mean).abs() < 1e-9);
    }

    #[test]
    fn test_scale_and_shift_identity() {
        let counts = vec![10, 20, 30];
        let hist = TokenHistogram::from_token_counts(&counts, 2);
        let identity = hist.scale_and_shift(1.0, 0.0);

        for (orig, trans) in hist.buckets().iter().zip(identity.buckets().iter()) {
            assert_eq!(orig.tau_min, trans.tau_min);
            assert_eq!(orig.tau_max, trans.tau_max);
            assert!((orig.average_token_count - trans.average_token_count).abs() < 1e-9);
        }
    }

    #[test]
    fn test_equi_depth_skewed_distribution() {
        let mut counts = vec![10; 90];
        counts.extend(vec![1000; 10]);

        let hist = TokenHistogram::from_token_counts(&counts, 5);
        assert!(!hist.buckets().is_empty());

        let total: usize = hist.buckets().iter().map(|b| b.tuple_count).sum();
        assert_eq!(total, counts.len());
    }
}
