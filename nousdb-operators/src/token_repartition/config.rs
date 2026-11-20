//! Configuration types for token-aware repartitioning.

/// Strategy for partitioning rows across partitions in token-aware repartitioning.
///
/// The strategies trade token balance against row distribution:
///
/// | Strategy | Token Balance | Row Distribution |
/// |----------|---------------|------------------|
/// | OptimalTokenBalance (default) | Best | Varies by tokens |
/// | EqualRowCounts | Good | Equal across partitions |
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionStrategy {
    /// Greedy bin-packing for token balance: each row goes to the partition
    /// with the fewest accumulated tokens. Row counts vary per partition.
    OptimalTokenBalance,

    /// Zigzag round-robin assignment for equal row counts across partitions.
    ///
    /// Uses `assign_rows_zigzag()` on the full wave working set: each wave collects
    /// one batch from every live input partition, merges them, then applies the zigzag
    /// sort-and-assign strategy on the merged rows. The partitioner state is reset
    /// between waves so each wave gets an independent zigzag pass.
    ///
    /// Each partition gets ⌈N/P⌉ rows per wave and emits one batch per wave.
    /// Suited to workloads that need predictable partition sizes.
    EqualRowCounts,
}

impl Default for PartitionStrategy {
    /// Returns the default strategy, `OptimalTokenBalance`, which prioritizes
    /// token balance.
    fn default() -> Self {
        Self::OptimalTokenBalance
    }
}

impl PartitionStrategy {
    /// Returns true if this strategy requires full materialization before emitting.
    ///
    /// With wave-barrier synchronization neither strategy accumulates: the
    /// coordinator collects one batch per live input partition (a "wave"),
    /// merges them, applies the strategy to the merged rows, and sends the
    /// results to the output channels immediately.
    pub fn requires_accumulation(&self) -> bool {
        false
    }

    /// Returns the target batch size for accumulation, if applicable.
    ///
    /// Always `None`: batch sizing for model calls belongs to the downstream
    /// operators (Extend, Selection), not to the partitioning strategy.
    pub fn target_batch_size(&self) -> Option<usize> {
        None
    }
}

/// Configuration for token-based partitioning.
///
/// Partitions by token counts rather than row counts, in one of two modes:
/// - fixed number of partitions (tokens distributed evenly)
/// - fixed token budget per partition (partition count derived from the total)
#[derive(Debug, Clone)]
pub struct TokenPartitionConfig {
    /// Columns to estimate tokens from.
    /// Only string/text columns should be specified here.
    pub token_columns: Vec<String>,

    /// Target number of partitions (optional).
    /// If specified, total tokens will be divided evenly across partitions.
    pub num_partitions: Option<usize>,

    /// Maximum tokens per partition (optional).
    /// If specified, partitions are sized to not exceed this budget.
    pub token_budget: Option<usize>,

    /// Partitioning strategy (default: OptimalTokenBalance).
    /// Determines how rows are assigned to partitions and how batches are emitted.
    pub strategy: PartitionStrategy,
}

impl TokenPartitionConfig {
    /// Create a config targeting a specific number of partitions.
    pub fn with_num_partitions(token_columns: Vec<String>, num_partitions: usize) -> Self {
        Self {
            token_columns,
            num_partitions: Some(num_partitions),
            token_budget: None,
            strategy: PartitionStrategy::default(),
        }
    }

    /// Create a config with a maximum token budget per partition.
    pub fn with_token_budget(token_columns: Vec<String>, token_budget: usize) -> Self {
        Self {
            token_columns,
            num_partitions: None,
            token_budget: Some(token_budget),
            strategy: PartitionStrategy::default(),
        }
    }

    /// Create a config with both constraints: at most `num_partitions`
    /// partitions, each with at most `token_budget` tokens.
    pub fn with_both(
        token_columns: Vec<String>,
        num_partitions: usize,
        token_budget: usize,
    ) -> Self {
        Self {
            token_columns,
            num_partitions: Some(num_partitions),
            token_budget: Some(token_budget),
            strategy: PartitionStrategy::default(),
        }
    }

    /// Set a custom partitioning strategy (builder pattern).
    pub fn with_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Compute the effective number of partitions and token budget given total tokens.
    ///
    /// Returns `(num_partitions, budget_per_partition)`.
    pub fn compute_partition_params(&self, total_tokens: usize) -> (usize, usize) {
        match (self.num_partitions, self.token_budget) {
            (Some(n), Some(budget)) => {
                // Both specified: use both as constraints
                (n, budget)
            }
            (Some(n), None) => {
                // Only num_partitions: divide tokens evenly
                let budget = total_tokens.div_ceil(n);
                (n, budget)
            }
            (None, Some(budget)) => {
                // Only budget: compute number of partitions
                let n = total_tokens.div_ceil(budget);
                (n.max(1), budget)
            }
            (None, None) => {
                // Neither specified: default to 1 partition
                (1, total_tokens)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_params_with_num_partitions() {
        let config = TokenPartitionConfig::with_num_partitions(vec!["text".into()], 4);
        let (n, budget) = config.compute_partition_params(1000);
        assert_eq!(n, 4);
        assert_eq!(budget, 250);
    }

    #[test]
    fn test_compute_params_with_budget() {
        let config = TokenPartitionConfig::with_token_budget(vec!["text".into()], 300);
        let (n, budget) = config.compute_partition_params(1000);
        assert_eq!(n, 4); // ceil(1000/300) = 4
        assert_eq!(budget, 300);
    }

    #[test]
    fn test_compute_params_with_both() {
        let config = TokenPartitionConfig::with_both(vec!["text".into()], 8, 200);
        let (n, budget) = config.compute_partition_params(1000);
        assert_eq!(n, 8);
        assert_eq!(budget, 200);
    }

    #[test]
    fn test_default_strategy() {
        let strategy = PartitionStrategy::default();
        assert!(matches!(strategy, PartitionStrategy::OptimalTokenBalance));
    }

    #[test]
    fn test_requires_accumulation() {
        let optimal = PartitionStrategy::OptimalTokenBalance;
        let equal = PartitionStrategy::EqualRowCounts;

        // Wave-based streaming: no accumulation needed
        assert!(!optimal.requires_accumulation());
        assert!(!equal.requires_accumulation());
    }

    #[test]
    fn test_target_batch_size() {
        let optimal = PartitionStrategy::OptimalTokenBalance;
        let equal = PartitionStrategy::EqualRowCounts;

        // Neither strategy owns batch sizing; downstream operators do.
        assert_eq!(optimal.target_batch_size(), None);
        assert_eq!(equal.target_batch_size(), None);
    }

    #[test]
    fn test_with_strategy() {
        let config = TokenPartitionConfig::with_num_partitions(vec!["text".into()], 4)
            .with_strategy(PartitionStrategy::EqualRowCounts);

        assert_eq!(config.strategy, PartitionStrategy::EqualRowCounts);
    }

    #[test]
    fn test_default_config_uses_default_strategy() {
        let config = TokenPartitionConfig::with_num_partitions(vec!["text".into()], 4);
        assert!(matches!(
            config.strategy,
            PartitionStrategy::OptimalTokenBalance
        ));
    }
}
