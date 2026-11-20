//! Row-to-partition assignment by token count.

/// Token-based partitioner using greedy bin-packing or zigzag assignment.
///
/// Supports two strategies:
/// - **Greedy bin-packing**: Assigns rows to minimize token imbalance (default)
/// - **Zigzag assignment**: Assigns rows to ensure equal row counts per partition
#[derive(Debug, Clone)]
pub struct TokenPartitioner {
    /// Number of partitions to create
    num_partitions: usize,
    /// Maximum tokens per partition (optional constraint)
    token_budget: Option<usize>,
    /// Current token count for each partition
    partition_tokens: Vec<usize>,
    /// Current row count for each partition (used in zigzag strategy)
    partition_row_counts: Vec<usize>,
    /// Row assignments: partition index for each row
    row_assignments: Vec<usize>,
}

impl TokenPartitioner {
    /// Create a new partitioner.
    ///
    /// # Arguments
    /// * `num_partitions` - Number of partitions to create
    /// * `token_budget` - Optional maximum tokens per partition
    pub fn new(num_partitions: usize, token_budget: Option<usize>) -> Self {
        Self {
            num_partitions,
            token_budget,
            partition_tokens: vec![0; num_partitions],
            partition_row_counts: vec![0; num_partitions],
            row_assignments: Vec::new(),
        }
    }

    /// Assign a row to the best partition.
    ///
    /// Uses greedy bin-packing: assigns to the partition with the most
    /// remaining capacity (least current tokens).
    ///
    /// # Arguments
    /// * `tokens` - Token count for this row
    ///
    /// # Returns
    /// The partition index this row was assigned to
    pub fn assign_row(&mut self, tokens: usize) -> usize {
        // Find partition with minimum current tokens
        let (min_partition, min_tokens) = self
            .partition_tokens
            .iter()
            .enumerate()
            .min_by_key(|(_, count)| *count)
            .map(|(idx, count)| (idx, *count))
            .unwrap_or((0, 0));

        // Check if we would exceed the budget
        // If so, we still assign to the partition with most capacity
        // (the budget is a soft constraint in greedy bin-packing)
        let target_partition = if let Some(budget) = self.token_budget {
            if min_tokens + tokens > budget && self.num_partitions > 1 {
                // All partitions would exceed budget, use the one with least tokens
                min_partition
            } else {
                min_partition
            }
        } else {
            min_partition
        };

        // Update partition tokens and record assignment
        self.partition_tokens[target_partition] += tokens;
        self.row_assignments.push(target_partition);

        target_partition
    }

    /// Assign multiple rows at once using greedy bin-packing.
    ///
    /// # Arguments
    /// * `token_counts` - Vector of token counts, one per row
    ///
    /// # Returns
    /// Vector of partition assignments, one per row
    pub fn assign_rows(&mut self, token_counts: &[usize]) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(token_counts.len());
        for &tokens in token_counts {
            assignments.push(self.assign_row(tokens));
        }
        assignments
    }

    /// Assign multiple rows using zigzag round-robin for equal row distribution.
    ///
    /// Sorts rows by token count (descending) and assigns them in a zigzag pattern:
    /// P0 → P1 → P2 → ... → Pn-1 → Pn-1 → ... → P2 → P1 → P0 (repeat)
    ///
    /// This ensures each partition gets approximately ⌈N/P⌉ rows while still
    /// considering token sizes for better balance than pure round-robin.
    ///
    /// # Arguments
    /// * `token_counts` - Vector of token counts, one per row
    ///
    /// # Returns
    /// Vector of partition assignments, one per row
    pub fn assign_rows_zigzag(&mut self, token_counts: &[usize]) -> Vec<usize> {
        let n_rows = token_counts.len();
        if n_rows == 0 {
            return Vec::new();
        }

        // Create (row_idx, token_count) pairs and sort by tokens descending
        let mut indexed_tokens: Vec<(usize, usize)> = token_counts
            .iter()
            .enumerate()
            .map(|(idx, &tokens)| (idx, tokens))
            .collect();
        indexed_tokens.sort_by_key(|&(_, tokens)| std::cmp::Reverse(tokens));

        // Initialize assignments vector with placeholder values
        let mut assignments = vec![0; n_rows];

        // Assign rows in zigzag pattern
        let mut forward = true;
        let mut current_partition = 0;

        for (original_idx, tokens) in indexed_tokens {
            // Assign this row to current partition
            assignments[original_idx] = current_partition;
            self.partition_tokens[current_partition] += tokens;
            self.partition_row_counts[current_partition] += 1;

            // Move to next partition in zigzag pattern
            if self.num_partitions == 1 {
                // Only one partition, all rows go there
                continue;
            }

            if forward {
                if current_partition == self.num_partitions - 1 {
                    // Reached the end: reverse direction, staying at the
                    // boundary for endpoint repetition: ...,P-2,P-1,P-1,P-2,...
                    forward = false;
                } else {
                    current_partition += 1;
                }
            } else {
                if current_partition == 0 {
                    // Reached the beginning: reverse direction, staying at the
                    // boundary for endpoint repetition: ...,1,0,0,1,...
                    forward = true;
                } else {
                    current_partition -= 1;
                }
            }
        }

        // Store assignments for later queries
        self.row_assignments = assignments.clone();

        assignments
    }

    /// Get the partition assignment for a specific row.
    ///
    /// # Arguments
    /// * `row_idx` - The row index (0-based, in order of assignment)
    ///
    /// # Returns
    /// The partition index for this row, or None if not assigned
    pub fn get_row_partition(&self, row_idx: usize) -> Option<usize> {
        self.row_assignments.get(row_idx).copied()
    }

    /// Get all row indices assigned to a specific partition.
    ///
    /// # Arguments
    /// * `partition` - The partition index
    ///
    /// # Returns
    /// Vector of row indices assigned to this partition
    pub fn get_partition_rows(&self, partition: usize) -> Vec<usize> {
        self.row_assignments
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == partition)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get the current token count for each partition.
    pub fn partition_tokens(&self) -> &[usize] {
        &self.partition_tokens
    }

    /// Get the current row count for each partition.
    pub fn partition_row_counts(&self) -> &[usize] {
        &self.partition_row_counts
    }

    /// Get the total number of rows assigned.
    pub fn total_rows(&self) -> usize {
        self.row_assignments.len()
    }

    /// Get the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Assign a single row to the partition with the fewest rows (streaming-compatible).
    ///
    /// Unlike `assign_rows_zigzag()` which requires all rows upfront for global sorting,
    /// this method works incrementally, one row at a time. It assigns to the partition
    /// with the fewest cumulative rows, breaking ties by fewest cumulative tokens.
    ///
    /// Over many batches, this converges to equal row distribution across partitions.
    ///
    /// # Arguments
    /// * `tokens` - Token count for this row
    ///
    /// # Returns
    /// The partition index this row was assigned to
    pub fn assign_row_balanced(&mut self, tokens: usize) -> usize {
        let target = self
            .partition_row_counts
            .iter()
            .zip(self.partition_tokens.iter())
            .enumerate()
            .min_by(|(_, (rows_a, toks_a)), (_, (rows_b, toks_b))| {
                rows_a.cmp(rows_b).then_with(|| toks_a.cmp(toks_b))
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.partition_tokens[target] += tokens;
        self.partition_row_counts[target] += 1;
        self.row_assignments.push(target);

        target
    }

    /// Reset all state between waves.
    ///
    /// Zeros partition token counts, row counts, and clears row assignments.
    /// Used by `EqualRowCounts` strategy which needs a clean slate per wave
    /// (zigzag sorts the current wave's rows, so past state is irrelevant).
    pub fn reset(&mut self) {
        self.partition_tokens.iter_mut().for_each(|t| *t = 0);
        self.partition_row_counts.iter_mut().for_each(|r| *r = 0);
        self.row_assignments.clear();
    }

    /// Calculate the imbalance ratio (max/min partition tokens).
    ///
    /// Returns 1.0 for perfectly balanced partitions, higher values
    /// indicate more imbalance.
    pub fn imbalance_ratio(&self) -> f64 {
        if self.partition_tokens.is_empty() {
            return 1.0;
        }

        let min = *self.partition_tokens.iter().min().unwrap_or(&1);
        let max = *self.partition_tokens.iter().max().unwrap_or(&1);

        if min == 0 {
            if max == 0 { 1.0 } else { f64::INFINITY }
        } else {
            max as f64 / min as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_assignment() {
        let mut partitioner = TokenPartitioner::new(3, None);

        // Assign rows with varying token counts
        let assignments = partitioner.assign_rows(&[100, 200, 150, 50, 300, 100]);

        // All 6 rows should be assigned
        assert_eq!(assignments.len(), 6);

        // Check that partitions are relatively balanced
        let tokens = partitioner.partition_tokens();
        assert!(tokens.iter().all(|&t| t > 0)); // All partitions used
    }

    #[test]
    fn test_imbalance_ratio() {
        let mut partitioner = TokenPartitioner::new(2, None);

        // Assign equal tokens to both partitions
        partitioner.assign_rows(&[100, 100]);

        // Should be well-balanced
        let ratio = partitioner.imbalance_ratio();
        assert!(ratio <= 1.5, "Expected balanced ratio, got {}", ratio);
    }

    #[test]
    fn test_get_partition_rows() {
        let mut partitioner = TokenPartitioner::new(2, None);

        // First row goes to partition 0 (both have 0 tokens)
        partitioner.assign_row(100);
        // Second row goes to partition 1 (partition 0 has 100, partition 1 has 0)
        partitioner.assign_row(50);
        // Third row goes to partition 1 (partition 0 has 100, partition 1 has 50)
        partitioner.assign_row(40);

        let p0_rows = partitioner.get_partition_rows(0);
        let p1_rows = partitioner.get_partition_rows(1);

        assert_eq!(p0_rows, vec![0]);
        assert_eq!(p1_rows, vec![1, 2]);
    }

    #[test]
    fn test_token_budget_constraint() {
        let mut partitioner = TokenPartitioner::new(2, Some(150));

        // First row fits in partition 0
        assert_eq!(partitioner.assign_row(100), 0);
        // Second row would exceed budget in partition 0, goes to partition 1
        assert_eq!(partitioner.assign_row(100), 1);
        // Third row: both partitions have 100 tokens, goes to partition 0 (first min)
        assert_eq!(partitioner.assign_row(50), 0);
    }

    #[test]
    fn test_zigzag_assignment_equal_distribution() {
        let mut partitioner = TokenPartitioner::new(3, None);

        // Assign 6 rows with descending token counts
        let tokens = vec![300, 200, 150, 100, 50, 40];
        let assignments = partitioner.assign_rows_zigzag(&tokens);

        // Verify all rows are assigned
        assert_eq!(assignments.len(), 6);

        // Zigzag pattern with endpoint repetition (boustrophedon):
        // Sorted order (descending): [300, 200, 150, 100, 50, 40]
        // Assignment order: P0, P1, P2, P2, P1, P0
        // So: 300→P0, 200→P1, 150→P2, 100→P2, 50→P1, 40→P0
        assert_eq!(assignments[0], 0, "Row 0 (300 tokens) should go to P0");
        assert_eq!(assignments[1], 1, "Row 1 (200 tokens) should go to P1");
        assert_eq!(assignments[2], 2, "Row 2 (150 tokens) should go to P2");
        assert_eq!(assignments[3], 2, "Row 3 (100 tokens) should go to P2");
        assert_eq!(assignments[4], 1, "Row 4 (50 tokens) should go to P1");
        assert_eq!(assignments[5], 0, "Row 5 (40 tokens) should go to P0");

        // Equal row distribution: each partition gets exactly ⌈6/3⌉ = 2 rows
        let row_counts = partitioner.partition_row_counts();
        assert_eq!(
            row_counts,
            &[2, 2, 2],
            "Zigzag with endpoint repetition gives equal rows"
        );

        // Verify token distribution
        let partition_tokens = partitioner.partition_tokens();
        // P0: 300+40=340, P1: 200+50=250, P2: 150+100=250
        assert_eq!(partition_tokens[0], 340);
        assert_eq!(partition_tokens[1], 250);
        assert_eq!(partition_tokens[2], 250);
    }

    #[test]
    fn test_zigzag_single_partition() {
        let mut partitioner = TokenPartitioner::new(1, None);

        let tokens = vec![100, 200, 150];
        let assignments = partitioner.assign_rows_zigzag(&tokens);

        // All rows should go to partition 0
        assert_eq!(assignments, vec![0, 0, 0]);
        assert_eq!(partitioner.partition_row_counts(), &[3]);
    }

    #[test]
    fn test_zigzag_more_partitions_than_rows() {
        let mut partitioner = TokenPartitioner::new(5, None);

        let tokens = vec![100, 200, 150];
        let assignments = partitioner.assign_rows_zigzag(&tokens);

        // Only 3 rows for 5 partitions
        assert_eq!(assignments.len(), 3);

        // Rows should be distributed across first few partitions
        let row_counts = partitioner.partition_row_counts();
        assert_eq!(row_counts.iter().sum::<usize>(), 3);
    }

    #[test]
    fn test_balanced_assignment_equal_distribution() {
        let mut partitioner = TokenPartitioner::new(3, None);

        // Assign 6 rows one at a time
        let tokens = [300, 200, 150, 100, 50, 40];
        let assignments: Vec<usize> = tokens
            .iter()
            .map(|&t| partitioner.assign_row_balanced(t))
            .collect();

        assert_eq!(assignments.len(), 6);

        // Each partition should get exactly 2 rows
        let row_counts = partitioner.partition_row_counts();
        assert_eq!(row_counts, &[2, 2, 2], "Expected equal row distribution");
    }

    #[test]
    fn test_balanced_assignment_ties_broken_by_tokens() {
        let mut partitioner = TokenPartitioner::new(2, None);

        // First two rows go to P0 and P1 (both start with 0 rows, 0 tokens)
        let p0 = partitioner.assign_row_balanced(100);
        let p1 = partitioner.assign_row_balanced(50);
        assert_eq!(p0, 0); // first with 0 rows
        assert_eq!(p1, 1); // second with 0 rows

        // Both partitions now have 1 row. P1 has fewer tokens (50 vs 100), so next goes to P1
        let p2 = partitioner.assign_row_balanced(30);
        assert_eq!(p2, 1);

        // Now P0 has 1 row, P1 has 2 rows → next goes to P0
        let p3 = partitioner.assign_row_balanced(200);
        assert_eq!(p3, 0);

        assert_eq!(partitioner.partition_row_counts(), &[2, 2]);
    }

    #[test]
    fn test_balanced_assignment_single_partition() {
        let mut partitioner = TokenPartitioner::new(1, None);

        for &t in &[100, 200, 150] {
            assert_eq!(partitioner.assign_row_balanced(t), 0);
        }
        assert_eq!(partitioner.partition_row_counts(), &[3]);
    }

    #[test]
    fn test_balanced_assignment_incremental_convergence() {
        // Simulate streaming: assign rows in multiple "waves" and verify
        // that row counts remain balanced throughout
        let mut partitioner = TokenPartitioner::new(4, None);

        // Wave 1: 8 rows
        for &t in &[100, 200, 50, 300, 150, 80, 120, 90] {
            partitioner.assign_row_balanced(t);
        }
        let counts = partitioner.partition_row_counts();
        assert_eq!(
            counts,
            &[2, 2, 2, 2],
            "After 8 rows, each of 4 partitions should have 2"
        );

        // Wave 2: 4 more rows
        for &t in &[110, 60, 250, 70] {
            partitioner.assign_row_balanced(t);
        }
        let counts = partitioner.partition_row_counts();
        assert_eq!(
            counts,
            &[3, 3, 3, 3],
            "After 12 rows, each of 4 partitions should have 3"
        );
    }
}
