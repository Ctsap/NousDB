//! Logical rule that inserts token-balanced repartitioning before semantic operators.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use nousdb_core::token_estimator::batch_optimizer::BatchBudgetModel;
use nousdb_core::token_estimator::histogram::TokenHistogram;
use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::generate::logical::GenerateNode;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;
use nousdb_operators::token_repartition::config::{PartitionStrategy, TokenPartitionConfig};
use nousdb_operators::token_repartition::logical::create_token_repartition;

use crate::utils::is_token_repartition;

/// Thresholds for the 5-condition repartition insertion funnel.
#[derive(Debug, Clone)]
pub struct RepartitionThresholds {
    /// Minimum tuples per partition to justify repartitioning (γ)
    pub min_tuples_per_partition: usize,
    /// Skew tolerance: α must exceed 1 + δ to trigger (δ)
    pub skew_tolerance: f64,
    /// Tail ratio threshold: Q95/Q50 must exceed this to trigger (θ_tail)
    pub tail_ratio_threshold: f64,
    /// Cost of BPE tokenization per character in seconds (c_BPE)
    pub cost_bpe_per_char: f64,
    /// Cost of sorting per element in seconds (c_sort)
    pub cost_sort: f64,
    /// Cost of LLM inference per token in seconds (λ_m)
    pub cost_llm_per_token: f64,
    /// Approximate characters per token
    pub chars_per_token: f64,
    /// Range narrowness threshold for zigzag strategy selection (ε)
    pub zigzag_epsilon: f64,
}

impl Default for RepartitionThresholds {
    fn default() -> Self {
        Self {
            min_tuples_per_partition: 10,
            skew_tolerance: 0.12,
            tail_ratio_threshold: 5.0,
            cost_bpe_per_char: 0.0001,
            cost_sort: 0.000001,
            cost_llm_per_token: 0.001,
            chars_per_token: 4.0,
            zigzag_epsilon: 0.15,
        }
    }
}

/// Rows the scan under `plan` reports, when it reports any.
///
/// Descends through single-input nodes, since `SELECT *` arrives as a
/// projection over the scan. Row-reducing nodes on the way down (a filter) make
/// the answer an upper bound, which is the safe direction because the result is
/// used to skip work. Returns `None` for a multi-input node or a source that
/// reports no statistics.
fn source_row_count(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::TableScan(scan) => scan.fetch.or_else(|| {
            datafusion::datasource::source_as_provider(&scan.source)
                .ok()?
                .statistics()?
                .num_rows
                .get_value()
                .copied()
        }),
        // A generated relation has at most its declared row count.
        LogicalPlan::Extension(ext) if ext.node.as_any().is::<GenerateNode>() => ext
            .node
            .as_any()
            .downcast_ref::<GenerateNode>()
            .map(|generate| generate.num_rows),
        other => match other.inputs().as_slice() {
            [only] => source_row_count(only),
            _ => None,
        },
    }
}

/// Optimizer rule that inserts `TokenRepartitionNode`s before semantic operators
/// to ensure balanced token distribution across partitions.
///
/// Runs BottomUp so that repartition nodes are inserted closest to data sources
/// first.
///
/// ## Insertion Logic
///
/// When a histogram and batch budget are available on the operator node (injected
/// by `HistogramInjectionRule`), a 5-condition funnel is evaluated:
/// 1. Multi-batch necessity (total tokens exceed context window)
/// 2. Sufficient parallelism (N/P ≥ γ)
/// 3. Token skew (CV-based α exceeds threshold)
/// 4. Tail heaviness (Q95/Q50 ratio exceeds threshold)
/// 5. Overhead amortization (savings > repartition cost)
///
/// When no histogram is available, repartition is always inserted when P > 1.
///
/// - **Before semantic joins**: Always insert on both inputs (joins are O(|R|x|S|),
///   balanced partitions are critical).
/// - **Skip**: If the input is already a `TokenRepartitionNode`.
#[derive(Debug, Default)]
pub struct TokenRepartitionInsertionRule {
    /// Target number of partitions. If `None`, uses `OptimizerConfig::target_partitions`.
    target_partitions: Option<usize>,
    /// Thresholds for the 5-condition insertion funnel.
    pub(crate) thresholds: RepartitionThresholds,
    /// When set, bypasses automatic strategy selection and always uses this strategy.
    forced_strategy: Option<PartitionStrategy>,
}

impl TokenRepartitionInsertionRule {
    /// Creates a rule that targets `target_partitions` partitions and selects the
    /// strategy automatically.
    pub fn new(target_partitions: usize) -> Self {
        Self {
            target_partitions: Some(target_partitions),
            thresholds: RepartitionThresholds::default(),
            forced_strategy: None,
        }
    }

    /// Creates a rule that always uses `strategy`, bypassing automatic strategy
    /// selection.
    pub fn with_strategy(strategy: PartitionStrategy) -> Self {
        Self {
            target_partitions: None,
            thresholds: RepartitionThresholds::default(),
            forced_strategy: Some(strategy),
        }
    }

    /// Creates a rule that always inserts repartition with the given strategy
    /// and partition count, bypassing the 5-gate funnel.
    pub fn forced(target_partitions: usize, strategy: PartitionStrategy) -> Self {
        Self {
            target_partitions: Some(target_partitions),
            thresholds: RepartitionThresholds::default(),
            forced_strategy: Some(strategy),
        }
    }

    /// Extract text/string column names from a schema for use as token columns.
    ///
    /// All three Arrow string encodings count: `Utf8`, `LargeUtf8` and
    /// `Utf8View`, the set `nousdb_core::arrow_util::strings::StringColumn`
    /// reads at execution time. Parquet scans and casts to `VARCHAR` often
    /// produce `Utf8View`.
    fn extract_text_columns(schema: &datafusion::common::DFSchemaRef) -> Vec<String> {
        use datafusion::arrow::datatypes::DataType;
        schema
            .fields()
            .iter()
            .filter(|f| {
                matches!(
                    f.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                )
            })
            .map(|f| f.name().clone())
            .collect()
    }

    /// Wrap a plan in a TokenRepartitionNode if it isn't already one.
    ///
    /// Returns `Transformed::no(input)` when it declines, so the caller can
    /// leave the operator untouched instead of rebuilding it around the same
    /// input.
    fn wrap_with_repartition(
        input: LogicalPlan,
        token_columns: Vec<String>,
        num_partitions: usize,
        strategy: PartitionStrategy,
    ) -> Result<Transformed<LogicalPlan>> {
        if is_token_repartition(&input) {
            return Ok(Transformed::no(input));
        }
        if token_columns.is_empty() {
            return Ok(Transformed::no(input));
        }
        // Nothing to balance across more partitions than there are rows (small
        // join build sides in particular). A row count is only available from a
        // source that reports statistics; an unknown cardinality proceeds.
        if let Some(rows) = source_row_count(&input)
            && rows < num_partitions
        {
            log::info!(
                "[TokenRepartitionInsertionRule] skipping insertion: input has {rows} rows \
                 for {num_partitions} partitions, so repartitioning is pure overhead"
            );
            return Ok(Transformed::no(input));
        }
        let config = TokenPartitionConfig::with_num_partitions(token_columns, num_partitions)
            .with_strategy(strategy);
        create_token_repartition(input, config).map(Transformed::yes)
    }

    fn get_target_partitions(&self, config: &dyn OptimizerConfig) -> usize {
        self.target_partitions
            .unwrap_or_else(|| config.options().execution.target_partitions)
    }

    /// Evaluate the 5-condition funnel to determine if repartitioning is warranted.
    ///
    /// Returns `Some(strategy)` if repartitioning should be inserted, `None` otherwise.
    fn should_insert_repartition(
        &self,
        histogram: &TokenHistogram,
        budget: &BatchBudgetModel,
        target_partitions: usize,
    ) -> Option<PartitionStrategy> {
        let stats = histogram.stats();
        let th = &self.thresholds;
        let n = stats.total_rows;
        let p = target_partitions;

        // Gate 1, Multi-Batch Necessity:
        // If all tokens fit in one context window, no repartition needed.
        let c_avail = budget
            .context_window_tokens
            .saturating_sub(budget.fixed_overhead_tokens);
        if stats.total_tokens <= c_avail as f64 {
            return None;
        }
        log::debug!(
            "N = {}, P = {}, Min Tuples = {}, Total Tokens = {:.0}, Context Window = {}, C_avail = {}",
            n,
            p,
            th.min_tuples_per_partition,
            stats.total_tokens,
            budget.context_window_tokens,
            c_avail
        );
        // Gate 2, Sufficient Parallelism:
        // Not enough tuples per partition to justify the overhead.
        if n / p < th.min_tuples_per_partition {
            return None;
        }

        // Gate 3, Token Skew (primary trigger):
        let cv = if stats.mean > 0.0 {
            stats.variance.sqrt() / stats.mean
        } else {
            0.0
        };
        log::debug!(
            "CV: {:.2}, Mean: {:.2}, Variance: {:.2}",
            cv,
            stats.mean,
            stats.variance
        );
        let ln_p = if p > 1 { (p as f64).ln() } else { 0.0 };
        let alpha = 1.0 + cv * ((p as f64 * 2.0 * ln_p) / n as f64).sqrt();
        let skew_triggered = alpha > 1.0 + th.skew_tolerance;

        // Gate 4, Tail Heaviness (independent trigger):
        let q50 = histogram.quantile(0.50);
        let q95 = histogram.quantile(0.95);
        let tail_ratio = if q50 > 0.0 { q95 / q50 } else { 0.0 };
        let tail_triggered = tail_ratio > th.tail_ratio_threshold;

        // If neither skew nor tail triggered, skip.
        if !skew_triggered && !tail_triggered {
            return None;
        }

        // Gate 5, Overhead Amortization:
        // savings from balanced partitions must exceed repartition cost
        let total_chars = stats.total_tokens * th.chars_per_token;
        let savings = stats.total_tokens * th.cost_llm_per_token * (alpha - 1.0);
        let sort_cost = if n > 1 {
            th.cost_sort * (n as f64) * (n as f64).ln()
        } else {
            0.0
        };
        let overhead = th.cost_bpe_per_char * total_chars + sort_cost;
        if savings <= overhead {
            return None;
        }

        // Strategy Selection:
        let buckets = histogram.buckets();
        if buckets.is_empty() {
            return Some(PartitionStrategy::default());
        }
        let tau_max = buckets.last().unwrap().tau_max as f64;
        let tau_min = buckets.first().unwrap().tau_min as f64;
        let avg_tokens_per_partition = stats.total_tokens / p as f64;

        if (tau_max - tau_min) < th.zigzag_epsilon * avg_tokens_per_partition {
            Some(PartitionStrategy::EqualRowCounts)
        } else {
            Some(PartitionStrategy::OptimalTokenBalance)
        }
    }
}

impl OptimizerRule for TokenRepartitionInsertionRule {
    fn name(&self) -> &str {
        "TokenRepartitionInsertionRule"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let target_partitions = self.get_target_partitions(config);
        log::debug!(
            "Applying {} with target_partitions = {}",
            self.name(),
            target_partitions
        );
        // Only insert repartition when there are multiple target partitions
        if target_partitions <= 1 {
            return Ok(Transformed::no(plan));
        }

        if let LogicalPlan::Extension(ext) = &plan {
            // Before semantic join: repartition the left (probe) input for parallelism.
            // The right (build) input stays unchanged, the join's
            // required_input_distribution() specifies SinglePartition for it,
            // so DataFusion's EnforceDistribution will coalesce automatically.
            if let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>() {
                if is_token_repartition(&join.left_input) {
                    return Ok(Transformed::no(plan));
                }

                let strategy = if let Some(ref forced) = self.forced_strategy {
                    forced.clone()
                } else {
                    // Use left histogram (or fall back to right/token_histogram as symmetric proxy)
                    let hist = join
                        .left_token_histogram
                        .as_ref()
                        .or(join.token_histogram.as_ref());
                    log::debug!(
                        "Evaluating join repartition: hist = {}, budget = {}",
                        hist.is_some(),
                        join.join_batch_budget.is_some()
                    );
                    match (hist, &join.join_batch_budget) {
                        (Some(hist), Some(budget)) => {
                            let compat_budget = BatchBudgetModel {
                                context_window_tokens: budget.context_window_tokens,
                                fixed_overhead_tokens: budget.fixed_overhead_tokens,
                                per_row_delimiter_tokens: 0,
                                expected_output_tokens_per_row: 0,
                                model_max_batch_size: join.model_max_batch_size,
                                quality_max_batch_size: join.quality_max_batch_size,
                                // The join's pair-level output bound is applied
                                // by `estimate_join_block_sizes_v2`; repeating
                                // it here would bound *rows* by a pair budget.
                                max_output_tokens: None,
                                reasoning_output_tokens_per_row: 0,
                            };
                            match self.should_insert_repartition(
                                hist,
                                &compat_budget,
                                target_partitions,
                            ) {
                                Some(s) => s,
                                None => return Ok(Transformed::no(plan)),
                            }
                        }
                        _ => PartitionStrategy::default(),
                    }
                };

                let left_cols = Self::extract_text_columns(join.left_input.schema());

                let new_left = Self::wrap_with_repartition(
                    join.left_input.clone(),
                    left_cols,
                    target_partitions,
                    strategy,
                )?;
                if !new_left.transformed {
                    return Ok(Transformed::no(plan));
                }

                // Clone, then swap the input, so every optimizer field set by
                // HistogramInjectionRule / BatchSizeRule rides along.
                let mut new_join = join.clone();
                new_join.left_input = new_left.data;

                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_join),
                })));
            }

            // Before extend: evaluate funnel or fallback
            if let Some(extend) = ext.node.as_any().downcast_ref::<ExtendNode>() {
                if is_token_repartition(&extend.input) {
                    return Ok(Transformed::no(plan));
                }

                let strategy = if let Some(ref forced) = self.forced_strategy {
                    forced.clone()
                } else {
                    match (&extend.token_histogram, &extend.batch_budget) {
                        (Some(hist), Some(budget)) => {
                            match self.should_insert_repartition(hist, budget, target_partitions) {
                                Some(s) => s,
                                None => return Ok(Transformed::no(plan)),
                            }
                        }
                        _ => PartitionStrategy::default(),
                    }
                };

                let token_cols = Self::extract_text_columns(extend.input.schema());
                let new_input = Self::wrap_with_repartition(
                    extend.input.clone(),
                    token_cols,
                    target_partitions,
                    strategy,
                )?;
                // A declined wrap leaves the node exactly as it came: the
                // histogram and budget HistogramInjectionRule just set are what
                // BatchSizeRule sizes from next.
                if !new_input.transformed {
                    return Ok(Transformed::no(plan));
                }

                let mut new_extend = extend.clone();
                new_extend.input = new_input.data;

                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_extend),
                })));
            }

            // Before selection: evaluate funnel or fallback
            if let Some(sel) = ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                if is_token_repartition(&sel.input) {
                    return Ok(Transformed::no(plan));
                }

                let strategy = if let Some(ref forced) = self.forced_strategy {
                    forced.clone()
                } else {
                    match (&sel.token_histogram, &sel.batch_budget) {
                        (Some(hist), Some(budget)) => {
                            match self.should_insert_repartition(hist, budget, target_partitions) {
                                Some(s) => s,
                                None => return Ok(Transformed::no(plan)),
                            }
                        }
                        _ => PartitionStrategy::default(),
                    }
                };

                let token_cols = Self::extract_text_columns(sel.input.schema());
                let new_input = Self::wrap_with_repartition(
                    sel.input.clone(),
                    token_cols,
                    target_partitions,
                    strategy,
                )?;
                if !new_input.transformed {
                    return Ok(Transformed::no(plan));
                }

                let mut new_sel = sel.clone();
                new_sel.input = new_input.data;

                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_sel),
                })));
            }
        }

        Ok(Transformed::no(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::DFSchema;
    use nousdb_core::token_estimator::batch_optimizer::BatchBudgetModel;
    use nousdb_core::token_estimator::histogram::TokenHistogram;
    use nousdb_operators::extend::logical::create_extend;
    use nousdb_operators::selection::logical::create_semantic_selection;
    fn make_table_scan() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("id", DataType::Int32, false),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    fn make_histogram(token_counts: &[usize]) -> TokenHistogram {
        TokenHistogram::from_token_counts(token_counts, 32)
    }

    fn make_budget(context_window: usize) -> BatchBudgetModel {
        BatchBudgetModel {
            context_window_tokens: context_window,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 5,
            expected_output_tokens_per_row: 20,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        }
    }

    fn make_extend_with_histogram(
        input: LogicalPlan,
        histogram: TokenHistogram,
        budget: BatchBudgetModel,
    ) -> LogicalPlan {
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let plan = create_extend(
            input,
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        if let LogicalPlan::Extension(ext) = &plan {
            let mut extend = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            extend.token_histogram = Some(histogram);
            extend.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(extend),
            })
        } else {
            panic!("Expected Extension node");
        }
    }

    #[allow(dead_code)]
    fn make_selection_with_histogram(
        input: LogicalPlan,
        histogram: TokenHistogram,
        budget: BatchBudgetModel,
    ) -> LogicalPlan {
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();

        if let LogicalPlan::Extension(ext) = &plan {
            let mut sel = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap()
                .clone();
            sel.token_histogram = Some(histogram);
            sel.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(sel),
            })
        } else {
            panic!("Expected Extension node");
        }
    }

    // Fallback path (no histogram).

    #[test]
    fn test_repartition_inserted_before_extend() {
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let plan = create_extend(
            input,
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        // The extend's input should now be a TokenRepartitionNode
        if let LogicalPlan::Extension(ext) = &result.data {
            let extend = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            assert!(is_token_repartition(&extend.input));
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_repartition_inserted_before_selection() {
        let input = make_table_scan();
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let sel = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap();
            assert!(is_token_repartition(&sel.input));
        }
    }

    #[test]
    fn test_skip_when_already_repartitioned() {
        let input = make_table_scan();
        let config_rp = TokenPartitionConfig::with_num_partitions(vec!["text".to_string()], 4);
        let repartitioned = create_token_repartition(input, config_rp).unwrap();

        let plan = create_semantic_selection(
            &repartitioned,
            "test-model",
            vec!["text".to_string()],
            "Keep positive",
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(!result.transformed);
    }

    #[test]
    fn test_skip_when_single_partition() {
        let input = make_table_scan();
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive",
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(1);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(!result.transformed);
    }

    // 5-condition funnel.

    #[test]
    fn test_skip_small_relation() {
        // Gate 1: total_tokens <= C_avail → skip
        // 100 rows × ~50 tokens each = ~5000 total tokens, context window = 128000
        let input = make_table_scan();
        let counts: Vec<usize> = vec![50; 100];
        let hist = make_histogram(&counts);
        let budget = make_budget(128000); // huge context, everything fits
        let plan = make_extend_with_histogram(input, hist, budget);

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            !result.transformed,
            "Should skip: all tokens fit in one context window"
        );
    }

    #[test]
    fn test_skip_few_tuples_per_partition() {
        // Gate 2: N/P < γ=15 → skip
        // N=10, P=4 → N/P = 2.5 < 15
        let input = make_table_scan();
        let counts: Vec<usize> = vec![500; 10];
        let hist = make_histogram(&counts);
        let budget = make_budget(4096); // small context so Gate 1 passes
        let plan = make_extend_with_histogram(input, hist, budget);

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            !result.transformed,
            "Should skip: not enough tuples per partition"
        );
    }

    #[test]
    fn test_skew_triggers_insertion() {
        // Gate 3: high CV → α > 1 + δ → triggers
        // Mix of small and large tokens creates high variance
        let input = make_table_scan();
        let mut counts: Vec<usize> = vec![50; 200];
        counts.extend(vec![5000; 50]);
        let hist = make_histogram(&counts);
        let budget = make_budget(4096);
        let plan = make_extend_with_histogram(input, hist, budget);

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            result.transformed,
            "Should insert: high token skew triggers repartition"
        );
    }

    #[test]
    fn test_tail_triggers_insertion() {
        // Gate 4: Q95/Q50 > θ_tail=5.0 → triggers independently
        // Heavy tail: most rows small, a few very large
        let input = make_table_scan();
        let mut counts: Vec<usize> = vec![100; 200];
        counts.extend(vec![10000; 20]); // heavy tail
        let hist = make_histogram(&counts);
        let budget = make_budget(4096);
        let plan = make_extend_with_histogram(input, hist, budget);

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            result.transformed,
            "Should insert: tail heaviness triggers repartition"
        );
    }

    #[test]
    fn test_uniform_distribution_skip() {
        // Gates 3+4: low CV and no tail → both fail → skip
        // All tokens nearly identical, but enough rows and small context to pass Gates 1-2
        let input = make_table_scan();
        let counts: Vec<usize> = (100..160).collect(); // 60 values, small range
        let hist = make_histogram(&counts);
        let budget = make_budget(2000); // small enough that Gate 1 passes
        let plan = make_extend_with_histogram(input, hist, budget);

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            !result.transformed,
            "Should skip: uniform distribution, no skew or tail"
        );
    }

    #[test]
    fn test_strategy_greedy_for_wide_range() {
        // Wide τ_max - τ_min → OptimalTokenBalance (Greedy)
        // Use should_insert_repartition directly with custom thresholds:
        // - Lower cost_bpe_per_char to relax Gate 5
        // - Lower zigzag_epsilon to make the narrow→zigzag threshold tighter,
        //   so moderate ranges select Greedy
        //
        // Data: 200×10 + 50×10000 → N=250, P=4
        //   τ_min=10, τ_max=10000, range=9990
        //   T_total ≈ 502000, T_total/P = 125500
        //   With ε=0.01: 0.01 × 125500 = 1255, range 9990 > 1255 → Greedy
        let mut rule = TokenRepartitionInsertionRule::new(4);
        rule.thresholds.cost_bpe_per_char = 0.00001;
        rule.thresholds.zigzag_epsilon = 0.01;

        let mut counts: Vec<usize> = vec![10; 200];
        counts.extend(vec![10000; 50]);
        let hist = make_histogram(&counts);
        let budget = make_budget(4096);

        let result = rule.should_insert_repartition(&hist, &budget, 4);
        assert!(result.is_some(), "Should trigger insertion with high skew");
        assert_eq!(
            result.unwrap(),
            PartitionStrategy::OptimalTokenBalance,
            "Wide range should select Greedy (OptimalTokenBalance)"
        );
    }

    #[test]
    fn test_strategy_zigzag_for_narrow_range() {
        // Zigzag requires: all 5 gates pass AND narrow bucket range.
        // Gate 5 demands alpha > 1.4 with default thresholds, which requires
        // extreme CV, incompatible with narrow range. So we use custom
        // thresholds with lower cost_bpe_per_char to relax Gate 5.
        //
        // Bimodal 100/350, N=200, P=4:
        //   CV = 250/450 ≈ 0.556, alpha ≈ 1.131 > 1.12 (Gate 3 passes)
        //   τ_max=350, τ_min=100, range=250
        //   T_total = 200*225 = 45000, T_total/P = 11250
        //   250 < 0.15 * 11250 = 1687 → narrow → Zigzag
        let mut rule = TokenRepartitionInsertionRule::new(4);
        rule.thresholds.cost_bpe_per_char = 0.00001; // 10x lower to relax Gate 5

        let bimodal: Vec<usize> = vec![100; 100].into_iter().chain(vec![350; 100]).collect();
        let hist = make_histogram(&bimodal);
        let budget = make_budget(2000);

        let result = rule.should_insert_repartition(&hist, &budget, 4);
        assert!(
            result.is_some(),
            "Should trigger insertion with moderate bimodal skew"
        );
        assert_eq!(
            result.unwrap(),
            PartitionStrategy::EqualRowCounts,
            "Narrow range relative to T_total/P should select Zigzag (EqualRowCounts)"
        );
    }

    #[test]
    fn test_fallback_without_histogram() {
        // No histogram on node → fallback: always inserts with default strategy
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let plan = create_extend(
            input,
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            result.transformed,
            "Should always insert without histogram (backward compat)"
        );
    }

    #[test]
    fn test_selection_fallback_without_histogram() {
        // Same as above but for selection nodes
        let input = make_table_scan();
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive",
        )
        .unwrap();

        let rule = TokenRepartitionInsertionRule::new(4);
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            result.transformed,
            "Should always insert without histogram (backward compat)"
        );
    }

    #[test]
    fn test_should_insert_direct_skew() {
        // Direct test of should_insert_repartition with high skew data
        let rule = TokenRepartitionInsertionRule::new(4);
        let mut counts: Vec<usize> = vec![50; 200];
        counts.extend(vec![5000; 50]);
        let hist = make_histogram(&counts);
        let budget = make_budget(4096);

        let result = rule.should_insert_repartition(&hist, &budget, 4);
        assert!(result.is_some(), "High skew should trigger insertion");
    }

    #[test]
    fn test_should_insert_direct_uniform_skip() {
        // Direct test: uniform data should not trigger
        let rule = TokenRepartitionInsertionRule::new(4);
        let counts: Vec<usize> = (100..160).collect();
        let hist = make_histogram(&counts);
        let budget = make_budget(2000);

        let result = rule.should_insert_repartition(&hist, &budget, 4);
        assert!(
            result.is_none(),
            "Uniform distribution should not trigger insertion"
        );
    }

    // String encodings.

    /// A scan whose only text column has the given Arrow string encoding.
    fn make_table_scan_with_text(text_type: DataType) -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("text", text_type, false),
            Field::new("id", DataType::Int32, false),
        ]);
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(DFSchema::try_from(schema).unwrap()),
        })
    }

    #[test]
    fn text_columns_include_every_string_encoding() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::LargeUtf8, true),
            Field::new("c", DataType::Utf8View, true),
            Field::new("id", DataType::Int32, false),
            Field::new("when", DataType::Date32, true),
            Field::new("score", DataType::Float32, true),
        ]);
        let schema = Arc::new(DFSchema::try_from(schema).unwrap());
        assert_eq!(
            TokenRepartitionInsertionRule::extract_text_columns(&schema),
            vec!["a", "b", "c"],
            "Utf8, LargeUtf8 and Utf8View are text; numeric and temporal columns are not"
        );
    }

    /// A scan whose only string column is a `Utf8View` or `LargeUtf8` still
    /// counts as having a text column, so the rule inserts the repartition.
    #[test]
    fn repartition_is_inserted_over_a_utf8view_or_largeutf8_text_column() {
        for text_type in [DataType::LargeUtf8, DataType::Utf8View] {
            let derived = Arc::new(Schema::new(vec![Field::new(
                "sentiment",
                DataType::Utf8,
                false,
            )]));
            let extend = create_extend(
                make_table_scan_with_text(text_type.clone()),
                "Classify sentiment",
                "test-model",
                vec!["text".to_string()],
                derived,
            )
            .unwrap();
            let selection = create_semantic_selection(
                &make_table_scan_with_text(text_type.clone()),
                "test-model",
                vec!["text".to_string()],
                "Keep positive reviews",
            )
            .unwrap();

            let rule = TokenRepartitionInsertionRule::new(4);
            let config = datafusion::optimizer::OptimizerContext::new();

            let result = rule.rewrite(extend, &config).unwrap();
            let LogicalPlan::Extension(ext) = &result.data else {
                panic!("{text_type:?}: expected an Extension node");
            };
            let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            assert!(
                is_token_repartition(&node.input),
                "{text_type:?}: extend input must be repartitioned"
            );

            let result = rule.rewrite(selection, &config).unwrap();
            let LogicalPlan::Extension(ext) = &result.data else {
                panic!("{text_type:?}: expected an Extension node");
            };
            let node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap();
            assert!(
                is_token_repartition(&node.input),
                "{text_type:?}: selection input must be repartitioned"
            );
        }
    }

    /// The histogram-guided path over a `Utf8View` column reaches the same
    /// verdict as over `Utf8`: skewed tokens insert a repartition node.
    #[test]
    fn the_funnel_judges_a_utf8view_column_like_a_utf8_one() {
        let mut counts: Vec<usize> = vec![50; 200];
        counts.extend(vec![5000; 50]);

        let mut inserted = Vec::new();
        for text_type in [DataType::Utf8, DataType::Utf8View] {
            let plan = make_extend_with_histogram(
                make_table_scan_with_text(text_type),
                make_histogram(&counts),
                make_budget(4096),
            );
            let rule = TokenRepartitionInsertionRule::new(4);
            let config = datafusion::optimizer::OptimizerContext::new();
            let result = rule.rewrite(plan, &config).unwrap();
            let LogicalPlan::Extension(ext) = &result.data else {
                panic!("expected an Extension node");
            };
            let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            inserted.push(is_token_repartition(&node.input));
        }
        assert_eq!(inserted, vec![true, true]);
    }

    // Annotations across the rebuild: HistogramInjectionRule runs just before
    // this rule and BatchSizeRule just after, so a rebuild that drops the
    // histogram or budget leaves the node to the planner's fallback sizing.

    fn forced_rule() -> TokenRepartitionInsertionRule {
        TokenRepartitionInsertionRule::forced(4, PartitionStrategy::OptimalTokenBalance)
    }

    fn scan_with(fields: Vec<Field>) -> LogicalPlan {
        let df_schema = DFSchema::try_from(Schema::new(fields)).unwrap();
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    /// A named scan; `fetch` is the row count `source_row_count` reads.
    fn named_scan(name: &str, fields: Vec<Field>, fetch: Option<usize>) -> LogicalPlan {
        use datafusion::logical_expr::{LogicalPlanBuilder, LogicalTableSource};
        let source = Arc::new(LogicalTableSource::new(Arc::new(Schema::new(fields))));
        LogicalPlanBuilder::scan_with_filters_fetch(name, source, None, vec![], fetch)
            .unwrap()
            .build()
            .unwrap()
    }

    fn text_fields(text_type: DataType) -> Vec<Field> {
        vec![
            Field::new("text", text_type, false),
            Field::new("id", DataType::Int32, false),
        ]
    }

    fn extend_of(plan: &LogicalPlan) -> &ExtendNode {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected an Extension node, got {plan}");
        };
        ext.node.as_any().downcast_ref::<ExtendNode>().unwrap()
    }

    fn selection_of(plan: &LogicalPlan) -> &SemanticSelectionNode {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected an Extension node, got {plan}");
        };
        ext.node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap()
    }

    fn join_of(plan: &LogicalPlan) -> &SemanticJoinNode {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected an Extension node, got {plan}");
        };
        ext.node
            .as_any()
            .downcast_ref::<SemanticJoinNode>()
            .unwrap()
    }

    /// Extend with every optimizer annotation populated, as the rule finds it
    /// after HistogramInjectionRule (and, on a later pass, BatchSizeRule).
    fn annotated_extend(input: LogicalPlan, input_col: &str) -> LogicalPlan {
        let plan = create_extend(
            input,
            "Classify sentiment",
            "test-model",
            vec![input_col.to_string()],
            Arc::new(Schema::new(vec![Field::new(
                "sentiment",
                DataType::Utf8,
                true,
            )])),
        )
        .unwrap();
        let mut node = extend_of(&plan).clone();
        node.token_histogram = Some(make_histogram(&[50; 100]));
        node.batch_budget = Some(make_budget(4096));
        node.optimal_batch_size = Some(7);
        LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })
    }

    fn annotated_selection(input: LogicalPlan, input_col: &str) -> LogicalPlan {
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec![input_col.to_string()],
            "Keep positive reviews",
        )
        .unwrap();
        let mut node = selection_of(&plan).clone();
        node.token_histogram = Some(make_histogram(&[50; 100]));
        node.batch_budget = Some(make_budget(4096));
        node.optimal_batch_size = Some(7);
        LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })
    }

    fn annotated_join(left: LogicalPlan) -> LogicalPlan {
        use nousdb_core::token_estimator::JoinBatchBudgetModel;
        let right = make_table_scan();
        let schema = left.schema().clone();
        let mut node = SemanticJoinNode::new(
            left,
            right,
            vec!["text".to_string()],
            vec!["text".to_string()],
            "test-model".to_string(),
            "the two entries refer to the same thing".to_string(),
            schema,
        );
        let budget = JoinBatchBudgetModel {
            context_window_tokens: 4096,
            fixed_overhead_tokens: 400,
            left_row_format_tokens: 12,
            right_row_format_tokens: 12,
            pair_reference_tokens: 15,
            output_tokens_per_pair: 8,
        };
        node.left_block_size = Some(3);
        node.right_block_size = Some(5);
        node.token_histogram = Some(make_histogram(&[40; 50]));
        node.batch_budget = Some(budget.clone());
        node.model_max_batch_size = Some(25);
        node.quality_max_batch_size = Some(16);
        node.model_max_output_tokens = Some(9000);
        node.left_token_histogram = Some(make_histogram(&[60; 50]));
        node.join_batch_budget = Some(budget);
        LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })
    }

    /// Every field but the rewritten input must come through untouched.
    ///
    /// Compared by `Debug` because the nodes' `PartialEq` ignores the
    /// annotations and the annotation types implement none. No field is listed
    /// by name, so fields added later are covered too.
    fn assert_only_input_changed<N: Clone + std::fmt::Debug>(
        before: &N,
        after: &N,
        take_input: impl Fn(&mut N, &N),
    ) {
        let mut expected = before.clone();
        take_input(&mut expected, after);
        assert_eq!(
            format!("{expected:?}"),
            format!("{after:?}"),
            "the rebuild changed a field other than the input"
        );
    }

    #[test]
    fn extend_annotations_survive_an_insertion() {
        let plan = annotated_extend(make_table_scan(), "text");
        let before = extend_of(&plan).clone();

        let result = forced_rule()
            .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
            .unwrap();

        assert!(result.transformed);
        let after = extend_of(&result.data);
        assert!(is_token_repartition(&after.input));
        assert!(after.token_histogram.is_some() && after.batch_budget.is_some());
        assert_eq!(after.optimal_batch_size, Some(7));
        assert_only_input_changed(&before, after, |e, a| e.input = a.input.clone());
    }

    #[test]
    fn selection_annotations_survive_an_insertion() {
        let plan = annotated_selection(make_table_scan(), "text");
        let before = selection_of(&plan).clone();

        let result = forced_rule()
            .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
            .unwrap();

        assert!(result.transformed);
        let after = selection_of(&result.data);
        assert!(is_token_repartition(&after.input));
        assert!(after.token_histogram.is_some() && after.batch_budget.is_some());
        assert_eq!(after.optimal_batch_size, Some(7));
        assert_only_input_changed(&before, after, |e, a| e.input = a.input.clone());
    }

    #[test]
    fn join_annotations_survive_an_insertion() {
        let plan = annotated_join(make_table_scan());
        let before = join_of(&plan).clone();

        let result = forced_rule()
            .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
            .unwrap();

        assert!(result.transformed);
        let after = join_of(&result.data);
        assert!(is_token_repartition(&after.left_input));
        assert!(!is_token_repartition(&after.right_input));
        assert_only_input_changed(&before, after, |e, a| e.left_input = a.left_input.clone());
    }

    /// Both ways `wrap_with_repartition` declines: the input has no text
    /// column to balance on, or fewer rows than partitions.
    fn declining_inputs() -> Vec<(&'static str, LogicalPlan, &'static str)> {
        vec![
            (
                "no text column",
                scan_with(vec![
                    Field::new("id", DataType::Int32, false),
                    Field::new("rating", DataType::Float64, false),
                ]),
                "id",
            ),
            (
                "fewer rows than partitions",
                named_scan("tiny", text_fields(DataType::Utf8), Some(2)),
                "text",
            ),
        ]
    }

    #[test]
    fn extend_is_not_rebuilt_when_wrapping_declines() {
        for (why, input, col) in declining_inputs() {
            let plan = annotated_extend(input, col);
            let before = extend_of(&plan).clone();

            let result = forced_rule()
                .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
                .unwrap();

            assert!(!result.transformed, "{why}: nothing was inserted");
            let after = extend_of(&result.data);
            assert!(!is_token_repartition(&after.input), "{why}");
            assert_eq!(format!("{before:?}"), format!("{after:?}"), "{why}");
        }
    }

    #[test]
    fn selection_is_not_rebuilt_when_wrapping_declines() {
        for (why, input, col) in declining_inputs() {
            let plan = annotated_selection(input, col);
            let before = selection_of(&plan).clone();

            let result = forced_rule()
                .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
                .unwrap();

            assert!(!result.transformed, "{why}: nothing was inserted");
            let after = selection_of(&result.data);
            assert!(!is_token_repartition(&after.input), "{why}");
            assert_eq!(format!("{before:?}"), format!("{after:?}"), "{why}");
        }
    }

    #[test]
    fn join_is_not_rebuilt_when_wrapping_declines() {
        let plan = annotated_join(named_scan("tiny", text_fields(DataType::Utf8), Some(2)));
        let before = join_of(&plan).clone();

        let result = forced_rule()
            .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
            .unwrap();

        assert!(!result.transformed);
        let after = join_of(&result.data);
        assert!(!is_token_repartition(&after.left_input));
        assert_eq!(format!("{before:?}"), format!("{after:?}"));
    }

    #[test]
    fn every_string_encoding_is_a_text_column() {
        let schema = Arc::new(
            DFSchema::try_from(Schema::new(vec![
                Field::new("body", DataType::Utf8View, true),
                Field::new("id", DataType::Int64, false),
                Field::new("title", DataType::LargeUtf8, true),
                Field::new("tags", DataType::Utf8, true),
                Field::new("score", DataType::Int32, true),
            ]))
            .unwrap(),
        );
        assert_eq!(
            TokenRepartitionInsertionRule::extract_text_columns(&schema),
            vec!["body", "title", "tags"]
        );
    }

    /// A `Utf8View`-only relation is repartitioned on its text column.
    #[test]
    fn utf8_view_text_is_repartitioned() {
        use nousdb_operators::token_repartition::logical::TokenRepartitionNode;

        let plan = annotated_extend(scan_with(text_fields(DataType::Utf8View)), "text");
        let result = forced_rule()
            .rewrite(plan, &datafusion::optimizer::OptimizerContext::new())
            .unwrap();

        assert!(result.transformed);
        let LogicalPlan::Extension(ext) = &extend_of(&result.data).input else {
            panic!("expected a TokenRepartitionNode under the extend");
        };
        let tr = ext
            .node
            .as_any()
            .downcast_ref::<TokenRepartitionNode>()
            .unwrap();
        assert_eq!(tr.token_columns, vec!["text"]);
    }

    // The rule chain: HistogramInjection, TokenRepartition, BatchSize.

    /// The batch size the chain ends with on an extend over `scan`, plus
    /// whether a repartition node sits under it.
    ///
    /// The catalog holds a histogram for `test_table.text`; the registry is
    /// empty, so the injection rule resolves no card and uses its fallback
    /// budget, which is the same for every caller here.
    fn chain_batch_size(scan: LogicalPlan, with_tr: bool, max_passes: u8) -> (Option<usize>, bool) {
        use crate::{BatchSizeOptimizationRule, HistogramInjectionRule};
        use datafusion::optimizer::{Optimizer, OptimizerContext};
        use nousdb_core::providers::registry::ModelRegistry;
        use nousdb_core::token_estimator::HistogramCatalog;

        let dir = tempfile::TempDir::new().unwrap();
        let table_dir = dir.path().join("test_table");
        std::fs::create_dir(&table_dir).unwrap();
        let counts: Vec<usize> = (0..400).map(|i| 30 + (i * 37) % 900).collect();
        std::fs::write(
            table_dir.join("text.json"),
            serde_json::to_string(&make_histogram(&counts)).unwrap(),
        )
        .unwrap();
        let catalog = Arc::new(HistogramCatalog::load(dir.path()).unwrap());

        let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = vec![Arc::new(
            HistogramInjectionRule::new(catalog, Arc::new(ModelRegistry::new())),
        )];
        if with_tr {
            rules.push(Arc::new(forced_rule()));
        }
        rules.push(Arc::new(BatchSizeOptimizationRule::default()));

        let plan = create_extend(
            scan,
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            Arc::new(Schema::new(vec![Field::new(
                "sentiment",
                DataType::Utf8,
                true,
            )])),
        )
        .unwrap();
        let optimized = Optimizer::with_rules(rules)
            .optimize(
                plan,
                &OptimizerContext::new().with_max_passes(max_passes),
                |_, _| {},
            )
            .unwrap();
        let extend = extend_of(&optimized);
        (
            extend.optimal_batch_size,
            is_token_repartition(&extend.input),
        )
    }

    #[test]
    fn the_chain_sizes_the_operator_whether_or_not_repartition_is_inserted() {
        // The reference: the same chain without repartitioning at all.
        let (reference, _) = chain_batch_size(
            named_scan("test_table", text_fields(DataType::Utf8), None),
            false,
            3,
        );
        let reference = reference.expect("injection + batch sizing alone must size the node");

        let cases = [
            // Inserted, and sized in the same pass.
            (
                "inserted, single pass",
                named_scan("test_table", text_fields(DataType::Utf8), None),
                1,
                true,
            ),
            // Declined: fewer rows than partitions.
            (
                "declined on row count",
                named_scan("test_table", text_fields(DataType::Utf8), Some(2)),
                3,
                false,
            ),
            // `Utf8View` text.
            (
                "Utf8View text",
                named_scan("test_table", text_fields(DataType::Utf8View), None),
                3,
                true,
            ),
        ];
        for (why, scan, passes, expect_tr) in cases {
            let (batch_size, has_tr) = chain_batch_size(scan, true, passes);
            assert_eq!(has_tr, expect_tr, "{why}: repartition node");
            assert_eq!(
                batch_size,
                Some(reference),
                "{why}: the optimizer batch size must survive the repartition rule"
            );
        }
    }
}
