//! Logical rule that sets batch sizes from token histograms.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use nousdb_core::token_estimator::{
    SafeBatchSizingConfig, estimate_join_block_sizes_v2, estimate_safe_batch_size,
};
use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;

/// Optimizer rule that sets `optimal_batch_size` on semantic operator logical
/// nodes from their `TokenHistogram` and `BatchBudgetModel`.
///
/// A node carrying both gets a batch size from the iterative
/// `estimate_safe_batch_size` algorithm. Nodes without histogram data are
/// skipped (the physical planner's fallback sizes them at execution time), as
/// are nodes that already have a batch size.
#[derive(Debug, Default)]
pub struct BatchSizeOptimizationRule {
    /// Sizing parameters shared by the extend, selection and join paths. The
    /// execution-time fallback in `compute_optimal_pair_batch_size` always uses
    /// the default `alpha`.
    sizing: SafeBatchSizingConfig,
}

impl BatchSizeOptimizationRule {
    /// Creates the rule with default sizing parameters (`alpha` = 0.01).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the overflow-probability target used by the quantile search.
    ///
    /// Smaller alpha gives a more conservative batch size: the search targets
    /// the `1 - alpha/k` quantile of the token distribution. The default is
    /// 0.01. Values outside `(0, 1)` are ignored.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        if alpha > 0.0 && alpha < 1.0 {
            self.sizing.alpha = alpha;
        } else {
            log::warn!(
                "[BatchSizeOptimizationRule] ignoring out-of-range alpha={alpha}; \
                 keeping alpha={}",
                self.sizing.alpha
            );
        }
        self
    }
}

impl OptimizerRule for BatchSizeOptimizationRule {
    fn name(&self) -> &str {
        "BatchSizeOptimizationRule"
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
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let LogicalPlan::Extension(ext) = &plan {
            // ExtendNode
            if let Some(extend) = ext.node.as_any().downcast_ref::<ExtendNode>() {
                if let Some(batch_size) = extend.optimal_batch_size {
                    log::info!(
                        "[BatchSizeOptimizationRule] ExtendNode already has pre-computed batch_size={}",
                        batch_size
                    );
                    return Ok(Transformed::no(plan));
                }
                let (Some(histogram), Some(budget)) =
                    (&extend.token_histogram, &extend.batch_budget)
                else {
                    return Ok(Transformed::no(plan));
                };
                let config = &self.sizing;
                let estimate = estimate_safe_batch_size(histogram, budget, config);
                let batch_size = estimate.final_batch_size.max(config.min_batch_size);
                let mut new_node = extend.clone();
                new_node.optimal_batch_size = Some(batch_size);
                log::info!(
                    "[BatchSizeOptimizationRule] Set optimizer-derived batch_size={} on ExtendNode \
                     (model_max_batch_size={:?}, quality_max_batch_size={:?}, bound={})",
                    batch_size,
                    budget.model_max_batch_size,
                    budget.quality_max_batch_size,
                    estimate.bound,
                );
                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_node),
                })));
            }

            // SemanticSelectionNode
            if let Some(sel) = ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                if let Some(batch_size) = sel.optimal_batch_size {
                    log::info!(
                        "[BatchSizeOptimizationRule] SemanticSelectionNode already has pre-computed batch_size={}",
                        batch_size
                    );
                    return Ok(Transformed::no(plan));
                }
                let (Some(histogram), Some(budget)) = (&sel.token_histogram, &sel.batch_budget)
                else {
                    return Ok(Transformed::no(plan));
                };
                let config = &self.sizing;
                let estimate = estimate_safe_batch_size(histogram, budget, config);
                let batch_size = estimate.final_batch_size.max(config.min_batch_size);
                let mut new_node = sel.clone();
                new_node.optimal_batch_size = Some(batch_size);
                log::info!(
                    "[BatchSizeOptimizationRule] Set optimizer-derived batch_size={} on SemanticSelectionNode \
                     (model_max_batch_size={:?}, quality_max_batch_size={:?}, bound={})",
                    batch_size,
                    budget.model_max_batch_size,
                    budget.quality_max_batch_size,
                    estimate.bound,
                );
                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_node),
                })));
            }

            // SemanticJoinNode
            if let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>() {
                if let (Some(left), Some(right)) = (join.left_block_size, join.right_block_size) {
                    log::info!(
                        "[BatchSizeOptimizationRule] SemanticJoinNode already has pre-computed block sizes: left={}, right={}",
                        left,
                        right
                    );
                    return Ok(Transformed::no(plan));
                }
                let Some(histogram) = &join.token_histogram else {
                    return Ok(Transformed::no(plan));
                };

                // Prefer `join_batch_budget`, falling back to `batch_budget`.
                let ref_budget = join
                    .join_batch_budget
                    .as_ref()
                    .or(join.batch_budget.as_ref());
                let Some(ref_budget) = ref_budget else {
                    return Ok(Transformed::no(plan));
                };

                let config = &self.sizing;

                // Reference-table cost model: rows are listed once and pairs are
                // ID references. The caps (capacity, quality, output budget)
                // bound the search, so the left/right ratio is chosen within them.
                let budget = ref_budget.clone();
                let output_budget = Some(
                    join.model_max_output_tokens
                        .unwrap_or_else(|| (budget.context_window_tokens * 3) / 10),
                );

                let estimate = estimate_join_block_sizes_v2(
                    join.left_token_histogram.as_ref(),
                    histogram,
                    &budget,
                    config,
                    join.model_max_batch_size,
                    join.quality_max_batch_size,
                    output_budget,
                );

                let mut new_node = join.clone();
                if new_node.left_block_size.is_none() {
                    new_node.left_block_size = Some(estimate.left_block);
                }
                if new_node.right_block_size.is_none() {
                    new_node.right_block_size = Some(estimate.right_block);
                }
                log::info!(
                    "[BatchSizeOptimizationRule] Set optimizer-derived block sizes on SemanticJoinNode: \
                     left={}, right={}, pairs={}, bound={} \
                     (model_max_batch_size={:?}, quality_max_batch_size={:?}, alpha={})",
                    estimate.left_block,
                    estimate.right_block,
                    estimate.pairs,
                    estimate.bound,
                    join.model_max_batch_size,
                    join.quality_max_batch_size,
                    config.alpha,
                );
                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_node),
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
    use datafusion::optimizer::Optimizer;
    use nousdb_core::token_estimator::{BatchBudgetModel, JoinBatchBudgetModel, TokenHistogram};
    use nousdb_operators::extend::logical::create_extend;
    use nousdb_operators::join::logical::create_semantic_join;
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

    /// Create a sample histogram and budget for testing (extend/selection).
    fn make_test_histogram_and_budget() -> (TokenHistogram, BatchBudgetModel) {
        make_test_histogram_and_capped_budget(None)
    }

    /// Same fixture, with an explicit `ModelCard.max_batch_size` ceiling.
    fn make_test_histogram_and_capped_budget(
        model_max_batch_size: Option<usize>,
    ) -> (TokenHistogram, BatchBudgetModel) {
        // Simulate rows with ~20-40 tokens each
        let token_counts: Vec<usize> = (0..100).map(|i| 20 + (i % 20)).collect();
        let histogram = TokenHistogram::from_token_counts(&token_counts, 32);
        let budget = BatchBudgetModel {
            context_window_tokens: 8192,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 10,
            model_max_batch_size,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        (histogram, budget)
    }

    // The card's row ceiling must bind even when the token budget would allow
    // a far larger batch.
    #[test]
    fn test_extend_respects_model_max_batch_size() {
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

        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_capped_budget(Some(3));
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        let LogicalPlan::Extension(ext) = &result.data else {
            panic!("Expected Extension node");
        };
        let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
        assert_eq!(node.optimal_batch_size, Some(3));
    }

    #[test]
    fn test_selection_respects_model_max_batch_size() {
        let input = make_table_scan();
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();

        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_capped_budget(Some(3));
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        let LogicalPlan::Extension(ext) = &result.data else {
            panic!("Expected Extension node");
        };
        let node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap();
        assert_eq!(node.optimal_batch_size, Some(3));
    }

    fn make_test_join_histogram_and_budget() -> (TokenHistogram, JoinBatchBudgetModel) {
        let token_counts: Vec<usize> = (0..100).map(|i| 20 + (i % 20)).collect();
        let histogram = TokenHistogram::from_token_counts(&token_counts, 32);
        let budget = JoinBatchBudgetModel {
            context_window_tokens: 8192,
            fixed_overhead_tokens: 200,
            left_row_format_tokens: 8,
            right_row_format_tokens: 8,
            pair_reference_tokens: 15,
            output_tokens_per_pair: 8,
        };
        (histogram, budget)
    }

    #[test]
    fn test_batch_size_set_on_extend() {
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

        // Attach histogram and budget to the ExtendNode
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_budget();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            assert!(node.optimal_batch_size.is_some());
            assert!(node.optimal_batch_size.unwrap() > 0);
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_batch_size_skips_when_already_set() {
        let input = make_table_scan();
        let input_schema = input.schema().clone();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));

        // Build an ExtendNode with batch size already set
        let mut all_fields: Vec<_> = input_schema.fields().to_vec();
        for f in derived.fields() {
            all_fields.push(f.clone());
        }
        let full_schema = Schema::new(all_fields);
        let df_schema = DFSchema::try_from(full_schema).unwrap();

        let mut node = ExtendNode::new(
            input,
            vec!["text".to_string()],
            "test-model".to_string(),
            "Classify sentiment".to_string(),
            derived,
            Arc::new(df_schema),
        );
        node.optimal_batch_size = Some(42);

        let plan = LogicalPlan::Extension(datafusion::logical_expr::Extension {
            node: Arc::new(node),
        });

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(!result.transformed);
    }

    #[test]
    fn test_batch_size_set_on_selection() {
        let input = make_table_scan();
        let plan = create_semantic_selection(
            &input,
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();

        // Attach histogram and budget to the SemanticSelectionNode
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_budget();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap();
            assert!(node.optimal_batch_size.is_some());
        } else {
            panic!("Expected Extension node");
        }
    }

    /// Integration test: run the full DataFusion optimizer pipeline with all
    /// NousDB rules and verify that `optimal_batch_size` survives multi-pass
    /// optimization (i.e., other rules don't clobber it).
    #[test]
    fn test_batch_size_survives_full_optimizer_pipeline() {
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

        // Attach histogram and budget
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_budget();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        // Build an optimizer with the batch-size rule plus DataFusion defaults
        let rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> =
            vec![Arc::new(BatchSizeOptimizationRule::default())];
        let optimizer = Optimizer::with_rules(rules);
        let config = datafusion::optimizer::OptimizerContext::new();

        let optimized = optimizer.optimize(plan, &config, |_, _| {}).unwrap();

        // Walk the optimized plan to find the ExtendNode
        fn find_extend(plan: &LogicalPlan) -> Option<&ExtendNode> {
            if let LogicalPlan::Extension(ext) = plan
                && let Some(node) = ext.node.as_any().downcast_ref::<ExtendNode>()
            {
                return Some(node);
            }
            for child in plan.inputs() {
                if let Some(found) = find_extend(child) {
                    return Some(found);
                }
            }
            None
        }

        let extend = find_extend(&optimized).expect("ExtendNode should survive optimization");
        assert!(
            extend.optimal_batch_size.is_some(),
            "optimal_batch_size must survive multi-pass optimization, got None"
        );
        assert!(extend.optimal_batch_size.unwrap() > 0);
    }

    #[test]
    fn test_rule_fills_batch_size_from_histogram() {
        // Verify the rule derives a batch size when histogram+budget are present
        // but optimal_batch_size is None (the gap-filling case)
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

        // Attach histogram and budget but leave optimal_batch_size as None
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            let (histogram, budget) = make_test_histogram_and_budget();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            assert!(node.optimal_batch_size.is_none());
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            assert!(
                node.optimal_batch_size.is_some(),
                "Rule should fill in batch size from histogram+budget"
            );
            assert!(node.optimal_batch_size.unwrap() > 0);
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_batch_size_set_on_join_with_reference_table_model() {
        let left_input = make_table_scan();
        let right_input = make_table_scan();
        let plan = create_semantic_join(
            &left_input,
            &right_input,
            vec!["text".to_string()],
            vec!["text".to_string()],
            "Are these related?",
            "test-model",
        )
        .unwrap();

        let (histogram, budget) = make_test_join_histogram_and_budget();

        // Attach histogram and budget to the SemanticJoinNode
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap()
                .clone();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap();
            assert!(node.left_block_size.is_some());
            assert!(node.right_block_size.is_some());

            let left = node.left_block_size.unwrap();
            let right = node.right_block_size.unwrap();
            // Reference-table model should produce reasonable block sizes
            assert!(left >= 1, "left_block should be >= 1, got {left}");
            assert!(right >= 1, "right_block should be >= 1, got {right}");
            // With 8192 token budget and ~30 token rows, should get blocks >> 1
            assert!(
                left * right > 1,
                "Reference-table model should produce more than 1 pair, got {left}x{right}"
            );
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_join_respects_model_max_batch_size_cap() {
        let left_input = make_table_scan();
        let right_input = make_table_scan();
        let plan = create_semantic_join(
            &left_input,
            &right_input,
            vec!["text".to_string()],
            vec!["text".to_string()],
            "Are these related?",
            "test-model",
        )
        .unwrap();

        let (histogram, budget) = make_test_join_histogram_and_budget();

        // Attach histogram, budget, and a tight max_batch_size cap
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap()
                .clone();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            node.model_max_batch_size = Some(9); // cap at 9 pairs
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap();
            let left = node.left_block_size.unwrap();
            let right = node.right_block_size.unwrap();
            assert!(
                left * right <= 9,
                "Pairs ({left}x{right}={}) should be capped at model_max_batch_size=9",
                left * right
            );
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_join_respects_output_budget_cap() {
        let left_input = make_table_scan();
        let right_input = make_table_scan();
        let plan = create_semantic_join(
            &left_input,
            &right_input,
            vec!["text".to_string()],
            vec!["text".to_string()],
            "Are these related?",
            "test-model",
        )
        .unwrap();

        let (histogram, budget) = make_test_join_histogram_and_budget();
        // budget.output_tokens_per_pair = 8 (from make_test_join_histogram_and_budget)
        // With max_output_tokens = 100: (100 - 20) / 8 = 10 pairs max
        let max_output_tokens: usize = 100;

        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap()
                .clone();
            node.token_histogram = Some(histogram);
            node.batch_budget = Some(budget);
            node.model_max_output_tokens = Some(max_output_tokens);
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension node");
        };

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap();
            let left = node.left_block_size.unwrap();
            let right = node.right_block_size.unwrap();
            let max_pairs_by_output = (max_output_tokens - 20) / 8; // = 10
            assert!(
                left * right <= max_pairs_by_output,
                "Pairs ({left}x{right}={}) should be capped by output budget ({max_pairs_by_output})",
                left * right
            );
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_rule_skips_when_no_histogram() {
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        // Create an ExtendNode without histogram/budget (the default)
        let plan = create_extend(
            input,
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let rule = BatchSizeOptimizationRule::default();
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        // Should not transform, no histogram data available
        assert!(!result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            assert!(node.optimal_batch_size.is_none());
        } else {
            panic!("Expected Extension node");
        }
    }
}
