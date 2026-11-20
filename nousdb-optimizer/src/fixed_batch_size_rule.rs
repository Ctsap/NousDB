//! Logical rule that sets a fixed batch size on semantic operators.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;

/// Optimizer rule that sets `optimal_batch_size` to a fixed constant on every
/// semantic operator node that does not carry one yet, for plans that want
/// fixed-size batching instead of histogram-guided sizing.
///
/// - `ExtendNode` / `SemanticSelectionNode`: `optimal_batch_size = batch_size`.
/// - `SemanticJoinNode`: `left_block_size` / `right_block_size` from the join
///   geometry, which is `(batch_size, batch_size)` unless
///   [`Self::with_join_blocks`] says otherwise. Only the unset side(s) are
///   filled.
///
/// A value already on the node is never overwritten, which keeps the rule
/// idempotent across DataFusion's multi-pass optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedBatchSizeRule {
    batch_size: usize,
    /// `(left_block_size, right_block_size)` stamped on joins.
    join_blocks: (usize, usize),
}

impl FixedBatchSizeRule {
    /// Fix every extend/selection batch at `batch_size` and every join block
    /// at `batch_size` on both sides (so a join sends `batch_size²` pairs per
    /// request).
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            join_blocks: (batch_size, batch_size),
        }
    }

    /// Stamp joins with an explicit block geometry instead of
    /// `(batch_size, batch_size)`: `left` rows per left block, `right` rows per
    /// right tile.
    pub fn with_join_blocks(mut self, left: usize, right: usize) -> Self {
        self.join_blocks = (left, right);
        self
    }
}

impl OptimizerRule for FixedBatchSizeRule {
    fn name(&self) -> &str {
        "FixedBatchSizeRule"
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
                if extend.optimal_batch_size.is_some() {
                    return Ok(Transformed::no(plan));
                }
                let mut new_node = extend.clone();
                new_node.optimal_batch_size = Some(self.batch_size);
                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_node),
                })));
            }

            // SemanticSelectionNode
            if let Some(sel) = ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                if sel.optimal_batch_size.is_some() {
                    return Ok(Transformed::no(plan));
                }
                let mut new_node = sel.clone();
                new_node.optimal_batch_size = Some(self.batch_size);
                return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_node),
                })));
            }

            // SemanticJoinNode, fill whichever block size is still unset
            if let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>() {
                if join.left_block_size.is_some() && join.right_block_size.is_some() {
                    return Ok(Transformed::no(plan));
                }
                let mut new_node = join.clone();
                if new_node.left_block_size.is_none() {
                    new_node.left_block_size = Some(self.join_blocks.0);
                }
                if new_node.right_block_size.is_none() {
                    new_node.right_block_size = Some(self.join_blocks.1);
                }
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
    use datafusion::optimizer::OptimizerContext;
    use nousdb_operators::extend::logical::create_extend;
    use nousdb_operators::join::logical::create_semantic_join;
    use nousdb_operators::selection::logical::create_semantic_selection;

    fn make_table_scan() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("id", DataType::Int32, false),
        ]);
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(DFSchema::try_from(schema).unwrap()),
        })
    }

    fn join_plan() -> LogicalPlan {
        create_semantic_join(
            &make_table_scan(),
            &make_table_scan(),
            vec!["text".to_string()],
            vec!["text".to_string()],
            "Are these related?",
            "test-model",
        )
        .unwrap()
    }

    fn apply(rule: &FixedBatchSizeRule, plan: LogicalPlan) -> Transformed<LogicalPlan> {
        rule.rewrite(plan, &OptimizerContext::new()).unwrap()
    }

    fn join_blocks(plan: &LogicalPlan) -> (Option<usize>, Option<usize>) {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected an Extension node");
        };
        let node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticJoinNode>()
            .expect("expected SemanticJoinNode");
        (node.left_block_size, node.right_block_size)
    }

    fn with_join_blocks_set(
        plan: LogicalPlan,
        left: Option<usize>,
        right: Option<usize>,
    ) -> LogicalPlan {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected an Extension node");
        };
        let mut node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticJoinNode>()
            .expect("expected SemanticJoinNode")
            .clone();
        node.left_block_size = left;
        node.right_block_size = right;
        LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })
    }

    #[test]
    fn new_fills_extend_and_selection() {
        let rule = FixedBatchSizeRule::new(7);

        let selection = create_semantic_selection(
            &make_table_scan(),
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();
        let result = apply(&rule, selection);
        assert!(result.transformed);
        let LogicalPlan::Extension(ext) = &result.data else {
            panic!("expected an Extension node");
        };
        let node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap();
        assert_eq!(node.optimal_batch_size, Some(7));

        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]));
        let extend = create_extend(
            make_table_scan(),
            "Classify sentiment",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();
        let result = apply(&rule, extend);
        assert!(result.transformed);
        let LogicalPlan::Extension(ext) = &result.data else {
            panic!("expected an Extension node");
        };
        let node = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
        assert_eq!(node.optimal_batch_size, Some(7));

        // A second application finds nothing to do.
        let again = apply(&rule, result.data);
        assert!(!again.transformed);
    }

    #[test]
    fn new_keeps_square_join_blocks() {
        let result = apply(&FixedBatchSizeRule::new(4), join_plan());
        assert!(result.transformed);
        assert_eq!(join_blocks(&result.data), (Some(4), Some(4)));
    }

    #[test]
    fn explicit_join_blocks_override_the_square_default() {
        let square = apply(&FixedBatchSizeRule::new(4), join_plan());
        let explicit = apply(
            &FixedBatchSizeRule::new(4).with_join_blocks(4, 4),
            join_plan(),
        );
        assert_eq!(join_blocks(&square.data), join_blocks(&explicit.data));

        let strip = apply(
            &FixedBatchSizeRule::new(4).with_join_blocks(1, 16),
            join_plan(),
        );
        assert_eq!(join_blocks(&strip.data), (Some(1), Some(16)));
    }

    #[test]
    fn a_preset_value_is_never_overwritten() {
        let rule = FixedBatchSizeRule::new(4).with_join_blocks(4, 4);

        // Both set: the rule declines.
        let both = with_join_blocks_set(join_plan(), Some(2), Some(3));
        let result = apply(&rule, both);
        assert!(!result.transformed);
        assert_eq!(join_blocks(&result.data), (Some(2), Some(3)));

        // One side set: only the other is filled.
        let left_only = with_join_blocks_set(join_plan(), Some(2), None);
        let result = apply(&rule, left_only);
        assert!(result.transformed);
        assert_eq!(join_blocks(&result.data), (Some(2), Some(4)));

        let right_only = with_join_blocks_set(join_plan(), None, Some(3));
        let result = apply(&rule, right_only);
        assert!(result.transformed);
        assert_eq!(join_blocks(&result.data), (Some(4), Some(3)));

        // Extend / selection with a size already set.
        let selection = create_semantic_selection(
            &make_table_scan(),
            "test-model",
            vec!["text".to_string()],
            "Keep positive reviews",
        )
        .unwrap();
        let LogicalPlan::Extension(ext) = selection else {
            panic!("expected an Extension node");
        };
        let mut node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap()
            .clone();
        node.optimal_batch_size = Some(9);
        let preset = LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        });
        let result = apply(&rule, preset);
        assert!(!result.transformed);
        let LogicalPlan::Extension(ext) = &result.data else {
            panic!("expected an Extension node");
        };
        let node = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap();
        assert_eq!(node.optimal_batch_size, Some(9));
    }
}
