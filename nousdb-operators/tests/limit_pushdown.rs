//! DataFusion's limit hook on the semantic nodes.
//!
//! `ExtendNode` declares `supports_limit_pushdown`, so DataFusion's
//! `PushDownLimit` applies a limit above an extend to its input too and the
//! model is called for the limited rows only. The other nodes do not: a
//! selection and a join are not one to one.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::DFSchemaRef;
use datafusion::error::Result;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore, table_scan,
};
use datafusion::optimizer::{Optimizer, OptimizerContext};
use nousdb_operators::extend::logical::{ExtendNode, create_extend};
use nousdb_operators::join::logical::create_semantic_join;
use nousdb_operators::selection::logical::create_semantic_selection;
use nousdb_operators::token_repartition::config::TokenPartitionConfig;
use nousdb_operators::token_repartition::logical::create_token_repartition;

fn scan(name: &str) -> LogicalPlan {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("review", DataType::Utf8, true),
    ]);
    table_scan(Some(name), &schema, None)
        .unwrap()
        .build()
        .unwrap()
}

fn extend(input: LogicalPlan) -> LogicalPlan {
    let derived = Arc::new(Schema::new(vec![Field::new(
        "sentiment",
        DataType::Utf8,
        true,
    )]));
    create_extend(input, "classify", "m", vec!["review".to_string()], derived).unwrap()
}

fn limit(skip: usize, fetch: usize, input: LogicalPlan) -> LogicalPlan {
    LogicalPlanBuilder::from(input)
        .limit(skip, Some(fetch))
        .unwrap()
        .build()
        .unwrap()
}

fn node(plan: &LogicalPlan) -> &dyn UserDefinedLogicalNode {
    let LogicalPlan::Extension(ext) = plan else {
        panic!(
            "expected an extension node, found\n{}",
            plan.display_indent()
        );
    };
    ext.node.as_ref()
}

fn lines(plan: &LogicalPlan) -> Vec<String> {
    plan.display_indent()
        .to_string()
        .lines()
        .map(|l| l.trim().to_string())
        .collect()
}

/// DataFusion's default logical rules, `passes` times at most.
fn optimize(plan: &LogicalPlan, passes: u8) -> LogicalPlan {
    let config = OptimizerContext::new().with_max_passes(passes);
    Optimizer::new()
        .optimize(plan.clone(), &config, |_, _| {})
        .unwrap()
}

#[test]
fn only_the_extend_declares_limit_pushdown() {
    assert!(node(&extend(scan("t"))).supports_limit_pushdown());

    let selection =
        create_semantic_selection(&scan("t"), "m", vec!["review".to_string()], "keep").unwrap();
    assert!(!node(&selection).supports_limit_pushdown());

    let join = create_semantic_join(
        &scan("a"),
        &scan("b"),
        vec!["a.review".to_string()],
        vec!["b.review".to_string()],
        "same?",
        "m",
    )
    .unwrap();
    assert!(!node(&join).supports_limit_pushdown());
}

/// Once token repartitioning sits under the extend, the limit is already
/// below it; a limit between the two would undo the repartitioning.
#[test]
fn an_extend_over_token_repartitioning_does_not_take_another_limit() {
    let repartitioned = create_token_repartition(
        scan("t"),
        TokenPartitionConfig::with_num_partitions(vec!["review".to_string()], 4),
    )
    .unwrap();
    assert!(!node(&extend(repartitioned)).supports_limit_pushdown());
}

#[test]
fn datafusion_pushes_the_limit_below_the_extend() {
    let plan = limit(0, 3, extend(scan("t")));
    assert_eq!(
        lines(&optimize(&plan, 1)),
        [
            "Limit: skip=0, fetch=3",
            "NousDB_Extend: instruction='classify', input_cols=[\"review\"], output_cols=[\"sentiment\"]",
            "Limit: skip=0, fetch=3",
            "TableScan: t projection=[id, review], fetch=3",
        ]
    );

    // An OFFSET stays above; the input keeps skip + fetch rows.
    let plan = limit(2, 3, extend(scan("t")));
    assert_eq!(
        lines(&optimize(&plan, 1)),
        [
            "Limit: skip=2, fetch=3",
            "NousDB_Extend: instruction='classify', input_cols=[\"review\"], output_cols=[\"sentiment\"]",
            "Limit: skip=0, fetch=5",
            "TableScan: t projection=[id, review], fetch=5",
        ]
    );
}

/// The optimizer runs its rules up to three times. Every pass pushes the
/// limit again; the extend merges it with the one already there, so the plan
/// is the same after one pass and after three.
#[test]
fn further_passes_do_not_stack_limits_under_the_extend() {
    let plan = limit(0, 3, extend(scan("t")));
    let once = lines(&optimize(&plan, 1));
    for passes in [2, 3] {
        assert_eq!(
            lines(&optimize(&plan, passes)),
            once,
            "after {passes} passes"
        );
    }

    // A tighter limit above a looser one written under the extend.
    let plan = limit(0, 3, extend(limit(0, 100, scan("t"))));
    let optimized = lines(&optimize(&plan, 3));
    assert_eq!(
        optimized.iter().filter(|l| l.starts_with("Limit:")).count(),
        2,
        "one limit above the extend and one below:\n{}",
        optimized.join("\n")
    );
    assert_eq!(optimized[2], "Limit: skip=0, fetch=3");
}

#[test]
fn rebuilding_an_extend_merges_limits_stacked_on_its_input() {
    let LogicalPlan::Extension(ext) = extend(scan("t")) else {
        unreachable!()
    };
    let stacked = limit(1, 2, limit(0, 10, limit(3, 50, scan("t"))));
    let rebuilt = ext
        .node
        .with_exprs_and_inputs(vec![], vec![stacked])
        .unwrap();
    let extend = rebuilt.as_any().downcast_ref::<ExtendNode>().unwrap();
    assert_eq!(
        lines(&extend.input),
        ["Limit: skip=4, fetch=2", "TableScan: t"],
        "skip 3 then 0 then 1, and the tightest fetch"
    );

    // An input that is not a stack of limits is passed through untouched.
    let single = limit(0, 2, scan("t"));
    let rebuilt = ext
        .node
        .with_exprs_and_inputs(vec![], vec![single.clone()])
        .unwrap();
    let extend = rebuilt.as_any().downcast_ref::<ExtendNode>().unwrap();
    assert_eq!(extend.input, single);
}

// --- What DataFusion does on its own -------------------------------------

/// The smallest extension node that allows limit pushdown.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
struct Passthrough {
    input: LogicalPlan,
}

impl UserDefinedLogicalNodeCore for Passthrough {
    fn name(&self) -> &str {
        "Passthrough"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Passthrough")
    }
    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(Self {
            input: inputs[0].clone(),
        })
    }
    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

/// Why `ExtendNode` merges limits itself: DataFusion 55.2 adds one `Limit`
/// under an extension node per optimizer pass and loses its own merge of
/// them (see `merge_stacked_limits` in `extend/logical.rs`). This pins that
/// behaviour with a node that does nothing. If a DataFusion upgrade makes it
/// fail, the merge in `ExtendNode::with_exprs_and_inputs` is no longer needed.
#[test]
fn datafusion_alone_stacks_limits_under_an_extension_node() {
    let passthrough = LogicalPlan::Extension(Extension {
        node: Arc::new(Passthrough { input: scan("t") }),
    });
    let plan = limit(0, 3, passthrough);
    for passes in [1usize, 2, 3] {
        let optimized = lines(&optimize(&plan, passes as u8));
        let below = optimized
            .iter()
            .skip_while(|l| l.as_str() != "Passthrough")
            .filter(|l| l.starts_with("Limit:"))
            .count();
        assert_eq!(
            below,
            passes,
            "{passes} pass(es) leave {passes} limit(s) under the node:\n{}",
            optimized.join("\n")
        );
    }
}
