//! Shared fixtures of the `rewrite_*` tests: a plain DataFusion session with
//! the semantic SQL functions and a few in-memory tables, and helpers that run
//! the analyzer and inspect the result. No model is involved: the tests assert
//! on logical plans only.

#![allow(dead_code)]

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::DFSchemaRef;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;

use nousdb_operators::{ExtendNode, SemanticJoinNode, SemanticSelectionNode};

/// Registers `name` with Utf8 columns, plus Int64 columns for names in `ints`.
fn table(ctx: &SessionContext, name: &str, columns: &[&str], ints: &[&str]) {
    let fields: Vec<Field> = columns
        .iter()
        .map(|c| {
            let ty = if ints.contains(c) {
                DataType::Int64
            } else {
                DataType::Utf8
            };
            Field::new(*c, ty, true)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Int64 => Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            _ => Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef,
        })
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    ctx.register_table(
        name,
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()),
    )
    .unwrap();
}

/// A plain session with the semantic SQL functions and the test tables:
/// `reviews(review, title, year)`, `papers(abstract, year)`, `a(title, price)`,
/// `b(name, title)` (shares `title` with `a`) and `c(name, kind)`.
pub fn session() -> SessionContext {
    let ctx = SessionContext::new();
    nousdb_sql::register(&ctx).unwrap();
    table(&ctx, "reviews", &["review", "title", "year"], &["year"]);
    table(&ctx, "papers", &["abstract", "year"], &["year"]);
    table(&ctx, "a", &["title", "price"], &["price"]);
    table(&ctx, "b", &["name", "title"], &[]);
    table(&ctx, "c", &["name", "kind"], &[]);
    ctx
}

/// Plans `sql` and runs the session's analyzer (DataFusion's rules, then the
/// semantic rewrite). Returns the schema the SQL planner gave the statement
/// and the analysed plan.
pub async fn try_analyze(ctx: &SessionContext, sql: &str) -> Result<(DFSchemaRef, LogicalPlan)> {
    let df = ctx.sql(sql).await?;
    let before = Arc::new(df.schema().clone());
    let plan = df.logical_plan().clone();
    let state = ctx.state();
    let analyzed = state
        .analyzer()
        .execute_and_check(plan, state.config_options(), |_, _| {})?;
    Ok((before, analyzed))
}

/// [`try_analyze`], which must succeed and keep the statement's schema
/// (names, types and qualifiers, in order).
pub async fn analyze(ctx: &SessionContext, sql: &str) -> LogicalPlan {
    let (before, plan) = try_analyze(ctx, sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should analyse: {e}"));
    assert!(
        plan.schema().logically_equivalent_names_and_types(&before),
        "`{sql}` changed the schema:\nbefore {before:?}\nafter {:?}",
        plan.schema()
    );
    assert_no_marker(&plan);
    plan
}

/// The error of planning or analysing `sql`, which must fail.
pub async fn analyze_err(ctx: &SessionContext, sql: &str) -> String {
    match try_analyze(ctx, sql).await {
        Ok((_, plan)) => panic!(
            "`{sql}` should fail, but analysed to\n{}",
            plan.display_indent()
        ),
        Err(e) => e.to_string(),
    }
}

/// Asserts that `msg` contains every one of `parts`.
pub fn assert_contains(msg: &str, parts: &[&str]) {
    for part in parts {
        assert!(msg.contains(part), "expected {part:?} in:\n{msg}");
    }
}

/// The plan as `EXPLAIN` prints it.
pub fn text(plan: &LogicalPlan) -> String {
    plan.display_indent().to_string()
}

/// Asserts that no marker function is left anywhere in `plan`.
pub fn assert_no_marker(plan: &LogicalPlan) {
    plan.apply_with_subqueries(|node| {
        node.apply_expressions(|e| {
            e.apply(|x| {
                assert!(
                    nousdb_sql::markers::marker_kind(x).is_none(),
                    "marker left in {}",
                    plan.display_indent()
                );
                Ok(TreeNodeRecursion::Continue)
            })
        })?;
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
}

/// `plan` as a NousDB node of type `T`, if it is one.
pub fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(ext) => ext.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// `plan` as an `ExtendNode`; panics otherwise.
pub fn extend(plan: &LogicalPlan) -> &ExtendNode {
    as_node::<ExtendNode>(plan)
        .unwrap_or_else(|| panic!("expected an ExtendNode, found\n{}", text(plan)))
}

/// `plan` as a `SemanticSelectionNode`; panics otherwise.
pub fn selection(plan: &LogicalPlan) -> &SemanticSelectionNode {
    as_node::<SemanticSelectionNode>(plan)
        .unwrap_or_else(|| panic!("expected a SemanticSelectionNode, found\n{}", text(plan)))
}

/// `plan` as a `SemanticJoinNode`; panics otherwise.
pub fn join(plan: &LogicalPlan) -> &SemanticJoinNode {
    as_node::<SemanticJoinNode>(plan)
        .unwrap_or_else(|| panic!("expected a SemanticJoinNode, found\n{}", text(plan)))
}

/// The single input of `plan`.
pub fn input(plan: &LogicalPlan) -> &LogicalPlan {
    let inputs = plan.inputs();
    assert_eq!(inputs.len(), 1, "expected one input:\n{}", text(plan));
    inputs[0]
}

/// The first node (pre-order, subqueries included) for which `f` holds.
pub fn find(plan: &LogicalPlan, f: impl Fn(&LogicalPlan) -> bool) -> Option<LogicalPlan> {
    let mut found = None;
    plan.apply_with_subqueries(|n| {
        if f(n) {
            found = Some(n.clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

/// How many NousDB nodes of type `T` the plan holds, subqueries included.
pub fn count<T: 'static>(plan: &LogicalPlan) -> usize {
    let mut n = 0;
    plan.apply_with_subqueries(|p| {
        if as_node::<T>(p).is_some() {
            n += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    n
}

/// The field names of a schema, qualified where they are.
pub fn field_names(plan: &LogicalPlan) -> Vec<String> {
    plan.schema()
        .iter()
        .map(|(q, f)| match q {
            Some(q) => format!("{q}.{}", f.name()),
            None => f.name().clone(),
        })
        .collect()
}
