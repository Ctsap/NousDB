//! The analyzer rule that replaces marker functions with NousDB's logical nodes.
//!
//! The rule runs after DataFusion's own analyzer rules (`TypeCoercion`
//! included) and rewrites bottom-up, subqueries included:
//!
//! - a `Projection` with `EXTEND` calls in its expressions becomes a stack of
//!   `ExtendNode`s (one per distinct call, first written innermost), with a
//!   residual projection on top only when it is not an identity
//!   (`projection`);
//! - a `Filter` whose top-level `AND`-ed conjuncts include `SEM_FILTER` calls
//!   becomes a relational `Filter` (if any relational conjunct is left) under
//!   one `SemanticSelectionNode` per call (`filter`); a `SEM_JOIN` conjunct
//!   over a comma/CROSS join builds a semantic join;
//! - an inner `Join` whose `ON` holds a `SEM_JOIN` call becomes a
//!   `SemanticJoinNode` (`join`).
//!
//! Every replacement has the schema (names, types and qualifiers, in order) of
//! the node it replaces, so no ancestor needs rewriting; the rule checks this.
//! A marker left anywhere else is an error naming where it is supported.

mod filter;
mod join;
mod projection;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::AnalyzerRule;

use crate::markers::{MarkerKind, marker_kind};

/// Rewrites `EXTEND`, `SEM_FILTER` and `SEM_JOIN` calls into `ExtendNode`,
/// `SemanticSelectionNode` and `SemanticJoinNode`.
#[derive(Debug, Default)]
pub struct SemanticFunctionRewriteRule;

impl SemanticFunctionRewriteRule {
    /// The rule's name, as listed by the session's analyzer.
    pub const NAME: &'static str = "semantic_function_rewrite";

    /// Creates the rule.
    pub fn new() -> Self {
        Self
    }
}

impl AnalyzerRule for SemanticFunctionRewriteRule {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        // Every query of the session passes through here.
        if !plan_contains_marker(&plan)? {
            return Ok(plan);
        }
        let plan = plan.transform_up_with_subqueries(rewrite_node)?.data;
        reject_leftover_markers(&plan)?;
        Ok(plan)
    }

    fn name(&self) -> &str {
        Self::NAME
    }
}

/// Rewrites one node whose own expressions hold markers; leaves every other
/// node alone. Checks the schema invariant on every replacement.
fn rewrite_node(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let (replacement, what) = match &plan {
        LogicalPlan::Projection(p) if any_marker(&p.expr)? => {
            (projection::rewrite(p)?, "Projection")
        }
        LogicalPlan::Filter(f) if expr_contains_marker(&f.predicate)? => {
            (filter::rewrite(f)?, "Filter")
        }
        LogicalPlan::Join(j)
            if j.filter
                .as_ref()
                .map(expr_contains_marker)
                .transpose()?
                .unwrap_or(false) =>
        {
            (join::rewrite_on(j)?, "Join")
        }
        _ => return Ok(Transformed::no(plan)),
    };
    if !replacement
        .schema()
        .logically_equivalent_names_and_types(plan.schema())
    {
        return Err(DataFusionError::Internal(format!(
            "{} changed the schema of a {what} node: before [{}], after [{}]",
            SemanticFunctionRewriteRule::NAME,
            describe_schema(&plan),
            describe_schema(&replacement)
        )));
    }
    Ok(Transformed::yes(replacement))
}

fn describe_schema(plan: &LogicalPlan) -> String {
    plan.schema()
        .iter()
        .map(|(q, f)| match q {
            Some(q) => format!("{q}.{}: {}", f.name(), f.data_type()),
            None => format!("{}: {}", f.name(), f.data_type()),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The first marker kind inside `expr` (pre-order), not looking into subquery
/// plans.
pub(crate) fn first_marker(expr: &Expr) -> Result<Option<MarkerKind>> {
    let mut found = None;
    expr.apply(|e| {
        if let Some(kind) = marker_kind(e) {
            found = Some(kind);
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// Whether `expr` holds a marker anywhere (not looking into subquery plans).
pub(crate) fn expr_contains_marker(expr: &Expr) -> Result<bool> {
    Ok(first_marker(expr)?.is_some())
}

fn any_marker(exprs: &[Expr]) -> Result<bool> {
    for e in exprs {
        if expr_contains_marker(e)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The first marker kind in any expression of `node` itself (not its inputs).
fn node_marker(node: &LogicalPlan) -> Result<Option<MarkerKind>> {
    let mut found = None;
    node.apply_expressions(|e| {
        found = first_marker(e)?;
        Ok(if found.is_some() {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(found)
}

/// Whether any node of `plan`, subqueries included, holds a marker.
fn plan_contains_marker(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        found = node_marker(node)?.is_some();
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(found)
}

/// Where each function is supported, for error messages.
pub(crate) fn supported_positions(kind: MarkerKind) -> &'static str {
    match kind {
        MarkerKind::Extend => {
            "in a SELECT list (e.g. SELECT *, EXTEND('...', 'model', column) AS x FROM t), \
             or as a relation, FROM EXTEND(t, ...)"
        }
        MarkerKind::SemFilter => "as a top-level AND-ed condition in WHERE",
        MarkerKind::SemJoin => {
            "in an INNER JOIN's ON clause, or in WHERE over a comma/CROSS join of two relations"
        }
    }
}

/// Fails on the first marker the rewrite left in the plan.
fn reject_leftover_markers(plan: &LogicalPlan) -> Result<()> {
    let mut error = None;
    plan.apply_with_subqueries(|node| {
        if let Some(kind) = node_marker(node)? {
            error = Some(DataFusionError::Plan(format!(
                "{name}(...) is not supported in {place}; {name} is supported {supported}",
                name = kind.sql_name(),
                place = node_kind(node),
                supported = supported_positions(kind),
            )));
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// A short description of the kind of plan node, for error messages.
fn node_kind(node: &LogicalPlan) -> String {
    match node {
        LogicalPlan::Aggregate(_) => {
            "an Aggregate (GROUP BY keys or aggregate function arguments)".into()
        }
        LogicalPlan::Sort(_) => "a Sort (ORDER BY)".into(),
        LogicalPlan::Window(_) => "a Window (window function arguments or OVER clause)".into(),
        LogicalPlan::Join(j) => format!("the ON clause of a {} JOIN", j.join_type),
        LogicalPlan::Distinct(_) => "a Distinct (DISTINCT ON)".into(),
        LogicalPlan::Limit(_) => "a Limit".into(),
        LogicalPlan::Values(_) => "a VALUES list".into(),
        LogicalPlan::Unnest(_) => "an Unnest".into(),
        LogicalPlan::Repartition(_) => "a Repartition".into(),
        LogicalPlan::TableScan(_) => "a TableScan's filters".into(),
        LogicalPlan::Projection(_) => "a Projection".into(),
        LogicalPlan::Filter(_) => "a Filter".into(),
        other => {
            let text = other.display().to_string();
            let kind = text.split(':').next().unwrap_or("").trim().to_string();
            format!("a {kind} node")
        }
    }
}
