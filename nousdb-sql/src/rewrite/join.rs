//! `SEM_JOIN`: an inner join's `ON` clause, or `WHERE` over a comma/CROSS
//! join, becomes a `SemanticJoinNode`. Both forms go through [`build`].

use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DFSchema, DFSchemaRef, JoinType};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::expr::{Exists, InSubquery, SetComparison};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, Filter, Join, LogicalPlan, Projection};
use nousdb_operators::create_semantic_join;

use super::filter::{Conjuncts, classify};
use super::first_marker;
use crate::args::SemJoinCall;
use crate::columns::bare_input_name;

fn plan_error<T>(msg: impl Into<String>) -> Result<T> {
    Err(DataFusionError::Plan(msg.into()))
}

/// Rewrites a `Join` whose `ON` filter holds a marker.
pub(super) fn rewrite_on(j: &Join) -> Result<LogicalPlan> {
    let filter = j.filter.as_ref().expect("the caller found a marker in it");
    if j.join_type != JoinType::Inner {
        // Name the function actually found (SEM_JOIN, or another marker).
        let kind = first_marker(filter)?.expect("the caller found a marker");
        if kind == crate::markers::MarkerKind::SemJoin {
            return plan_error(format!(
                "SEM_JOIN supports only INNER joins; found {} JOIN",
                j.join_type
            ));
        }
    }
    let Conjuncts {
        sem_filters,
        sem_joins,
        relational,
    } = classify(filter, "a JOIN's ON clause")?;
    if !sem_filters.is_empty() {
        return plan_error(
            "SEM_FILTER is not supported in a JOIN's ON clause; move it to WHERE \
             (a JOIN b ON SEM_JOIN(...) WHERE SEM_FILTER(...))",
        );
    }
    if j.join_type != JoinType::Inner {
        return plan_error(format!(
            "SEM_JOIN supports only INNER joins; found {} JOIN",
            j.join_type
        ));
    }
    let call = match sem_joins.len() {
        1 => sem_joins.into_iter().next().expect("one call"),
        n => {
            return plan_error(format!(
                "only one SEM_JOIN per join is supported, found {n} in the ON clause; combine \
                 the conditions into one instruction"
            ));
        }
    };
    if !j.on.is_empty() {
        return plan_error(
            "SEM_JOIN cannot be combined with USING, NATURAL or equi-join keys; write \
             a JOIN b ON SEM_JOIN(...)",
        );
    }
    build(&j.left, &j.right, &j.schema, call, relational, "ON")
}

/// Builds the semantic join of `left` and `right` for `call`, with the
/// relational conjuncts that came with it (placed by [`placement`]: a `Filter`
/// on one side, a `Filter` above the join, or the two-sided error), and
/// restores `stock_schema` (the stock join's schema) with a projection if the
/// node renamed a right column. The `Filter` above, if any, sits on top of
/// that projection, so it is built against `stock_schema`.
///
/// `clause` (`ON` or `WHERE`) is where the call was written.
pub(super) fn build(
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    stock_schema: &DFSchemaRef,
    call: SemJoinCall,
    relational: Vec<Expr>,
    clause: &str,
) -> Result<LogicalPlan> {
    let (ls, rs) = (left.schema(), right.schema());
    let mut left_attrs = Vec::new();
    let mut right_attrs = Vec::new();
    for c in &call.inputs {
        match (ls.has_column(c), rs.has_column(c)) {
            (true, true) => {
                return plan_error(format!(
                    "SEM_JOIN: column {} is found on both sides of the join; qualify it with \
                     its table name or alias",
                    c.flat_name()
                ));
            }
            (true, false) => left_attrs.push(bare_input_name(c, ls, "SEM_JOIN")?),
            (false, true) => right_attrs.push(bare_input_name(c, rs, "SEM_JOIN")?),
            (false, false) => {
                return plan_error(format!(
                    "SEM_JOIN: column {} is not a column of either side of the join",
                    c.flat_name()
                ));
            }
        }
    }
    if left_attrs.is_empty() || right_attrs.is_empty() {
        let missing = if left_attrs.is_empty() {
            "left"
        } else {
            "right"
        };
        let hint = if clause == "WHERE" {
            "in WHERE, the two relations SEM_JOIN compares must be the two sides of the last \
             comma/CROSS join in FROM: the last relation listed and the join of all the ones \
             before it (relations are not reordered, so FROM a, l, b WHERE SEM_JOIN(.., a.x, \
             b.y) works and FROM a, b, l does not), or write a JOIN ... ON SEM_JOIN(...) per pair"
        } else {
            "over more than two relations, write a JOIN ... ON SEM_JOIN(...) per pair"
        };
        return plan_error(format!(
            "SEM_JOIN needs at least one column from each side of the join; none is from the \
             {missing} side ({hint})"
        ));
    }

    for (name, side) in left_attrs
        .iter()
        .map(|n| (n, "left"))
        .chain(right_attrs.iter().map(|n| (n, "right")))
    {
        if name.contains('.') {
            return plan_error(format!(
                "SEM_JOIN: the {side} input column \"{name}\" has a '.' in its name, which \
                 SEM_JOIN cannot bind (it identifies columns by name and splits on '.'); \
                 rename it in a subquery, e.g. (SELECT \"{name}\" AS {}, ... FROM t)",
                name.replace('.', "_")
            ));
        }
    }

    let mut left_filters = Vec::new();
    let mut right_filters = Vec::new();
    let mut above = Vec::new();
    for conjunct in relational {
        match placement(&conjunct, ls, rs) {
            Placement::Left => left_filters.push(conjunct),
            Placement::Right => right_filters.push(conjunct),
            Placement::Above => above.push(conjunct),
            Placement::BothSides => {
                return plan_error(format!(
                    "a condition over both sides of the join ({conjunct}) cannot be combined with \
                     SEM_JOIN in {clause}: the model would be evaluated over the whole cross product. \
                     Join on it relationally and filter semantically instead: \
                     a JOIN b ON a.k = b.k WHERE SEM_FILTER('...', 'model', a.x, b.y)"
                ));
            }
        }
    }
    let left_plan = with_filter(left, left_filters)?;
    let right_plan = with_filter(right, right_filters)?;

    let node = create_semantic_join(
        &left_plan,
        &right_plan,
        left_attrs,
        right_attrs,
        call.instruction,
        call.model,
    )?;

    // Restore the stock names only when the node renamed something, so that
    // nothing sits between the node and a selection above it otherwise.
    let renamed = node
        .schema()
        .iter()
        .zip(stock_schema.iter())
        .any(|((nq, nf), (sq, sf))| nq != sq || nf.name() != sf.name());
    let joined = if renamed {
        restore_names(node, stock_schema)?
    } else {
        node
    };
    // Conditions kept above are built against the stock join's schema, which
    // `joined` has (above the fix-up projection when there is one).
    with_filter(&Arc::new(joined), above)
}

/// A projection over `node` restoring `stock_schema`'s names and qualifiers.
fn restore_names(node: LogicalPlan, stock_schema: &DFSchemaRef) -> Result<LogicalPlan> {
    let node_schema = node.schema().clone();
    let exprs = node_schema
        .iter()
        .zip(stock_schema.iter())
        .map(|((nq, nf), (sq, sf))| {
            let column = Expr::Column(Column::new(nq.cloned(), nf.name()));
            if nq == sq && nf.name() == sf.name() {
                column
            } else {
                column.alias_qualified(sq.cloned(), sf.name())
            }
        })
        .collect();
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(node),
    )?))
}

/// Where a relational conjunct written next to a `SEM_JOIN` goes.
#[derive(Debug, PartialEq, Eq)]
enum Placement {
    /// A `Filter` on the left input.
    Left,
    /// A `Filter` on the right input.
    Right,
    /// A `Filter` above the semantic join (and above its fix-up projection).
    Above,
    /// A plain condition over both sides: an error (cross-product message).
    BothSides,
}

/// Places one relational conjunct of a semantic join, from the columns of this
/// query level it reads: its own `Column`s plus the outer-reference columns
/// of every subquery nested in it.
///
/// - correlated (holds an outer reference: this query block is itself a
///   correlated subquery), volatile, or reading no column: above the join
///   (decorrelation cannot pull a predicate up through the semantic node, a
///   volatile one must be drawn once per pair, and a constant one has no side);
/// - every column on the left (else the right) side: that side;
/// - columns of both sides through a subquery: above the join;
/// - columns of both sides otherwise: [`Placement::BothSides`].
fn placement(conjunct: &Expr, ls: &DFSchema, rs: &DFSchema) -> Placement {
    if conjunct.contains_outer() || conjunct.is_volatile() {
        return Placement::Above;
    }
    let (columns, has_subquery) = columns_read(conjunct);
    if columns.is_empty() {
        return Placement::Above;
    }
    if columns.iter().all(|c| ls.has_column(c)) {
        Placement::Left
    } else if columns.iter().all(|c| rs.has_column(c)) {
        Placement::Right
    } else if has_subquery {
        Placement::Above
    } else {
        Placement::BothSides
    }
}

/// The columns `expr` reads at this query level (its own `Column`s and the
/// `outer_ref_columns` of its subqueries), and whether it holds a subquery.
fn columns_read(expr: &Expr) -> (Vec<Column>, bool) {
    let mut columns: Vec<Column> = expr.column_refs().into_iter().cloned().collect();
    let mut has_subquery = false;
    expr.apply(|e| {
        let subquery = match e {
            Expr::Exists(Exists { subquery, .. })
            | Expr::InSubquery(InSubquery { subquery, .. })
            | Expr::SetComparison(SetComparison { subquery, .. })
            | Expr::ScalarSubquery(subquery) => subquery,
            _ => return Ok(TreeNodeRecursion::Continue),
        };
        has_subquery = true;
        for outer in &subquery.outer_ref_columns {
            outer.apply(|x| {
                if let Expr::OuterReferenceColumn(_, c) = x {
                    columns.push(c.clone());
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("the closure is infallible");
    (columns, has_subquery)
}

fn with_filter(input: &Arc<LogicalPlan>, conjuncts: Vec<Expr>) -> Result<LogicalPlan> {
    Ok(match conjunction(conjuncts) {
        Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, input.clone())?),
        None => input.as_ref().clone(),
    })
}
