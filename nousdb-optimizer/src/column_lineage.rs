//! Plan-time column lineage: which base-table column does an operator's input
//! column read?
//!
//! `HistogramCatalog` is keyed by *registered table name* and *source column
//! name*, while a semantic operator names its columns as they appear in its
//! input, possibly qualified by a `SubqueryAlias` (`drama_movies.title`) or
//! renamed by a `Projection` (`review AS text`). [`trace_column`] follows one
//! column down through the plan by field position until it reaches the
//! `TableScan` that produces it, so the catalog can be asked about the column
//! it actually stores.
//!
//! Tracing is deliberately conservative: anything that computes a value
//! (a projection expression, an aggregate, a derived extend column) or merges
//! several relations into one column (a union) stops it, and the caller
//! decides what to do instead.

use datafusion::common::{Column, DFSchema, JoinType};
use datafusion::logical_expr::{Distinct, Expr, LogicalPlan};

use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::generate::GenerateNode;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;
use nousdb_operators::token_repartition::TokenRepartitionNode;

/// A column as stored by a registered table: what the histogram catalog keys on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BaseColumn {
    /// Unqualified registered table name (`TableScan::table_name.table()`).
    pub table: String,
    /// The column's name in that table.
    pub column: String,
}

/// Why a column could not be traced to a single base column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Untraced {
    /// No field of the schema matches the column.
    NotFound,
    /// More than one field matches an unqualified name.
    Ambiguous,
    /// The column is computed or merged here; the payload names the node.
    Opaque(String),
}

impl std::fmt::Display for Untraced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Untraced::NotFound => write!(f, "not in the input schema"),
            Untraced::Ambiguous => write!(f, "ambiguous unqualified name"),
            Untraced::Opaque(what) => write!(f, "computed by {what}"),
        }
    }
}

/// Trace `column`, as resolved against `plan`'s output schema, to the base
/// column of the `TableScan` that produces it.
pub(crate) fn trace_column(plan: &LogicalPlan, column: &Column) -> Result<BaseColumn, Untraced> {
    let idx = unique_index(plan.schema(), column)?;
    trace_index(plan, idx)
}

/// The position of `column` in `schema`, matched the way DataFusion matches
/// it (`DFSchema::index_of_column_by_name`), except that an unqualified name
/// matching several fields is reported instead of resolving to the first.
fn unique_index(schema: &DFSchema, column: &Column) -> Result<usize, Untraced> {
    let mut matches = schema
        .iter()
        .enumerate()
        .filter(|(_, (qualifier, field))| {
            field.name() == &column.name
                && match (&column.relation, qualifier) {
                    (Some(wanted), Some(have)) => wanted.resolved_eq(have),
                    (Some(_), None) => false,
                    (None, _) => true,
                }
        })
        .map(|(idx, _)| idx);
    match (matches.next(), matches.next()) {
        (Some(idx), None) => Ok(idx),
        (None, _) => Err(Untraced::NotFound),
        (Some(_), Some(_)) => Err(Untraced::Ambiguous),
    }
}

/// Trace output field `idx` of `plan` to its base column.
fn trace_index(plan: &LogicalPlan, idx: usize) -> Result<BaseColumn, Untraced> {
    match plan {
        LogicalPlan::TableScan(scan) => Ok(BaseColumn {
            table: scan.table_name.table().to_string(),
            column: scan.projected_schema.field(idx).name().clone(),
        }),
        // Re-qualifies every field, keeping positions.
        LogicalPlan::SubqueryAlias(alias) => trace_index(&alias.input, idx),
        LogicalPlan::Projection(proj) => match unalias(&proj.expr[idx]) {
            Expr::Column(col) => trace_column(&proj.input, col),
            _ => Err(Untraced::Opaque("a projection expression".to_string())),
        },
        // Same schema as the input.
        LogicalPlan::Filter(f) => trace_index(&f.input, idx),
        LogicalPlan::Limit(l) => trace_index(&l.input, idx),
        LogicalPlan::Sort(s) => trace_index(&s.input, idx),
        LogicalPlan::Repartition(r) => trace_index(&r.input, idx),
        LogicalPlan::Distinct(Distinct::All(input)) => trace_index(input, idx),
        // Input fields first, then one field per window expression.
        LogicalPlan::Window(w) if idx < w.input.schema().fields().len() => {
            trace_index(&w.input, idx)
        }
        // Group columns first (DataFusion rewrites `SELECT DISTINCT` into
        // this); a plain column group key passes its values through. Grouping
        // sets add an internal grouping-id field, so they are not traced.
        LogicalPlan::Aggregate(agg)
            if idx < agg.group_expr.len()
                && !agg
                    .group_expr
                    .iter()
                    .any(|e| matches!(e, Expr::GroupingSet(_))) =>
        {
            match unalias(&agg.group_expr[idx]) {
                Expr::Column(col) => trace_column(&agg.input, col),
                _ => Err(Untraced::Opaque("a group-by expression".to_string())),
            }
        }
        // Field layout of DataFusion's `build_join_schema`.
        LogicalPlan::Join(join) => {
            let left_len = join.left.schema().fields().len();
            let right_len = join.right.schema().fields().len();
            match join.join_type {
                JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                    if idx < left_len {
                        trace_index(&join.left, idx)
                    } else {
                        trace_index(&join.right, idx - left_len)
                    }
                }
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark if idx < left_len => {
                    trace_index(&join.left, idx)
                }
                JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
                    if idx < right_len =>
                {
                    trace_index(&join.right, idx)
                }
                _ => Err(Untraced::Opaque("a mark join".to_string())),
            }
        }
        LogicalPlan::Union(_) => Err(Untraced::Opaque("a union".to_string())),
        LogicalPlan::Extension(ext) => {
            let node = ext.node.as_any();
            if let Some(extend) = node.downcast_ref::<ExtendNode>() {
                // Input fields first, then the derived columns.
                if idx < extend.input.schema().fields().len() {
                    trace_index(&extend.input, idx)
                } else {
                    Err(Untraced::Opaque("an extend".to_string()))
                }
            } else if let Some(sel) = node.downcast_ref::<SemanticSelectionNode>() {
                trace_index(&sel.input, idx)
            } else if let Some(tr) = node.downcast_ref::<TokenRepartitionNode>() {
                trace_index(&tr.input, idx)
            } else if let Some(join) = node.downcast_ref::<SemanticJoinNode>() {
                // `build_join_schema`: every left field, then every right
                // field (renamed on collision, but in place).
                let left_len = join.left_input.schema().fields().len();
                if idx < left_len {
                    trace_index(&join.left_input, idx)
                } else {
                    trace_index(&join.right_input, idx - left_len)
                }
            } else if node.downcast_ref::<GenerateNode>().is_some() {
                Err(Untraced::Opaque("a generate".to_string()))
            } else {
                Err(Untraced::Opaque(format!("extension {}", ext.node.name())))
            }
        }
        // The one-line display starts with the node name ("Aggregate: ...").
        other => Err(Untraced::Opaque(
            other
                .display()
                .to_string()
                .split(':')
                .next()
                .unwrap_or_default()
                .to_string(),
        )),
    }
}

/// The expression under any number of `AS` aliases.
fn unalias(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = &alias.expr;
    }
    expr
}
