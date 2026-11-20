//! `SEM_FILTER` (and `SEM_JOIN` over a comma/CROSS join) in `WHERE` or
//! `HAVING`: a `Filter` becomes a relational `Filter` under one
//! `SemanticSelectionNode` per call, with the correlated conjuncts (those
//! holding an outer reference) in a `Filter` above the selections.

use std::sync::Arc;

use datafusion::common::JoinType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, Filter, LogicalPlan};
use nousdb_operators::create_semantic_selection;

use super::{first_marker, join, supported_positions};
use crate::args::{SemFilterCall, SemJoinCall};
use crate::columns::bare_input_name;
use crate::markers::{MarkerKind, marker_kind};

/// The conjuncts of a predicate (a `Filter`'s or a join's `ON`), by kind.
#[derive(Default)]
pub(super) struct Conjuncts {
    /// Bare `SEM_FILTER` calls, in written order.
    pub sem_filters: Vec<SemFilterCall>,
    /// Bare `SEM_JOIN` calls, in written order.
    pub sem_joins: Vec<SemJoinCall>,
    /// Conjuncts with no marker anywhere inside, in written order.
    pub relational: Vec<Expr>,
}

/// Splits `predicate` on `AND` and parses every marker conjunct. A marker
/// anywhere else in the predicate (under `OR`/`NOT`, inside another
/// expression) and `EXTEND` anywhere are errors; `clause` names the clause.
pub(super) fn classify(predicate: &Expr, clause: &str) -> Result<Conjuncts> {
    let mut out = Conjuncts::default();
    for conjunct in split_conjunction(predicate) {
        match marker_kind(conjunct) {
            Some(MarkerKind::SemFilter) => {
                out.sem_filters.push(SemFilterCall::parse(args(conjunct))?)
            }
            Some(MarkerKind::SemJoin) => out.sem_joins.push(SemJoinCall::parse(args(conjunct))?),
            Some(MarkerKind::Extend) => return extend_in_predicate(clause),
            None => match first_marker(conjunct)? {
                None => out.relational.push(conjunct.clone()),
                Some(MarkerKind::Extend) => return extend_in_predicate(clause),
                Some(kind) => {
                    return Err(DataFusionError::Plan(format!(
                        "{name} must be a whole condition AND-ed with the others in {clause}; it \
                         cannot appear under OR or NOT or inside another expression (found in: \
                         {conjunct}). {name} is supported {supported}",
                        name = kind.sql_name(),
                        supported = supported_positions(kind),
                    )));
                }
            },
        }
    }
    Ok(out)
}

fn args(marker: &Expr) -> &[Expr] {
    match marker {
        Expr::ScalarFunction(f) => &f.args,
        _ => unreachable!("marker_kind matched a scalar function"),
    }
}

fn extend_in_predicate<T>(clause: &str) -> Result<T> {
    Err(DataFusionError::Plan(format!(
        "EXTEND is not supported in a predicate ({clause}); compute it in a subquery's SELECT \
         list and filter on the column, e.g. SELECT * FROM (SELECT *, EXTEND('...', 'model', \
         column) AS x FROM t) WHERE x = '...'. EXTEND is supported {}",
        supported_positions(MarkerKind::Extend)
    )))
}

pub(super) fn rewrite(f: &Filter) -> Result<LogicalPlan> {
    let Conjuncts {
        sem_filters,
        sem_joins,
        relational,
    } = classify(&f.predicate, "WHERE")?;
    // A correlated conjunct (this Filter is in a correlated subquery) stays
    // above every semantic node: DataFusion's decorrelation pulls predicates
    // up from the top of the subquery and cannot pass an extension node.
    let (correlated, relational): (Vec<Expr>, Vec<Expr>) =
        relational.into_iter().partition(Expr::contains_outer);

    let mut plan = match sem_joins.len() {
        0 => match conjunction(relational) {
            Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, f.input.clone())?),
            None => f.input.as_ref().clone(),
        },
        1 => {
            let call = sem_joins.into_iter().next().expect("one call");
            match f.input.as_ref() {
                LogicalPlan::Join(j)
                    if j.join_type == JoinType::Inner && j.on.is_empty() && j.filter.is_none() =>
                {
                    join::build(&j.left, &j.right, &j.schema, call, relational, "WHERE")?
                }
                _ => {
                    return Err(DataFusionError::Plan(
                        "SEM_JOIN in WHERE is supported only directly over a comma or CROSS \
                         join of two relations (FROM a, b WHERE SEM_JOIN(...)); use \
                         JOIN ... ON SEM_JOIN(...)"
                            .into(),
                    ));
                }
            }
        }
        n => {
            return Err(DataFusionError::Plan(format!(
                "only one SEM_JOIN per join is supported, found {n} in WHERE; combine the \
                 conditions into one instruction"
            )));
        }
    };

    // First written innermost; inputs are checked against the Filter's input
    // schema, which every relational step preserves.
    let schema = f.input.schema().clone();
    for call in sem_filters {
        let inputs = call
            .inputs
            .iter()
            .map(|c| bare_input_name(c, &schema, "SEM_FILTER"))
            .collect::<Result<Vec<_>>>()?;
        plan = create_semantic_selection(&plan, call.model, inputs, call.instruction)?;
    }
    if let Some(predicate) = conjunction(correlated) {
        plan = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(plan))?);
    }
    Ok(plan)
}
