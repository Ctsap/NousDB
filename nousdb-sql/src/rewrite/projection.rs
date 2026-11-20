//! `EXTEND` in a SELECT list: a `Projection` becomes a stack of `ExtendNode`s,
//! with a residual projection only when it is not an identity.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use nousdb_operators::create_extend;

use super::expr_contains_marker;
use crate::args::ExtendCall;
use crate::columns::bare_input_name;
use crate::markers::{MarkerKind, marker_kind};

/// The name given to a derived column that has no spec name and no alias.
const GENERATED_NAME: &str = "result";

/// One distinct `EXTEND` call of the SELECT list.
struct Occurrence {
    /// The marker expression, as written (the dedupe key).
    expr: Expr,
    call: ExtendCall,
    /// The alias of the first occurrence that is a whole aliased expression.
    alias: Option<String>,
    /// The derived column's name, once chosen.
    derived: String,
}

pub(super) fn rewrite(p: &Projection) -> Result<LogicalPlan> {
    reject_boolean_markers(&p.expr)?;

    let mut calls: Vec<Occurrence> = Vec::new();
    for e in &p.expr {
        let top_alias = match e {
            Expr::Alias(a) if marker_kind(&a.expr) == Some(MarkerKind::Extend) => {
                Some((a.expr.as_ref(), a.name.clone()))
            }
            _ => None,
        };
        let mut error = None;
        e.apply(|sub| {
            if marker_kind(sub) != Some(MarkerKind::Extend) {
                return Ok(TreeNodeRecursion::Continue);
            }
            let alias = top_alias
                .as_ref()
                .filter(|(m, _)| *m == sub)
                .map(|(_, a)| a.clone());
            if let Some(existing) = calls.iter_mut().find(|o| &o.expr == sub) {
                if existing.alias.is_none() {
                    existing.alias = alias;
                }
            } else {
                let Expr::ScalarFunction(f) = sub else {
                    unreachable!("marker_kind matched a scalar function")
                };
                match ExtendCall::parse(&f.args) {
                    Ok(call) => calls.push(Occurrence {
                        expr: sub.clone(),
                        call,
                        alias,
                        derived: String::new(),
                    }),
                    Err(e) => {
                        error = Some(e);
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
            }
            // A marker's arguments hold no other marker that parses.
            Ok(TreeNodeRecursion::Jump)
        })?;
        if let Some(e) = error {
            return Err(e);
        }
    }

    // Stack one extend per call, first written innermost.
    let input_schema = p.input.schema().clone();
    let mut used: HashSet<String> = input_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut plan = p.input.as_ref().clone();
    for occ in &mut calls {
        let inputs = occ
            .call
            .inputs
            .iter()
            .map(|c| bare_input_name(c, &input_schema, "EXTEND"))
            .collect::<Result<Vec<_>>>()?;
        let wanted = occ
            .call
            .output
            .name
            .clone()
            .or_else(|| occ.alias.clone())
            .unwrap_or_else(|| GENERATED_NAME.to_string());
        let name = free_name(&wanted, &used);
        used.insert(name.clone());
        let derived = Arc::new(Schema::new(vec![Field::new(
            &name,
            occ.call.output.data_type.clone(),
            true,
        )]));
        plan = create_extend(
            plan,
            occ.call.instruction.clone(),
            occ.call.model.clone(),
            inputs,
            derived,
        )?;
        occ.derived = name;
    }

    // Residual expressions: each marker becomes its derived column, every
    // output keeps its name and qualifier.
    let preserver = NamePreserver::new_for_projection();
    let residual = p
        .expr
        .iter()
        .map(|e| {
            if !expr_contains_marker(e)? {
                return Ok(e.clone());
            }
            let saved = preserver.save(e);
            let replaced = e
                .clone()
                .unalias()
                .transform_down(|sub| match calls.iter().find(|o| o.expr == sub) {
                    Some(o) => Ok(Transformed::new(
                        Expr::Column(Column::new_unqualified(o.derived.clone())),
                        true,
                        TreeNodeRecursion::Jump,
                    )),
                    None => Ok(Transformed::no(sub)),
                })?
                .data;
            Ok(saved.restore(replaced))
        })
        .collect::<Result<Vec<_>>>()?;

    if is_identity(&residual, &plan) {
        return Ok(plan);
    }
    Ok(LogicalPlan::Projection(Projection::try_new(
        residual,
        Arc::new(plan),
    )?))
}

/// `wanted` if no column of the input or earlier derived column has that
/// name, else the first free `wanted_1`, `wanted_2`, ...
fn free_name(wanted: &str, used: &HashSet<String>) -> String {
    if !used.contains(wanted) {
        return wanted.to_string();
    }
    (1..)
        .map(|i| format!("{wanted}_{i}"))
        .find(|n| !used.contains(n))
        .expect("an unbounded range has a free name")
}

/// Whether `exprs` are exactly the fields of `plan`, in order, as plain columns.
fn is_identity(exprs: &[Expr], plan: &LogicalPlan) -> bool {
    let schema = plan.schema();
    exprs.len() == schema.fields().len()
        && exprs.iter().zip(schema.iter()).all(|(e, (q, f))| match e {
            Expr::Column(c) => c.relation.as_ref() == q && &c.name == f.name(),
            _ => false,
        })
}

/// `SEM_FILTER` / `SEM_JOIN` produce no column; reject them in a SELECT list.
fn reject_boolean_markers(exprs: &[Expr]) -> Result<()> {
    for e in exprs {
        let mut found = None;
        e.apply(|sub| {
            if let Some(k @ (MarkerKind::SemFilter | MarkerKind::SemJoin)) = marker_kind(sub) {
                found = Some(k);
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if let Some(kind) = found {
            let name = kind.sql_name();
            let place = match kind {
                MarkerKind::SemFilter => "to filter rows, put it in WHERE",
                _ => "to join, put it in an INNER JOIN's ON clause",
            };
            return Err(DataFusionError::Plan(format!(
                "{name} is not supported in a SELECT list: {place}; to get the model's yes/no \
                 answer as a column, use EXTEND('...', 'model', column, 'Boolean')"
            )));
        }
    }
    Ok(())
}
