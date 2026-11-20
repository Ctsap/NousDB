//! Logical rule applying algebraic rewrites to semantic operators.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DFSchema};
use datafusion::datasource::source_as_provider;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, Projection};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::generate::logical::GenerateNode;
use nousdb_operators::join::logical::{SemanticJoinNode, build_join_schema};
use nousdb_operators::selection::logical::SemanticSelectionNode;

/// Optimizer rule that restructures plan trees containing semantic operators
/// based on algebraic properties of the operators.
///
/// Runs TopDown so rewrites are applied from the root towards leaves.
///
/// ## Rewrites
///
/// **A: Selection pushdown before Extend**
/// `σ_sel(extend_E(R))` → `extend_E(σ_sel(R))`
/// when selection's input columns don't reference extended columns.
///
/// **B: Selection pushdown before Join**
/// `σ_sel(R ⋈ S)` → `σ_sel(R) ⋈ S` or `R ⋈ σ_sel(S)`
/// when selection columns come only from one side.
///
/// **C: Selection reordering**
/// `σ_expensive(σ_cheap(R))` → `σ_cheap(σ_expensive(R))`
/// where fewer input columns = cheaper.
///
/// **D: Join commutativity**
/// `R ⋈ S` → `S ⋈ R` when R is smaller (put larger on streaming/left side).
///
/// **E: Extend-Selection Fusion**
/// `π(σ_sel(ε_E(R)))` → fused `σ(R)` when selection depends on E,
/// projection removes E, and both use the same model.
///
/// **F: Extend-Join Fusion**
/// `π(ε_E(R) ⋈ S)` → fused `R ⋈ S` when E is only used as join key,
/// projection removes E, and extend/join use the same model.
#[derive(Debug)]
pub struct SemanticRewritingRule;

impl OptimizerRule for SemanticRewritingRule {
    fn name(&self) -> &str {
        "SemanticRewritingRule"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Rewrites E & F: fusion rules operate on Projection nodes
        if let LogicalPlan::Projection(ref proj) = plan {
            if let Some(result) = try_extend_selection_fusion(proj)? {
                return Ok(Transformed::yes(result));
            }
            if let Some(result) = try_extend_join_fusion(proj)? {
                return Ok(Transformed::yes(result));
            }
        }

        if let LogicalPlan::Extension(ext) = &plan {
            // Try rewrites on selection nodes
            if let Some(sel) = ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                if let Some(result) = try_pushdown_selection_before_extend(sel)? {
                    return Ok(Transformed::yes(result));
                }
                if let Some(result) = try_pushdown_selection_before_join(sel)? {
                    return Ok(Transformed::yes(result));
                }
                if let Some(result) = try_reorder_selections(sel)? {
                    return Ok(Transformed::yes(result));
                }
            }

            // Try rewrites on join nodes
            if let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>()
                && let Some(result) = try_join_commutativity(join)?
            {
                return Ok(Transformed::yes(result));
            }
        }

        Ok(Transformed::no(plan))
    }
}

/// Rewrite A: Push selection below extend when the selection doesn't reference
/// any of the extended (derived) columns.
///
/// Before: σ_sel(extend_E(R))
/// After:  extend_E(σ_sel(R))
fn try_pushdown_selection_before_extend(
    sel: &SemanticSelectionNode,
) -> Result<Option<LogicalPlan>> {
    if let LogicalPlan::Extension(child_ext) = &sel.input
        && let Some(extend) = child_ext.node.as_any().downcast_ref::<ExtendNode>()
    {
        // Check if any of the selection's input columns reference extended columns
        let derived_col_names: Vec<&str> = extend
            .derived_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        let references_derived = sel
            .input_schema
            .iter()
            .any(|col| derived_col_names.contains(&col.as_str()));

        if !references_derived {
            // Safe to push selection below extend
            // Build: extend_E(σ_sel(R))
            let new_selection = SemanticSelectionNode::new(
                extend.input.clone(),
                sel.model.clone(),
                sel.input_schema.clone(),
                sel.prompt.clone(),
                extend.input.schema().clone(),
            );

            let selection_plan = LogicalPlan::Extension(Extension {
                node: Arc::new(new_selection),
            });

            let mut new_extend = ExtendNode::new(
                selection_plan,
                extend.input_schema.clone(),
                extend.model.clone(),
                extend.prompt.clone(),
                extend.derived_schema.clone(),
                extend.schema.clone(),
            );
            new_extend.optimal_batch_size = extend.optimal_batch_size;

            return Ok(Some(LogicalPlan::Extension(Extension {
                node: Arc::new(new_extend),
            })));
        }
    }
    Ok(None)
}

/// Rewrite B: Push selection below join when selection columns come from only
/// one side of the join.
///
/// Before: σ_sel(R ⋈ S)
/// After:  σ_sel(R) ⋈ S  (when sel columns come only from R)
///    or:  R ⋈ σ_sel(S)  (when sel columns come only from S)
fn try_pushdown_selection_before_join(sel: &SemanticSelectionNode) -> Result<Option<LogicalPlan>> {
    if let LogicalPlan::Extension(child_ext) = &sel.input
        && let Some(join) = child_ext.node.as_any().downcast_ref::<SemanticJoinNode>()
    {
        let left_schema = join.left_input.schema();
        let right_schema = join.right_input.schema();
        let left_len = left_schema.fields().len();

        // Resolve each selection column against the join's output schema (the
        // selection's input) and find which side produces it. A right column
        // that the join renamed (`title_1`) is not a column of the right input
        // under that name, so a selection reading it is not pushed down.
        let mut all_from_left = true;
        let mut all_from_right = true;
        for col in &sel.input_schema {
            let Some(idx) = resolve_index(&join.schema, &Column::new_unqualified(col)) else {
                return Ok(None);
            };
            if idx < left_len {
                all_from_right = false;
                if left_schema.field(idx).name() != col {
                    all_from_left = false;
                }
            } else {
                all_from_left = false;
                if right_schema.field(idx - left_len).name() != col {
                    all_from_right = false;
                }
            }
        }

        if all_from_left {
            // Push selection to left side: σ_sel(R) ⋈ S
            let mut new_selection = SemanticSelectionNode::new(
                join.left_input.clone(),
                sel.model.clone(),
                sel.input_schema.clone(),
                sel.prompt.clone(),
                left_schema.clone(),
            );
            new_selection.optimal_batch_size = sel.optimal_batch_size;
            let selection_plan = LogicalPlan::Extension(Extension {
                node: Arc::new(new_selection),
            });

            let output_schema = build_join_schema(selection_plan.schema(), right_schema)?;
            let mut new_join = SemanticJoinNode::new(
                selection_plan,
                join.right_input.clone(),
                join.left_join_attrs.clone(),
                join.right_join_attrs.clone(),
                join.model.clone(),
                join.prompt.clone(),
                output_schema,
            );
            new_join.left_block_size = join.left_block_size;
            new_join.right_block_size = join.right_block_size;

            return Ok(Some(LogicalPlan::Extension(Extension {
                node: Arc::new(new_join),
            })));
        } else if all_from_right {
            // Push selection to right side: R ⋈ σ_sel(S)
            let mut new_selection = SemanticSelectionNode::new(
                join.right_input.clone(),
                sel.model.clone(),
                sel.input_schema.clone(),
                sel.prompt.clone(),
                right_schema.clone(),
            );
            new_selection.optimal_batch_size = sel.optimal_batch_size;
            let selection_plan = LogicalPlan::Extension(Extension {
                node: Arc::new(new_selection),
            });

            let output_schema = build_join_schema(left_schema, selection_plan.schema())?;
            let mut new_join = SemanticJoinNode::new(
                join.left_input.clone(),
                selection_plan,
                join.left_join_attrs.clone(),
                join.right_join_attrs.clone(),
                join.model.clone(),
                join.prompt.clone(),
                output_schema,
            );
            new_join.left_block_size = join.left_block_size;
            new_join.right_block_size = join.right_block_size;

            return Ok(Some(LogicalPlan::Extension(Extension {
                node: Arc::new(new_join),
            })));
        }
    }
    Ok(None)
}

/// Rewrite C: Reorder adjacent selections so the cheaper one runs first.
///
/// Before: σ_expensive(σ_cheap(R))
/// After:  σ_cheap(σ_expensive(R))
///
/// Heuristic: fewer input columns = cheaper (less data for the LLM to process).
fn try_reorder_selections(outer: &SemanticSelectionNode) -> Result<Option<LogicalPlan>> {
    if let LogicalPlan::Extension(child_ext) = &outer.input
        && let Some(inner) = child_ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
    {
        // If outer is cheaper (fewer input columns) than inner, swap them
        // so the cheaper filter runs first (closest to the data source).
        if outer.input_schema.len() < inner.input_schema.len() {
            // Build: inner(outer(R))
            // outer becomes the new inner (runs first)
            let mut new_inner = SemanticSelectionNode::new(
                inner.input.clone(),
                outer.model.clone(),
                outer.input_schema.clone(),
                outer.prompt.clone(),
                inner.input.schema().clone(),
            );
            new_inner.optimal_batch_size = outer.optimal_batch_size;
            let inner_plan = LogicalPlan::Extension(Extension {
                node: Arc::new(new_inner),
            });

            let mut new_outer = SemanticSelectionNode::new(
                inner_plan,
                inner.model.clone(),
                inner.input_schema.clone(),
                inner.prompt.clone(),
                inner.schema.clone(),
            );
            new_outer.optimal_batch_size = inner.optimal_batch_size;

            return Ok(Some(LogicalPlan::Extension(Extension {
                node: Arc::new(new_outer),
            })));
        }
    }
    Ok(None)
}

/// Rewrite D: Join commutativity, put the larger relation on the left (streaming)
/// side and the smaller on the right (build) side.
///
/// Before: R ⋈ S where |R| < |S|
/// After:  S ⋈ R (swap so the larger table streams, smaller is materialized)
fn try_join_commutativity(join: &SemanticJoinNode) -> Result<Option<LogicalPlan>> {
    // Use row count from statistics if available
    let left_rows = get_row_count_estimate(&join.left_input);
    let right_rows = get_row_count_estimate(&join.right_input);

    if let (Some(left_count), Some(right_count)) = (left_rows, right_rows)
        && left_count < right_count
    {
        // Swap: put larger relation on left (streaming side)
        let output_schema = build_join_schema(join.right_input.schema(), join.left_input.schema())?;

        let mut new_join = SemanticJoinNode::new(
            join.right_input.clone(),
            join.left_input.clone(),
            join.right_join_attrs.clone(),
            join.left_join_attrs.clone(),
            join.model.clone(),
            join.prompt.clone(),
            output_schema,
        );
        // Carry every plan-time annotation across the swap, sides included.
        // This rule runs before `HistogramInjectionRule` and
        // `BatchSizeOptimizationRule`, so they are normally `None` here, but
        // dropping them would leave the join unsized (and uncertifiable) if
        // the rule order changed.
        new_join.left_block_size = join.right_block_size;
        new_join.right_block_size = join.left_block_size;
        new_join.token_histogram = join.left_token_histogram.clone();
        new_join.left_token_histogram = join.token_histogram.clone();
        new_join.batch_budget = join.batch_budget.clone();
        new_join.join_batch_budget = join.join_batch_budget.clone();
        new_join.model_max_batch_size = join.model_max_batch_size;
        new_join.model_max_output_tokens = join.model_max_output_tokens;

        log::debug!(
            "[SemanticRewritingRule] Rewrite D: swapping join operands \
                 (left {left_count} rows < right {right_count} rows)"
        );

        let swapped = LogicalPlan::Extension(Extension {
            node: Arc::new(new_join),
        });

        // Swapping the operands swaps the output columns with them, and an
        // optimizer rule may not change its plan's schema (DataFusion asserts
        // this after every rule). Project the original column order back on top.
        return Ok(Some(restore_column_order(
            swapped,
            join.schema.as_ref(),
            join.left_input.schema().fields().len(),
        )?));
    }

    Ok(None)
}

/// Whether every projection expression only selects a column (possibly
/// renamed with an alias), the shape fusion rewrites E and F require.
fn is_column_selection(proj: &Projection) -> bool {
    proj.expr
        .iter()
        .all(|expr| projected_column(expr).is_some())
}

/// Rewrite E: Extend-Selection Fusion
///
/// Pattern: `π_cols(σ_sel(ε_E(R)))` where:
/// - the projection only selects (and possibly renames) columns
/// - Selection depends on at least one column from E (derived_schema)
/// - Projection removes all E columns
/// - Both operators use the same model
///
/// Result: `π_cols(σ_fused(R))` with a composed prompt that first extends then
/// filters. The projection is rebuilt from the original expressions (so
/// qualifiers and aliases survive) and dropped only when it selects every
/// column of R in order.
fn try_extend_selection_fusion(proj: &Projection) -> Result<Option<LogicalPlan>> {
    if !is_column_selection(proj) {
        return Ok(None);
    }

    // proj.input is the selection
    let sel = match proj.input.as_ref() {
        LogicalPlan::Extension(ext) => {
            match ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                Some(s) => s,
                None => return Ok(None),
            }
        }
        _ => return Ok(None),
    };

    // sel.input is the extend
    let extend = match &sel.input {
        LogicalPlan::Extension(ext) => match ext.node.as_any().downcast_ref::<ExtendNode>() {
            Some(e) => e,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Same model requirement
    if sel.model != extend.model {
        return Ok(None);
    }

    let derived_col_names: Vec<String> = extend
        .derived_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    // Selection must depend on at least one derived column
    let depends_on_derived = sel
        .input_schema
        .iter()
        .any(|col| derived_col_names.contains(col));
    if !depends_on_derived {
        return Ok(None);
    }

    // The extend's output is R's fields, then the derived ones, so a projected
    // column at position >= |R| is derived. The projection must keep none.
    let r_plan = &extend.input;
    let r_len = r_plan.schema().fields().len();
    for expr in &proj.expr {
        let keeps_derived = projected_column(expr)
            .and_then(|col| resolve_index(proj.input.schema(), col))
            .is_none_or(|idx| idx >= r_len);
        if keeps_derived {
            return Ok(None);
        }
    }

    // Compose prompt
    let composed_prompt = format!(
        "First, for each row, {}. Then, using those results, {}",
        extend.prompt, sel.prompt
    );

    // Fused input_schema: union of extend's and selection's input columns (deduped)
    let mut fused_input_cols = extend.input_schema.clone();
    for col in &sel.input_schema {
        if !derived_col_names.contains(col) && !fused_input_cols.contains(col) {
            fused_input_cols.push(col.clone());
        }
    }

    // Build fused selection on R (extend's original input)
    let fused_selection = SemanticSelectionNode::new(
        r_plan.clone(),
        sel.model.clone(),
        fused_input_cols,
        composed_prompt,
        r_plan.schema().clone(),
    );

    let fused_plan = LogicalPlan::Extension(Extension {
        node: Arc::new(fused_selection),
    });

    // R's fields keep their positions in the extend's output.
    reproject(proj, fused_plan, |idx| (idx < r_len).then_some(idx))
}

/// Rewrite F: Extend-Join Fusion
///
/// Pattern: `π_cols(ε_E(R) ⋈ S)` where:
/// - the projection only selects (and possibly renames) columns
/// - Extend's derived columns appear in join's left_join_attrs (E used as join keys)
/// - Projection removes all E columns
/// - Extend and join use the same model
///
/// Also handles symmetric case where extend is on the right input. The
/// projection is rebuilt from the original expressions, rebound by position
/// to the fused join's fields (so qualifiers, aliases and the original names
/// of renamed right columns survive).
fn try_extend_join_fusion(proj: &Projection) -> Result<Option<LogicalPlan>> {
    if !is_column_selection(proj) {
        return Ok(None);
    }

    let join = match proj.input.as_ref() {
        LogicalPlan::Extension(ext) => match ext.node.as_any().downcast_ref::<SemanticJoinNode>() {
            Some(j) => j,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Try left fusion: extend on left input
    if let Some(result) = try_extend_join_fusion_left(proj, join)? {
        return Ok(Some(result));
    }

    // Try right fusion: extend on right input
    if let Some(result) = try_extend_join_fusion_right(proj, join)? {
        return Ok(Some(result));
    }

    Ok(None)
}

/// Whether some projected column of `proj` (over `join`) is at a position in
/// `derived` (the extend's derived columns in the join output), or does not
/// resolve.
fn projection_keeps(proj: &Projection, derived: std::ops::Range<usize>) -> bool {
    proj.expr.iter().any(|expr| {
        projected_column(expr)
            .and_then(|col| resolve_index(proj.input.schema(), col))
            .is_none_or(|idx| derived.contains(&idx))
    })
}

fn try_extend_join_fusion_left(
    proj: &Projection,
    join: &SemanticJoinNode,
) -> Result<Option<LogicalPlan>> {
    let extend = match &join.left_input {
        LogicalPlan::Extension(ext) => match ext.node.as_any().downcast_ref::<ExtendNode>() {
            Some(e) => e,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Same model
    if extend.model != join.model {
        return Ok(None);
    }

    let derived_col_names: Vec<String> = extend
        .derived_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    // Derived columns must appear in left_join_attrs
    let derived_in_join_keys = derived_col_names
        .iter()
        .any(|col| join.left_join_attrs.contains(col));
    if !derived_in_join_keys {
        return Ok(None);
    }

    // Join output: R's fields, the derived ones, then S's.
    let r_plan = &extend.input;
    let r_len = r_plan.schema().fields().len();
    let extend_len = join.left_input.schema().fields().len();

    // Projection must not include any derived columns
    if projection_keeps(proj, r_len..extend_len) {
        return Ok(None);
    }

    // Compose prompt
    let composed_prompt = format!(
        "First, compute from left row: {}. Then evaluate join condition: {}",
        extend.prompt, join.prompt
    );

    // New left join attrs: replace derived columns with extend's input columns
    let mut new_left_join_attrs: Vec<String> = join
        .left_join_attrs
        .iter()
        .filter(|attr| !derived_col_names.contains(attr))
        .cloned()
        .collect();
    for col in &extend.input_schema {
        if !new_left_join_attrs.contains(col) {
            new_left_join_attrs.push(col.clone());
        }
    }

    // Build new join on R (extend's original input) and S
    let output_schema = build_join_schema(r_plan.schema(), join.right_input.schema())?;

    let mut new_join = SemanticJoinNode::new(
        r_plan.clone(),
        join.right_input.clone(),
        new_left_join_attrs,
        join.right_join_attrs.clone(),
        join.model.clone(),
        composed_prompt,
        output_schema,
    );
    new_join.left_block_size = join.left_block_size;
    new_join.right_block_size = join.right_block_size;

    let join_plan = LogicalPlan::Extension(Extension {
        node: Arc::new(new_join),
    });

    // R keeps its positions; S shifts left by the number of derived columns.
    reproject(proj, join_plan, |idx| {
        if idx < r_len {
            Some(idx)
        } else if idx >= extend_len {
            Some(idx - (extend_len - r_len))
        } else {
            None
        }
    })
}

fn try_extend_join_fusion_right(
    proj: &Projection,
    join: &SemanticJoinNode,
) -> Result<Option<LogicalPlan>> {
    let extend = match &join.right_input {
        LogicalPlan::Extension(ext) => match ext.node.as_any().downcast_ref::<ExtendNode>() {
            Some(e) => e,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Same model
    if extend.model != join.model {
        return Ok(None);
    }

    let derived_col_names: Vec<String> = extend
        .derived_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    // Derived columns must appear in right_join_attrs
    let derived_in_join_keys = derived_col_names
        .iter()
        .any(|col| join.right_join_attrs.contains(col));
    if !derived_in_join_keys {
        return Ok(None);
    }

    // Join output: R's fields, S's, then the derived ones.
    let s_plan = &extend.input;
    let left_len = join.left_input.schema().fields().len();
    let s_end = left_len + s_plan.schema().fields().len();
    let total = join.schema.fields().len();

    // Projection must not include any derived columns
    if projection_keeps(proj, s_end..total) {
        return Ok(None);
    }

    // Compose prompt
    let composed_prompt = format!(
        "First, compute from right row: {}. Then evaluate join condition: {}",
        extend.prompt, join.prompt
    );

    // New right join attrs: replace derived columns with extend's input columns
    let mut new_right_join_attrs: Vec<String> = join
        .right_join_attrs
        .iter()
        .filter(|attr| !derived_col_names.contains(attr))
        .cloned()
        .collect();
    for col in &extend.input_schema {
        if !new_right_join_attrs.contains(col) {
            new_right_join_attrs.push(col.clone());
        }
    }

    // Build new join on R and S (extend's original input)
    let output_schema = build_join_schema(join.left_input.schema(), s_plan.schema())?;

    let mut new_join = SemanticJoinNode::new(
        join.left_input.clone(),
        s_plan.clone(),
        join.left_join_attrs.clone(),
        new_right_join_attrs,
        join.model.clone(),
        composed_prompt,
        output_schema,
    );
    new_join.left_block_size = join.left_block_size;
    new_join.right_block_size = join.right_block_size;

    let join_plan = LogicalPlan::Extension(Extension {
        node: Arc::new(new_join),
    });

    // R and S keep their positions; only the trailing derived columns go.
    reproject(proj, join_plan, |idx| (idx < s_end).then_some(idx))
}

/// Project the swapped join's columns back into the layout of the join it
/// replaces.
///
/// Used by the commutativity rewrite, which exchanges the two operands and
/// with them the two halves of the output schema. Matching is by position:
/// original field `i < left_len` (a left field) is swapped field
/// `right_len + i`, and original right field `i` is swapped field
/// `i - left_len`. Matching by name would be wrong when both sides share a
/// column name, because the swap moves the `_1` rename to the other side.
/// Each output keeps the original field's qualifier and name (an alias where
/// the swapped field is named differently), so the schema is unchanged.
fn restore_column_order(
    plan: LogicalPlan,
    original: &DFSchema,
    left_len: usize,
) -> Result<LogicalPlan> {
    let swapped = Arc::clone(plan.schema());
    let right_len = original.fields().len() - left_len;
    let exprs: Vec<Expr> = (0..original.fields().len())
        .map(|i| {
            let j = if i < left_len {
                right_len + i
            } else {
                i - left_len
            };
            column_named_as(&swapped, j, original, i)
        })
        .collect();
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(plan),
    )?))
}

/// A reference to field `j` of `schema`, named (qualifier included) as field
/// `i` of `target`: the bare column when the two agree, otherwise the column
/// aliased to the target's qualified name.
fn column_named_as(schema: &DFSchema, j: usize, target: &DFSchema, i: usize) -> Expr {
    let (qualifier, field) = schema.qualified_field(j);
    let column = Expr::Column(Column::new(qualifier.cloned(), field.name()));
    let (target_qualifier, target_field) = target.qualified_field(i);
    if qualifier == target_qualifier && field.name() == target_field.name() {
        column
    } else {
        column.alias_qualified(target_qualifier.cloned(), target_field.name())
    }
}

/// The column a projection expression selects: a plain column, or a plain
/// column under an alias. `None` for anything that computes a value.
fn projected_column(expr: &Expr) -> Option<&Column> {
    match expr {
        Expr::Column(col) => Some(col),
        Expr::Alias(alias) => match alias.expr.as_ref() {
            Expr::Column(col) => Some(col),
            _ => None,
        },
        _ => None,
    }
}

/// The position `column` resolves to in `schema`, as DataFusion resolves a
/// column (a qualifier must match; an unqualified name matching several
/// fields resolves only if exactly one of them is unqualified). `None` when
/// it does not resolve to exactly one field.
fn resolve_index(schema: &DFSchema, column: &Column) -> Option<usize> {
    let matches: Vec<(usize, bool)> = schema
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
        .map(|(idx, (qualifier, _))| (idx, qualifier.is_none()))
        .collect();
    match matches.as_slice() {
        [(idx, _)] => Some(*idx),
        _ if column.relation.is_none() => {
            let mut unqualified = matches.iter().filter(|(_, bare)| *bare);
            match (unqualified.next(), unqualified.next()) {
                (Some((idx, _)), None) => Some(*idx),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Rebuild `proj` over `new_input`, where `remap` takes the position of each
/// selected column in `proj`'s input to its position in `new_input` (`None`:
/// the column has no counterpart, so the rewrite is abandoned).
///
/// Each expression is rebound to the counterpart field and keeps its output
/// name: an alias stays as written, and a column whose counterpart is named
/// differently is aliased to the original qualified name. The projection is
/// dropped when it selects every field of `new_input` in order. Returns
/// `None` unless the result has exactly `proj`'s schema.
fn reproject(
    proj: &Projection,
    new_input: LogicalPlan,
    remap: impl Fn(usize) -> Option<usize>,
) -> Result<Option<LogicalPlan>> {
    let old_schema = proj.input.schema();
    let new_schema = Arc::clone(new_input.schema());
    let mut exprs = Vec::with_capacity(proj.expr.len());
    let mut identity = proj.expr.len() == new_schema.fields().len();
    for (pos, expr) in proj.expr.iter().enumerate() {
        let Some(col) = projected_column(expr) else {
            return Ok(None);
        };
        let Some(old_idx) = resolve_index(old_schema, col) else {
            return Ok(None);
        };
        let Some(new_idx) = remap(old_idx) else {
            return Ok(None);
        };
        let rebound = match expr {
            Expr::Alias(alias) => {
                let (qualifier, field) = new_schema.qualified_field(new_idx);
                let mut alias = alias.clone();
                alias.expr = Box::new(Expr::Column(Column::new(qualifier.cloned(), field.name())));
                Expr::Alias(alias)
            }
            _ => column_named_as(&new_schema, new_idx, old_schema, old_idx),
        };
        identity &= new_idx == pos && matches!(rebound, Expr::Column(_));
        exprs.push(rebound);
    }

    let plan = if identity {
        new_input
    } else {
        match Projection::try_new(exprs, Arc::new(new_input)) {
            Ok(p) => LogicalPlan::Projection(p),
            Err(_) => return Ok(None),
        }
    };
    if !plan
        .schema()
        .logically_equivalent_names_and_types(&proj.schema)
    {
        return Ok(None);
    }
    Ok(Some(plan))
}

/// Try to extract a row count estimate from a logical plan.
///
/// A pushed-down `fetch` bounds what the scan produces, so it wins over the
/// table's own cardinality when both are known. This runs before
/// `HistogramInjectionRule`, so table statistics are the only other source.
///
/// Sources that report no statistics yield `None` and the caller declines to
/// rewrite: an unknown cardinality must never be read as small. A CSV
/// `ListingTable` and a plain `MemTable` report nothing; a provider that wants
/// its joins reordered must implement `TableProvider::statistics`.
fn get_row_count_estimate(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::TableScan(scan) => scan.fetch.or_else(|| {
            source_as_provider(&scan.source)
                .ok()?
                .statistics()?
                .num_rows
                .get_value()
                .copied()
        }),
        LogicalPlan::EmptyRelation(rel) => Some(if rel.produce_one_row { 1 } else { 0 }),
        LogicalPlan::Limit(_limit) => get_row_count_estimate(&_limit.input),
        // A generated relation has at most its declared row count.
        LogicalPlan::Extension(ext) => ext
            .node
            .as_any()
            .downcast_ref::<GenerateNode>()
            .map(|generate| generate.num_rows),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::DFSchema;
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

    fn make_right_table() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Int32, false),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    fn make_empty_relation(fields: Vec<(&str, DataType)>, produce_one_row: bool) -> LogicalPlan {
        let schema = Schema::new(
            fields
                .into_iter()
                .map(|(name, dt)| Field::new(name, dt, false))
                .collect::<Vec<_>>(),
        );
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row,
            schema: Arc::new(df_schema),
        })
    }

    fn apply_rule(plan: LogicalPlan) -> Transformed<LogicalPlan> {
        let rule = SemanticRewritingRule;
        let config = datafusion::optimizer::OptimizerContext::new();
        rule.rewrite(plan, &config).unwrap()
    }

    // Rewrites A and C.

    #[test]
    fn test_pushdown_selection_before_extend() {
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let plan = create_semantic_selection(
            &extend_plan,
            "llama3.1:latest",
            vec!["text".to_string()],
            "Keep English only",
        )
        .unwrap();

        let result = apply_rule(plan);

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            assert!(ext.node.as_any().downcast_ref::<ExtendNode>().is_some());
            let extend = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
            if let LogicalPlan::Extension(child_ext) = &extend.input {
                assert!(
                    child_ext
                        .node
                        .as_any()
                        .downcast_ref::<SemanticSelectionNode>()
                        .is_some()
                );
            } else {
                panic!("Expected selection as child of extend");
            }
        } else {
            panic!("Expected Extension node");
        }
    }

    #[test]
    fn test_no_pushdown_when_referencing_derived_column() {
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let plan = create_semantic_selection(
            &extend_plan,
            "llama3.1:latest",
            vec!["sentiment".to_string()],
            "Keep positive",
        )
        .unwrap();

        let result = apply_rule(plan);
        assert!(!result.transformed);
    }

    #[test]
    fn test_selection_reordering() {
        let input = make_table_scan();

        let inner = create_semantic_selection(
            &input,
            "llama3.1:latest",
            vec!["text".to_string(), "id".to_string()],
            "Complex filter",
        )
        .unwrap();

        let outer = create_semantic_selection(
            &inner,
            "llama3.1:latest",
            vec!["text".to_string()],
            "Simple filter",
        )
        .unwrap();

        let result = apply_rule(outer);

        assert!(result.transformed);
        if let LogicalPlan::Extension(ext) = &result.data {
            let outer_sel = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap();
            assert_eq!(outer_sel.input_schema.len(), 2);
            if let LogicalPlan::Extension(inner_ext) = &outer_sel.input {
                let inner_sel = inner_ext
                    .node
                    .as_any()
                    .downcast_ref::<SemanticSelectionNode>()
                    .unwrap();
                assert_eq!(inner_sel.input_schema.len(), 1);
            }
        }
    }

    // Rewrite B: Selection pushdown before Join

    #[test]
    fn test_pushdown_selection_before_join_left() {
        let left = make_table_scan();
        let right = make_right_table();

        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["text".to_string()],
            vec!["name".to_string()],
            "Match text to name",
            "llama3.1:latest",
        )
        .unwrap();

        // Selection on "text", left-side column only
        let plan = create_semantic_selection(
            &join_plan,
            "llama3.1:latest",
            vec!["text".to_string()],
            "Keep English only",
        )
        .unwrap();

        let result = apply_rule(plan);
        assert!(result.transformed);

        // Result: σ_sel(R) ⋈ S, join with selection pushed to left
        if let LogicalPlan::Extension(ext) = &result.data {
            let join = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .expect("Expected join at top");
            // Left input should be a selection
            if let LogicalPlan::Extension(left_ext) = &join.left_input {
                let sel = left_ext
                    .node
                    .as_any()
                    .downcast_ref::<SemanticSelectionNode>()
                    .expect("Expected selection as left input of join");
                assert_eq!(sel.prompt, "Keep English only");
            } else {
                panic!("Expected Extension (selection) as left input of join");
            }
        } else {
            panic!("Expected Extension (join) at top");
        }
    }

    #[test]
    fn test_pushdown_selection_before_join_right() {
        let left = make_table_scan();
        let right = make_right_table();

        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["text".to_string()],
            vec!["name".to_string()],
            "Match text to name",
            "llama3.1:latest",
        )
        .unwrap();

        // Selection on "name", right-side column only
        let plan = create_semantic_selection(
            &join_plan,
            "llama3.1:latest",
            vec!["name".to_string()],
            "Keep short names",
        )
        .unwrap();

        let result = apply_rule(plan);
        assert!(result.transformed);

        if let LogicalPlan::Extension(ext) = &result.data {
            let join = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .expect("Expected join at top");
            // Right input should be a selection
            if let LogicalPlan::Extension(right_ext) = &join.right_input {
                let sel = right_ext
                    .node
                    .as_any()
                    .downcast_ref::<SemanticSelectionNode>()
                    .expect("Expected selection as right input of join");
                assert_eq!(sel.prompt, "Keep short names");
            } else {
                panic!("Expected Extension (selection) as right input of join");
            }
        } else {
            panic!("Expected Extension (join) at top");
        }
    }

    #[test]
    fn test_no_pushdown_selection_both_sides() {
        let left = make_table_scan();
        let right = make_right_table();

        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["text".to_string()],
            vec!["name".to_string()],
            "Match text to name",
            "llama3.1:latest",
        )
        .unwrap();

        // Selection references columns from both sides, no pushdown
        let plan = create_semantic_selection(
            &join_plan,
            "llama3.1:latest",
            vec!["text".to_string(), "name".to_string()],
            "Complex filter on both sides",
        )
        .unwrap();

        let result = apply_rule(plan);
        assert!(!result.transformed);
    }

    // Rewrite D: Join commutativity

    #[test]
    fn test_join_commutativity_swap() {
        // Left: 0 rows, Right: 1 row → should swap so larger is on left
        let left = make_empty_relation(
            vec![("a", DataType::Utf8)],
            false, // 0 rows
        );
        let right = make_empty_relation(
            vec![("b", DataType::Utf8)],
            true, // 1 row
        );

        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["a".to_string()],
            vec!["b".to_string()],
            "Match a to b",
            "llama3.1:latest",
        )
        .unwrap();

        let original_schema = join_plan.schema().clone();
        let result = apply_rule(join_plan);
        assert!(result.transformed);

        // The swap exchanges the two halves of the output schema, so the rewrite
        // projects the original column order back on top. An optimizer rule that
        // changed its plan's schema would trip DataFusion's post-rule invariant
        // check and fail the query.
        assert_eq!(
            result.data.schema(),
            &original_schema,
            "Rewrite D must preserve the output schema"
        );

        let LogicalPlan::Projection(proj) = &result.data else {
            panic!("Expected the swapped join to be wrapped in a Projection");
        };

        // After swap: right (1 row) is now left, left (0 rows) is now right
        if let LogicalPlan::Extension(ext) = proj.input.as_ref() {
            let join = ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .expect("Expected join");
            // Left join attrs should now be "b" (was right), right should be "a" (was left)
            assert_eq!(join.left_join_attrs, vec!["b".to_string()]);
            assert_eq!(join.right_join_attrs, vec!["a".to_string()]);
        } else {
            panic!("Expected Extension (join) under the Projection");
        }
    }

    /// A table that reports an exact row count through
    /// `TableProvider::statistics`, as a provider must for Rewrite D to see it.
    #[test]
    fn row_count_estimate_reads_provider_statistics() {
        use datafusion::arrow::datatypes::SchemaRef;
        use datafusion::catalog::{Session, TableProvider};
        use datafusion::common::Statistics;
        use datafusion::common::stats::Precision;
        use datafusion::datasource::provider_as_source;
        use datafusion::logical_expr::{LogicalPlanBuilder, TableType};
        use datafusion::physical_plan::ExecutionPlan;

        #[derive(Debug)]
        struct CountedTable(SchemaRef, usize);

        #[async_trait::async_trait]
        impl TableProvider for CountedTable {
            fn schema(&self) -> SchemaRef {
                self.0.clone()
            }
            fn table_type(&self) -> TableType {
                TableType::Base
            }
            fn statistics(&self) -> Option<Statistics> {
                Some(Statistics {
                    num_rows: Precision::Exact(self.1),
                    total_byte_size: Precision::Absent,
                    column_statistics: Statistics::unknown_column(&self.0),
                })
            }
            async fn scan(
                &self,
                _state: &dyn Session,
                _projection: Option<&Vec<usize>>,
                _filters: &[Expr],
                _limit: Option<usize>,
            ) -> Result<Arc<dyn ExecutionPlan>> {
                unimplemented!("planning only")
            }
        }
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
        let source = provider_as_source(Arc::new(CountedTable(schema, 4242)));
        let scan = LogicalPlanBuilder::scan("t", source, None)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(get_row_count_estimate(&scan), Some(4242));
    }

    #[test]
    fn test_join_commutativity_no_swap() {
        // Left: 1 row, Right: 0 rows → left is already larger, no swap
        let left = make_empty_relation(
            vec![("a", DataType::Utf8)],
            true, // 1 row
        );
        let right = make_empty_relation(
            vec![("b", DataType::Utf8)],
            false, // 0 rows
        );

        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["a".to_string()],
            vec!["b".to_string()],
            "Match a to b",
            "llama3.1:latest",
        )
        .unwrap();

        let result = apply_rule(join_plan);
        assert!(!result.transformed);
    }

    // Rewrite E: Extend-Selection Fusion

    #[test]
    fn test_extend_selection_fusion() {
        // Build: π_{text,id}(σ_sentiment(ε_sentiment(R)))
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        // Selection depends on "sentiment" (derived column)
        let sel_plan = create_semantic_selection(
            &extend_plan,
            "llama3.1:latest",
            vec!["sentiment".to_string()],
            "Keep positive only",
        )
        .unwrap();

        // Projection removes "sentiment", keeps only base columns
        let proj_exprs = vec![Expr::Column("text".into()), Expr::Column("id".into())];
        let proj = Projection::try_new(proj_exprs, Arc::new(sel_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(result.transformed, "Fusion should have been applied");

        // Result should be a fused selection on R (no extend, no projection needed
        // since we select all of R's columns)
        if let LogicalPlan::Extension(ext) = &result.data {
            let sel = ext
                .node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .expect("Expected fused selection");
            assert!(
                sel.prompt.contains("First, for each row,"),
                "Prompt should be composed"
            );
            assert!(
                sel.prompt.contains("Classify sentiment"),
                "Prompt should include extend instruction"
            );
            assert!(
                sel.prompt.contains("Keep positive only"),
                "Prompt should include selection instruction"
            );
            // Input should be R (EmptyRelation), not the extend
            assert!(
                matches!(sel.input, LogicalPlan::EmptyRelation(_)),
                "Fused selection should operate on R directly"
            );
        } else {
            panic!(
                "Expected fused selection (Extension), got: {:?}",
                result.data
            );
        }
    }

    #[test]
    fn test_no_fusion_when_projection_keeps_extended() {
        // Projection includes "sentiment" → no fusion
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let sel_plan = create_semantic_selection(
            &extend_plan,
            "llama3.1:latest",
            vec!["sentiment".to_string()],
            "Keep positive only",
        )
        .unwrap();

        // Projection keeps "sentiment", fusion should not apply
        let proj_exprs = vec![
            Expr::Column("text".into()),
            Expr::Column("sentiment".into()),
        ];
        let proj = Projection::try_new(proj_exprs, Arc::new(sel_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(
            !result.transformed,
            "Fusion should not apply when projection keeps E"
        );
    }

    #[test]
    fn test_no_fusion_when_models_differ() {
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "model-a",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let sel_plan = create_semantic_selection(
            &extend_plan,
            "model-b",
            vec!["sentiment".to_string()],
            "Keep positive only",
        )
        .unwrap();

        let proj_exprs = vec![Expr::Column("text".into()), Expr::Column("id".into())];
        let proj = Projection::try_new(proj_exprs, Arc::new(sel_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(
            !result.transformed,
            "Fusion should not apply when models differ"
        );
    }

    #[test]
    fn test_no_fusion_when_selection_independent() {
        // Selection doesn't reference derived columns → not a fusion candidate
        // (Rewrite A pushdown would apply instead)
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            input,
            "Classify sentiment",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        // Selection on "text", does NOT depend on "sentiment"
        let sel_plan = create_semantic_selection(
            &extend_plan,
            "llama3.1:latest",
            vec!["text".to_string()],
            "Keep English only",
        )
        .unwrap();

        let proj_exprs = vec![Expr::Column("text".into()), Expr::Column("id".into())];
        let proj = Projection::try_new(proj_exprs, Arc::new(sel_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(
            !result.transformed,
            "Fusion should not apply when selection is independent of E"
        );
    }

    // Rewrite F: Extend-Join Fusion

    #[test]
    fn test_extend_join_left_fusion() {
        // Build: π_{text,id,name,score}(ε_category(R) ⋈_{category,name} S)
        let r = make_table_scan(); // text, id
        let s = make_right_table(); // name, score

        let derived = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            r,
            "Categorize the text",
            "llama3.1:latest",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let join_plan = create_semantic_join(
            &extend_plan,
            &s,
            vec!["category".to_string()],
            vec!["name".to_string()],
            "Match category to name",
            "llama3.1:latest",
        )
        .unwrap();

        // Projection removes "category", keeps base columns from both sides
        let proj_exprs = vec![
            Expr::Column("text".into()),
            Expr::Column("id".into()),
            Expr::Column("name".into()),
            Expr::Column("score".into()),
        ];
        let proj = Projection::try_new(proj_exprs, Arc::new(join_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(result.transformed, "Left fusion should have been applied");

        // Result should be a join on R and S (no extend) possibly wrapped in projection
        let join_node = match &result.data {
            LogicalPlan::Extension(ext) => ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .expect("Expected fused join"),
            LogicalPlan::Projection(proj) => {
                if let LogicalPlan::Extension(ext) = proj.input.as_ref() {
                    ext.node
                        .as_any()
                        .downcast_ref::<SemanticJoinNode>()
                        .expect("Expected fused join under projection")
                } else {
                    panic!("Expected Extension under Projection");
                }
            }
            other => panic!("Expected join or projection, got: {:?}", other),
        };

        assert!(
            join_node.prompt.contains("First, compute from left row:"),
            "Prompt should be composed"
        );
        assert!(
            join_node.prompt.contains("Categorize the text"),
            "Prompt should include extend instruction"
        );
        // Left input should be R (EmptyRelation), not the extend
        assert!(
            matches!(join_node.left_input, LogicalPlan::EmptyRelation(_)),
            "Fused join should operate on R directly"
        );
    }

    #[test]
    fn test_extend_join_right_fusion() {
        // Symmetric: extend on the right input
        let r = make_table_scan(); // text, id
        let s = make_right_table(); // name, score

        let derived = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        let extend_plan = create_extend(
            s,
            "Categorize the name",
            "llama3.1:latest",
            vec!["name".to_string()],
            derived,
        )
        .unwrap();

        let join_plan = create_semantic_join(
            &r,
            &extend_plan,
            vec!["text".to_string()],
            vec!["category".to_string()],
            "Match text to category",
            "llama3.1:latest",
        )
        .unwrap();

        // Projection removes "category", keeps base columns
        let proj_exprs = vec![
            Expr::Column("text".into()),
            Expr::Column("id".into()),
            Expr::Column("name".into()),
            Expr::Column("score".into()),
        ];
        let proj = Projection::try_new(proj_exprs, Arc::new(join_plan)).unwrap();
        let plan = LogicalPlan::Projection(proj);

        let result = apply_rule(plan);
        assert!(result.transformed, "Right fusion should have been applied");

        let join_node = match &result.data {
            LogicalPlan::Extension(ext) => ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .expect("Expected fused join"),
            LogicalPlan::Projection(proj) => {
                if let LogicalPlan::Extension(ext) = proj.input.as_ref() {
                    ext.node
                        .as_any()
                        .downcast_ref::<SemanticJoinNode>()
                        .expect("Expected fused join under projection")
                } else {
                    panic!("Expected Extension under Projection");
                }
            }
            other => panic!("Expected join or projection, got: {:?}", other),
        };

        assert!(
            join_node.prompt.contains("First, compute from right row:"),
            "Prompt should be composed for right fusion"
        );
        assert!(
            join_node.prompt.contains("Categorize the name"),
            "Prompt should include extend instruction"
        );
        // Right input should be S (EmptyRelation), not the extend
        assert!(
            matches!(join_node.right_input, LogicalPlan::EmptyRelation(_)),
            "Fused join should operate on S directly"
        );
    }

    // Qualified schemas: every rewrite returns a plan with the qualified schema
    // of the plan it replaces (DataFusion compares qualifiers after each rule).

    fn qualified_relation(
        qualifier: Option<&str>,
        fields: &[&str],
        produce_one_row: bool,
    ) -> LogicalPlan {
        let schema = Schema::new(
            fields
                .iter()
                .map(|name| Field::new(*name, DataType::Utf8, true))
                .collect::<Vec<_>>(),
        );
        let df_schema = match qualifier {
            Some(q) => DFSchema::try_from_qualified_schema(q, &schema).unwrap(),
            None => DFSchema::try_from(schema).unwrap(),
        };
        LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row,
            schema: Arc::new(df_schema),
        })
    }

    fn derived(name: &str) -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]))
    }

    fn qualified_names(schema: &DFSchema) -> Vec<String> {
        schema
            .iter()
            .map(|(q, f)| match q {
                Some(q) => format!("{q}.{}", f.name()),
                None => f.name().clone(),
            })
            .collect()
    }

    /// Runs the rule through DataFusion's `Optimizer`, which asserts after the
    /// rule that the plan's (qualified) schema did not change.
    fn optimize(plan: LogicalPlan) -> LogicalPlan {
        let optimizer =
            datafusion::optimizer::Optimizer::with_rules(vec![Arc::new(SemanticRewritingRule)]);
        let config = datafusion::optimizer::OptimizerContext::new().with_max_passes(1);
        optimizer.optimize(plan, &config, |_, _| {}).unwrap()
    }

    fn col(qualifier: Option<&str>, name: &str) -> Expr {
        Expr::Column(Column::new(qualifier, name))
    }

    fn project(exprs: Vec<Expr>, input: LogicalPlan) -> LogicalPlan {
        LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(input)).unwrap())
    }

    /// The column a projection expression reads and the name it outputs.
    fn read_and_named(expr: &Expr) -> (String, String) {
        let read = projected_column(expr).unwrap().flat_name();
        let named = match expr {
            Expr::Alias(a) => match &a.relation {
                Some(r) => format!("{r}.{}", a.name),
                None => a.name.clone(),
            },
            other => projected_column(other).unwrap().flat_name(),
        };
        (read, named)
    }

    fn swapped_join(plan: &LogicalPlan) -> (&Projection, &SemanticJoinNode) {
        let LogicalPlan::Projection(proj) = plan else {
            panic!("expected a Projection over the swapped join, got {plan:?}");
        };
        let LogicalPlan::Extension(ext) = proj.input.as_ref() else {
            panic!("expected the swapped join under the Projection");
        };
        let join = ext
            .node
            .as_any()
            .downcast_ref::<SemanticJoinNode>()
            .unwrap();
        (proj, join)
    }

    #[test]
    fn rewrite_d_restores_a_qualified_schema_with_colliding_names_by_position() {
        // a: 0 rows, b: 1 row, so the operands swap.
        let left = qualified_relation(Some("a"), &["id", "x"], false);
        let right = qualified_relation(Some("b"), &["id", "y"], true);
        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["x".to_string()],
            vec!["y".to_string()],
            "Match",
            "m",
        )
        .unwrap();
        let original = join_plan.schema().clone();
        assert_eq!(qualified_names(&original), ["a.id", "a.x", "b.id_1", "b.y"]);

        let result = optimize(join_plan);
        assert_eq!(result.schema().as_ref(), original.as_ref());

        let (proj, join) = swapped_join(&result);
        // The swapped join puts b first, so the rename moves to a's `id`.
        assert_eq!(
            qualified_names(&join.schema),
            ["b.id", "b.y", "a.id_1", "a.x"]
        );
        let mapping: Vec<_> = proj.expr.iter().map(read_and_named).collect();
        assert_eq!(
            mapping,
            [
                ("a.id_1".to_string(), "a.id".to_string()),
                ("a.x".to_string(), "a.x".to_string()),
                ("b.id".to_string(), "b.id_1".to_string()),
                ("b.y".to_string(), "b.y".to_string()),
            ]
        );
    }

    /// Before the positional restore, `id` was matched by name to the swapped
    /// join's `id` (the right input's column) and `id_1` to the left input's,
    /// so the two output columns carried each other's values.
    #[test]
    fn rewrite_d_does_not_exchange_the_values_of_two_columns_of_one_name() {
        let left = qualified_relation(None, &["id"], false);
        let right = qualified_relation(None, &["id"], true);
        let join_plan = create_semantic_join(
            &left,
            &right,
            vec!["id".into()],
            vec!["id".into()],
            "M",
            "m",
        )
        .unwrap();
        let original = join_plan.schema().clone();
        assert_eq!(qualified_names(&original), ["id", "id_1"]);

        let result = optimize(join_plan);
        assert_eq!(result.schema().as_ref(), original.as_ref());

        let (proj, join) = swapped_join(&result);
        // In the swapped join the original left input is the right side, whose
        // `id` is renamed `id_1`; the output `id` must read that one.
        let swapped_right_start = join.left_input.schema().fields().len();
        for (i, expr) in proj.expr.iter().enumerate() {
            let read = resolve_index(&join.schema, projected_column(expr).unwrap()).unwrap();
            let from_original_left = read >= swapped_right_start;
            assert_eq!(
                from_original_left,
                i == 0,
                "output {i} reads the wrong side"
            );
        }
        let mapping: Vec<_> = proj.expr.iter().map(read_and_named).collect();
        assert_eq!(
            mapping,
            [
                ("id_1".to_string(), "id".to_string()),
                ("id".to_string(), "id_1".to_string()),
            ]
        );
    }

    fn qualified_extend_selection() -> LogicalPlan {
        let input = qualified_relation(Some("t"), &["text", "id"], false);
        let extend_plan = create_extend(
            input,
            "Classify",
            "m",
            vec!["text".into()],
            derived("sentiment"),
        )
        .unwrap();
        create_semantic_selection(&extend_plan, "m", vec!["sentiment".into()], "Keep positive")
            .unwrap()
    }

    fn fused_selection_input(plan: &LogicalPlan) -> &LogicalPlan {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected the fused selection, got {plan:?}");
        };
        let sel = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .expect("fused selection");
        &sel.input
    }

    #[test]
    fn rewrite_e_over_a_qualified_input_drops_an_identity_projection() {
        let plan = project(
            vec![col(Some("t"), "text"), col(Some("t"), "id")],
            qualified_extend_selection(),
        );
        let expected = plan.schema().clone();
        let result = optimize(plan);
        assert_eq!(qualified_names(result.schema()), ["t.text", "t.id"]);
        assert_eq!(result.schema().as_ref(), expected.as_ref());
        assert!(matches!(
            fused_selection_input(&result),
            LogicalPlan::EmptyRelation(_)
        ));
    }

    #[test]
    fn rewrite_e_keeps_a_projection_that_reorders_or_renames() {
        for exprs in [
            vec![col(Some("t"), "id"), col(Some("t"), "text")],
            vec![col(None, "text").alias("body"), col(Some("t"), "id")],
        ] {
            let plan = project(exprs, qualified_extend_selection());
            let expected = plan.schema().clone();
            let result = optimize(plan);
            assert_eq!(result.schema().as_ref(), expected.as_ref());
            let LogicalPlan::Projection(proj) = &result else {
                panic!("the projection must survive, got {result:?}");
            };
            assert!(matches!(
                fused_selection_input(&proj.input),
                LogicalPlan::EmptyRelation(_)
            ));
        }
    }

    #[test]
    fn rewrite_e_still_declines_a_projection_keeping_a_derived_column_under_an_alias() {
        let plan = project(
            vec![col(Some("t"), "text"), col(None, "sentiment").alias("s")],
            qualified_extend_selection(),
        );
        assert!(!apply_rule(plan).transformed);
    }

    fn fused_join(plan: &LogicalPlan) -> &SemanticJoinNode {
        let node = match plan {
            LogicalPlan::Projection(p) => p.input.as_ref(),
            other => other,
        };
        let LogicalPlan::Extension(ext) = node else {
            panic!("expected the fused join, got {plan:?}");
        };
        ext.node
            .as_any()
            .downcast_ref::<SemanticJoinNode>()
            .unwrap()
    }

    #[test]
    fn rewrite_f_left_over_qualified_inputs_keeps_the_schema() {
        let r = qualified_relation(Some("r"), &["text", "id"], false);
        let s = qualified_relation(Some("s"), &["name", "score"], false);
        let extend_plan = create_extend(
            r,
            "Categorize",
            "m",
            vec!["text".into()],
            derived("category"),
        )
        .unwrap();
        let join_plan = create_semantic_join(
            &extend_plan,
            &s,
            vec!["category".into()],
            vec!["name".into()],
            "M",
            "m",
        )
        .unwrap();
        let plan = project(
            vec![
                col(Some("r"), "text"),
                col(Some("r"), "id"),
                col(Some("s"), "name"),
                col(Some("s"), "score"),
            ],
            join_plan,
        );
        let expected = plan.schema().clone();
        let result = optimize(plan);
        assert_eq!(result.schema().as_ref(), expected.as_ref());
        assert_eq!(
            qualified_names(result.schema()),
            ["r.text", "r.id", "s.name", "s.score"]
        );
        // Identity over the fused join: no projection left.
        assert!(matches!(result, LogicalPlan::Extension(_)));
        assert!(matches!(
            fused_join(&result).left_input,
            LogicalPlan::EmptyRelation(_)
        ));
    }

    /// The derived column collides with a right column, which the original join
    /// renamed `s.name_1`; the fused join has no collision, so the projection
    /// reads `s.name` and keeps the original output name.
    #[test]
    fn rewrite_f_left_maps_a_renamed_right_column_by_position() {
        let r = qualified_relation(Some("r"), &["text", "id"], false);
        let s = qualified_relation(Some("s"), &["name", "score"], false);
        let extend_plan =
            create_extend(r, "Categorize", "m", vec!["text".into()], derived("name")).unwrap();
        let join_plan = create_semantic_join(
            &extend_plan,
            &s,
            vec!["name".into()],
            vec!["s.name".into()],
            "M",
            "m",
        )
        .unwrap();
        assert_eq!(
            qualified_names(join_plan.schema()),
            ["r.text", "r.id", "name", "s.name_1", "s.score"]
        );
        let plan = project(
            vec![col(Some("r"), "text"), col(Some("s"), "name_1")],
            join_plan,
        );
        let expected = plan.schema().clone();
        let result = optimize(plan);
        assert_eq!(result.schema().as_ref(), expected.as_ref());
        let LogicalPlan::Projection(proj) = &result else {
            panic!("expected a projection, got {result:?}");
        };
        let mapping: Vec<_> = proj.expr.iter().map(read_and_named).collect();
        assert_eq!(
            mapping,
            [
                ("r.text".to_string(), "r.text".to_string()),
                ("s.name".to_string(), "s.name_1".to_string()),
            ]
        );
    }

    #[test]
    fn rewrite_f_right_over_qualified_inputs_keeps_the_schema() {
        let r = qualified_relation(Some("r"), &["text", "id"], false);
        let s = qualified_relation(Some("s"), &["name", "score"], false);
        let extend_plan = create_extend(
            s,
            "Categorize",
            "m",
            vec!["name".into()],
            derived("category"),
        )
        .unwrap();
        let join_plan = create_semantic_join(
            &r,
            &extend_plan,
            vec!["text".into()],
            vec!["category".into()],
            "M",
            "m",
        )
        .unwrap();
        let plan = project(
            vec![col(Some("s"), "name"), col(Some("r"), "text")],
            join_plan,
        );
        let expected = plan.schema().clone();
        let result = optimize(plan);
        assert_eq!(result.schema().as_ref(), expected.as_ref());
        assert_eq!(qualified_names(result.schema()), ["s.name", "r.text"]);
        assert!(matches!(result, LogicalPlan::Projection(_)));
        assert!(matches!(
            fused_join(&result).right_input,
            LogicalPlan::EmptyRelation(_)
        ));
    }

    #[test]
    fn rewrites_a_and_b_keep_qualified_schemas() {
        // A: σ(ε(R)) over a qualified R.
        let input = qualified_relation(Some("t"), &["text", "id"], false);
        let extend_plan = create_extend(
            input,
            "Classify",
            "m",
            vec!["text".into()],
            derived("sentiment"),
        )
        .unwrap();
        let plan =
            create_semantic_selection(&extend_plan, "m", vec!["text".into()], "English").unwrap();
        let expected = plan.schema().clone();
        let result = optimize(plan);
        assert!(matches!(&result, LogicalPlan::Extension(e) if e.node.name() == "ExtendNode"));
        assert_eq!(result.schema().as_ref(), expected.as_ref());

        // B: σ(R ⋈ S), pushed to either side.
        for (col_name, side) in [("text", "left"), ("score", "right")] {
            let r = qualified_relation(Some("r"), &["text", "id"], false);
            let s = qualified_relation(Some("s"), &["name", "score"], false);
            let join_plan =
                create_semantic_join(&r, &s, vec!["text".into()], vec!["name".into()], "M", "m")
                    .unwrap();
            let plan =
                create_semantic_selection(&join_plan, "m", vec![col_name.into()], "Keep").unwrap();
            let expected = plan.schema().clone();
            let result = optimize(plan);
            assert_eq!(result.schema().as_ref(), expected.as_ref(), "{side}");
            let join = fused_join(&result);
            let pushed = if side == "left" {
                &join.left_input
            } else {
                &join.right_input
            };
            assert!(matches!(pushed, LogicalPlan::Extension(_)), "{side}");
        }
    }

    /// A selection reading a right column the join renamed (`x_1` is the right
    /// input's `x`) is not pushed down: on the right input, `x_1` is a
    /// different column.
    #[test]
    fn rewrite_b_does_not_push_a_selection_on_a_renamed_right_column() {
        let left = qualified_relation(None, &["x"], false);
        let right = qualified_relation(None, &["x", "x_1"], false);
        let join_plan =
            create_semantic_join(&left, &right, vec!["x".into()], vec!["x".into()], "M", "m")
                .unwrap();
        assert_eq!(qualified_names(join_plan.schema()), ["x", "x_1", "x_1_1"]);
        let plan = create_semantic_selection(&join_plan, "m", vec!["x_1".into()], "Keep").unwrap();
        assert!(!apply_rule(plan).transformed);
    }
}
