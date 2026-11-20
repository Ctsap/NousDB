//! Logical rule that attaches token histograms and batch budgets to semantic operators.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::Column;
use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use nousdb_core::certification::OperatorKind;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::utils::{flat_static_prompt_tokens, join_static_prompt_tokens};
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::token_estimator::{
    BatchBudgetModel, HistogramCatalog, JoinBatchBudgetModel, TokenEstimator, TokenHistogram,
    estimate_output_tokens_per_row, row_format_tokens,
};
use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::join::join_schema_validation::validate_column_exists;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;

use crate::column_lineage::{BaseColumn, Untraced, trace_column};

/// Optimizer rule that injects pre-computed token histograms and batch budget
/// models into semantic operator nodes.
///
/// This rule must run before `BatchSizeOptimizationRule`, which reads what it
/// injects. It finds semantic operators (Extend, Selection, Join), traces each
/// operator column to the `TableScan` column it reads (through aliases and
/// renaming projections, see `histogram_source`), and fills `token_histogram`
/// and `batch_budget` from the `HistogramCatalog`.
///
/// The batch budget is computed from the `ModelCard` of each operator's model,
/// with per-operator output token estimation. If the model is not in the
/// registry, a conservative fallback budget is used. Nodes that already have
/// histograms are skipped.
pub struct HistogramInjectionRule {
    catalog: Arc<HistogramCatalog>,
    model_registry: Arc<ModelRegistry>,
    /// The reasoning mode whose prompt templates execution renders; sizing
    /// measures those templates.
    reasoning: ReasoningMode,
}

impl std::fmt::Debug for HistogramInjectionRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistogramInjectionRule")
            .field("catalog", &self.catalog)
            .field("model_registry", &"<ModelRegistry>")
            .field("reasoning", &self.reasoning)
            .finish()
    }
}

impl HistogramInjectionRule {
    /// Creates the rule over `catalog`, resolving model cards through
    /// `model_registry`.
    pub fn new(catalog: Arc<HistogramCatalog>, model_registry: Arc<ModelRegistry>) -> Self {
        Self {
            catalog,
            model_registry,
            reasoning: ReasoningMode::default(),
        }
    }

    /// Sizes batches against the templates of `reasoning` instead of the baseline.
    pub fn with_reasoning_mode(mut self, reasoning: ReasoningMode) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// The table the histogram was looked up in, and the histogram if the
    /// catalog has one (see [`histogram_source`]).
    fn lookup_histogram(
        &self,
        op: &str,
        input: &LogicalPlan,
        columns: &[String],
        qualified: bool,
    ) -> (Option<String>, Option<TokenHistogram>) {
        match histogram_source(op, input, columns, qualified) {
            HistogramSource::Lineage { table, columns } => {
                let histogram = self.catalog.get_combined(&table, &columns);
                (Some(table), histogram)
            }
            HistogramSource::FirstScan { table, columns } => {
                let histogram = table
                    .as_ref()
                    .and_then(|t| self.catalog.get_combined(t, &columns));
                (table, histogram)
            }
        }
    }
}

fn resolve_model_card(registry: &ModelRegistry, model_id: &str) -> Option<ModelCard> {
    registry.model_card(model_id)
}

/// Measure the static overhead of a flat (extend/selection) prompt by
/// rendering the real templates with zero rows, as the join path does.
///
/// A render failure falls back to `instruction.len() / 3 + 100` rather than
/// failing the plan.
fn flat_static_overhead(
    system_template: &str,
    user_template: &str,
    instruction: &str,
    output_schema: Option<&SchemaRef>,
) -> usize {
    TokenEstimator::default_estimator()
        .ok()
        .and_then(|est| {
            flat_static_prompt_tokens(
                system_template,
                user_template,
                instruction,
                output_schema,
                &est,
            )
            .ok()
        })
        .unwrap_or_else(|| {
            log::warn!(
                "[HistogramInjectionRule] failed to measure '{}' template overhead; \
                 falling back to a conservative constant",
                user_template
            );
            instruction.len() / 3 + 100
        })
}

/// Build a `BatchBudgetModel` for an Extend operator from its `ModelCard`.
///
/// `{{ output_schema }}` is interpolated from the wire fields, so a
/// prompt-only field (`rationale`) is measured too.
fn build_budget_for_extend(
    card: &ModelCard,
    extend: &ExtendNode,
    reasoning: ReasoningMode,
) -> BatchBudgetModel {
    let wire_schema: SchemaRef = Arc::new(Schema::new(
        reasoning
            .prompt_only_fields()
            .into_iter()
            .map(Arc::new)
            .chain(extend.derived_schema.fields().iter().cloned())
            .collect::<Vec<_>>(),
    ));
    BatchBudgetModel {
        context_window_tokens: card.effective_input_budget(),
        fixed_overhead_tokens: flat_static_overhead(
            &reasoning.system_template(OperatorKind::Extend),
            reasoning.user_template(OperatorKind::Extend),
            &extend.prompt,
            Some(&wire_schema),
        ),
        per_row_delimiter_tokens: 2,
        expected_output_tokens_per_row: estimate_output_tokens_per_row(
            extend.derived_schema.fields(),
        ),
        model_max_batch_size: Some(card.max_batch_size),
        quality_max_batch_size: card.quality_max_batch_size,
        max_output_tokens: card.max_output_tokens,
        reasoning_output_tokens_per_row: reasoning.reasoning_output_tokens_per_row(),
    }
}

/// Build a `BatchBudgetModel` for a Selection operator from its `ModelCard`.
fn build_budget_for_selection(
    card: &ModelCard,
    sel: &SemanticSelectionNode,
    reasoning: ReasoningMode,
) -> BatchBudgetModel {
    BatchBudgetModel {
        context_window_tokens: card.effective_input_budget(),
        fixed_overhead_tokens: flat_static_overhead(
            &reasoning.system_template(OperatorKind::Selection),
            reasoning.user_template(OperatorKind::Selection),
            &sel.prompt,
            None,
        ),
        per_row_delimiter_tokens: 2,
        expected_output_tokens_per_row: 4, // boolean true/false per row
        model_max_batch_size: Some(card.max_batch_size),
        quality_max_batch_size: card.quality_max_batch_size,
        max_output_tokens: card.max_output_tokens,
        reasoning_output_tokens_per_row: reasoning.reasoning_output_tokens_per_row(),
    }
}

/// Build a `JoinBatchBudgetModel` for the reference-table prompt format.
///
/// The static overhead is measured by rendering the real join templates with
/// the actual condition and zero rows; per-row formatting is derived from the
/// join columns rendered. Row content tokens come from the histograms when
/// `BatchSizeOptimizationRule` runs.
///
/// `JoinBatchBudgetModel` has no `reasoning_output_tokens_per_row` field:
/// `output_tokens_per_pair` is used only for the output-budget cap in
/// `estimate_join_block_sizes_v2`, so the reasoning mode's per-item cost is
/// folded straight into it.
fn build_ref_table_budget_for_join(
    context_window_tokens: usize,
    join: &SemanticJoinNode,
    reasoning: ReasoningMode,
) -> JoinBatchBudgetModel {
    let estimator = TokenEstimator::default_estimator().ok();
    let fixed_overhead_tokens = estimator
        .as_ref()
        .and_then(|est| {
            join_static_prompt_tokens(
                &reasoning.system_template(OperatorKind::Join),
                reasoning.user_template(OperatorKind::Join),
                &join.prompt,
                est,
            )
            .ok()
        })
        .unwrap_or_else(|| {
            log::warn!(
                "[HistogramInjectionRule] failed to measure join template overhead; \
                 falling back to a conservative constant"
            );
            join.prompt.len() / 3 + 450
        });
    let (left_fmt, right_fmt) = match &estimator {
        Some(est) => (
            row_format_tokens(&join.left_join_attrs, est),
            row_format_tokens(&join.right_join_attrs, est),
        ),
        None => (
            8 + 4 * join.left_join_attrs.len(),
            8 + 4 * join.right_join_attrs.len(),
        ),
    };
    JoinBatchBudgetModel {
        context_window_tokens,
        fixed_overhead_tokens,
        left_row_format_tokens: left_fmt,
        right_row_format_tokens: right_fmt,
        pair_reference_tokens: 15,
        output_tokens_per_pair: 8 + reasoning.reasoning_output_tokens_per_row(),
    }
}

/// Conservative fallback budget when model is not found in registry.
fn fallback_batch_budget() -> BatchBudgetModel {
    BatchBudgetModel {
        context_window_tokens: 8192,
        fixed_overhead_tokens: 200,
        per_row_delimiter_tokens: 2,
        expected_output_tokens_per_row: 24,
        // No card was resolved, so there is no row ceiling to enforce.
        model_max_batch_size: None,
        quality_max_batch_size: None,
        max_output_tokens: None,
        reasoning_output_tokens_per_row: 0,
    }
}

/// Where a semantic operator's histogram is looked up.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HistogramSource {
    /// The traced columns all come from `table`; `columns` are their names
    /// there (columns that did not trace are left out).
    Lineage { table: String, columns: Vec<String> },
    /// Nothing traced to a single table: the first `TableScan` and the
    /// column names as the operator wrote them.
    FirstScan {
        table: Option<String>,
        columns: Vec<String>,
    },
}

/// Decide where the histogram for `columns` of `input` lives.
///
/// Each column is traced to the base column its `TableScan` stores
/// ([`trace_column`]), through `SubqueryAlias` qualifiers, renaming
/// projections and the passthrough nodes, so `drama_movies.title` over an
/// aliased scan of `movies_with_reviews` asks the catalog for
/// `movies_with_reviews.title`. Columns that do not trace (computed,
/// derived by an extend, ambiguous) are left out, as `get_combined` already
/// leaves out columns it has no histogram for. When no column traces, or the
/// traced ones span several tables, this falls back to the first-scan lookup.
///
/// `qualified` parses the names the way `create_semantic_join` does
/// (`table.column` or `column`); extend and selection columns are always
/// unqualified names.
fn histogram_source(
    op: &str,
    input: &LogicalPlan,
    columns: &[String],
    qualified: bool,
) -> HistogramSource {
    let mut traced: Vec<BaseColumn> = Vec::new();
    let mut untraced: Vec<String> = Vec::new();
    for name in columns {
        let column = if qualified {
            validate_column_exists(name, input.schema(), "input")
        } else {
            Ok(Column::new_unqualified(name))
        };
        match column
            .map_err(|_| Untraced::NotFound)
            .and_then(|c| trace_column(input, &c))
        {
            Ok(base) => traced.push(base),
            Err(why) => untraced.push(format!("{name} ({why})")),
        }
    }

    match traced.first().map(|b| b.table.clone()) {
        Some(table) if traced.iter().all(|b| b.table == table) => {
            let base: Vec<String> = traced.into_iter().map(|b| b.column).collect();
            log::debug!(
                "[HistogramInjectionRule] {op}: columns {columns:?} traced to '{table}' {base:?} \
                 (lineage; untraced: {untraced:?})"
            );
            HistogramSource::Lineage {
                table,
                columns: base,
            }
        }
        _ => {
            let table = extract_table_name(input);
            let why = if traced.is_empty() {
                format!("no column traced: {untraced:?}")
            } else {
                let tables: Vec<&str> = traced.iter().map(|b| b.table.as_str()).collect();
                format!("columns span tables {tables:?}")
            };
            log::debug!(
                "[HistogramInjectionRule] {op}: {why}; falling back to first scan {table:?} \
                 with columns {columns:?}"
            );
            HistogramSource::FirstScan {
                table,
                columns: columns.to_vec(),
            }
        }
    }
}

/// Returns the table name of the first `TableScan` found below `plan`.
fn extract_table_name(plan: &LogicalPlan) -> Option<String> {
    match plan {
        LogicalPlan::TableScan(scan) => Some(scan.table_name.table().to_string()),
        LogicalPlan::Extension(ext) => {
            for input in ext.node.inputs() {
                if let Some(name) = extract_table_name(input) {
                    return Some(name);
                }
            }
            None
        }
        _ => {
            for input in plan.inputs() {
                if let Some(name) = extract_table_name(input) {
                    return Some(name);
                }
            }
            None
        }
    }
}

impl OptimizerRule for HistogramInjectionRule {
    fn name(&self) -> &str {
        "HistogramInjectionRule"
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
                if extend.token_histogram.is_some() {
                    return Ok(Transformed::no(plan));
                }

                let (table_name, histogram) =
                    self.lookup_histogram("ExtendNode", &extend.input, &extend.input_schema, false);
                if let Some(ref table) = table_name
                    && let Some(histogram) = histogram
                {
                    let budget = match resolve_model_card(&self.model_registry, &extend.model) {
                        Some(card) => build_budget_for_extend(&card, extend, self.reasoning),
                        None => {
                            log::warn!(
                                "[HistogramInjectionRule] Model '{}' not found in registry, using fallback budget",
                                extend.model
                            );
                            fallback_batch_budget()
                        }
                    };

                    let mut new_node = extend.clone();
                    new_node.token_histogram = Some(histogram);
                    new_node.batch_budget = Some(budget);
                    log::info!(
                        "[HistogramInjectionRule] Injected histogram into ExtendNode for table '{}'",
                        table
                    );
                    return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                        node: Arc::new(new_node),
                    })));
                }

                log::info!(
                    "[HistogramInjectionRule] No histogram found for ExtendNode (table={:?})",
                    table_name
                );
                return Ok(Transformed::no(plan));
            }

            // SemanticSelectionNode
            if let Some(sel) = ext.node.as_any().downcast_ref::<SemanticSelectionNode>() {
                if sel.token_histogram.is_some() {
                    return Ok(Transformed::no(plan));
                }

                let (table_name, histogram) = self.lookup_histogram(
                    "SemanticSelectionNode",
                    &sel.input,
                    &sel.input_schema,
                    false,
                );
                if let Some(ref table) = table_name
                    && let Some(histogram) = histogram
                {
                    let budget = match resolve_model_card(&self.model_registry, &sel.model) {
                        Some(card) => build_budget_for_selection(&card, sel, self.reasoning),
                        None => {
                            log::warn!(
                                "[HistogramInjectionRule] Model '{}' not found in registry, using fallback budget",
                                sel.model
                            );
                            fallback_batch_budget()
                        }
                    };

                    let mut new_node = sel.clone();
                    new_node.token_histogram = Some(histogram);
                    new_node.batch_budget = Some(budget);
                    log::info!(
                        "[HistogramInjectionRule] Injected histogram into SemanticSelectionNode for table '{}'",
                        table
                    );
                    return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                        node: Arc::new(new_node),
                    })));
                }

                return Ok(Transformed::no(plan));
            }

            // SemanticJoinNode
            if let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>() {
                if join.token_histogram.is_some() {
                    return Ok(Transformed::no(plan));
                }

                // The right side's histogram is required (as in
                // `compute_optimal_pair_batch_size`); the left side's is optional.
                let (right_table, histogram) = self.lookup_histogram(
                    "SemanticJoinNode right",
                    &join.right_input,
                    &join.right_join_attrs,
                    true,
                );
                if let Some(ref table) = right_table
                    && let Some(histogram) = histogram
                {
                    let (ref_budget, max_batch_size, quality_max_batch_size, max_output_tokens) =
                        match resolve_model_card(&self.model_registry, &join.model) {
                            Some(card) => (
                                build_ref_table_budget_for_join(
                                    card.effective_input_budget(),
                                    join,
                                    self.reasoning,
                                ),
                                Some(card.max_batch_size),
                                card.quality_max_batch_size,
                                card.max_output_tokens,
                            ),
                            None => {
                                log::warn!(
                                    "[HistogramInjectionRule] Model '{}' not found in registry, using fallback budget",
                                    join.model
                                );
                                // No card, so no ceilings; a conservative
                                // context window still applies.
                                (
                                    build_ref_table_budget_for_join(8192, join, self.reasoning),
                                    None,
                                    None,
                                    None,
                                )
                            }
                        };

                    // Try to look up left-side histogram for asymmetric block sizing
                    let (_, left_histogram) = self.lookup_histogram(
                        "SemanticJoinNode left",
                        &join.left_input,
                        &join.left_join_attrs,
                        true,
                    );

                    let mut new_node = join.clone();
                    new_node.token_histogram = Some(histogram);
                    new_node.batch_budget = Some(ref_budget.clone());
                    new_node.join_batch_budget = Some(ref_budget);
                    new_node.left_token_histogram = left_histogram;
                    new_node.model_max_batch_size = max_batch_size;
                    new_node.quality_max_batch_size = quality_max_batch_size;
                    new_node.model_max_output_tokens = max_output_tokens;
                    log::info!(
                        "[HistogramInjectionRule] Injected histogram into SemanticJoinNode for right table '{}' (left_histogram={})",
                        table,
                        new_node.left_token_histogram.is_some()
                    );
                    return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                        node: Arc::new(new_node),
                    })));
                }

                return Ok(Transformed::no(plan));
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
    use nousdb_core::prompt::reasoning::RATIONALE_OUTPUT_TOKENS_PER_ROW;
    use nousdb_core::token_estimator::{
        SafeBatchSizingConfig, SizingBound, TokenHistogram, estimate_safe_batch_size,
    };
    use nousdb_operators::extend::logical::create_extend;
    use std::fs;
    use tempfile::TempDir;

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

    fn setup_catalog() -> (TempDir, Arc<HistogramCatalog>, Arc<ModelRegistry>) {
        let dir = TempDir::new().unwrap();
        let table_dir = dir.path().join("test_table");
        fs::create_dir(&table_dir).unwrap();

        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 20)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let json = serde_json::to_string(&hist).unwrap();
        fs::write(table_dir.join("text.json"), &json).unwrap();

        let catalog = Arc::new(HistogramCatalog::load(dir.path()).unwrap());
        // Empty registry triggers fallback path
        let registry = Arc::new(ModelRegistry::new());
        (dir, catalog, registry)
    }

    #[test]
    fn test_injection_skips_when_no_table_match() {
        let (_dir, catalog, registry) = setup_catalog();
        let rule = HistogramInjectionRule::new(catalog, registry);

        // EmptyRelation has no TableScan, so no table name can be found
        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let plan = create_extend(
            input,
            "Classify",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(!result.transformed);
    }

    #[test]
    fn test_injection_skips_when_already_has_histogram() {
        let (_dir, catalog, registry) = setup_catalog();
        let rule = HistogramInjectionRule::new(catalog, registry);

        let input = make_table_scan();
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            false,
        )]));
        let plan = create_extend(
            input,
            "Classify",
            "test-model",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();

        // Pre-populate histogram
        let plan = if let LogicalPlan::Extension(ext) = plan {
            let mut node = ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone();
            node.token_histogram = Some(TokenHistogram::from_token_counts(&[10, 20], 4));
            node.batch_budget = Some(fallback_batch_budget());
            LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })
        } else {
            panic!("Expected Extension");
        };

        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(!result.transformed);
    }

    /// A vLLM-style card: context 32768, max output 9000, row ceilings of 25.
    fn qwen3_8b_card() -> ModelCard {
        ModelCard::new("Qwen/Qwen3-8B", 32768)
            .with_max_output_tokens(9000)
            .with_max_batch_size(25)
            .with_quality_max_batch_size(25)
    }

    fn selection_node() -> SemanticSelectionNode {
        let input = make_table_scan();
        let schema = input.schema().clone();
        SemanticSelectionNode::new(
            input,
            "Qwen/Qwen3-8B".to_string(),
            vec!["text".to_string()],
            "the review expresses a positive opinion about the movie".to_string(),
            schema,
        )
    }

    fn extend_node() -> ExtendNode {
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]));
        let plan = create_extend(
            make_table_scan(),
            "the review expresses a positive opinion about the movie",
            "Qwen/Qwen3-8B",
            vec!["text".to_string()],
            derived,
        )
        .unwrap();
        match plan {
            LogicalPlan::Extension(ext) => ext
                .node
                .as_any()
                .downcast_ref::<ExtendNode>()
                .unwrap()
                .clone(),
            _ => panic!("Expected Extension"),
        }
    }

    fn join_node() -> SemanticJoinNode {
        let left = make_table_scan();
        let right = make_table_scan();
        let schema = left.schema().clone();
        SemanticJoinNode::new(
            left,
            right,
            vec!["text".to_string()],
            vec!["text".to_string()],
            "Qwen/Qwen3-8B".to_string(),
            "the two entries refer to the same thing".to_string(),
            schema,
        )
    }

    /// Review-length-shaped token counts (60 to 299 tokens per row).
    fn imdb_histogram() -> TokenHistogram {
        let counts: Vec<usize> = (0..1000).map(|i| 60 + (i % 240)).collect();
        TokenHistogram::from_token_counts(&counts, 32)
    }

    /// The reasoning mode's per-item generation cost must reach the budget
    /// model, or a large batch could truncate the JSON array and lose the
    /// whole request.
    #[test]
    fn the_rationale_arm_charges_its_output_tokens_to_every_budget() {
        let card = qwen3_8b_card();
        let sel = selection_node();
        let ext = extend_node();
        let join_reasoning = |mode: ReasoningMode| {
            // `JoinBatchBudgetModel` has no reasoning field; the arm's cost is
            // folded into `output_tokens_per_pair`, which the estimator reads
            // for the output cap and nothing else.
            let node = join_node();
            build_ref_table_budget_for_join(card.effective_input_budget(), &node, mode)
                .output_tokens_per_pair
        };

        for mode in ReasoningMode::all() {
            let expected = mode.reasoning_output_tokens_per_row();
            assert_eq!(
                build_budget_for_selection(&card, &sel, mode).reasoning_output_tokens_per_row,
                expected,
                "selection/{mode:?}"
            );
            assert_eq!(
                build_budget_for_extend(&card, &ext, mode).reasoning_output_tokens_per_row,
                expected,
                "extend/{mode:?}"
            );
            assert_eq!(join_reasoning(mode), 8 + expected, "join/{mode:?}");
        }

        assert_eq!(
            build_budget_for_selection(&card, &sel, ReasoningMode::Rationale)
                .reasoning_output_tokens_per_row,
            RATIONALE_OUTPUT_TOKENS_PER_ROW,
            "the constant must be the one that reaches sizing"
        );
    }

    /// The output cap is computed but must not decide k here: `(9000 - 20) /
    /// (4 + 48) = 172` rows for selection, far above the card's quality ceiling
    /// of 25. If this fails, the rationale mode has started costing throughput
    /// in throughput and the constant is not the thing to adjust, the cap is.
    #[test]
    fn the_rationale_output_cap_does_not_move_k_on_qwen3_8b() {
        let card = qwen3_8b_card();
        let sel = selection_node();
        let hist = imdb_histogram();
        let config = SafeBatchSizingConfig::default();

        let off = estimate_safe_batch_size(
            &hist,
            &build_budget_for_selection(&card, &sel, ReasoningMode::Off),
            &config,
        );
        let budget = build_budget_for_selection(&card, &sel, ReasoningMode::Rationale);
        let rationale = estimate_safe_batch_size(&hist, &budget, &config);

        // The cap exists and is charged...
        assert_eq!(
            budget.reasoning_output_tokens_per_row,
            RATIONALE_OUTPUT_TOKENS_PER_ROW
        );
        let per_row =
            budget.expected_output_tokens_per_row + budget.reasoning_output_tokens_per_row;
        assert_eq!(per_row, 52);
        assert_eq!(
            (9000 - 20) / per_row,
            172,
            "the output-cap arithmetic moved"
        );

        // ...and loses to the quality ceiling, so k is unchanged.
        assert_eq!(off.final_batch_size, 25);
        assert_eq!(off.bound, SizingBound::QualityCap);
        assert_eq!(
            rationale.final_batch_size, off.final_batch_size,
            "the rationale budget moved k from {} to {}",
            off.final_batch_size, rationale.final_batch_size
        );
        assert_eq!(rationale.bound, SizingBound::QualityCap);
    }

    /// Extend has a wider answer than selection, so its output cap is closer to
    /// binding, but at one derived Utf8 column it is still 112 rows against a
    /// quality ceiling of 25. This pins the crossover (12 short-named Utf8
    /// columns with a rationale, 14 without).
    #[test]
    fn the_rationale_output_cap_binds_for_extend_only_past_twelve_derived_columns() {
        let card = qwen3_8b_card();
        let hist = imdb_histogram();
        let config = SafeBatchSizingConfig::default();

        let ext = extend_node();
        let budget = build_budget_for_extend(&card, &ext, ReasoningMode::Rationale);
        assert_eq!(budget.expected_output_tokens_per_row, 32); // 6 + 2 + 24
        assert_eq!((9000 - 20) / (32 + RATIONALE_OUTPUT_TOKENS_PER_ROW), 112);
        let est = estimate_safe_batch_size(&hist, &budget, &config);
        assert_eq!(est.final_batch_size, 25);
        assert_eq!(est.bound, SizingBound::QualityCap);

        // Sweep the derived-column count with the same 9-character name, so
        // `estimate_output_tokens_per_row` adds 26 tokens per column.
        let cap_at = |columns: usize, reasoning: usize| {
            let fields: Vec<Field> = (0..columns)
                .map(|i| Field::new(format!("sentimen{i}"), DataType::Utf8, true))
                .collect();
            let per_row = estimate_output_tokens_per_row(&Schema::new(fields).fields().clone());
            (9000 - 20) / (per_row + reasoning)
        };
        assert!(cap_at(11, RATIONALE_OUTPUT_TOKENS_PER_ROW) >= 25);
        assert!(cap_at(12, RATIONALE_OUTPUT_TOKENS_PER_ROW) < 25);
        assert!(cap_at(13, 0) >= 25);
        assert!(cap_at(14, 0) < 25);
    }

    use datafusion::logical_expr::{LogicalPlanBuilder, cast, col, table_scan};
    use nousdb_operators::join::create_semantic_join;
    use nousdb_operators::selection::create_semantic_selection;

    /// `movies(title, review)` with 200-row histograms, `people(name, bio)`
    /// with a 50-row `name` histogram, so `total_rows` tells them apart.
    fn lineage_catalog() -> (TempDir, Arc<HistogramCatalog>) {
        let dir = TempDir::new().unwrap();
        let write = |table: &str, column: &str, counts: Vec<usize>| {
            let table_dir = dir.path().join(table);
            fs::create_dir_all(&table_dir).unwrap();
            let hist = TokenHistogram::from_token_counts(&counts, 32);
            fs::write(
                table_dir.join(format!("{column}.json")),
                serde_json::to_string(&hist).unwrap(),
            )
            .unwrap();
        };
        write("movies", "title", (0..200).map(|i| 4 + i % 8).collect());
        write(
            "movies",
            "review",
            (0..200).map(|i| 120 + i % 240).collect(),
        );
        write("people", "name", (0..50).map(|i| 3 + i % 4).collect());
        let catalog = Arc::new(HistogramCatalog::load(dir.path()).unwrap());
        (dir, catalog)
    }

    fn scan(table: &str, columns: [&str; 2]) -> LogicalPlanBuilder {
        let schema = Schema::new(
            columns
                .iter()
                .map(|c| Field::new(*c, DataType::Utf8, false))
                .collect::<Vec<_>>(),
        );
        table_scan(Some(table), &schema, None).unwrap()
    }

    fn movies() -> LogicalPlanBuilder {
        scan("movies", ["title", "review"])
    }

    fn people() -> LogicalPlanBuilder {
        scan("people", ["name", "bio"])
    }

    /// `join_demo`'s side: `LIMIT n` → `.alias(alias)` → `select_columns`.
    fn aliased_movies(alias: &str, rows: usize) -> LogicalPlan {
        movies()
            .limit(0, Some(rows))
            .unwrap()
            .alias(alias)
            .unwrap()
            .project(vec![
                col(format!("{alias}.title")),
                col(format!("{alias}.review")),
            ])
            .unwrap()
            .build()
            .unwrap()
    }

    fn names(columns: &[&str]) -> Vec<String> {
        columns.iter().map(|c| c.to_string()).collect()
    }

    fn lineage(table: &str, columns: &[&str]) -> HistogramSource {
        HistogramSource::Lineage {
            table: table.to_string(),
            columns: names(columns),
        }
    }

    fn first_scan(table: &str, columns: &[&str]) -> HistogramSource {
        HistogramSource::FirstScan {
            table: Some(table.to_string()),
            columns: names(columns),
        }
    }

    #[test]
    fn an_unqualified_column_traces_to_its_scan() {
        let input = movies().build().unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["review"]), false),
            lineage("movies", &["review"])
        );
    }

    #[test]
    fn a_table_qualified_column_traces_to_its_scan() {
        let input = movies().build().unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["movies.title"]), true),
            lineage("movies", &["title"])
        );
    }

    /// The qualifier is an alias, not a catalog table, so the lookup must not
    /// ask the right table for `drama_movies.title`.
    #[test]
    fn an_alias_qualified_column_traces_through_the_alias() {
        let (_dir, catalog) = lineage_catalog();
        let input = aliased_movies("drama_movies", 20);
        let columns = names(&["drama_movies.title", "drama_movies.review"]);

        let first_scan_table = extract_table_name(&input).unwrap();
        assert_eq!(first_scan_table, "movies");
        assert!(
            catalog.get_combined(&first_scan_table, &columns).is_none(),
            "the pre-lineage lookup finds nothing for qualified names"
        );

        assert_eq!(
            histogram_source("test", &input, &columns, true),
            lineage("movies", &["title", "review"])
        );
    }

    #[test]
    fn a_projection_renamed_column_traces_to_its_source_name() {
        let input = movies()
            .project(vec![col("review").alias("body")])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["body"]), false),
            lineage("movies", &["review"])
        );
    }

    #[test]
    fn a_column_from_the_second_scan_traces_to_that_table() {
        let input = movies()
            .cross_join(people().build().unwrap())
            .unwrap()
            .build()
            .unwrap();
        // The first-scan guess would ask `movies` for `name`.
        assert_eq!(extract_table_name(&input).as_deref(), Some("movies"));
        assert_eq!(
            histogram_source("test", &input, &names(&["name"]), false),
            lineage("people", &["name"])
        );
    }

    #[test]
    fn a_semantic_join_output_traces_to_both_inputs() {
        let join = create_semantic_join(
            &movies().build().unwrap(),
            &people().build().unwrap(),
            names(&["review"]),
            names(&["name"]),
            "the review mentions the person",
            "test-model",
        )
        .unwrap();
        assert_eq!(
            histogram_source("test", &join, &names(&["name"]), false),
            lineage("people", &["name"])
        );
        assert_eq!(
            histogram_source("test", &join, &names(&["title"]), false),
            lineage("movies", &["title"])
        );
    }

    /// An extend-derived column has no base column; the traced ones still
    /// count, the way `get_combined` skips a column it has no histogram for.
    #[test]
    fn an_untraced_column_is_left_out_of_a_lineage_lookup() {
        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]));
        let extend = create_extend(
            movies().build().unwrap(),
            "Classify",
            "test-model",
            names(&["review"]),
            derived,
        )
        .unwrap();
        assert_eq!(
            histogram_source("test", &extend, &names(&["review", "sentiment"]), false),
            lineage("movies", &["review"])
        );
    }

    #[test]
    fn computed_columns_fall_back_to_the_first_scan() {
        let input = movies()
            .project(vec![cast(col("review"), DataType::Utf8).alias("body")])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["body"]), false),
            first_scan("movies", &["body"])
        );
    }

    /// A union merges two columns into one: not traced, so the lookup falls back
    /// to the first scan's histogram.
    #[test]
    fn a_union_falls_back_to_the_first_scan() {
        let input = movies()
            .union(movies().build().unwrap())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["review"]), false),
            first_scan("movies", &["review"])
        );
    }

    #[test]
    fn columns_from_two_tables_fall_back_to_the_first_scan() {
        let input = movies()
            .cross_join(people().build().unwrap())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            histogram_source("test", &input, &names(&["review", "name"]), false),
            first_scan("movies", &["review", "name"])
        );
    }

    fn rewrite_join(rule: &HistogramInjectionRule, plan: LogicalPlan) -> SemanticJoinNode {
        let config = datafusion::optimizer::OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(result.transformed, "the join must get its histograms");
        match result.data {
            LogicalPlan::Extension(ext) => ext
                .node
                .as_any()
                .downcast_ref::<SemanticJoinNode>()
                .unwrap()
                .clone(),
            other => panic!("expected the join, got {other}"),
        }
    }

    #[test]
    fn an_alias_qualified_join_gets_both_histograms() {
        let (_dir, catalog) = lineage_catalog();
        let rule = HistogramInjectionRule::new(catalog, Arc::new(ModelRegistry::new()));
        let plan = create_semantic_join(
            &aliased_movies("drama_movies", 20),
            &aliased_movies("comedy_movies", 10),
            names(&["drama_movies.title", "drama_movies.review"]),
            names(&["comedy_movies.title", "comedy_movies.review"]),
            "the reviews discuss similar themes",
            "test-model",
        )
        .unwrap();

        let join = rewrite_join(&rule, plan);
        assert_eq!(join.token_histogram.map(|h| h.total_rows()), Some(200));
        assert_eq!(join.left_token_histogram.map(|h| h.total_rows()), Some(200));
        assert!(join.join_batch_budget.is_some());
    }

    #[test]
    fn a_table_qualified_join_reads_each_sides_table() {
        let (_dir, catalog) = lineage_catalog();
        let rule = HistogramInjectionRule::new(catalog, Arc::new(ModelRegistry::new()));
        let plan = create_semantic_join(
            &movies().build().unwrap(),
            &people().build().unwrap(),
            names(&["movies.review"]),
            names(&["people.name"]),
            "the review mentions the person",
            "test-model",
        )
        .unwrap();

        let join = rewrite_join(&rule, plan);
        assert_eq!(join.token_histogram.map(|h| h.total_rows()), Some(50));
        assert_eq!(join.left_token_histogram.map(|h| h.total_rows()), Some(200));
    }

    #[test]
    fn projection_renamed_extend_and_selection_columns_get_the_source_histogram() {
        let (_dir, catalog) = lineage_catalog();
        let rule = HistogramInjectionRule::new(catalog, Arc::new(ModelRegistry::new()));
        let config = datafusion::optimizer::OptimizerContext::new();
        let renamed = movies()
            .project(vec![col("review").alias("body")])
            .unwrap()
            .build()
            .unwrap();

        let derived = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]));
        let extend = create_extend(
            renamed.clone(),
            "Classify",
            "test-model",
            names(&["body"]),
            derived,
        )
        .unwrap();
        let LogicalPlan::Extension(ext) = rule.rewrite(extend, &config).unwrap().data else {
            panic!("expected the extend");
        };
        let extend = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
        assert_eq!(
            extend.token_histogram.as_ref().map(|h| h.total_rows()),
            Some(200)
        );

        let sel = create_semantic_selection(&renamed, "test-model", names(&["body"]), "positive")
            .unwrap();
        let LogicalPlan::Extension(ext) = rule.rewrite(sel, &config).unwrap().data else {
            panic!("expected the selection");
        };
        let sel = ext
            .node
            .as_any()
            .downcast_ref::<SemanticSelectionNode>()
            .unwrap();
        assert_eq!(
            sel.token_histogram.as_ref().map(|h| h.total_rows()),
            Some(200)
        );
    }
}
