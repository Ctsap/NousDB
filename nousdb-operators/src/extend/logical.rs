//! Logical plan node and constructor for the Extend operator.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::utils::combine_limit;
use datafusion::common::{DFSchema, DFSchemaRef, TableReference};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    Expr, Extension, FetchType, Limit, LogicalPlan, SkipType, UserDefinedLogicalNodeCore, lit,
};
use std::sync::Arc;

use nousdb_core::token_estimator::{BatchBudgetModel, TokenHistogram};

use crate::token_repartition::logical::TokenRepartitionNode;

/// Logical plan node for the Extend operator: adds columns derived by a model.
#[derive(Debug, Clone)]
pub struct ExtendNode {
    /// The input plan.
    pub input: LogicalPlan,

    /// Schema of the derived columns.
    pub derived_schema: Arc<Schema>,

    /// Output schema of this node (input schema + derived columns). Input
    /// columns keep their qualifiers; derived columns are unqualified.
    pub schema: DFSchemaRef,

    /// Model identifier (a logical choice, not a deployment detail).
    pub model: String,

    /// Names of the input columns shown to the model.
    pub input_schema: Vec<String>,

    /// Natural-language instruction for the model.
    pub prompt: String,

    /// Optimizer-determined batch size for LLM calls.
    ///
    /// Never exceeds `ModelCard.max_batch_size`: `BatchSizeOptimizationRule`
    /// clamps the histogram-guided value by it (through
    /// `BatchBudgetModel::model_max_batch_size`), and when this is `None` the
    /// physical planner's fallback heuristic clamps by the same field.
    pub optimal_batch_size: Option<usize>,

    /// Pre-computed token histogram for histogram-based batch sizing.
    pub token_histogram: Option<TokenHistogram>,

    /// Pre-computed batch budget model for histogram-based batch sizing.
    pub batch_budget: Option<BatchBudgetModel>,
}

impl ExtendNode {
    /// Creates a node with no optimizer annotations set.
    pub fn new(
        input: LogicalPlan,
        input_schema: Vec<String>,
        model: String,
        prompt: String,
        derived_schema: Arc<Schema>,
        schema: DFSchemaRef,
    ) -> Self {
        Self {
            input,
            input_schema,
            model,
            prompt,
            derived_schema,
            schema,
            optimal_batch_size: None,
            token_histogram: None,
            batch_budget: None,
        }
    }

    /// Names of the input columns shown to the model.
    pub fn input_schema(&self) -> &Vec<String> {
        &self.input_schema
    }

    /// Output schema (input columns followed by the derived columns).
    pub fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    /// Model identifier.
    pub fn model(&self) -> &String {
        &self.model
    }

    /// Natural-language instruction for the model.
    pub fn prompt(&self) -> &String {
        &self.prompt
    }

    /// The input plan.
    pub fn input(&self) -> &LogicalPlan {
        &self.input
    }

    fn fingerprint(&self) -> String {
        // Child plan equality is left out; DataFusion compares children separately.
        let exprs_txt = self
            .input_schema
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("|");
        format!("Extend|schema={}|attrs={}", self.schema, exprs_txt)
    }
}

// `optimal_batch_size` is excluded from Hash/PartialEq so that setting it in
// the optimizer does not change the plan signature and trigger extra passes.
impl PartialEq for ExtendNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.derived_schema == other.derived_schema
            && self.schema == other.schema
            && self.model == other.model
            && self.input_schema == other.input_schema
            && self.prompt == other.prompt
    }
}

impl Eq for ExtendNode {}

impl Hash for ExtendNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.derived_schema.hash(state);
        self.schema.hash(state);
        self.model.hash(state);
        self.input_schema.hash(state);
        self.prompt.hash(state);
    }
}

impl PartialOrd for ExtendNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.fingerprint().cmp(&other.fingerprint()))
    }
}

impl UserDefinedLogicalNodeCore for ExtendNode {
    fn name(&self) -> &str {
        "ExtendNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    /// DataFusion's limit hook: a limit above an extend may also be applied
    /// to its input. The extend maps rows one to one and in order (a failed
    /// request NULL-fills its rows, it never drops them), so the first `n`
    /// output rows come from the first `n` input rows. DataFusion's
    /// `PushDownLimit` then adds `Limit(skip + fetch)` under this node and
    /// keeps the original limit above it, so the model is called for the
    /// limited rows only.
    ///
    /// Not once token repartitioning sits under the extend. `PushDownLimit`
    /// runs before the insertion rule in every pass, so by then the limit is
    /// already below the repartition node; a later pass would put a second
    /// limit between the two, whose single output partition undoes the
    /// repartitioning (and the insertion rule would repartition again).
    fn supports_limit_pushdown(&self) -> bool {
        !is_token_repartition(&self.input)
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.input_schema
            .iter()
            .map(|attr| Expr::Column(attr.clone().into()))
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NousDB_Extend: instruction='{}', input_cols={:?}, output_cols={:?}",
            self.prompt,
            self.input_schema,
            self.derived_schema
                .fields()
                .iter()
                .map(|field| field.name())
                .collect::<Vec<_>>()
        )?;
        if let Some(bs) = self.optimal_batch_size {
            write!(f, ", batch_size={}", bs)?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self, DataFusionError> {
        if inputs.len() != 1 {
            return Err(datafusion_common::DataFusionError::Internal(
                "NousDB_Extend expects exactly one input".to_string(),
            ));
        }
        // DataFusion's `PushDownLimit` hands this node a new `Limit` over its
        // input on every optimizer pass; see `merge_stacked_limits`.
        let mut node = ExtendNode::new(
            merge_stacked_limits(inputs[0].clone()),
            self.input_schema.clone(),
            self.model.clone(),
            self.prompt.clone(),
            self.derived_schema.clone(),
            self.schema.clone(),
        );
        node.optimal_batch_size = self.optimal_batch_size;
        node.token_histogram = self.token_histogram.clone();
        node.batch_budget = self.batch_budget.clone();
        Ok(node)
    }
}

/// Whether `plan` is a token repartition node.
fn is_token_repartition(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::Extension(ext)
        if ext.node.as_any().downcast_ref::<TokenRepartitionNode>().is_some())
}

/// Collapses `Limit(Limit(X))` into one limit over `X`, as DataFusion's
/// `PushDownLimit` does (`combine_limit`).
///
/// It is done here because DataFusion 55.2 loses that merge under an
/// extension node. `supports_limit_pushdown` makes `PushDownLimit` put a new
/// `Limit` over this node's input on every optimizer pass. The rule then
/// merges the two stacked limits, but it reports the merged plan as
/// unchanged whenever the scan below already carries the fetch
/// (`original_limit` in `push_down_limit.rs`), and the optimizer rebuilds an
/// extension node only when a child reports a change (`map_children_mut` in
/// `optimizer.rs`). The merge is dropped, and the plan gains one `Limit` per
/// pass. A node with no behaviour of its own shows the same
/// (`datafusion_alone_stacks_limits_under_an_extension_node` in
/// `tests/limit_pushdown.rs`).
fn merge_stacked_limits(plan: LogicalPlan) -> LogicalPlan {
    let LogicalPlan::Limit(outer) = &plan else {
        return plan;
    };
    let LogicalPlan::Limit(inner) = outer.input.as_ref() else {
        return plan;
    };
    // Only literal limits can be combined; anything else is left as written.
    let (
        Ok(SkipType::Literal(outer_skip)),
        Ok(FetchType::Literal(outer_fetch)),
        Ok(SkipType::Literal(inner_skip)),
        Ok(FetchType::Literal(inner_fetch)),
    ) = (
        outer.get_skip_type(),
        outer.get_fetch_type(),
        inner.get_skip_type(),
        inner.get_fetch_type(),
    )
    else {
        return plan;
    };
    let (skip, fetch) = combine_limit(outer_skip, outer_fetch, inner_skip, inner_fetch);
    // The inner input may itself be a limit (three stacked after three passes).
    merge_stacked_limits(LogicalPlan::Limit(Limit {
        skip: Some(Box::new(lit(skip as i64))),
        fetch: fetch.map(|rows| Box::new(lit(rows as i64))),
        input: Arc::clone(&inner.input),
    }))
}

/// Creates an Extend logical plan that adds model-derived columns to `input`.
///
/// * `input`: the input logical plan
/// * `user_instruction`: natural-language description of what to derive
/// * `model`: model identifier
/// * `input_columns`: columns from the input shown to the model
/// * `derived_schema`: schema of the new columns
///
/// Returns an error if an input column is missing from the input schema or a
/// derived column name already exists in it.
///
/// # Example
///
/// ```ignore
/// let derived_schema = Schema::new(vec![
///     Field::new("sentiment", DataType::Utf8, false),
/// ]);
///
/// let extended = create_extend(
///     base_plan,
///     "Classify this movie review as positive or negative",
///     vec!["review_text".to_string()],
///     derived_schema,
/// )?;
/// ```
pub fn create_extend(
    input: LogicalPlan,
    user_instruction: impl Into<String>,
    model: impl Into<String>,
    input_columns: Vec<String>,
    derived_schema: Arc<Schema>,
) -> DFResult<LogicalPlan> {
    let input_schema = input.schema();
    for col_name in &input_columns {
        if input_schema.field_with_unqualified_name(col_name).is_err() {
            return Err(DataFusionError::Plan(format!(
                "Column '{}' not found in input schema. Available columns: {:?}",
                col_name,
                input_schema
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            )));
        }
    }

    let mut all_fields = input_schema.fields().to_vec();
    // Input columns keep their qualifiers (`t.review` must still resolve
    // above the node); the derived columns are unqualified.
    let mut qualifiers: Vec<Option<TableReference>> = input_schema
        .iter()
        .map(|(qualifier, _)| qualifier.cloned())
        .collect();

    // Check for name collisions before adding new fields
    for new_field in derived_schema.fields() {
        if input_schema
            .field_with_unqualified_name(new_field.name())
            .is_ok()
        {
            return Err(DataFusionError::Plan(format!(
                "Output column '{}' already exists in input schema. \
                 Choose a different name for the extended attribute.",
                new_field.name()
            )));
        }
        all_fields.push(new_field.clone());
        qualifiers.push(None);
    }

    // The Arrow schema is exactly the input's fields plus the derived ones
    // (no metadata): the physical planner checks `ExtendExec`'s schema
    // against it. `DFSchema::try_from` does not run `check_names`, and neither
    // does `with_field_specific_qualified_schema`, so inputs accepted before
    // qualifiers were kept are still accepted.
    let full_schema = Schema::new(all_fields);
    let df_schema = DFSchema::try_from(full_schema)
        .and_then(|schema| schema.with_field_specific_qualified_schema(qualifiers))
        .map_err(|e| DataFusionError::Plan(format!("Failed to create DFSchema: {}", e)))?;

    let node = ExtendNode {
        input,
        prompt: user_instruction.into(),
        model: model.into(),
        input_schema: input_columns,
        derived_schema,
        schema: Arc::new(df_schema),
        optimal_batch_size: None,
        token_histogram: None,
        batch_budget: None,
    };

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}
