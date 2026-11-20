//! Logical plan node and constructors for the semantic join.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef, TableReference};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use std::collections::HashSet;
use std::sync::Arc;

use nousdb_core::token_estimator::{JoinBatchBudgetModel, TokenHistogram};

use crate::join::join_schema_validation::validate_column_exists;

/// Logical plan node for the semantic join: a join whose predicate is
/// evaluated by a language model over the join attributes of each left x right
/// row pair.
#[derive(Debug, Clone)]
pub struct SemanticJoinNode {
    /// Logical plan producing the left relation.
    pub left_input: LogicalPlan,

    /// Logical plan producing the right relation.
    pub right_input: LogicalPlan,

    /// Join attribute names from left relation
    pub left_join_attrs: Vec<String>,

    /// Join attribute names from right relation
    pub right_join_attrs: Vec<String>,

    /// Output schema of this node.
    pub schema: DFSchemaRef,

    /// Identifier of the model that evaluates the predicate.
    pub model: String,

    /// Natural language description of the join condition. At execution it is
    /// combined with the values of each row pair to form the model prompt.
    pub prompt: String,

    /// Optimizer-determined left block size for join evaluation.
    /// When `None`, the physical operator uses default sizing.
    pub left_block_size: Option<usize>,

    /// Optimizer-determined right block size for join evaluation.
    /// When `None`, the physical operator uses default sizing.
    pub right_block_size: Option<usize>,

    /// Token histogram for histogram-based batch sizing.
    pub token_histogram: Option<TokenHistogram>,

    /// Batch budget model for histogram-based batch sizing.
    pub batch_budget: Option<JoinBatchBudgetModel>,

    /// Model's max pairs per LLM call (from ModelCard.max_batch_size).
    /// Populated by HistogramInjectionRule, used by BatchSizeOptimizationRule.
    pub model_max_batch_size: Option<usize>,

    /// Model's accuracy-derived pair ceiling
    /// (from ModelCard.quality_max_batch_size).
    /// Populated by HistogramInjectionRule, used by BatchSizeOptimizationRule.
    pub quality_max_batch_size: Option<usize>,

    /// Model's max output tokens (from ModelCard.max_output_tokens).
    /// Used to cap pairs by output budget.
    pub model_max_output_tokens: Option<usize>,

    /// Pre-computed token histogram for the left relation (optional).
    /// Used by BatchSizeOptimizationRule for asymmetric block sizing.
    /// When None, the right histogram is used as a symmetric proxy.
    pub left_token_histogram: Option<TokenHistogram>,

    /// Join batch budget model used by BatchSizeOptimizationRule for the
    /// O(L+R) cost model.
    pub join_batch_budget: Option<JoinBatchBudgetModel>,
}

impl SemanticJoinNode {
    /// Creates a node over `left_input` and `right_input`.
    ///
    /// `left_join_attrs` and `right_join_attrs` name the columns that enter the
    /// prompt, `model` identifies the model, `prompt` is the natural language
    /// join condition and `schema` is the output schema (all columns from both
    /// inputs). Optimizer-populated fields start as `None`.
    pub fn new(
        left_input: LogicalPlan,
        right_input: LogicalPlan,
        left_join_attrs: Vec<String>,
        right_join_attrs: Vec<String>,
        model: String,
        prompt: String,
        schema: DFSchemaRef,
    ) -> Self {
        Self {
            left_input,
            right_input,
            left_join_attrs,
            right_join_attrs,
            schema,
            model,
            prompt,
            left_block_size: None,
            right_block_size: None,
            token_histogram: None,
            batch_budget: None,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            model_max_output_tokens: None,
            left_token_histogram: None,
            join_batch_budget: None,
        }
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &String {
        &self.model
    }

    /// Returns the join condition prompt.
    pub fn prompt(&self) -> &String {
        &self.prompt
    }

    /// Returns the left join attributes.
    pub fn left_join_attrs(&self) -> &Vec<String> {
        &self.left_join_attrs
    }

    /// Returns the right join attributes.
    pub fn right_join_attrs(&self) -> &Vec<String> {
        &self.right_join_attrs
    }

    /// Fingerprint used to order nodes. It covers the parameters that affect the
    /// join's semantics (schema, attributes, prompt, model) and excludes the
    /// input plans, which DataFusion compares separately.
    fn fingerprint(&self) -> String {
        let left_cols = self.left_join_attrs.join("|");
        let right_cols = self.right_join_attrs.join("|");

        format!(
            "SemanticJoin|schema={}|left_attrs={}|right_attrs={}|prompt={}|model={}",
            self.schema, left_cols, right_cols, self.prompt, self.model,
        )
    }
}

// `left_block_size` and `right_block_size` are excluded from Hash/PartialEq so
// that the batch-size optimizer rule does not change the plan signature and
// trigger extra optimizer passes.
impl PartialEq for SemanticJoinNode {
    fn eq(&self, other: &Self) -> bool {
        self.left_input == other.left_input
            && self.right_input == other.right_input
            && self.left_join_attrs == other.left_join_attrs
            && self.right_join_attrs == other.right_join_attrs
            && self.schema == other.schema
            && self.model == other.model
            && self.prompt == other.prompt
    }
}

impl Eq for SemanticJoinNode {}

impl Hash for SemanticJoinNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.left_input.hash(state);
        self.right_input.hash(state);
        self.left_join_attrs.hash(state);
        self.right_join_attrs.hash(state);
        self.schema.hash(state);
        self.model.hash(state);
        self.prompt.hash(state);
    }
}

impl PartialOrd for SemanticJoinNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.fingerprint().cmp(&other.fingerprint()))
    }
}

impl UserDefinedLogicalNodeCore for SemanticJoinNode {
    fn name(&self) -> &str {
        "SemanticJoin"
    }

    /// The left and right input plans.
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.left_input, &self.right_input]
    }

    /// All columns of both input relations.
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    /// Column references to the join attributes of both relations, so that
    /// projection pushdown and column pruning keep them.
    fn expressions(&self) -> Vec<Expr> {
        let mut exprs = Vec::new();

        for attr in &self.left_join_attrs {
            exprs.push(Expr::Column(attr.clone().into()));
        }

        for attr in &self.right_join_attrs {
            exprs.push(Expr::Column(attr.clone().into()));
        }

        exprs
    }

    /// Formats the node's key parameters for `EXPLAIN` output.
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NousDB_SemanticJoin: model={}, left_attrs={:?}, right_attrs={:?}, prompt='{}'",
            self.model, self.left_join_attrs, self.right_join_attrs, self.prompt,
        )?;
        if let Some(lbs) = self.left_block_size {
            write!(f, ", left_block_size={}", lbs)?;
        }
        if let Some(rbs) = self.right_block_size {
            write!(f, ", right_block_size={}", rbs)?;
        }
        Ok(())
    }

    /// Rebuilds the node over new inputs, preserving all other parameters.
    ///
    /// The join condition is evaluated by a model rather than by expressions,
    /// so `_exprs` is ignored. Exactly two inputs are required.
    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self, DataFusionError> {
        if inputs.len() != 2 {
            return Err(DataFusionError::Internal(
                "SemanticJoin expects exactly two inputs".to_string(),
            ));
        }

        let mut node = SemanticJoinNode::new(
            inputs[0].clone(),
            inputs[1].clone(),
            self.left_join_attrs.clone(),
            self.right_join_attrs.clone(),
            self.model.clone(),
            self.prompt.clone(),
            self.schema.clone(),
        );
        node.left_block_size = self.left_block_size;
        node.right_block_size = self.right_block_size;
        node.token_histogram = self.token_histogram.clone();
        node.batch_budget = self.batch_budget.clone();
        node.model_max_batch_size = self.model_max_batch_size;
        node.quality_max_batch_size = self.quality_max_batch_size;
        node.model_max_output_tokens = self.model_max_output_tokens;
        node.left_token_histogram = self.left_token_histogram.clone();
        node.join_batch_budget = self.join_batch_budget.clone();
        Ok(node)
    }
}

/// Creates a semantic join logical plan wrapped in a `LogicalPlan::Extension`.
///
/// The join attributes must exist in their respective input schemas. The
/// output schema is the left columns followed by the right columns; right
/// columns whose names collide with a left column get a numeric suffix (see
/// [`build_join_schema`]).
///
/// # Errors
///
/// Returns an error if any join attribute does not exist in its schema.
///
/// # Example
///
/// ```ignore
/// let directors_plan = ctx.table("directors").await?.logical_plan();
/// let reviews_plan = ctx.table("reviews").await?.logical_plan();
///
/// let join_plan = create_semantic_join(
///     &directors_plan,
///     &reviews_plan,
///     vec!["name".to_string(), "signature_style".to_string()],
///     vec!["review_text".to_string()],
///     "The review discusses or mentions this director",
///     "llama-3.1-8b",
/// )?;
/// ```
pub fn create_semantic_join(
    left_input: &LogicalPlan,
    right_input: &LogicalPlan,
    left_join_attrs: Vec<String>,
    right_join_attrs: Vec<String>,
    prompt: impl Into<String>,
    model: impl Into<String>,
) -> DFResult<LogicalPlan, DataFusionError> {
    let left_schema = left_input.schema();
    let right_schema = right_input.schema();

    for col in &left_join_attrs {
        validate_column_exists(col, left_schema, "left relation")?;
    }

    for col in &right_join_attrs {
        validate_column_exists(col, right_schema, "right relation")?;
    }

    let output_schema = build_join_schema(left_schema, right_schema)?;

    log::debug!("Semantic join output schema: {}", output_schema);

    let node = SemanticJoinNode::new(
        left_input.clone(),
        right_input.clone(),
        left_join_attrs,
        right_join_attrs,
        model.into(),
        prompt.into(),
        output_schema,
    );

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}

/// Builds the output schema of a semantic join: all left fields, then all
/// right fields. A right field whose name collides with an existing field is
/// renamed with the first free numeric suffix (`name_1`, `name_2`, ...); left
/// names are never changed. Collisions are decided by bare name, whatever the
/// qualifiers.
///
/// Every field keeps its input's qualifier (a renamed right field keeps the
/// right field's), so `l.title` and `r.title_1` resolve above the join. The
/// Arrow schema carries no qualifiers and no metadata.
pub fn build_join_schema(
    left_schema: &DFSchemaRef,
    right_schema: &DFSchemaRef,
) -> DFResult<DFSchemaRef> {
    let mut used_names = HashSet::new();

    let mut output_fields: Vec<Arc<Field>> = Vec::new();
    let mut qualifiers: Vec<Option<TableReference>> = Vec::new();

    // Left fields keep their names, so an unqualified name that exists on both
    // sides resolves to the left column, as in SQL.
    for (qualifier, field) in left_schema.iter() {
        let field_name = field.name().to_string();
        used_names.insert(field_name.clone());
        output_fields.push(field.clone());
        qualifiers.push(qualifier.cloned());
    }

    for (qualifier, field) in right_schema.iter() {
        let original_name = field.name().to_string();
        qualifiers.push(qualifier.cloned());

        if used_names.contains(&original_name) {
            let new_name = find_available_name(&original_name, &used_names);

            // Keep data type, nullability and metadata.
            let renamed_field = Arc::new(
                Field::new(
                    new_name.clone(),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );

            used_names.insert(new_name);
            output_fields.push(renamed_field);
        } else {
            used_names.insert(original_name);
            output_fields.push(field.clone());
        }
    }

    let fields_for_schema: Vec<Field> = output_fields
        .iter()
        .map(|arc_field| (**arc_field).clone())
        .collect();

    let arrow_schema = Schema::new(fields_for_schema);

    // Neither `try_from` nor `with_field_specific_qualified_schema` runs
    // `check_names`, which keeps this as lenient as the unqualified schema was.
    let df_schema =
        DFSchema::try_from(arrow_schema)?.with_field_specific_qualified_schema(qualifiers)?;

    Ok(Arc::new(df_schema))
}

fn find_available_name(base_name: &str, used_names: &HashSet<String>) -> String {
    let mut suffix = 1;

    loop {
        let candidate = format!("{}_{}", base_name, suffix);

        if !used_names.contains(&candidate) {
            return candidate;
        }

        suffix += 1;
    }
}

/// Maps each right-relation column name to its name in the output schema,
/// where right columns start at index `left_field_count`.
pub fn get_right_column_mapping(
    right_schema: &DFSchemaRef,
    output_schema: &DFSchemaRef,
    left_field_count: usize,
) -> Vec<(String, String)> {
    let mut mapping = Vec::new();

    for (i, right_field) in right_schema.fields().iter().enumerate() {
        let original_name = right_field.name().to_string();
        let output_field = &output_schema.fields()[left_field_count + i];
        let output_name = output_field.name().to_string();

        mapping.push((original_name, output_name));
    }

    mapping
}
