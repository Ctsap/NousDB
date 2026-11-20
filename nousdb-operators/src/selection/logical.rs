//! Logical plan node and constructor for semantic selection.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use datafusion::common::DFSchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use std::sync::Arc;

use nousdb_core::token_estimator::{BatchBudgetModel, TokenHistogram};

/// Logical plan node for semantic selection: keeps the rows the model accepts.
#[derive(Debug, Clone)]
pub struct SemanticSelectionNode {
    /// The input plan.
    pub input: LogicalPlan,

    /// Output schema of this node (same as the input schema).
    pub schema: DFSchemaRef,

    /// Model identifier.
    pub model: String,

    /// Names of the input columns shown to the model.
    pub input_schema: Vec<String>,

    /// Natural-language filter condition.
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

impl SemanticSelectionNode {
    /// Creates a node with no optimizer annotations set.
    pub fn new(
        input: LogicalPlan,
        model: String,
        input_schema: Vec<String>,
        prompt: String,
        schema: DFSchemaRef,
    ) -> Self {
        Self {
            input,
            model,
            input_schema,
            prompt,
            schema,
            optimal_batch_size: None,
            token_histogram: None,
            batch_budget: None,
        }
    }

    /// Model identifier.
    pub fn model(&self) -> &String {
        &self.model
    }

    /// Natural-language filter condition.
    pub fn prompt(&self) -> &String {
        &self.prompt
    }

    /// Fingerprint used to order nodes: node type, schema, instruction, input
    /// columns and model. The input plan is excluded because DataFusion
    /// compares children separately.
    fn fingerprint(&self) -> String {
        let cols_txt = self
            .input_schema
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("|");

        format!(
            "SemanticSelection|schema={}|instruction={}|cols={}|model={}",
            self.schema, self.prompt, cols_txt, self.model
        )
    }
}

// `optimal_batch_size` is excluded from Hash/PartialEq so that setting it in
// the optimizer does not change the plan signature and trigger extra passes.
impl PartialEq for SemanticSelectionNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.schema == other.schema
            && self.model == other.model
            && self.input_schema == other.input_schema
            && self.prompt == other.prompt
    }
}

impl Eq for SemanticSelectionNode {}

impl Hash for SemanticSelectionNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.schema.hash(state);
        self.model.hash(state);
        self.input_schema.hash(state);
        self.prompt.hash(state);
    }
}

impl PartialOrd for SemanticSelectionNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.fingerprint().cmp(&other.fingerprint()))
    }
}

impl UserDefinedLogicalNodeCore for SemanticSelectionNode {
    fn name(&self) -> &str {
        "SemanticSelection"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
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
            "NousDB_SemanticSelection: model={}, instruction={}, input_schema={:?}",
            self.model, self.prompt, self.input_schema,
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
        let mut node = SemanticSelectionNode::new(
            inputs[0].clone(),
            self.model.clone(),
            self.input_schema.clone(),
            self.prompt.clone(),
            self.schema.clone(),
        );
        node.optimal_batch_size = self.optimal_batch_size;
        node.token_histogram = self.token_histogram.clone();
        node.batch_budget = self.batch_budget.clone();
        Ok(node)
    }
}

/// Creates a SemanticSelection logical plan that keeps the rows of `input`
/// accepted by the model. The output schema equals the input schema.
///
/// * `input`: the input logical plan
/// * `model`: model identifier
/// * `input_columns`: columns shown to the model
/// * `user_instruction`: natural-language filter condition
///
/// Returns an error if an input column is missing from the input schema.
pub fn create_semantic_selection(
    input: &LogicalPlan,
    model: impl Into<String>,
    input_columns: Vec<String>,
    user_instruction: impl Into<String>,
) -> DFResult<LogicalPlan, DataFusionError> {
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

    let node = SemanticSelectionNode {
        input: input.clone(),
        schema: input.schema().clone(),
        model: model.into(),
        input_schema: input_columns,
        prompt: user_instruction.into(),
        optimal_batch_size: None,
        token_histogram: None,
        batch_budget: None,
    };

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}
