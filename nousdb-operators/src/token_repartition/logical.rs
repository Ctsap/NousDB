//! Logical plan node and constructor for token repartitioning.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::common::DFSchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::token_repartition::config::{PartitionStrategy, TokenPartitionConfig};

/// Logical node for token-based repartitioning.
///
/// Repartitions data by estimated token counts rather than row counts: each
/// row's tokens are estimated and rows are assigned to partitions so that each
/// partition has approximately equal total tokens.
#[derive(Debug, Clone)]
pub struct TokenRepartitionNode {
    /// The input logical plan
    pub input: LogicalPlan,

    /// Columns to estimate tokens from
    pub token_columns: Vec<String>,

    /// Target number of partitions (if specified)
    pub num_partitions: Option<usize>,

    /// Maximum tokens per partition (if specified)
    pub token_budget: Option<usize>,

    /// Partitioning strategy
    pub strategy: PartitionStrategy,

    /// Output schema (same as input, repartitioning doesn't change schema)
    pub schema: DFSchemaRef,
}

// Custom PartialEq that handles PartitionStrategy comparison
impl PartialEq for TokenRepartitionNode {
    fn eq(&self, other: &Self) -> bool {
        self.token_columns == other.token_columns
            && self.num_partitions == other.num_partitions
            && self.token_budget == other.token_budget
            && self.strategy == other.strategy
            && self.schema == other.schema
        // Input plans are not compared.
    }
}

impl Eq for TokenRepartitionNode {}

// Custom Hash implementation
impl Hash for TokenRepartitionNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.token_columns.hash(state);
        self.num_partitions.hash(state);
        self.token_budget.hash(state);
        // Hash the strategy
        match &self.strategy {
            PartitionStrategy::OptimalTokenBalance => {
                "OptimalTokenBalance".hash(state);
            }
            PartitionStrategy::EqualRowCounts => {
                "EqualRowCounts".hash(state);
            }
        }
        // Input and schema are not hashed.
    }
}

impl TokenRepartitionNode {
    /// Creates a node whose output schema equals the input schema.
    pub fn new(
        input: LogicalPlan,
        token_columns: Vec<String>,
        num_partitions: Option<usize>,
        token_budget: Option<usize>,
        strategy: PartitionStrategy,
    ) -> Self {
        let schema = input.schema().clone();
        Self {
            input,
            token_columns,
            num_partitions,
            token_budget,
            strategy,
            schema,
        }
    }

    fn fingerprint(&self) -> String {
        let strategy_str = match &self.strategy {
            PartitionStrategy::OptimalTokenBalance => "OptimalTokenBalance".to_string(),
            PartitionStrategy::EqualRowCounts => "EqualRowCounts".to_string(),
        };
        format!(
            "TokenRepartition|columns={:?}|partitions={:?}|budget={:?}|strategy={}|schema={}",
            self.token_columns, self.num_partitions, self.token_budget, strategy_str, self.schema
        )
    }
}

impl PartialOrd for TokenRepartitionNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.fingerprint().cmp(&other.fingerprint()))
    }
}

impl UserDefinedLogicalNodeCore for TokenRepartitionNode {
    fn name(&self) -> &str {
        "TokenRepartitionNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.token_columns
            .iter()
            .map(|col| Expr::Column(col.clone().into()))
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let strategy_str = match &self.strategy {
            PartitionStrategy::OptimalTokenBalance => "OptimalTokenBalance".to_string(),
            PartitionStrategy::EqualRowCounts => "EqualRowCounts".to_string(),
        };
        write!(
            f,
            "TokenRepartition: columns={:?}, partitions={:?}, budget={:?}, strategy={}",
            self.token_columns, self.num_partitions, self.token_budget, strategy_str
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self, DataFusionError> {
        if inputs.len() != 1 {
            return Err(DataFusionError::Internal(
                "TokenRepartitionNode expects exactly one input".to_string(),
            ));
        }
        Ok(TokenRepartitionNode::new(
            inputs[0].clone(),
            self.token_columns.clone(),
            self.num_partitions,
            self.token_budget,
            self.strategy.clone(),
        ))
    }
}

/// Creates a TokenRepartition logical plan that repartitions `input` by token
/// counts so that model workload is balanced across partitions.
///
/// `config` specifies the token columns and the partition count and/or token
/// budget. Returns an error if a column is missing from the input schema or
/// neither a partition count nor a budget is given.
///
/// # Example
///
/// ```ignore
/// use nousdb_operators::token_repartition::{TokenPartitionConfig, create_token_repartition};
///
/// let config = TokenPartitionConfig::with_num_partitions(
///     vec!["reviewText".to_string()],
///     8,
/// );
/// let repartitioned = create_token_repartition(input_plan, config)?;
/// ```
pub fn create_token_repartition(
    input: LogicalPlan,
    config: TokenPartitionConfig,
) -> DFResult<LogicalPlan> {
    // Validate that the specified columns exist in the input schema
    let input_schema = input.schema();
    for col_name in &config.token_columns {
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

    // At least one of num_partitions or token_budget must be specified
    if config.num_partitions.is_none() && config.token_budget.is_none() {
        return Err(DataFusionError::Plan(
            "TokenRepartition requires either num_partitions or token_budget to be specified"
                .to_string(),
        ));
    }

    let node = TokenRepartitionNode::new(
        input,
        config.token_columns,
        config.num_partitions,
        config.token_budget,
        config.strategy,
    );

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}
