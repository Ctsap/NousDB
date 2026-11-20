//! Extension trait adding the semantic operators to `LogicalPlanBuilder`.

use datafusion::arrow::datatypes::Schema;
use datafusion::error::Result;
use datafusion::logical_expr::LogicalPlanBuilder;
use nousdb_operators::extend::create_extend;
use nousdb_operators::join::create_semantic_join;
use nousdb_operators::selection::create_semantic_selection;

use std::sync::Arc;

/// Extension trait that adds NousDB operators to [`LogicalPlanBuilder`], so
/// they chain like the built-in DataFusion operators.
///
/// ```ignore
/// use datafusion::prelude::*;
/// use nousdb_planner::plan_builder::NousDBPlanBuilderExt;
///
/// let plan = LogicalPlanBuilder::scan("reviews", table_source, None)?
///     .semantic_select(
///         "llama3.1".to_string(),
///         vec!["review_text".to_string()],
///         "The review is positive".to_string(),
///     )?
///     .build()?;
/// ```
pub trait NousDBPlanBuilderExt: Sized {
    /// Adds the columns of `output_schema` to the relation by applying `model`
    /// to `input_columns` under `user_instruction` (the Extend operator).
    fn extend(
        self,
        output_schema: Arc<Schema>,
        user_instruction: impl Into<String>,
        model: impl Into<String>,
        input_columns: Vec<String>,
    ) -> Result<Self>;

    /// Keeps the rows for which `model` judges `filter_instruction` true over
    /// `input_columns`. The schema is unchanged.
    fn semantic_select(
        self,
        model: impl Into<String>,
        input_columns: Vec<String>,
        filter_instruction: impl Into<String>,
    ) -> Result<Self>;

    /// Joins `self` with `right`, keeping the pairs for which `model` judges
    /// `join_condition` true over the given left and right columns.
    ///
    /// Candidate pairs number |left| x |right|, so filter both sides first when
    /// possible. The output schema is the left columns followed by the right
    /// columns.
    fn semantic_join(
        self,
        right: LogicalPlanBuilder,
        left_join_columns: Vec<String>,
        right_join_columns: Vec<String>,
        join_condition: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self>;
}

impl NousDBPlanBuilderExt for LogicalPlanBuilder {
    fn extend(
        self,
        output_schema: Arc<Schema>,
        user_instruction: impl Into<String>,
        model: impl Into<String>,
        input_columns: Vec<String>,
    ) -> Result<Self> {
        let input_plan = self.build()?;

        let extended_plan = create_extend(
            input_plan,
            user_instruction,
            model,
            input_columns,
            output_schema,
        )?;

        Ok(LogicalPlanBuilder::from(extended_plan))
    }

    fn semantic_select(
        self,
        model: impl Into<String>,
        input_columns: Vec<String>,
        filter_instruction: impl Into<String>,
    ) -> Result<Self> {
        let input_plan = self.build()?;

        let filtered_plan = create_semantic_selection(
            &input_plan,
            model.into(),
            input_columns.clone(),
            filter_instruction.into(),
        )?;

        Ok(LogicalPlanBuilder::from(filtered_plan))
    }

    fn semantic_join(
        self,
        right: LogicalPlanBuilder,
        left_join_columns: Vec<String>,
        right_join_columns: Vec<String>,
        join_condition: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        let left_plan = self.build()?;
        let right_plan = right.build()?;

        let joined_plan = create_semantic_join(
            &left_plan,
            &right_plan,
            left_join_columns,
            right_join_columns,
            join_condition.into(),
            model.into(),
        )?;

        Ok(LogicalPlanBuilder::from(joined_plan))
    }
}
