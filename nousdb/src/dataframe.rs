//! [`NousDBDataFrame`]: a DataFusion `DataFrame` wrapper that chains NousDB's
//! semantic operators (extend, select, join) with the relational methods.

use std::ops::Deref;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::DFSchema;
use datafusion::dataframe::DataFrame;
use datafusion::error::Result;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{ExplainOption, Expr, LogicalPlan, SortExpr};
use datafusion::prelude::*;

use nousdb_operators::{create_extend, create_semantic_join, create_semantic_selection};

/// Converts the forms accepted for the output schema of `extend()` into an
/// Arrow [`Schema`]:
///
/// ```ignore
/// // Single column (most common case)
/// .extend(("sentiment", DataType::Utf8), ...)
///
/// // Multiple columns
/// .extend(vec![("sentiment", DataType::Utf8), ("score", DataType::Float64)], ...)
///
/// // Full schema control
/// .extend(Arc::new(Schema::new(vec![...])), ...)
/// ```
pub trait IntoOutputSchema {
    /// Convert this type into an Arrow Schema.
    fn into_output_schema(self) -> Result<Arc<Schema>>;
}

/// Single column: `("name", DataType::Utf8)`
impl IntoOutputSchema for (&str, DataType) {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        let (name, dtype) = self;
        Ok(Arc::new(Schema::new(vec![Field::new(name, dtype, true)])))
    }
}

/// Single column from owned String: `(String, DataType)`
impl IntoOutputSchema for (String, DataType) {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        let (name, dtype) = self;
        Ok(Arc::new(Schema::new(vec![Field::new(name, dtype, true)])))
    }
}

/// Multiple columns: `vec![("a", DataType::Utf8), ("b", DataType::Int64)]`
impl IntoOutputSchema for Vec<(&str, DataType)> {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        let fields: Vec<Field> = self
            .into_iter()
            .map(|(name, dtype)| Field::new(name, dtype, true))
            .collect();
        Ok(Arc::new(Schema::new(fields)))
    }
}

/// Multiple columns from owned Strings
impl IntoOutputSchema for Vec<(String, DataType)> {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        let fields: Vec<Field> = self
            .into_iter()
            .map(|(name, dtype)| Field::new(name, dtype, true))
            .collect();
        Ok(Arc::new(Schema::new(fields)))
    }
}

/// Direct Schema passthrough
impl IntoOutputSchema for Schema {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        Ok(Arc::new(self))
    }
}

/// Passes an `Arc<Schema>` (a `SchemaRef`) through unchanged.
impl IntoOutputSchema for Arc<Schema> {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        Ok(self)
    }
}

/// Single Field
impl IntoOutputSchema for Field {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        Ok(Arc::new(Schema::new(vec![self])))
    }
}

/// Multiple Fields
impl IntoOutputSchema for Vec<Field> {
    fn into_output_schema(self) -> Result<Arc<Schema>> {
        Ok(Arc::new(Schema::new(self)))
    }
}

/// Converts the forms accepted for the input columns of a semantic operator
/// into a list of column names:
///
/// ```ignore
/// // Single column
/// .extend(..., "review_text")?
///
/// // Slice of strings
/// .extend(..., &["title", "description"])?
///
/// // Array
/// .extend(..., ["title", "description"])?
///
/// // Vec
/// .extend(..., vec!["title".to_string(), "description".to_string()])?
/// ```
pub trait IntoColumnList {
    /// Convert this type into a Vec of column names.
    fn into_column_list(self) -> Vec<String>;
}

/// Single column: `"column_name"`
impl IntoColumnList for &str {
    fn into_column_list(self) -> Vec<String> {
        vec![self.to_string()]
    }
}

/// Single column from owned String
impl IntoColumnList for String {
    fn into_column_list(self) -> Vec<String> {
        vec![self]
    }
}

/// Slice of string refs: `&["col1", "col2"]`
impl IntoColumnList for &[&str] {
    fn into_column_list(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

/// Fixed-size array: `["col1", "col2"]`
impl<const N: usize> IntoColumnList for [&str; N] {
    fn into_column_list(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

/// Vec of owned Strings
impl IntoColumnList for Vec<String> {
    fn into_column_list(self) -> Vec<String> {
        self
    }
}

/// Vec of string refs
impl IntoColumnList for Vec<&str> {
    fn into_column_list(self) -> Vec<String> {
        self.into_iter().map(|s| s.to_string()).collect()
    }
}

/// A DataFrame wrapper that provides fluent access to NousDB semantic operators.
///
/// `NousDBDataFrame` wraps DataFusion's `DataFrame` and adds `extend()`,
/// `semantic_select()` and `semantic_join()`. It dereferences to `DataFrame`, so
/// the other DataFrame methods are available; those that return a new frame
/// have wrappers here that keep the `NousDBDataFrame` type.
///
/// ```ignore
/// use nousdb::prelude::*;
///
/// let results = ctx.nousdb_sql("SELECT * FROM reviews").await?
///     .extend(
///         ("sentiment", DataType::Utf8),
///         "llama3.1:latest",
///         "Classify the sentiment as positive, negative, or neutral",
///         ["reviewText"]
///     )?
///     .filter(col("sentiment").eq(lit("positive")))?
///     .collect().await?;
/// ```
#[derive(Clone)]
pub struct NousDBDataFrame {
    /// The underlying DataFusion DataFrame.
    df: DataFrame,
    /// Session state used to build new DataFrames after transformations
    /// (cheap to clone).
    session_state: SessionState,
}

impl NousDBDataFrame {
    /// Create a new NousDBDataFrame from a DataFrame and SessionState.
    ///
    /// Usually obtained through [`NousDBContext`](crate::NousDBContext) methods.
    pub fn new(df: DataFrame, session_state: SessionState) -> Self {
        Self { df, session_state }
    }

    /// Get the underlying DataFrame.
    pub fn into_inner(self) -> DataFrame {
        self.df
    }

    /// Get the underlying logical plan.
    pub fn logical_plan(&self) -> &LogicalPlan {
        self.df.logical_plan()
    }

    /// Get the schema of this DataFrame.
    pub fn schema(&self) -> &DFSchema {
        self.df.schema()
    }

    /// Extend the DataFrame with new columns computed by a model.
    ///
    /// The model is applied to `input_columns` of each row under `instruction`
    /// and returns the columns of `output_schema`.
    ///
    /// # Arguments
    ///
    /// * `output_schema` - Schema for the new columns. Accepts:
    ///   - `("name", DataType)` for a single column
    ///   - `vec![("name1", DataType1), ...]` for multiple columns
    ///   - `Schema` or `Arc<Schema>` for full control
    /// * `model` - Model identifier (e.g., "llama3.1:latest")
    /// * `instruction` - Natural language instruction for the model
    /// * `input_columns` - Which columns to feed to the model
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = ctx.nousdb_table("reviews").await?
    ///     .extend(
    ///         ("sentiment", DataType::Utf8),
    ///         "llama3.1:latest",
    ///         "Classify this review as positive, negative, or neutral",
    ///         ["reviewText"]
    ///     )?;
    /// ```
    pub fn extend<S, I>(
        self,
        output_schema: S,
        model: impl Into<String>,
        instruction: impl Into<String>,
        input_columns: I,
    ) -> Result<Self>
    where
        S: IntoOutputSchema,
        I: IntoColumnList,
    {
        let derived_schema = output_schema.into_output_schema()?;
        let input_cols = input_columns.into_column_list();

        let plan = create_extend(
            self.df.logical_plan().clone(),
            instruction,
            model,
            input_cols,
            derived_schema,
        )?;

        Ok(Self::new(
            DataFrame::new(self.session_state.clone(), plan),
            self.session_state,
        ))
    }

    /// Filter rows using a model-evaluated predicate.
    ///
    /// The model evaluates each row and only rows judged true are kept. The
    /// schema is unchanged.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier
    /// * `instruction` - Natural language description of which rows to keep
    /// * `input_columns` - Which columns the model should evaluate
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = ctx.nousdb_table("reviews").await?
    ///     .semantic_select(
    ///         "llama3.1:latest",
    ///         "Keep only reviews that discuss the plot or storyline",
    ///         ["reviewText"]
    ///     )?;
    /// ```
    pub fn semantic_select<I>(
        self,
        model: impl Into<String>,
        instruction: impl Into<String>,
        input_columns: I,
    ) -> Result<Self>
    where
        I: IntoColumnList,
    {
        let input_cols = input_columns.into_column_list();

        let plan =
            create_semantic_selection(self.df.logical_plan(), model, input_cols, instruction)?;

        Ok(Self::new(
            DataFrame::new(self.session_state.clone(), plan),
            self.session_state,
        ))
    }

    /// Join with another DataFrame using a model-evaluated condition.
    ///
    /// The model judges each candidate (left, right) pair from the given
    /// columns; only pairs judged true are kept.
    ///
    /// # Arguments
    ///
    /// * `right` - The right DataFrame to join with
    /// * `left_columns` - Columns from self to use in the join condition
    /// * `right_columns` - Columns from right to use in the join condition
    /// * `model` - Model identifier
    /// * `instruction` - Natural language description of the join condition
    ///
    /// # Example
    ///
    /// ```ignore
    /// let reviews = ctx.nousdb_table("reviews").await?;
    /// let directors = ctx.nousdb_table("directors").await?;
    ///
    /// let joined = reviews.semantic_join(
    ///     directors,
    ///     ["reviewText"],
    ///     ["name", "style"],
    ///     "llama3.1:latest",
    ///     "The review mentions or references this director's work"
    /// )?;
    /// ```
    pub fn semantic_join<L, R>(
        self,
        right: NousDBDataFrame,
        left_columns: L,
        right_columns: R,
        model: impl Into<String>,
        instruction: impl Into<String>,
    ) -> Result<Self>
    where
        L: IntoColumnList,
        R: IntoColumnList,
    {
        let left_cols = left_columns.into_column_list();
        let right_cols = right_columns.into_column_list();

        let plan = create_semantic_join(
            self.df.logical_plan(),
            right.df.logical_plan(),
            left_cols,
            right_cols,
            instruction,
            model,
        )?;

        Ok(Self::new(
            DataFrame::new(self.session_state.clone(), plan),
            self.session_state,
        ))
    }

    /// Apply a filter expression.
    ///
    /// Wraps `DataFrame::filter`, keeping the `NousDBDataFrame` type.
    pub fn filter(self, predicate: Expr) -> Result<Self> {
        let new_df = self.df.filter(predicate)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Select specific expressions.
    ///
    /// Wraps `DataFrame::select`, keeping the `NousDBDataFrame` type.
    pub fn select(self, expr_list: Vec<Expr>) -> Result<Self> {
        let new_df = self.df.select(expr_list)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Select specific columns by name.
    ///
    /// Wraps `DataFrame::select_columns`, keeping the `NousDBDataFrame` type.
    pub fn select_columns(self, columns: &[&str]) -> Result<Self> {
        let new_df = self.df.select_columns(columns)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Limit the number of rows.
    ///
    /// Wraps `DataFrame::limit`, keeping the `NousDBDataFrame` type.
    pub fn limit(self, skip: usize, fetch: Option<usize>) -> Result<Self> {
        let new_df = self.df.limit(skip, fetch)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Sort by specified expressions.
    ///
    /// Wraps `DataFrame::sort`, keeping the `NousDBDataFrame` type.
    pub fn sort(self, expr: Vec<SortExpr>) -> Result<Self> {
        let new_df = self.df.sort(expr)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Perform a traditional join with another DataFrame.
    ///
    /// Wraps `DataFrame::join`, keeping the `NousDBDataFrame` type.
    pub fn join(
        self,
        right: NousDBDataFrame,
        join_type: JoinType,
        left_cols: &[&str],
        right_cols: &[&str],
        filter: Option<Expr>,
    ) -> Result<Self> {
        let new_df = self
            .df
            .join(right.df, join_type, left_cols, right_cols, filter)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Aggregate with grouping.
    ///
    /// Wraps `DataFrame::aggregate`, keeping the `NousDBDataFrame` type.
    pub fn aggregate(self, group_expr: Vec<Expr>, aggr_expr: Vec<Expr>) -> Result<Self> {
        let new_df = self.df.aggregate(group_expr, aggr_expr)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Set an alias for this DataFrame.
    pub fn alias(self, alias: &str) -> Result<Self> {
        let new_df = self.df.alias(alias)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Add a column expression.
    pub fn with_column(self, name: &str, expr: Expr) -> Result<Self> {
        let new_df = self.df.with_column(name, expr)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Drop columns by name.
    pub fn drop_columns(self, columns: &[&str]) -> Result<Self> {
        let new_df = self.df.drop_columns(columns)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Get distinct rows.
    pub fn distinct(self) -> Result<Self> {
        let new_df = self.df.distinct()?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Union with another DataFrame.
    pub fn union(self, other: NousDBDataFrame) -> Result<Self> {
        let new_df = self.df.union(other.df)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Intersect with another DataFrame.
    pub fn intersect(self, other: NousDBDataFrame) -> Result<Self> {
        let new_df = self.df.intersect(other.df)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Except (set difference) with another DataFrame.
    pub fn except(self, other: NousDBDataFrame) -> Result<Self> {
        let new_df = self.df.except(other.df)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Collect all results into RecordBatches.
    pub async fn collect(self) -> Result<Vec<RecordBatch>> {
        self.df.collect().await
    }

    /// Display results to stdout.
    pub async fn show(self) -> Result<()> {
        self.df.show().await
    }

    /// Display a limited number of rows.
    pub async fn show_limit(self, num: usize) -> Result<()> {
        self.df.show_limit(num).await
    }

    /// Get the count of rows.
    pub async fn count(self) -> Result<usize> {
        self.df.count().await
    }

    /// Explain the query plan.
    pub async fn explain(self, verbose: bool, analyze: bool) -> Result<Self> {
        let new_df = self.df.explain(verbose, analyze)?;
        Ok(Self::new(new_df, self.session_state))
    }

    /// Explain the query plan with options.
    pub fn explain_with_options(self, options: ExplainOption) -> Result<Self> {
        let new_df = self.df.explain_with_options(options)?;
        Ok(Self::new(new_df, self.session_state))
    }
}

impl Deref for NousDBDataFrame {
    type Target = DataFrame;

    fn deref(&self) -> &Self::Target {
        &self.df
    }
}

/// Extension trait for converting DataFusion DataFrames to NousDBDataFrame.
///
/// This trait provides a convenient way to upgrade a regular DataFrame
/// to a NousDBDataFrame when you need access to semantic operators.
pub trait NousDBDataFrameExt {
    /// Convert this DataFrame into a NousDBDataFrame.
    fn into_nousdb(self, session_state: SessionState) -> NousDBDataFrame;
}

impl NousDBDataFrameExt for DataFrame {
    fn into_nousdb(self, session_state: SessionState) -> NousDBDataFrame {
        NousDBDataFrame::new(self, session_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_output_schema_single_column() {
        let schema = ("name", DataType::Utf8).into_output_schema().unwrap();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "name");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_into_output_schema_multiple_columns() {
        let schema = vec![("name", DataType::Utf8), ("score", DataType::Float64)]
            .into_output_schema()
            .unwrap();

        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "name");
        assert_eq!(schema.field(1).name(), "score");
    }

    #[test]
    fn test_into_column_list_single() {
        let cols = "column".into_column_list();
        assert_eq!(cols, vec!["column".to_string()]);
    }

    #[test]
    fn test_into_column_list_array() {
        let cols = ["col1", "col2"].into_column_list();
        assert_eq!(cols, vec!["col1".to_string(), "col2".to_string()]);
    }

    #[test]
    fn test_into_column_list_slice() {
        let arr = ["col1", "col2"];
        let cols = arr.as_slice().into_column_list();
        assert_eq!(cols, vec!["col1".to_string(), "col2".to_string()]);
    }
}
