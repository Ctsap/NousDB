//! Logical plan node and constructor for the Generate operator.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::json::WriterBuilder;
use datafusion::arrow::json::writer::LineDelimited;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};

/// Logical plan node for the semantic Generate operator: a source with no input
/// whose rows the model composes from an instruction, a declared schema and a
/// row count, optionally steered by a few example rows.
///
/// Per-row generation from context is Extend instead. Execution-only settings
/// (rows per request, request window) are resolved by the physical exec and
/// never stored here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenerateNode {
    /// The generated relation's schema. Every field is nullable.
    pub schema: DFSchemaRef,

    /// Model identifier.
    pub model: String,

    /// What to generate.
    pub instruction: String,

    /// How many rows to generate. An upper bound on the output
    /// cardinality: requests the model fails are dropped, not NULL-filled.
    pub num_rows: usize,

    /// Few-shot example rows, one JSON object per row in schema order.
    /// Pre-rendered so the node stays `Hash` (a `RecordBatch` is not).
    pub examples: Vec<String>,
}

impl GenerateNode {
    /// The generated relation's Arrow schema.
    pub fn output_schema(&self) -> Arc<Schema> {
        self.schema.inner().clone()
    }
}

// DFSchema has no ordering, so compare its rendering after the plain fields.
impl PartialOrd for GenerateNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            (
                &self.model,
                &self.instruction,
                self.num_rows,
                &self.examples,
            )
                .cmp(&(
                    &other.model,
                    &other.instruction,
                    other.num_rows,
                    &other.examples,
                ))
                .then_with(|| self.schema.to_string().cmp(&other.schema.to_string())),
        )
    }
}

impl UserDefinedLogicalNodeCore for GenerateNode {
    fn name(&self) -> &str {
        "GenerateNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NousDB_Generate: instruction='{}', num_rows={}, output_cols={:?}, examples={}",
            self.instruction,
            self.num_rows,
            self.schema
                .fields()
                .iter()
                .map(|field| field.name())
                .collect::<Vec<_>>(),
            self.examples.len()
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self, DataFusionError> {
        if !inputs.is_empty() {
            return Err(DataFusionError::Internal(
                "NousDB_Generate is a leaf and takes no inputs".to_string(),
            ));
        }
        Ok(self.clone())
    }
}

/// The JSON type a generated field is declared as in the prompt and decoded
/// from, or `None` for Arrow types `Prompt::parse_response` cannot build.
pub(crate) fn json_type_name(data_type: &DataType) -> Option<&'static str> {
    match data_type {
        DataType::Utf8 => Some("string"),
        DataType::Int64 => Some("integer"),
        DataType::Float64 => Some("number"),
        DataType::Boolean => Some("boolean"),
        _ => None,
    }
}

/// Create a Generate logical plan: a table of `num_rows` rows of
/// `output_schema`, composed by `model` following `instruction`.
///
/// `examples` are optional few-shot rows. Their columns must match the schema
/// by name and type (any order); they are shown to the model as illustrations
/// and are not part of the output.
///
/// Fails when `num_rows` is zero, the schema is empty, a field's type is not
/// one of Utf8 / Int64 / Float64 / Boolean (the types the response decoder
/// builds), or an example batch does not fit the schema. Fields are made
/// nullable: a model may answer `null`, and the decoder maps it to NULL.
///
/// # Example
///
/// ```ignore
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("name", DataType::Utf8, true),
///     Field::new("founded", DataType::Int64, true),
/// ]));
/// let plan = create_generate(schema, "llama3.1:latest", "fictional startups", 12, &[])?;
/// ```
pub fn create_generate(
    output_schema: Arc<Schema>,
    model: impl Into<String>,
    instruction: impl Into<String>,
    num_rows: usize,
    examples: &[RecordBatch],
) -> DFResult<LogicalPlan> {
    if num_rows == 0 {
        return Err(DataFusionError::Plan(
            "Generate needs num_rows >= 1".to_string(),
        ));
    }
    if output_schema.fields().is_empty() {
        return Err(DataFusionError::Plan(
            "Generate needs at least one output field".to_string(),
        ));
    }
    for field in output_schema.fields() {
        if json_type_name(field.data_type()).is_none() {
            return Err(DataFusionError::Plan(format!(
                "Generate field '{}' has unsupported type {:?}; use Utf8, Int64, Float64 or Boolean",
                field.name(),
                field.data_type()
            )));
        }
    }

    let fields: Vec<Field> = output_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        output_schema.metadata().clone(),
    ));

    let examples = render_example_rows(&schema, examples)?;

    let df_schema = DFSchema::try_from(schema.as_ref().clone())
        .map_err(|e| DataFusionError::Plan(format!("Failed to create DFSchema: {}", e)))?;

    let node = GenerateNode {
        schema: Arc::new(df_schema),
        model: model.into(),
        instruction: instruction.into(),
        num_rows,
        examples,
    };

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}

/// Render example batches as one JSON object per row, columns in `schema`
/// order, numbers and booleans unquoted (the shape the model answers in).
fn render_example_rows(schema: &Arc<Schema>, examples: &[RecordBatch]) -> DFResult<Vec<String>> {
    let mut rows = Vec::new();

    for batch in examples {
        if batch.num_rows() == 0 {
            continue;
        }
        let batch_schema = batch.schema();

        let mut indices = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let idx = batch_schema.index_of(field.name()).map_err(|_| {
                DataFusionError::Plan(format!(
                    "Generate example rows are missing column '{}'",
                    field.name()
                ))
            })?;
            let found = batch_schema.field(idx).data_type();
            if found != field.data_type() {
                return Err(DataFusionError::Plan(format!(
                    "Generate example column '{}' is {:?}, expected {:?}",
                    field.name(),
                    found,
                    field.data_type()
                )));
            }
            indices.push(idx);
        }
        if batch.num_columns() != indices.len() {
            let extra: Vec<&String> = batch_schema
                .fields()
                .iter()
                .map(|f| f.name())
                .filter(|name| schema.index_of(name).is_err())
                .collect();
            return Err(DataFusionError::Plan(format!(
                "Generate example rows have columns not in the output schema: {:?}",
                extra
            )));
        }

        let projected = batch.project(&indices)?;
        let aligned = RecordBatch::try_new(schema.clone(), projected.columns().to_vec())?;

        let mut writer = WriterBuilder::new()
            .with_explicit_nulls(true)
            .build::<_, LineDelimited>(Vec::new());
        writer.write(&aligned)?;
        writer.finish()?;
        let text = String::from_utf8(writer.into_inner()).map_err(|e| {
            DataFusionError::Execution(format!("Example rows are not UTF-8: {}", e))
        })?;

        rows.extend(
            text.lines()
                .filter(|line| !line.is_empty())
                .map(str::to_string),
        );
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int64Array, StringArray};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("founded", DataType::Int64, false),
        ]))
    }

    fn node(plan: &LogicalPlan) -> &GenerateNode {
        match plan {
            LogicalPlan::Extension(ext) => ext
                .node
                .as_any()
                .downcast_ref::<GenerateNode>()
                .expect("GenerateNode"),
            other => panic!("expected an extension, got {other:?}"),
        }
    }

    #[test]
    fn builds_a_leaf_with_nullable_fields() {
        let plan = create_generate(schema(), "m", "startups", 5, &[]).unwrap();
        let node = node(&plan);
        assert!(node.inputs().is_empty());
        assert_eq!(node.num_rows, 5);
        assert!(node.schema.fields().iter().all(|f| f.is_nullable()));
        assert!(node.examples.is_empty());
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(create_generate(schema(), "m", "x", 0, &[]).is_err());
        assert!(create_generate(Arc::new(Schema::empty()), "m", "x", 3, &[]).is_err());
        let int32 = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)]));
        let err = create_generate(int32, "m", "x", 3, &[]).unwrap_err();
        assert!(err.to_string().contains("Int64"), "{err}");
    }

    #[test]
    fn renders_examples_as_typed_json_in_schema_order() {
        // Columns deliberately out of schema order.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("founded", DataType::Int64, true),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![Some(2019), None])),
                Arc::new(StringArray::from(vec!["Acme", "Globex"])),
            ],
        )
        .unwrap();

        let plan = create_generate(schema(), "m", "startups", 5, &[batch]).unwrap();
        assert_eq!(
            node(&plan).examples,
            vec![
                r#"{"name":"Acme","founded":2019}"#.to_string(),
                r#"{"name":"Globex","founded":null}"#.to_string(),
            ]
        );
    }

    #[test]
    fn rejects_examples_that_do_not_fit_the_schema() {
        let missing = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["Acme"]))],
        )
        .unwrap();
        assert!(create_generate(schema(), "m", "x", 3, &[missing]).is_err());

        let extra = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("founded", DataType::Int64, true),
                Field::new("ceo", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["Acme"])),
                Arc::new(Int64Array::from(vec![2019])),
                Arc::new(StringArray::from(vec!["Wile"])),
            ],
        )
        .unwrap();
        let err = create_generate(schema(), "m", "x", 3, &[extra]).unwrap_err();
        assert!(err.to_string().contains("ceo"), "{err}");
    }

    #[test]
    fn clone_path_keeps_every_field_and_refuses_inputs() {
        let plan = create_generate(schema(), "m", "startups", 7, &[]).unwrap();
        let node = node(&plan);
        let rebuilt = node.with_exprs_and_inputs(vec![], vec![]).unwrap();
        assert_eq!(&rebuilt, node);
        assert!(
            node.with_exprs_and_inputs(vec![], vec![plan.clone()])
                .is_err()
        );
    }
}
