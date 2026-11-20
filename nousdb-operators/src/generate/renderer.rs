//! Prompt context construction for the Generate operator.

use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DFResult};
use minijinja::value::Value as MValue;
use nousdb_core::prompt::renderer::BatchRenderer;
use std::collections::HashMap;

use crate::generate::logical::json_type_name;

/// Renders one generate request.
///
/// A generate request has no input rows. The stream hands `predict` a
/// zero-column "slot" batch of `num_rows` rows, and each request is the slice
/// `[start_row, start_row + batch_size)` of it: the batch only carries the
/// slice's position, which becomes the "rows a–b of N" framing in the prompt's
/// last section. Everything else rendered here is identical across requests.
#[derive(Debug, Clone)]
pub struct GenerateBatchRenderer {
    instruction: String,
    /// `(name, JSON type)` per output field, in schema order.
    fields: Vec<(String, &'static str)>,
    /// Constant-size two-row response example with typed placeholders.
    format_example: String,
    examples: Vec<String>,
    num_rows: usize,
}

impl GenerateBatchRenderer {
    /// Creates a renderer for `num_rows` rows of `output_schema`. Fails if a
    /// field has a type the response decoder cannot build.
    pub fn new(
        instruction: String,
        output_schema: &Schema,
        examples: Vec<String>,
        num_rows: usize,
    ) -> DFResult<Self> {
        let fields = output_schema
            .fields()
            .iter()
            .map(|f| {
                json_type_name(f.data_type())
                    .map(|t| (f.name().clone(), t))
                    .ok_or_else(|| {
                        DataFusionError::Plan(format!(
                            "Generate field '{}' has unsupported type {:?}",
                            f.name(),
                            f.data_type()
                        ))
                    })
            })
            .collect::<DFResult<Vec<_>>>()?;

        let format_example = format_example(&fields);

        Ok(Self {
            instruction,
            fields,
            format_example,
            examples,
            num_rows,
        })
    }
}

/// Builds a response example such as `[{"a": "...", "n": 123}, {"a": "...", "n": 123}]`:
/// two objects whatever the request size, so the prompt prefix does not depend on it.
fn format_example(fields: &[(String, &'static str)]) -> String {
    let object = fields
        .iter()
        .map(|(name, json_type)| {
            let placeholder = match *json_type {
                "integer" => "123",
                "number" => "1.5",
                "boolean" => "true",
                _ => "\"...\"",
            };
            format!("\"{name}\": {placeholder}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{{{object}}}, {{{object}}}]")
}

impl BatchRenderer for GenerateBatchRenderer {
    fn prepare_context(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> DFResult<HashMap<String, MValue>> {
        let end_row = (start_row + batch_size).min(batch.num_rows());
        if end_row <= start_row {
            return Err(DataFusionError::Execution(format!(
                "Generate request slice [{start_row}, {}) is empty or beyond {} rows",
                start_row + batch_size,
                batch.num_rows()
            )));
        }
        let row_count = end_row - start_row;

        let fields: Vec<MValue> = self
            .fields
            .iter()
            .map(|(name, json_type)| {
                MValue::from_iter([
                    ("name", MValue::from(name.clone())),
                    ("type", MValue::from(*json_type)),
                ])
            })
            .collect();
        let examples: Vec<MValue> = self
            .examples
            .iter()
            .map(|e| MValue::from(e.clone()))
            .collect();

        let mut context = HashMap::new();
        context.insert(
            "instruction".to_string(),
            MValue::from(self.instruction.clone()),
        );
        context.insert("fields".to_string(), MValue::from(fields));
        context.insert(
            "format_example".to_string(),
            MValue::from(self.format_example.clone()),
        );
        context.insert("examples".to_string(), MValue::from(examples));
        context.insert("row_start".to_string(), MValue::from(start_row + 1));
        context.insert("row_end".to_string(), MValue::from(end_row));
        context.insert("row_count".to_string(), MValue::from(row_count));
        context.insert("total_rows".to_string(), MValue::from(self.num_rows));

        Ok(context)
    }

    fn context_variables(&self) -> Vec<String> {
        [
            "instruction",
            "fields",
            "format_example",
            "examples",
            "row_start",
            "row_end",
            "row_count",
            "total_rows",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn name(&self) -> String {
        "GenerateRenderer".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field};

    #[test]
    fn format_example_is_typed_and_size_independent() {
        let schema = Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("founded", DataType::Int64, true),
            Field::new("score", DataType::Float64, true),
            Field::new("public", DataType::Boolean, true),
        ]);
        let renderer = GenerateBatchRenderer::new("x".into(), &schema, vec![], 50).unwrap();
        assert_eq!(
            renderer.format_example,
            r#"[{"name": "...", "founded": 123, "score": 1.5, "public": true}, {"name": "...", "founded": 123, "score": 1.5, "public": true}]"#
        );
    }
}
