//! Prompt context construction for the Extend operator.

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DFResult};
use minijinja::value::Value as MValue;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::prompt::utils::*;
use std::collections::HashMap;

/// Builds the prompt context for an Extend request from a window of input rows.
#[derive(Debug, Clone)]
pub struct ExtendBatchRenderer {
    instruction: String,

    input_columns: Vec<String>,

    output_columns: Vec<String>,
}

impl ExtendBatchRenderer {
    /// Creates a renderer for `instruction` over `input_columns`, asking the
    /// model to produce `output_columns`.
    pub fn new(
        instruction: String,
        input_columns: Vec<String>,
        output_columns: Vec<String>,
    ) -> Self {
        Self {
            instruction,
            input_columns,
            output_columns,
        }
    }

    /// Names of the input columns rendered into each row.
    pub fn input_columns(&self) -> &[String] {
        &self.input_columns
    }

    fn extract_rows(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        end_row: usize,
    ) -> DFResult<Vec<Vec<(String, String)>>> {
        let mut rows = Vec::with_capacity(end_row - start_row);

        for row_idx in start_row..end_row {
            let row_data = extract_row_values(batch, row_idx, &self.input_columns)?;
            rows.push(row_data);
        }

        Ok(rows)
    }
}

impl BatchRenderer for ExtendBatchRenderer {
    fn prepare_context(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> DFResult<HashMap<String, MValue>> {
        let end_row = (start_row + batch_size).min(batch.num_rows());
        let actual_batch_size = end_row - start_row;

        if actual_batch_size == 0 {
            return Err(DataFusionError::Execution(
                "Batch size is zero or start_row is out of bounds".to_string(),
            ));
        }

        if start_row >= batch.num_rows() {
            return Err(DataFusionError::Execution(format!(
                "start_row {} is beyond batch size {}",
                start_row,
                batch.num_rows()
            )));
        }

        // One ordered (column, value) list per row, in input_columns order.
        let rows = self.extract_rows(batch, start_row, end_row)?;

        let mj_rows = rows_to_minijinja(rows);

        let mut context = HashMap::new();

        context.insert(
            "instruction".to_string(),
            MValue::from(self.instruction.clone()),
        );

        context.insert("row_count".to_string(), MValue::from(actual_batch_size));

        let input_cols: Vec<MValue> = self
            .input_columns
            .iter()
            .map(|col| MValue::from(col.clone()))
            .collect();
        context.insert("input_columns".to_string(), MValue::from(input_cols));

        let output_cols: Vec<MValue> = self
            .output_columns
            .iter()
            .map(|col| MValue::from(col.clone()))
            .collect();
        context.insert("output_columns".to_string(), MValue::from(output_cols));

        context.insert("batch_data".to_string(), MValue::from(mj_rows));

        Ok(context)
    }

    fn context_variables(&self) -> Vec<String> {
        vec![
            "instruction".to_string(),
            "row_count".to_string(),
            "input_columns".to_string(),
            "output_columns".to_string(),
            "batch_data".to_string(),
        ]
    }

    fn name(&self) -> String {
        "ExtendRenderer".to_string()
    }

    fn estimate_context_size(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> Option<usize> {
        let end_row = (start_row + batch_size).min(batch.num_rows());
        let actual_batch_size = end_row - start_row;

        // Context size in bytes: instruction, row count, column names, and a
        // rough 50 bytes per column per row for the data.
        let instruction_size = self.instruction.len();

        let row_count_size = 10;

        let input_cols_size: usize = self.input_columns.iter().map(|c| c.len()).sum();

        let output_cols_size: usize = self.output_columns.iter().map(|c| c.len()).sum();

        let data_size = actual_batch_size * self.input_columns.len() * 50;

        Some(instruction_size + row_count_size + input_cols_size + output_cols_size + data_size)
    }
}
