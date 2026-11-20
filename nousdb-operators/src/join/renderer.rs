//! Prompt renderer for the semantic join, using a reference-table format.
//!
//! Left rows are listed once as a reference table (L1, L2, ...), right rows
//! once as another (R1, R2, ...), and pairs reference rows by compact ids
//! ("Pair 1: L1 ↔ R2"). Token cost is
//! O(L × left_tokens + R × right_tokens + L×R × ~15) rather than
//! O(L×R × row_tokens), since each row's content appears once regardless of
//! how many pairs reference it. Each pair is still evaluated independently.

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DFResult};
use minijinja::value::Value as MValue;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::prompt::utils::*;
use std::collections::HashMap;

/// Renderer for the semantic join.
///
/// Left and right rows are deduplicated into reference tables and pairs are
/// listed as compact id references, which removes the repetition inherent in
/// cross-product blocks.
///
/// The renderer reads the pair batch positionally: the first
/// `left_col_names.len()` columns are the left side, the next
/// `right_col_names.len()` are the right side, and anything after that is
/// ignored. That lets the calibration prologue append a row-identity column to
/// a probe input without touching the prompt.
#[derive(Debug, Clone)]
pub struct SemanticJoinRenderer {
    /// The join condition as natural language.
    join_condition: String,

    /// Actual column names from the left relation, in pair-batch column order.
    /// Its length is the left side's column count.
    left_col_names: Vec<String>,

    /// Actual column names from the right relation, in pair-batch column order.
    /// Its length is the right side's column count.
    right_col_names: Vec<String>,
}

impl SemanticJoinRenderer {
    /// Creates a renderer for `join_condition` with the column names of each side.
    pub fn new(
        join_condition: String,
        left_col_names: Vec<String>,
        right_col_names: Vec<String>,
    ) -> Self {
        Self {
            join_condition,
            left_col_names,
            right_col_names,
        }
    }
}

impl BatchRenderer for SemanticJoinRenderer {
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
                "Batch size is zero or start_row out of bounds".to_string(),
            ));
        }

        let num_left_cols = self.left_col_names.len();
        let num_right_cols = self.right_col_names.len();

        // Deduplication structures for left and right rows.
        // Maps row content (as Vec<String>) to 1-indexed ID.
        let mut left_unique: Vec<Vec<(String, String)>> = Vec::new();
        let mut left_content_to_id: HashMap<Vec<String>, usize> = HashMap::new();

        let mut right_unique: Vec<Vec<(String, String)>> = Vec::new();
        let mut right_content_to_id: HashMap<Vec<String>, usize> = HashMap::new();

        // Pair references (1-indexed IDs)
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(actual_batch_size);

        for row_idx in start_row..end_row {
            // Extract and deduplicate left row
            let mut left_values = Vec::with_capacity(num_left_cols);
            let mut left_key = Vec::with_capacity(num_left_cols);
            for i in 0..num_left_cols {
                let array = batch.column(i);
                let value = extract_value_from_array(array, row_idx)?;
                let col_name = self
                    .left_col_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("col_{}", i));
                left_key.push(value.clone());
                left_values.push((col_name, value));
            }

            let left_id = if let Some(&id) = left_content_to_id.get(&left_key) {
                id
            } else {
                let id = left_unique.len() + 1; // 1-indexed
                left_unique.push(left_values);
                left_content_to_id.insert(left_key, id);
                id
            };

            // Extract and deduplicate right row
            let mut right_values = Vec::with_capacity(num_right_cols);
            let mut right_key = Vec::with_capacity(num_right_cols);
            for i in 0..num_right_cols {
                let col_idx = num_left_cols + i;
                let array = batch.column(col_idx);
                let value = extract_value_from_array(array, row_idx)?;
                let col_name = self
                    .right_col_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("col_{}", i));
                right_key.push(value.clone());
                right_values.push((col_name, value));
            }

            let right_id = if let Some(&id) = right_content_to_id.get(&right_key) {
                id
            } else {
                let id = right_unique.len() + 1; // 1-indexed
                right_unique.push(right_values);
                right_content_to_id.insert(right_key, id);
                id
            };

            pairs.push((left_id, right_id));
        }

        // Convert left/right rows to MiniJinja values (column order preserved)
        let mj_left_rows: Vec<MValue> = left_unique
            .into_iter()
            .map(|row| MValue::from_iter(row.into_iter().map(|(k, v)| (k, MValue::from(v)))))
            .collect();

        let mj_right_rows: Vec<MValue> = right_unique
            .into_iter()
            .map(|row| MValue::from_iter(row.into_iter().map(|(k, v)| (k, MValue::from(v)))))
            .collect();

        // Convert pairs to MiniJinja values
        let mj_pairs: Vec<MValue> = pairs
            .into_iter()
            .map(|(left_id, right_id)| {
                let mut pair = HashMap::new();
                pair.insert("left_id".to_string(), MValue::from(left_id));
                pair.insert("right_id".to_string(), MValue::from(right_id));
                MValue::from(pair)
            })
            .collect();

        // Build final context
        let mut context = HashMap::new();
        context.insert(
            "join_condition".to_string(),
            MValue::from(self.join_condition.clone()),
        );
        context.insert("pair_count".to_string(), MValue::from(actual_batch_size));
        context.insert("left_rows".to_string(), MValue::from(mj_left_rows));
        context.insert("right_rows".to_string(), MValue::from(mj_right_rows));
        context.insert("pairs".to_string(), MValue::from(mj_pairs));

        Ok(context)
    }

    fn context_variables(&self) -> Vec<String> {
        vec![
            "join_condition".to_string(),
            "pair_count".to_string(),
            "left_rows".to_string(),
            "right_rows".to_string(),
            "pairs".to_string(),
        ]
    }

    fn name(&self) -> String {
        "SemanticJoinRenderer".to_string()
    }

    fn estimate_context_size(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> Option<usize> {
        let end_row = (start_row + batch_size).min(batch.num_rows());
        let actual_batch_size = end_row - start_row;

        let condition_size = self.join_condition.len();
        let num_left_cols = self.left_col_names.len();
        let num_right_cols = self.right_col_names.len();

        // In a cross-product block of L×R pairs, unique left ≈ sqrt(pairs)
        // and unique right ≈ sqrt(pairs). Each unique row listed once (~100 bytes/col).
        // Each pair reference is ~20 bytes ("- Pair N: L1 ↔ R2\n").
        let approx_unique = (actual_batch_size as f64).sqrt().ceil() as usize;
        let ref_table_size =
            approx_unique * num_left_cols * 100 + approx_unique * num_right_cols * 100;
        let pair_ref_size = actual_batch_size * 20;

        Some(condition_size + ref_table_size + pair_ref_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{ArrayRef, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a synthetic pair batch simulating a 2×3 cross-product block.
    /// Left rows: ("Alice", "Bob"), Right rows: ("X", "Y", "Z")
    /// Pairs: (Alice,X), (Alice,Y), (Alice,Z), (Bob,X), (Bob,Y), (Bob,Z)
    fn make_pair_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("left_0", DataType::Utf8, false),
            Field::new("right_0", DataType::Utf8, false),
        ]));
        let left_col = Arc::new(StringArray::from(vec![
            "Alice", "Alice", "Alice", "Bob", "Bob", "Bob",
        ])) as ArrayRef;
        let right_col = Arc::new(StringArray::from(vec!["X", "Y", "Z", "X", "Y", "Z"])) as ArrayRef;
        RecordBatch::try_new(schema, vec![left_col, right_col]).unwrap()
    }

    fn make_renderer() -> SemanticJoinRenderer {
        SemanticJoinRenderer::new(
            "test condition".to_string(),
            vec!["name".to_string()],
            vec!["item".to_string()],
        )
    }

    /// Columns past the declared left+right split are never read, so a
    /// calibration probe input can carry a row-identity column without
    /// disturbing the prompt.
    #[test]
    fn trailing_columns_are_not_read() {
        use datafusion::arrow::array::UInt32Array;

        let batch = make_pair_batch();
        let mut fields = batch.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(
            "__nousdb_probe_id",
            DataType::UInt32,
            false,
        )));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(UInt32Array::from_iter_values(0..6)) as ArrayRef);
        let augmented = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

        let renderer = make_renderer();
        let plain = renderer.prepare_context(&batch, 0, 6).unwrap();
        let with_id = renderer.prepare_context(&augmented, 0, 6).unwrap();

        // The reference tables are ordered sequences, so compare them directly;
        // `pairs` holds unordered maps whose debug form is not stable, so
        // compare the two ids the template actually reads.
        for key in ["join_condition", "pair_count", "left_rows", "right_rows"] {
            assert_eq!(
                plain.get(key).unwrap().to_string(),
                with_id.get(key).unwrap().to_string(),
                "context variable '{key}' changed when a trailing column was added"
            );
        }
        let ids = |ctx: &HashMap<String, MValue>| -> Vec<(String, String)> {
            ctx.get("pairs")
                .unwrap()
                .try_iter()
                .unwrap()
                .map(|p| {
                    (
                        p.get_attr("left_id").unwrap().to_string(),
                        p.get_attr("right_id").unwrap().to_string(),
                    )
                })
                .collect()
        };
        assert_eq!(ids(&plain), ids(&with_id));
    }

    #[test]
    fn test_dedup_produces_correct_unique_counts() {
        let batch = make_pair_batch();
        let renderer = make_renderer();
        let ctx = renderer.prepare_context(&batch, 0, 6).unwrap();

        // Should have 2 unique left rows and 3 unique right rows
        let left_rows = ctx.get("left_rows").unwrap();
        let right_rows = ctx.get("right_rows").unwrap();
        // MValue doesn't expose len() directly, but we can check via pair_count
        assert_eq!(ctx.get("pair_count").unwrap().to_string(), "6");
        // Verify left_rows is a sequence of length 2
        let left_seq: Vec<MValue> = left_rows.try_iter().unwrap().collect();
        assert_eq!(left_seq.len(), 2);
        // Verify right_rows is a sequence of length 3
        let right_seq: Vec<MValue> = right_rows.try_iter().unwrap().collect();
        assert_eq!(right_seq.len(), 3);
    }

    #[test]
    fn test_pair_ids_are_correct() {
        let batch = make_pair_batch();
        let renderer = make_renderer();
        let ctx = renderer.prepare_context(&batch, 0, 6).unwrap();

        let pairs: Vec<MValue> = ctx.get("pairs").unwrap().try_iter().unwrap().collect();
        assert_eq!(pairs.len(), 6);

        // First 3 pairs should reference L1 (Alice)
        for pair in &pairs[..3] {
            let left_id = pair.get_attr("left_id").unwrap();
            assert_eq!(left_id.to_string(), "1");
        }
        // Last 3 pairs should reference L2 (Bob)
        for pair in &pairs[3..6] {
            let left_id = pair.get_attr("left_id").unwrap();
            assert_eq!(left_id.to_string(), "2");
        }

        // Right IDs cycle: R1, R2, R3, R1, R2, R3
        for (i, pair) in pairs.iter().enumerate() {
            let right_id = pair.get_attr("right_id").unwrap();
            let expected = (i % 3) + 1;
            assert_eq!(right_id.to_string(), expected.to_string());
        }
    }

    #[test]
    fn test_single_pair_no_dedup_needed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("left_0", DataType::Utf8, false),
            Field::new("right_0", DataType::Utf8, false),
        ]));
        let left_col = Arc::new(StringArray::from(vec!["Alice"])) as ArrayRef;
        let right_col = Arc::new(StringArray::from(vec!["X"])) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![left_col, right_col]).unwrap();

        let renderer = make_renderer();
        let ctx = renderer.prepare_context(&batch, 0, 1).unwrap();

        let left_rows: Vec<MValue> = ctx.get("left_rows").unwrap().try_iter().unwrap().collect();
        let right_rows: Vec<MValue> = ctx.get("right_rows").unwrap().try_iter().unwrap().collect();
        let pairs: Vec<MValue> = ctx.get("pairs").unwrap().try_iter().unwrap().collect();

        assert_eq!(left_rows.len(), 1);
        assert_eq!(right_rows.len(), 1);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_context_variables() {
        let renderer = make_renderer();
        let vars = renderer.context_variables();
        assert!(vars.contains(&"left_rows".to_string()));
        assert!(vars.contains(&"right_rows".to_string()));
        assert!(vars.contains(&"pairs".to_string()));
        assert!(vars.contains(&"join_condition".to_string()));
        assert!(vars.contains(&"pair_count".to_string()));
    }

    #[test]
    fn test_estimate_context_size_smaller_with_dedup() {
        let batch = make_pair_batch();
        let renderer = make_renderer();

        let estimate = renderer.estimate_context_size(&batch, 0, 6).unwrap();
        // With 6 pairs, the estimate should account for ~sqrt(6)≈3 unique rows
        // per side plus 6 pair references, which is less than 6 * 2 columns * 100
        let naive_estimate = "test condition".len() + 6 * 2 * 100;
        assert!(
            estimate < naive_estimate,
            "Reference-table estimate ({}) should be less than inline estimate ({})",
            estimate,
            naive_estimate
        );
    }
}
