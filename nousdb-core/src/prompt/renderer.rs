//! The [`BatchRenderer`] trait implemented by operators.

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DFResult;
use minijinja::value::Value as MValue;
use std::collections::HashMap;

/// Builds the template context for one batch of rows.
pub trait BatchRenderer: Send + Sync {
    /// Prepares the context variables for this operator's prompt template.
    ///
    /// Transforms a slice of an Arrow `RecordBatch` into a map of variable
    /// names to MiniJinja values. The structure depends on the operator's
    /// template.
    ///
    /// * `batch`: the input data
    /// * `start_row`: first row of the slice (inclusive)
    /// * `batch_size`: number of rows to process
    ///
    /// # Errors
    ///
    /// Returns a `DataFusionError` if a column is missing, a data type
    /// conversion fails, or the slice exceeds the batch
    /// (`start_row + batch_size > num_rows`).
    ///
    /// # Example
    ///
    /// For semantic selection the result is shaped like:
    ///
    /// ```text
    /// {
    ///   "expression": "sentiment is positive",
    ///   "row_count": 32,
    ///   "batch_data": [{"review_text": "Great!", ...}, ...]
    /// }
    /// ```
    fn prepare_context(
        &self,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> DFResult<HashMap<String, MValue>>;

    /// Names of the variables [`prepare_context`](Self::prepare_context)
    /// provides.
    fn context_variables(&self) -> Vec<String>;

    /// Renderer name, by default the implementing type's name.
    fn name(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }

    /// Estimated size of the context for the given slice, if the renderer can
    /// compute it cheaply. The default returns `None`.
    fn estimate_context_size(
        &self,
        _batch: &RecordBatch,
        _start_row: usize,
        _batch_size: usize,
    ) -> Option<usize> {
        None
    }
}
