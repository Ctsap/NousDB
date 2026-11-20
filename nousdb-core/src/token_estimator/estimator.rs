//! Token counting with tiktoken BPE tokenizers.

use std::collections::HashMap;
use std::io::ErrorKind::Other;
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use tiktoken_rs::CoreBPE;

use crate::arrow_util::StringColumn;

/// Process-wide cache of built BPE tables, keyed by encoding name.
///
/// Building a tokenizer (e.g. `cl100k_base`) decodes and hashes a vocabulary of
/// about 100k entries, which takes hundreds of milliseconds and tens of MB, so
/// every [`TokenEstimator::new`] shares one build per encoding.
fn bpe_cache() -> &'static Mutex<HashMap<String, Arc<CoreBPE>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<CoreBPE>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Configuration for the token estimator.
#[derive(Debug, Clone)]
pub struct TokenEstimatorConfig {
    /// The tiktoken encoding to use: `cl100k_base` (default), `p50k_base`,
    /// `r50k_base` or `o200k_base`.
    pub encoding: String,
}

impl Default for TokenEstimatorConfig {
    fn default() -> Self {
        Self {
            encoding: "cl100k_base".to_string(),
        }
    }
}

/// Estimates tokens for `RecordBatch` rows using a tiktoken BPE tokenizer.
pub struct TokenEstimator {
    bpe: Arc<CoreBPE>,
}

impl std::fmt::Debug for TokenEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenEstimator")
            .field("bpe", &"<CoreBPE>")
            .finish()
    }
}

impl TokenEstimator {
    /// Creates a `TokenEstimator` with the given configuration.
    pub fn new(config: &TokenEstimatorConfig) -> Result<Self> {
        let bpe = Self::get_bpe(&config.encoding)?;
        Ok(Self { bpe })
    }

    /// Creates a `TokenEstimator` with the default configuration (`cl100k_base`).
    pub fn default_estimator() -> Result<Self> {
        Self::new(&TokenEstimatorConfig::default())
    }

    /// Returns the BPE tokenizer for an encoding name, using the process-wide
    /// cache so the vocabulary table is built at most once.
    fn get_bpe(encoding: &str) -> Result<Arc<CoreBPE>> {
        if let Some(bpe) = bpe_cache()
            .lock()
            .expect("BPE cache lock poisoned")
            .get(encoding)
        {
            return Ok(bpe.clone());
        }
        let built = Self::build_bpe(encoding)?;
        Ok(bpe_cache()
            .lock()
            .expect("BPE cache lock poisoned")
            .entry(encoding.to_string())
            .or_insert_with(|| Arc::new(built))
            .clone())
    }

    /// Builds a BPE tokenizer from scratch (cache miss only).
    fn build_bpe(encoding: &str) -> Result<CoreBPE> {
        match encoding {
            "cl100k_base" => tiktoken_rs::cl100k_base().map_err(|e| {
                DataFusionError::External(Box::new(std::io::Error::new(Other, e.to_string())))
            }),
            "p50k_base" => tiktoken_rs::p50k_base().map_err(|e| {
                DataFusionError::External(Box::new(std::io::Error::new(Other, e.to_string())))
            }),
            "r50k_base" => tiktoken_rs::r50k_base().map_err(|e| {
                DataFusionError::External(Box::new(std::io::Error::new(Other, e.to_string())))
            }),
            "o200k_base" => tiktoken_rs::o200k_base().map_err(|e| {
                DataFusionError::External(Box::new(std::io::Error::new(Other, e.to_string())))
            }),
            _ => Err(DataFusionError::Configuration(format!(
                "Unknown tiktoken encoding: {}. Supported: cl100k_base, p50k_base, r50k_base, o200k_base",
                encoding
            ))),
        }
    }

    /// Estimates tokens for a single text string.
    pub fn estimate_text_tokens(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }

    /// Estimates tokens for row `row_idx` of `batch`, summed over `columns`.
    /// Non-string columns are charged one token.
    ///
    /// Returns an error if a column is not in the batch schema.
    pub fn estimate_row_tokens(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
        columns: &[String],
    ) -> Result<usize> {
        let mut total_tokens = 0;

        for col_name in columns {
            let col_idx = batch.schema().index_of(col_name).map_err(|_| {
                DataFusionError::Plan(format!("Column '{}' not found in batch schema", col_name))
            })?;

            let array = batch.column(col_idx);

            // Handle string columns in whichever encoding they arrive:
            // matching only `Utf8` would score a `Utf8View` text value as a
            // single token and mis-size the batch.
            if let Some(string_array) = StringColumn::try_new(array.as_ref()) {
                if let Some(text) = string_array.get(row_idx) {
                    total_tokens += self.estimate_text_tokens(text);
                }
            } else {
                // Non-string columns use a fixed estimate.
                total_tokens += 1;
            }
        }

        Ok(total_tokens)
    }

    /// Estimates tokens for every row of `batch`, summed over `columns`.
    ///
    /// Returns one token count per row.
    pub fn estimate_batch_tokens(
        &self,
        batch: &RecordBatch,
        columns: &[String],
    ) -> Result<Vec<usize>> {
        let num_rows = batch.num_rows();
        let mut token_counts = vec![0usize; num_rows];

        // Resolve column indices and downcast once per batch, not per row.
        for col_name in columns {
            let col_idx = batch.schema().index_of(col_name).map_err(|_| {
                DataFusionError::Plan(format!("Column '{}' not found in batch schema", col_name))
            })?;
            let array = batch.column(col_idx);

            if let Some(string_array) = StringColumn::try_new(array.as_ref()) {
                for (row_idx, count) in token_counts.iter_mut().enumerate() {
                    if let Some(text) = string_array.get(row_idx) {
                        *count += self.estimate_text_tokens(text);
                    }
                }
            } else {
                // Non-string columns use a fixed per-row estimate
                for count in token_counts.iter_mut() {
                    *count += 1;
                }
            }
        }

        Ok(token_counts)
    }

    /// Estimates tokens for every row of several batches.
    ///
    /// Returns a flat vector of `(batch_idx, row_idx, token_count)` tuples.
    pub fn estimate_batches_tokens(
        &self,
        batches: &[RecordBatch],
        columns: &[String],
    ) -> Result<Vec<(usize, usize, usize)>> {
        let mut results = Vec::new();

        for (batch_idx, batch) in batches.iter().enumerate() {
            let token_counts = self.estimate_batch_tokens(batch, columns)?;
            for (row_idx, tokens) in token_counts.into_iter().enumerate() {
                results.push((batch_idx, row_idx, tokens));
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::StringArray;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_test_batch() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("text", DataType::Utf8, false)]);
        let text_array = StringArray::from(vec![
            "Hello world",
            "This is a longer piece of text that should have more tokens",
            "Short",
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(text_array)]).unwrap()
    }

    #[test]
    fn test_estimate_text_tokens() {
        let estimator = TokenEstimator::default_estimator().unwrap();
        let tokens = estimator.estimate_text_tokens("Hello world");
        assert!(tokens > 0);
        assert!(tokens < 10); // "Hello world" is typically 2-3 tokens
    }

    #[test]
    fn test_estimate_batch_tokens() {
        let estimator = TokenEstimator::default_estimator().unwrap();
        let batch = create_test_batch();
        let columns = vec!["text".to_string()];

        let token_counts = estimator.estimate_batch_tokens(&batch, &columns).unwrap();
        assert_eq!(token_counts.len(), 3);

        // Verify relative ordering: longer text should have more tokens
        assert!(token_counts[1] > token_counts[0]);
        assert!(token_counts[1] > token_counts[2]);
    }

    #[test]
    fn test_estimate_row_tokens() {
        let estimator = TokenEstimator::default_estimator().unwrap();
        let batch = create_test_batch();
        let columns = vec!["text".to_string()];

        let tokens_0 = estimator.estimate_row_tokens(&batch, 0, &columns).unwrap();
        let tokens_1 = estimator.estimate_row_tokens(&batch, 1, &columns).unwrap();

        assert!(tokens_1 > tokens_0);
    }
}
