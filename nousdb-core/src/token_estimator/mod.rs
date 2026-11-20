//! Token estimation and batch sizing.

pub mod batch_optimizer;
pub mod catalog;
pub mod estimator;
pub mod histogram;
pub mod preparation;

pub use batch_optimizer::{
    BatchBudgetModel, GENERATE_UTF8_TOKENS_PER_FIELD, JoinBatchBudgetModel, JoinBatchSizingResult,
    JoinBlockEstimate, SafeBatchEstimate, SafeBatchSizingConfig, SizingBound,
    compute_optimal_pair_batch_size, estimate_generation_tokens_per_row,
    estimate_join_block_sizes_v2, estimate_output_tokens_per_row, estimate_safe_batch_size,
    generate_chunk_size, row_format_tokens,
};
pub use catalog::HistogramCatalog;
pub use estimator::{TokenEstimator, TokenEstimatorConfig};
pub use histogram::{HistogramBucket, HistogramStats, TokenHistogram};
pub use preparation::{
    HISTOGRAM_BUCKETS, PreparationReport, PreparedHistogram, column_histogram,
    histogram_string_columns, list_csv_files, prepare_csv_folder, read_csv_for_histograms,
    write_histogram,
};
