//! Optimizer rules for [NousDB](https://docs.rs/nousdb) query plans.
//!
//! [`semantic_rule_chain`] assembles the default chain from a
//! [`SemanticOptimizerConfig`]. In order:
//!
//! 1. [`SemanticRewritingRule`]: algebraic rewrites that push selections
//!    down, reorder them, commute joins and fuse operators.
//! 2. [`HistogramInjectionRule`] (when a histogram catalog is configured):
//!    attaches token-length histograms of the input columns.
//! 3. [`TokenRepartitionInsertionRule`]: balances partitions by token count.
//! 4. [`BatchSizeOptimizationRule`]: picks how many rows go into each model
//!    request, or [`FixedBatchSizeRule`] for a constant size.
//!
//! [`CertifyBatchSizeRule`] is a physical rule that replaces the batch size
//! with one certified to keep answer quality, when quality certification is
//! enabled.

#![warn(missing_docs)]

pub mod batch_size_rule;
pub mod certify_batch_size_rule;
mod column_lineage;
pub mod fixed_batch_size_rule;
pub mod histogram_injection_rule;
pub mod semantic_chain;
pub mod semantic_rewriting_rule;
pub mod token_repartition_rule;
mod utils;

pub use batch_size_rule::BatchSizeOptimizationRule;
pub use certify_batch_size_rule::CertifyBatchSizeRule;
pub use fixed_batch_size_rule::FixedBatchSizeRule;
pub use histogram_injection_rule::HistogramInjectionRule;
pub use semantic_chain::{
    BatchSizingChoice, SemanticOptimizerConfig, TokenRepartitionChoice, semantic_rule_chain,
};
pub use semantic_rewriting_rule::SemanticRewritingRule;
pub use token_repartition_rule::TokenRepartitionInsertionRule;
