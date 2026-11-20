//! The semantic operators of [NousDB](https://docs.rs/nousdb): logical plan
//! nodes and the physical operators that execute them.
//!
//! | Operator | Logical node | Physical operator |
//! |---|---|---|
//! | Extend: add model-derived columns | [`ExtendNode`] | [`ExtendExec`] |
//! | Semantic selection: keep rows the model accepts | [`SemanticSelectionNode`] | [`SemanticSelectionExec`] |
//! | Semantic join: pair rows the model matches | [`SemanticJoinNode`] | [`SemanticJoinExec`] |
//! | Generate: rows the model writes from scratch | [`GenerateNode`] | [`GenerateExec`] |
//! | Token repartition: balance partitions by token count | [`TokenRepartitionNode`] | [`TokenRepartitionExec`] |
//!
//! Most programs build these through the `nousdb` crate's `NousDBDataFrame`
//! rather than directly.

#![warn(missing_docs)]

/// Quality certification: the calibration prologue that certifies a batch size.
pub mod certification;
/// Extend: adds model-derived columns.
pub mod extend;
/// Generate: rows written by a model from a description.
pub mod generate;
/// Semantic join: pairs rows a model matches.
pub mod join;
/// Execution metrics reported by the semantic operators.
pub mod metrics;
/// Semantic selection: keeps rows a model accepts.
pub mod selection;
/// Token repartition: balances partitions by token count.
pub mod token_repartition;

pub use certification::{
    CalibrationOutcome, CertifyRuntime, Disposition, ProbeCalibrationContext, ProbeExecFactory,
    ProbeObservation, ProbeScorer, RequestOutcome, RequestOutcomeSink, StreamCertification,
    run_probe_calibration,
};
pub use extend::{ExtendExec, ExtendNode, create_extend};
pub use generate::{GenerateExec, GenerateNode, create_generate};
pub use join::{SemanticJoinExec, SemanticJoinNode, create_semantic_join};
pub use metrics::{
    CalibrationMetrics, SemanticJoinMetrics, SemanticOperatorMetrics, TokenRepartitionMetrics,
    WallClockTimer,
};
pub use selection::{SemanticSelectionExec, SemanticSelectionNode, create_semantic_selection};
pub use token_repartition::{TokenRepartitionExec, TokenRepartitionNode, create_token_repartition};
