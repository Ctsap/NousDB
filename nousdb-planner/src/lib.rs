//! Query planning for [NousDB](https://docs.rs/nousdb).
//!
//! [`NousDBQueryPlanner`] plugs into DataFusion and turns NousDB's logical
//! nodes into physical operators, resolving each operator's model through a
//! [`ModelRegistry`](nousdb_core::ModelRegistry).
//! [`NousDBPlanBuilderExt`] adds the semantic operators to DataFusion's
//! `LogicalPlanBuilder`.

#![warn(missing_docs)]

pub mod physical;
pub mod plan_builder;
pub mod query_planner;

pub use physical::NousDBPhysicalPlanner;
pub use plan_builder::NousDBPlanBuilderExt;
pub use query_planner::NousDBQueryPlanner;
