//! DataFusion query planner that installs the NousDB extension planner.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result as DFResult;
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use crate::physical::NousDBPhysicalPlanner;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::registry::ModelRegistry;

/// DataFusion [`QueryPlanner`] that plans NousDB operators through
/// [`NousDBPhysicalPlanner`] and everything else with the default planner.
pub struct NousDBQueryPlanner {
    physical_planner: DefaultPhysicalPlanner,
}

impl NousDBQueryPlanner {
    /// Creates a planner that resolves models through `model_registry`, with the
    /// default [`ReasoningMode`].
    pub fn new(model_registry: Arc<ModelRegistry>) -> Self {
        Self::with_reasoning_mode(model_registry, ReasoningMode::default())
    }

    /// Creates a planner whose semantic operators run under `reasoning`. The mode
    /// is fixed at construction because the extension planner is shared into
    /// the `DefaultPhysicalPlanner`.
    pub fn with_reasoning_mode(
        model_registry: Arc<ModelRegistry>,
        reasoning: ReasoningMode,
    ) -> Self {
        let nousdb_planner: Arc<dyn ExtensionPlanner + Send + Sync> =
            Arc::new(NousDBPhysicalPlanner::new(model_registry).with_reasoning_mode(reasoning));

        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![nousdb_planner]);

        Self { physical_planner }
    }
}

// Manual impl: `DefaultPhysicalPlanner` does not implement `Debug`.
impl fmt::Debug for NousDBQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NousDBQueryPlanner")
            .field("physical_planner", &"<DefaultPhysicalPlanner (redacted)>")
            .finish()
    }
}

#[async_trait]
impl QueryPlanner for NousDBQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        self.physical_planner
            .create_physical_plan(logical_plan, session)
            .await
    }
}
