//! Extension planner that builds physical operators for NousDB logical nodes.

use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::ExecutionPlanProperties;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_operators::extend::logical::ExtendNode;
use nousdb_operators::extend::physical::ExtendExec;
use nousdb_operators::generate::logical::GenerateNode;
use nousdb_operators::generate::physical::GenerateExec;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::join::physical::SemanticJoinExec;
use nousdb_operators::selection::logical::SemanticSelectionNode;
use nousdb_operators::selection::physical::SemanticSelectionExec;
use nousdb_operators::token_repartition::logical::TokenRepartitionNode;
use nousdb_operators::token_repartition::physical::TokenRepartitionExec;
use std::sync::Arc;

/// Extension planner that turns NousDB logical nodes into physical operators.
///
/// For each semantic operator it resolves the model's provider and
/// [`ModelCard`] from the registry and passes the card to the physical exec,
/// so execution needs no further async lookups. The session's
/// [`ReasoningMode`] is passed to the exec for prompt rendering; it is not
/// stored on the logical nodes because their `with_exprs_and_inputs` clone
/// paths would drop it silently.
pub struct NousDBPhysicalPlanner {
    model_registry: Arc<ModelRegistry>,
    reasoning: ReasoningMode,
}

impl NousDBPhysicalPlanner {
    /// The provider serving `model`, and the model's card.
    fn resolve(&self, model: &str) -> Result<(Arc<dyn ModelProvider>, ModelCard)> {
        Ok(self.model_registry.resolve(model)?)
    }

    /// Creates a planner that resolves models through `model_registry`, with the
    /// default [`ReasoningMode`].
    pub fn new(model_registry: Arc<ModelRegistry>) -> Self {
        Self {
            model_registry,
            reasoning: ReasoningMode::default(),
        }
    }

    /// Sets the reasoning mode every semantic operator built by this planner
    /// runs under.
    pub fn with_reasoning_mode(mut self, reasoning: ReasoningMode) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Fallback batch size used when the optimizer produced no value (for
    /// example when there is no histogram catalog).
    ///
    /// Assumes about 50 tokens per column per row, reserves 20% of the context
    /// window for system and output, and clamps to
    /// `[1, min(max_batch_size, quality_max_batch_size, 200)]`, so the model
    /// card's row ceilings always apply.
    fn estimate_batch_size(num_input_columns: usize, card: &ModelCard) -> usize {
        let usable_tokens = (card.context_window as f64 * 0.8) as usize;
        let tokens_per_row = (num_input_columns * 50).max(100);
        let batch_size = usable_tokens / tokens_per_row;
        let card_cap = card
            .max_batch_size
            .min(card.quality_max_batch_size.unwrap_or(usize::MAX));
        batch_size.clamp(1, card_cap.clamp(1, 200))
    }

    async fn plan_extend(
        &self,
        extend_node: &ExtendNode,
        physical_children: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (provider, model_card) = self.resolve(&extend_node.model)?;
        let provider_name = provider.name().to_string();

        log::debug!(
            "Planning Extend operator: model='{}', provider='{}', columns={:?}",
            extend_node.model,
            provider_name,
            extend_node.input_schema
        );

        let optimal_batch_size = extend_node.optimal_batch_size.or_else(|| {
            Some(Self::estimate_batch_size(
                extend_node.input_schema.len(),
                &model_card,
            ))
        });

        log::debug!(
            "[NousDBPhysicalPlanner] plan_extend: optimal_batch_size={:?} (from_optimizer={})",
            optimal_batch_size,
            extend_node.optimal_batch_size.is_some()
        );

        // Per-partition in-flight window from the card (1 = serial dispatch)
        let request_window = model_card.max_concurrent_requests.unwrap_or(1);

        let physical = Arc::new(
            ExtendExec::new(
                physical_children[0].clone(),
                extend_node.prompt.clone(),
                extend_node.input_schema.clone(),
                extend_node.derived_schema.clone(),
                model_card.id.clone(),
                provider,
                model_card,
                Arc::clone(extend_node.schema.inner()),
                optimal_batch_size,
            )
            .with_request_window(request_window)
            .with_reasoning_mode(self.reasoning),
        );

        Ok(physical)
    }

    /// Generate is a leaf: the exec sizes its own requests from the card, and
    /// it does not take the session's `ReasoningMode`.
    async fn plan_generate(
        &self,
        generate_node: &GenerateNode,
        physical_children: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !physical_children.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "Generate is a leaf but was planned with {} children",
                physical_children.len()
            )));
        }

        let (provider, model_card) = self.resolve(&generate_node.model)?;
        let provider_name = provider.name().to_string();

        // Per-partition in-flight window from the card (1 = serial dispatch)
        let request_window = model_card.max_concurrent_requests.unwrap_or(1);

        let physical = GenerateExec::new(
            generate_node.output_schema(),
            generate_node.instruction.clone(),
            generate_node.examples.clone(),
            generate_node.num_rows,
            model_card.id.clone(),
            provider,
            model_card,
        )
        .with_request_window(request_window);

        log::debug!(
            "Planning Generate operator: model='{}', provider='{}', num_rows={}, chunk_size={}",
            generate_node.model,
            provider_name,
            generate_node.num_rows,
            physical.chunk_size()
        );

        Ok(Arc::new(physical))
    }

    async fn plan_semantic_selection(
        &self,
        selection_node: &SemanticSelectionNode,
        physical_children: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (provider, model_card) = self.resolve(&selection_node.model)?;
        let provider_name = provider.name().to_string();

        let optimal_batch_size = selection_node.optimal_batch_size.or_else(|| {
            Some(Self::estimate_batch_size(
                selection_node.input_schema.len(),
                &model_card,
            ))
        });

        log::debug!(
            "Planning SemanticSelection operator: model='{}', provider='{}', columns={:?}, optimal_batch_size={:?} (from_optimizer={})",
            selection_node.model,
            provider_name,
            selection_node.input_schema,
            optimal_batch_size,
            selection_node.optimal_batch_size.is_some()
        );

        // Per-partition in-flight window from the card (1 = serial dispatch)
        let request_window = model_card.max_concurrent_requests.unwrap_or(1);

        let physical = Arc::new(
            SemanticSelectionExec::new(
                physical_children[0].clone(),
                selection_node.prompt.clone(),
                selection_node.input_schema.clone(),
                model_card.id.clone(),
                provider,
                model_card,
                Arc::clone(selection_node.schema.inner()),
                optimal_batch_size,
            )
            .with_request_window(request_window)
            .with_reasoning_mode(self.reasoning),
        );

        Ok(physical)
    }

    async fn plan_semantic_join(
        &self,
        join_node: &SemanticJoinNode,
        physical_children: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (provider, model_card) = self.resolve(&join_node.model)?;
        let provider_name = provider.name().to_string();

        log::debug!(
            "Planning SemanticJoin operator: model='{}', provider='{}', columns={:?}",
            join_node.model,
            provider_name,
            join_node
                .left_join_attrs
                .iter()
                .chain(join_node.right_join_attrs.iter())
                .cloned()
                .collect::<Vec<_>>()
        );

        let predicted_batch_size = match (join_node.left_block_size, join_node.right_block_size) {
            (Some(l), Some(r)) => Some(l * r),
            _ => {
                let total_cols = join_node.left_join_attrs.len() + join_node.right_join_attrs.len();
                // For a join the card's ceilings count pairs.
                let batch = Self::estimate_batch_size(total_cols, &model_card);
                Some(batch)
            }
        };

        // Per-partition in-flight window from the card (1 = serial dispatch)
        let request_window = model_card.max_concurrent_requests.unwrap_or(1);

        let physical = SemanticJoinExec::try_new(
            physical_children[0].clone(),
            physical_children[1].clone(),
            join_node.left_join_attrs.clone(),
            join_node.right_join_attrs.clone(),
            Arc::clone(join_node.schema.inner()),
            provider,
            model_card.id.clone(),
            model_card,
            join_node.prompt.clone(),
            predicted_batch_size,
            join_node.left_block_size,
            join_node.right_block_size,
        )?
        .with_request_window(request_window)
        .with_reasoning_mode(self.reasoning);

        Ok(Arc::new(physical))
    }

    async fn plan_token_repartition(
        &self,
        node: &TokenRepartitionNode,
        physical_children: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let num_partitions = node
            .num_partitions
            .unwrap_or_else(|| physical_children[0].output_partitioning().partition_count());

        log::debug!(
            "Planning TokenRepartition operator: columns={:?}, partitions={}, budget={:?}, strategy={:?}",
            node.token_columns,
            num_partitions,
            node.token_budget,
            node.strategy
        );

        let physical = TokenRepartitionExec::new(
            physical_children[0].clone(),
            node.token_columns.clone(),
            num_partitions,
            node.token_budget,
            node.strategy.clone(),
        )?;

        Ok(Arc::new(physical))
    }
}

#[async_trait]
impl ExtensionPlanner for NousDBPhysicalPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn datafusion::logical_expr::UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session: &dyn Session,
        // NousDB's execs build no physical expressions, so there is nothing to forward this to.
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(extend_node) = node.as_any().downcast_ref::<ExtendNode>() {
            let physical = self.plan_extend(extend_node, physical_inputs).await?;
            return Ok(Some(physical));
        }

        if let Some(generate_node) = node.as_any().downcast_ref::<GenerateNode>() {
            let physical = self.plan_generate(generate_node, physical_inputs).await?;
            return Ok(Some(physical));
        }

        if let Some(selection_node) = node.as_any().downcast_ref::<SemanticSelectionNode>() {
            let physical = self
                .plan_semantic_selection(selection_node, physical_inputs)
                .await?;
            return Ok(Some(physical));
        }

        if let Some(join_node) = node.as_any().downcast_ref::<SemanticJoinNode>() {
            let physical = self.plan_semantic_join(join_node, physical_inputs).await?;
            return Ok(Some(physical));
        }

        if let Some(repartition_node) = node.as_any().downcast_ref::<TokenRepartitionNode>() {
            let physical = self
                .plan_token_repartition(repartition_node, physical_inputs)
                .await?;
            return Ok(Some(physical));
        }

        Ok(None)
    }
}
