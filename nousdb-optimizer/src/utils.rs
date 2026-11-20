use datafusion::logical_expr::LogicalPlan;
use nousdb_operators::token_repartition::logical::TokenRepartitionNode;

/// Whether `plan` is a `TokenRepartitionNode`.
pub(crate) fn is_token_repartition(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Extension(ext) => ext
            .node
            .as_any()
            .downcast_ref::<TokenRepartitionNode>()
            .is_some(),
        _ => false,
    }
}
