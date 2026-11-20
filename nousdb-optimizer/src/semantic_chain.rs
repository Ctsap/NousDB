//! Turns a plain-data optimizer configuration into the logical rule list of a
//! NousDB session.
//!
//! Callers that cannot build Rust rule objects (language bindings, config
//! files) pick from a closed set of choices here. `NousDBContextBuilder::build`
//! uses [`semantic_rule_chain`] for its default path too, and
//! [`SemanticOptimizerConfig::default`] reproduces the standard chain.

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::OptimizerRule;

use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::token_estimator::HistogramCatalog;
use nousdb_operators::token_repartition::config::PartitionStrategy;

use crate::batch_size_rule::BatchSizeOptimizationRule;
use crate::fixed_batch_size_rule::FixedBatchSizeRule;
use crate::histogram_injection_rule::HistogramInjectionRule;
use crate::semantic_rewriting_rule::SemanticRewritingRule;
use crate::token_repartition_rule::TokenRepartitionInsertionRule;

/// Whether and how token repartitioning is inserted.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TokenRepartitionChoice {
    /// `TokenRepartitionInsertionRule::default()`: the session's
    /// `target_partitions`, with the five-gate insertion funnel when a
    /// histogram is available (it may decline), unconditional otherwise.
    #[default]
    Auto,
    /// No repartition rule; DataFusion's round-robin applies.
    Off,
    /// `TokenRepartitionInsertionRule::forced(partitions, strategy)`: pins the
    /// partition count and bypasses the funnel, so "on" means "inserted".
    Forced {
        /// Number of partitions to repartition into.
        partitions: usize,
        /// How rows are assigned to partitions.
        strategy: PartitionStrategy,
    },
}

/// How the batch size of each semantic operator is chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchSizingChoice {
    /// `BatchSizeOptimizationRule::default()` (feasibility size `k*_α` from the
    /// injected histograms), with `alpha`, the context-overflow probability,
    /// when set. Operators without a histogram keep the planner's card-derived
    /// fallback.
    Auto {
        /// Overflow-probability target; `None` keeps the rule's default.
        alpha: Option<f64>,
    },
    /// `FixedBatchSizeRule::new(batch_size)`: every extend/selection runs at
    /// `batch_size`; joins at `join_blocks = (left, right)`, or
    /// `(batch_size, batch_size)` when `None`.
    Fixed {
        /// Rows per request for extend and selection operators.
        batch_size: usize,
        /// `(left, right)` join block sizes; `None` uses `(batch_size, batch_size)`.
        join_blocks: Option<(usize, usize)>,
    },
}

impl Default for BatchSizingChoice {
    fn default() -> Self {
        Self::Auto { alpha: None }
    }
}

/// Plain-data description of the NousDB part of the logical optimizer chain.
///
/// `Default` is the standard chain: algebraic rewrites on, automatic token
/// repartitioning, histogram-guided batch sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticOptimizerConfig {
    /// Register `SemanticRewritingRule` (algebraic rewrites A–F).
    pub rewrites: bool,
    /// Whether and how token repartitioning is inserted.
    pub token_repartition: TokenRepartitionChoice,
    /// How batch sizes are chosen.
    pub batch_sizing: BatchSizingChoice,
}

impl Default for SemanticOptimizerConfig {
    fn default() -> Self {
        Self {
            rewrites: true,
            token_repartition: TokenRepartitionChoice::default(),
            batch_sizing: BatchSizingChoice::default(),
        }
    }
}

/// Build the logical optimizer chain for `cfg`, in registration order:
///
/// 1. DataFusion's own logical rules (`SessionStateBuilder::new().build().optimizers()`)
/// 2. `SemanticRewritingRule`, if `cfg.rewrites`
/// 3. `HistogramInjectionRule`, if a `histogram_catalog` is given
/// 4. the token-repartition rule per `cfg.token_repartition`, never when `certifying`
/// 5. the batch-sizing rule per `cfg.batch_sizing`
///
/// Rule order matters (rewrites settle structure before anything is
/// annotated; injection precedes both repartitioning, whose funnel reads the
/// histogram, and sizing, whose estimator does). `reasoning` is the session's
/// resolved arm and must be the one execution renders. `certifying` is whether
/// the session enables quality certification.
pub fn semantic_rule_chain(
    cfg: &SemanticOptimizerConfig,
    histogram_catalog: Option<Arc<HistogramCatalog>>,
    registry: Arc<ModelRegistry>,
    reasoning: ReasoningMode,
    certifying: bool,
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let default_state = SessionStateBuilder::new().build();
    let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = default_state.optimizers().to_vec();

    if cfg.rewrites {
        rules.push(Arc::new(SemanticRewritingRule));
    }

    if let Some(catalog) = histogram_catalog {
        // Batch sizing must measure the templates execution sends, so the
        // injection rule renders prompts under the session's reasoning mode.
        rules.push(Arc::new(
            HistogramInjectionRule::new(catalog, registry).with_reasoning_mode(reasoning),
        ));
    }

    // Certification and token repartitioning are mutually exclusive (a
    // certified batch distribution must match what executes, and token-ordered
    // partitions bias the pooled calibration sample), so the rule is left out
    // whatever the choice. `NousDBContextBuilder::build` applies the same
    // exclusion to a caller-supplied rule list by name.
    if certifying {
        if matches!(cfg.token_repartition, TokenRepartitionChoice::Forced { .. }) {
            log::info!(
                "[semantic_rule_chain] quality certification is enabled; not inserting the \
                 requested forced TokenRepartitionInsertionRule"
            );
        }
    } else {
        match &cfg.token_repartition {
            TokenRepartitionChoice::Auto => {
                rules.push(Arc::new(TokenRepartitionInsertionRule::default()));
            }
            TokenRepartitionChoice::Off => {}
            TokenRepartitionChoice::Forced {
                partitions,
                strategy,
            } => {
                rules.push(Arc::new(TokenRepartitionInsertionRule::forced(
                    *partitions,
                    strategy.clone(),
                )));
            }
        }
    }

    match &cfg.batch_sizing {
        BatchSizingChoice::Auto { alpha } => {
            let mut rule = BatchSizeOptimizationRule::default();
            if let Some(alpha) = alpha {
                rule = rule.with_alpha(*alpha);
            }
            rules.push(Arc::new(rule));
        }
        BatchSizingChoice::Fixed {
            batch_size,
            join_blocks,
        } => {
            let mut rule = FixedBatchSizeRule::new(*batch_size);
            if let Some((left, right)) = join_blocks {
                rule = rule.with_join_blocks(*left, *right);
            }
            rules.push(Arc::new(rule));
        }
    }

    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(rules: &[Arc<dyn OptimizerRule + Send + Sync>]) -> Vec<String> {
        rules.iter().map(|r| r.name().to_string()).collect()
    }

    fn chain(cfg: &SemanticOptimizerConfig, certifying: bool) -> Vec<String> {
        names(&semantic_rule_chain(
            cfg,
            None,
            Arc::new(ModelRegistry::new()),
            ReasoningMode::Off,
            certifying,
        ))
    }

    fn df_defaults() -> Vec<String> {
        names(SessionStateBuilder::new().build().optimizers())
    }

    #[test]
    fn default_config_is_the_standard_chain() {
        let mut expected = df_defaults();
        expected.extend([
            "SemanticRewritingRule".to_string(),
            "TokenRepartitionInsertionRule".to_string(),
            "BatchSizeOptimizationRule".to_string(),
        ]);
        assert_eq!(chain(&SemanticOptimizerConfig::default(), false), expected);
    }

    #[test]
    fn certifying_drops_the_repartition_rule_whatever_the_choice() {
        let forced = SemanticOptimizerConfig {
            token_repartition: TokenRepartitionChoice::Forced {
                partitions: 4,
                strategy: PartitionStrategy::EqualRowCounts,
            },
            ..Default::default()
        };
        for cfg in [SemanticOptimizerConfig::default(), forced] {
            assert!(
                !chain(&cfg, true).contains(&"TokenRepartitionInsertionRule".to_string()),
                "{cfg:?}"
            );
        }
    }

    #[test]
    fn each_choice_swaps_exactly_its_own_entry() {
        // Everything off or fixed: DataFusion's rules, untouched, then the
        // fixed-size rule and nothing else.
        let minimal = SemanticOptimizerConfig {
            rewrites: false,
            token_repartition: TokenRepartitionChoice::Off,
            batch_sizing: BatchSizingChoice::Fixed {
                batch_size: 5,
                join_blocks: Some((1, 5)),
            },
        };
        let mut expected = df_defaults();
        expected.push("FixedBatchSizeRule".to_string());
        assert_eq!(chain(&minimal, false), expected);

        // Forced repartition keeps its slot between rewrites and sizing.
        let forced = SemanticOptimizerConfig {
            token_repartition: TokenRepartitionChoice::Forced {
                partitions: 4,
                strategy: PartitionStrategy::OptimalTokenBalance,
            },
            ..Default::default()
        };
        assert_eq!(
            chain(&forced, false),
            chain(&SemanticOptimizerConfig::default(), false)
        );
    }

    #[test]
    fn config_types_are_plain_data() {
        let a = SemanticOptimizerConfig::default();
        let b = a.clone();
        assert_eq!(a, b);
        assert_ne!(
            a,
            SemanticOptimizerConfig {
                batch_sizing: BatchSizingChoice::Auto { alpha: Some(0.05) },
                ..Default::default()
            }
        );
    }
}
