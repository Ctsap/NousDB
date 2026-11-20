//! Physical optimizer rule for quality-certified batch sizing.
//!
//! Runs after `BatchSizeOptimizationRule` and after all of DataFusion's own
//! physical rewrites (`with_physical_optimizer_rule` appends to the default
//! set). It reads the exec's resolved `optimal_batch_size` as the feasibility
//! size `k*_alpha` and decides how the final batch size is determined:
//!
//! - Certificate-cache hit: resolve a fixed size `min(k*_alpha, cached k)` now,
//!   with no execution-time prologue.
//! - Cardinality gate (optional): inputs too small to amortize the probe cost
//!   run at `k = 1`.
//! - Otherwise: attach a `CertificationSpec`; the stream's calibration prologue
//!   measures fidelity on the realized input and picks `k̂`.
//!
//! Certificates are specific to a plan position, which is why this is a
//! physical rule appended last: an earlier rewrite that moved the operator
//! would invalidate what the calibration measured.
//!
//! `SemanticSelectionExec`, `ExtendExec` and `SemanticJoinExec` are certified;
//! the spec and cache machinery is operator-generic (see
//! `nousdb_core::certification::OperatorKind`). Extend agreement is exact match
//! after canonical normalization of production-coerced values. A join certifies
//! pairs per request: the rule requires both plan-time block sizes
//! (`k*_alpha = L·R`; the runtime-sizing path stays uncertified because a spec
//! must exist at plan time), the cardinality gate counts estimated pairs
//! `|L|·|R|`, and a certified budget is decomposed back to block geometry
//! keeping the tile width (the stream and `with_fixed_certified_size` share
//! `decompose_certified_pairs`).

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, StatisticsArgs, StatisticsContext,
};

use nousdb_core::certification::{
    Budget, CertKey, CertificateCache, CertificationSpec, OperatorKind, PackingPolicy,
    QualityCertConfig, budget_for, build_ladder, feasible_budget, fnv1a_64,
};
use nousdb_operators::certification::{CertifyRuntime, PrefixPool};
use nousdb_operators::extend::ExtendExec;
use nousdb_operators::join::SemanticJoinExec;
use nousdb_operators::join::stream::OnceAsync;
use nousdb_operators::selection::SemanticSelectionExec;

/// The operator's partition count: the number of streams that donate to its
/// calibration sample.
fn partitions_of(node: &Arc<dyn ExecutionPlan>) -> usize {
    node.output_partitioning().partition_count().max(1)
}

/// Operator-independent outcome of the certification decision for one node.
enum CertifyAction {
    /// Leave the node untouched.
    Skip,
    /// Resolve a plan-time fixed certified size (cache hit or gate).
    Fixed(usize),
    /// Attach a spec; the stream's prologue picks `k̂` at execution time.
    Attach(Arc<CertificationSpec>),
}

/// See the module docs. Construct with a session-scoped cache and register
/// via `SessionStateBuilder::with_physical_optimizer_rule` (which appends
/// after the DataFusion defaults).
#[derive(Debug)]
pub struct CertifyBatchSizeRule {
    cfg: QualityCertConfig,
    cache: Arc<CertificateCache>,
}

impl CertifyBatchSizeRule {
    /// Creates the rule with certification settings `cfg` and a session-scoped
    /// certificate `cache`.
    pub fn new(cfg: QualityCertConfig, cache: Arc<CertificateCache>) -> Self {
        Self { cfg, cache }
    }

    /// The certificate cache this rule reads and fills.
    pub fn cache(&self) -> &Arc<CertificateCache> {
        &self.cache
    }

    /// `|O_q|` for the query-level budget split: certified semantic operator
    /// instances in the plan (selection + extend + join). Omitting a
    /// certifiable operator would inflate every instance's per-rung risk
    /// budget, so this list must match `certify_node`'s downcasts.
    fn count_semantic_execs(plan: &Arc<dyn ExecutionPlan>) -> usize {
        let mut count = usize::from(
            plan.is::<SemanticSelectionExec>()
                || plan.is::<ExtendExec>()
                || plan.is::<SemanticJoinExec>(),
        );
        for child in plan.children() {
            count += Self::count_semantic_execs(child);
        }
        count
    }

    /// Cache key for one operator instance: provider, model, operator kind,
    /// packing policy, and a stable hash over the prompt-template identity
    /// with its bound parameters (instruction + input columns).
    fn cert_key_for(exec: &SemanticSelectionExec) -> CertKey {
        // Template names come from the exec's own mode, so they match what
        // SemanticSelectionStream renders. The mode label is hashed too: a
        // different system prompt changes agreement rates, so a certificate
        // must not be reused under another reasoning mode.
        let reasoning = exec.reasoning_mode();
        let system = reasoning.system_template(OperatorKind::Selection);
        let mut parts: Vec<&str> = vec![
            reasoning.label(),
            &system,
            reasoning.user_template(OperatorKind::Selection),
            exec.filter_instruction(),
        ];
        parts.extend(exec.input_columns().iter().map(|s| s.as_str()));
        CertKey {
            provider: exec.provider_name().to_string(),
            model_id: exec.model_id().to_string(),
            operator_kind: OperatorKind::Selection,
            template_hash: fnv1a_64(&parts),
            packing_policy: PackingPolicy::Flat,
        }
    }

    /// Cache key for an extend instance. The derived schema is a bound template
    /// parameter (it becomes the response-format JSON schema and the output
    /// columns), so its field names and types are part of the identity.
    fn cert_key_for_extend(exec: &ExtendExec) -> CertKey {
        // Template names match what ExtendStream renders; the mode label rides
        // alongside them (see `cert_key_for`).
        let reasoning = exec.reasoning_mode();
        let mut parts: Vec<String> = vec![
            reasoning.label().to_string(),
            reasoning.system_template(OperatorKind::Extend),
            reasoning.user_template(OperatorKind::Extend).to_string(),
            exec.user_instruction().to_string(),
        ];
        parts.extend(exec.input_columns().iter().cloned());
        for field in exec.output_schema().fields() {
            parts.push(field.name().clone());
            // The textual DataType is stable enough for a session-scoped cache.
            parts.push(field.data_type().to_string());
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        CertKey {
            provider: exec.provider_name().to_string(),
            model_id: exec.model_id().to_string(),
            operator_kind: OperatorKind::Extend,
            template_hash: fnv1a_64(&refs),
            packing_policy: PackingPolicy::Flat,
        }
    }

    /// Certification scores agreement on production-coerced values. A derived
    /// type outside the set the array builders accept fails at execution
    /// anyway, so certifying it would spend n(1+L) probe calls to conclude
    /// k = 1. Skipping never mints a certificate.
    fn extend_output_types_supported(exec: &ExtendExec) -> bool {
        exec.output_schema().fields().iter().all(|f| {
            matches!(
                f.data_type(),
                DataType::Boolean | DataType::Int64 | DataType::Float64 | DataType::Utf8
            )
        })
    }

    /// Row estimate of the operator's input (pre-operator cardinality).
    fn input_row_estimate(exec: &dyn ExecutionPlan) -> Option<usize> {
        let child = exec.children().first().copied()?;
        let stats = StatisticsContext::new()
            .compute(child.as_ref(), &StatisticsArgs::new())
            .ok()?;
        stats.num_rows.get_value().copied()
    }

    /// Cache key for a join instance. The certified item is a (left, right)
    /// pair; both sides' column names are bound template parameters (the
    /// reference-table headers), tagged with side markers so a column moving
    /// across sides changes the key. Packing is `JoinTile` carrying this
    /// plan's tile width: the block-nested-loop distribution `D_k` does not
    /// transfer to or from the sequential coalescing of selection and extend,
    /// nor between tile widths. The caller skips execs that already carry a
    /// certificate or a fixed size, so the width read here is never one
    /// rewritten by `with_fixed_certified_size`.
    fn cert_key_for_join(exec: &SemanticJoinExec) -> CertKey {
        // Template names match what SemanticJoinStream renders.
        let (left_cols, right_cols) = exec.join_column_names();
        let reasoning = exec.reasoning_mode();
        let mut parts: Vec<String> = vec![
            reasoning.label().to_string(),
            reasoning.system_template(OperatorKind::Join),
            reasoning.user_template(OperatorKind::Join).to_string(),
            exec.join_condition().to_string(),
            "left".to_string(),
        ];
        parts.extend(left_cols);
        parts.push("right".to_string());
        parts.extend(right_cols);
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        CertKey {
            provider: exec.provider_name().to_string(),
            model_id: exec.model_id().to_string(),
            operator_kind: OperatorKind::Join,
            template_hash: fnv1a_64(&refs),
            packing_policy: PackingPolicy::JoinTile {
                right_block: exec.planned_block_sizes().map_or(1, |(_, r)| r),
            },
        }
    }

    /// Estimated pair cardinality `|L|·|R|`, the join's item count for the
    /// cardinality gate (a "row" of the certified join operator is a pair).
    fn join_pair_estimate(exec: &SemanticJoinExec) -> Option<usize> {
        let rows = |child: &Arc<dyn ExecutionPlan>| -> Option<usize> {
            let stats = StatisticsContext::new()
                .compute(child.as_ref(), &StatisticsArgs::new())
                .ok()?;
            stats.num_rows.get_value().copied()
        };
        let children = exec.children();
        let left = rows(children.first().copied()?)?;
        let right = rows(children.get(1).copied()?)?;
        left.checked_mul(right)
    }

    /// Build the execution-time attachment for one operator instance.
    ///
    /// `partitions` is the operator's partition count, which sizes each
    /// partition's donation quota in the shared [`PrefixPool`]. It is resolved
    /// here because the pool is per instance and only the plan knows how many
    /// partitions contribute to it.
    fn runtime_for(&self, spec: Arc<CertificationSpec>, partitions: usize) -> CertifyRuntime {
        CertifyRuntime {
            pool: Arc::new(PrefixPool::new(spec.sample_target(), partitions)),
            spec,
            cache: self.cache.clone(),
            once: Arc::new(OnceAsync::default()),
        }
    }

    /// Operator-independent certification decision, shared by every
    /// certifiable operator branch in `certify_node`.
    fn decide(
        &self,
        op: &str,
        k_star_alpha: Option<usize>,
        cache_key: impl FnOnce() -> CertKey,
        input_rows: Option<usize>,
        num_semantic: usize,
    ) -> CertifyAction {
        // k*_alpha from BatchSizeOptimizationRule or the planner's fallback;
        // certifying a heuristic size is still sound, since fidelity is
        // measured for whatever k the ladder tops out at.
        let Some(k_star_alpha) = k_star_alpha else {
            return CertifyAction::Skip;
        };
        if k_star_alpha <= 1 {
            // phi(1) = 1 by definition; nothing to certify.
            return CertifyAction::Skip;
        }

        if build_ladder(k_star_alpha).is_empty() {
            return CertifyAction::Skip;
        }

        let cache_key = cache_key();

        // Certificate-cache hit: resolve now, clamped by the CURRENT query's
        // feasibility. A cached failed ladder resolves to k = 1.
        if let Some(cert) = self.cache.get(&cache_key) {
            let k = cert.certified_k.unwrap_or(1).min(k_star_alpha);
            log::info!(
                "[CertifyBatchSizeRule] {op}: certificate cache hit (certified_k={:?}) => fixed k={k}, no prologue",
                cert.certified_k
            );
            return CertifyAction::Fixed(k);
        }

        // Cardinality gate: certification cost is |R|-independent but the
        // savings scale with |R|; below the break-even run exact semantics.
        // Absent statistics, proceed with certification rather than guessing.
        if self.cfg.min_rows_for_certification > 0
            && let Some(rows) = input_rows
            && rows < self.cfg.min_rows_for_certification
        {
            log::info!(
                "[CertifyBatchSizeRule] {op}: input estimate {rows} rows < gate {} => fixed k=1",
                self.cfg.min_rows_for_certification
            );
            return CertifyAction::Fixed(1);
        }

        // Size the ladder and the probe budget to the sample that will be
        // available: pinning the top at `k*_alpha` needs `n * k*_alpha` rows,
        // which a large planner-chosen batch size cannot meet. Absent
        // statistics, plan the full ladder and let the runtime retarget it
        // against the realized sample.
        let delta_share = self.cfg.delta_share(num_semantic);
        let budget = match input_rows {
            Some(rows) => match feasible_budget(
                k_star_alpha,
                delta_share,
                self.cfg.eta,
                self.cfg.tolerated_disagreements,
                self.cfg.max_probes,
                rows,
            ) {
                Some(budget) => budget,
                None => {
                    log::info!(
                        "[CertifyBatchSizeRule] {op}: {rows} input rows cannot seat a probe \
                         design at any rung of k*_alpha={k_star_alpha} (eta={}, delta'={}); \
                         fixed k=1 without probing",
                        self.cfg.eta,
                        delta_share,
                    );
                    return CertifyAction::Fixed(1);
                }
            },
            None => {
                let ladder = build_ladder(k_star_alpha);
                let Some((delta_prime, n, s_star)) =
                    budget_for(&self.cfg, ladder.len(), num_semantic)
                else {
                    log::warn!(
                        "[CertifyBatchSizeRule] degenerate budget for eta={}, delta={}, L={}; \
                         skipping certification of {op}",
                        self.cfg.eta,
                        self.cfg.delta,
                        ladder.len()
                    );
                    return CertifyAction::Skip;
                };
                Budget {
                    ladder,
                    delta_prime,
                    n,
                    s_star,
                }
            }
        };

        let tolerance = budget.tolerance();
        let (n, s_star, delta_prime) = (budget.n, budget.s_star, budget.delta_prime);
        let spec = Arc::new(CertificationSpec {
            ladder: budget.ladder,
            eta: self.cfg.eta,
            delta_prime,
            n,
            s_star,
            seed: self.cfg.seed,
            cache_key,
            delta_share,
            tolerated: self.cfg.tolerated_disagreements,
            max_probes: self.cfg.max_probes,
        });
        log::info!(
            "[CertifyBatchSizeRule] {op}: k*_alpha={k_star_alpha}, input_rows={input_rows:?} => \
             attaching spec: ladder={:?}, n={n}, s*={s_star}, tolerated_disagreements={tolerance}, \
             delta'={delta_prime:.6}, min_items={}, sample_target={}",
            spec.ladder,
            spec.min_items(),
            spec.sample_target(),
        );
        CertifyAction::Attach(spec)
    }

    fn certify_node(
        &self,
        node: Arc<dyn ExecutionPlan>,
        num_semantic: usize,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        if let Some(exec) = node.downcast_ref::<SemanticSelectionExec>() {
            // Idempotence: a node this rule already decided is left alone.
            if exec.certification().is_some() || exec.certified_fixed().is_some() {
                return Ok(Transformed::no(node));
            }
            let action = self.decide(
                "SemanticSelectionExec",
                exec.optimal_batch_size(),
                || Self::cert_key_for(exec),
                Self::input_row_estimate(exec),
                num_semantic,
            );
            return Ok(match action {
                CertifyAction::Skip => Transformed::no(node),
                CertifyAction::Fixed(k) => Transformed::yes(Arc::new(
                    exec.with_fixed_certified_size(k),
                )
                    as Arc<dyn ExecutionPlan>),
                CertifyAction::Attach(spec) => {
                    let runtime = self.runtime_for(spec, partitions_of(&node));
                    Transformed::yes(
                        Arc::new(exec.with_certification(runtime)) as Arc<dyn ExecutionPlan>
                    )
                }
            });
        }

        if let Some(exec) = node.downcast_ref::<ExtendExec>() {
            if exec.certification().is_some() || exec.certified_fixed().is_some() {
                return Ok(Transformed::no(node));
            }
            if !Self::extend_output_types_supported(exec) {
                log::warn!(
                    "[CertifyBatchSizeRule] ExtendExec derived schema has output types \
                     outside {{Boolean, Int64, Float64, Utf8}}; skipping certification"
                );
                return Ok(Transformed::no(node));
            }
            let action = self.decide(
                "ExtendExec",
                exec.optimal_batch_size(),
                || Self::cert_key_for_extend(exec),
                Self::input_row_estimate(exec),
                num_semantic,
            );
            return Ok(match action {
                CertifyAction::Skip => Transformed::no(node),
                CertifyAction::Fixed(k) => Transformed::yes(Arc::new(
                    exec.with_fixed_certified_size(k),
                )
                    as Arc<dyn ExecutionPlan>),
                CertifyAction::Attach(spec) => {
                    let runtime = self.runtime_for(spec, partitions_of(&node));
                    Transformed::yes(
                        Arc::new(exec.with_certification(runtime)) as Arc<dyn ExecutionPlan>
                    )
                }
            });
        }

        if let Some(exec) = node.downcast_ref::<SemanticJoinExec>() {
            if exec.certification().is_some() || exec.certified_fixed().is_some() {
                return Ok(Transformed::no(node));
            }
            // Gate on plan-time geometry, never on `predicted_batch_size`:
            // the planner always fills the latter (with a fallback estimate)
            // even when no geometry was resolved, and a spec must describe
            // the geometry that will actually run. The runtime-computed
            // block-size path therefore stays uncertified.
            let Some((l, r)) = exec.planned_block_sizes() else {
                return Ok(Transformed::no(node));
            };
            let action = self.decide(
                "SemanticJoinExec",
                Some(l * r),
                || Self::cert_key_for_join(exec),
                Self::join_pair_estimate(exec),
                num_semantic,
            );
            return Ok(match action {
                CertifyAction::Skip => Transformed::no(node),
                CertifyAction::Fixed(k) => Transformed::yes(Arc::new(
                    exec.with_fixed_certified_size(k),
                )
                    as Arc<dyn ExecutionPlan>),
                CertifyAction::Attach(spec) => {
                    let runtime = self.runtime_for(spec, partitions_of(&node));
                    Transformed::yes(
                        Arc::new(exec.with_certification(runtime)) as Arc<dyn ExecutionPlan>
                    )
                }
            });
        }

        Ok(Transformed::no(node))
    }
}

impl PhysicalOptimizerRule for CertifyBatchSizeRule {
    fn name(&self) -> &str {
        "certify_batch_size"
    }

    fn schema_check(&self) -> bool {
        true // the rule never changes schemas
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let num_semantic = if self.cfg.query_level_budget {
            Self::count_semantic_execs(&plan)
        } else {
            1
        };
        plan.transform_up(|node| self.certify_node(node, num_semantic))
            .map(|t| t.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
    use nousdb_core::certification::DEFAULT_MAX_PROBES;
    use nousdb_core::certification::{Certificate, RungResult};
    use nousdb_core::prompt::prompt::Prompt;
    use nousdb_core::prompt::renderer::BatchRenderer;
    use nousdb_core::providers::metadata::ModelCard;
    use nousdb_core::providers::metrics::PredictResult;
    use nousdb_core::providers::provider::ModelProvider;

    /// Rebuilds `plan` over `children`, recomputing its properties.
    fn rebuild(
        plan: Arc<dyn ExecutionPlan>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    /// Plan-time-only mock: the rule never invokes the model.
    #[derive(Debug)]
    struct PlanTimeMockProvider;

    #[async_trait::async_trait]
    impl ModelProvider for PlanTimeMockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        fn models(&self) -> Vec<ModelCard> {
            vec![]
        }
        fn model(&self, _model_id: &str) -> Option<ModelCard> {
            None
        }
        async fn predict(
            &self,
            _model_id: &str,
            _prompt: &Prompt,
            _batch: &RecordBatch,
            _start_row: usize,
            _batch_size: usize,
            _renderer: &(dyn BatchRenderer + Send + Sync),
        ) -> Result<PredictResult> {
            unreachable!("plan-time tests must never call predict")
        }
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
    }

    fn memory_input(rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = test_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn selection_exec(
        input: Arc<dyn ExecutionPlan>,
        k_star_alpha: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(SemanticSelectionExec::new(
            input,
            "is positive".to_string(),
            vec!["val".to_string()],
            "test-model".to_string(),
            Arc::new(PlanTimeMockProvider),
            ModelCard::new("mock", 8192),
            test_schema(),
            k_star_alpha,
        ))
    }

    fn derived_schema_with(dt: DataType) -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("out", dt, true)]))
    }

    fn extend_exec_with_output(
        input: Arc<dyn ExecutionPlan>,
        k_star_alpha: Option<usize>,
        derived: Arc<Schema>,
    ) -> Arc<dyn ExecutionPlan> {
        let full: Vec<Field> = test_schema()
            .fields()
            .iter()
            .chain(derived.fields().iter())
            .map(|f| f.as_ref().clone())
            .collect();
        Arc::new(ExtendExec::new(
            input,
            "classify the value".to_string(),
            vec!["val".to_string()],
            derived,
            "test-model".to_string(),
            Arc::new(PlanTimeMockProvider),
            ModelCard::new("mock", 8192),
            Arc::new(Schema::new(full)),
            k_star_alpha,
        ))
    }

    fn extend_exec(
        input: Arc<dyn ExecutionPlan>,
        k_star_alpha: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        extend_exec_with_output(input, k_star_alpha, derived_schema_with(DataType::Utf8))
    }

    fn downcast_extend(plan: &Arc<dyn ExecutionPlan>) -> &ExtendExec {
        plan.downcast_ref::<ExtendExec>()
            .expect("expected ExtendExec at plan root")
    }

    /// Demo operating point. `max_probes: 0` pins `n` to the requested budget,
    /// so these tests exercise the ladder and `delta'` plumbing rather than the
    /// sample-spending policy (see `probe_budget_grows_into_the_sample`).
    fn demo_cfg() -> QualityCertConfig {
        QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            max_probes: 0,
            ..Default::default()
        }
    }

    fn rule(cfg: QualityCertConfig) -> CertifyBatchSizeRule {
        CertifyBatchSizeRule::new(cfg, Arc::new(CertificateCache::default()))
    }

    fn downcast(plan: &Arc<dyn ExecutionPlan>) -> &SemanticSelectionExec {
        plan.downcast_ref::<SemanticSelectionExec>()
            .expect("expected SemanticSelectionExec at plan root")
    }

    fn certificate(k: Option<usize>) -> Certificate {
        Certificate {
            certified_k: k,
            per_rung: vec![RungResult {
                k: k.unwrap_or(2),
                gating: true,
                x: 36,
                n: 36,
                scored: 36,
                errors: 0,
                lower_bound: 0.9,
                curtailed: false,
            }],
            eta: 0.10,
            delta_prime: 0.025,
            n: 36,
            n_planned: 36,
        }
    }

    #[test]
    fn attaches_spec_with_demo_budget() {
        let rule = rule(demo_cfg());
        let plan = selection_exec(memory_input(1000), Some(16));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast(&optimized);

        let spec = &exec.certification().expect("spec should be attached").spec;
        assert_eq!(spec.ladder, vec![16, 8, 4, 2]);
        assert_eq!(spec.n, 36);
        assert_eq!(spec.s_star, 36);
        assert!((spec.delta_prime - 0.025).abs() < 1e-12);
        assert_eq!(spec.min_items(), 36 + 15);
        assert_eq!(exec.certified_fixed(), None);
        // k*_alpha itself is untouched until the prologue decides.
        assert_eq!(exec.optimal_batch_size(), Some(16));
    }

    /// Surplus sample is spent on probes. The requested budget at this
    /// operating point is the bare floor `n = 36`, which tolerates nothing;
    /// 1000 rows seat the probe ceiling, and that slack is what lets a rung
    /// absorb a disagreement instead of dying on its first.
    #[test]
    fn probe_budget_grows_into_the_sample() {
        let cfg = QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            ..Default::default()
        };
        let rule = rule(cfg);
        let plan = selection_exec(memory_input(1000), Some(16));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let spec = &downcast(&optimized)
            .certification()
            .expect("spec should be attached")
            .spec;

        assert_eq!(spec.ladder, vec![16, 8, 4, 2], "the top is still k*_alpha");
        assert_eq!(spec.n, DEFAULT_MAX_PROBES);
        assert!(
            spec.n - spec.s_star >= 1,
            "n={} s*={} still tolerates nothing",
            spec.n,
            spec.s_star
        );
        assert!(spec.min_items() <= 1000);
    }

    /// When `k*_alpha` is far larger than the sample can seat, the feasibility
    /// pass drops the ladder top instead of falling back to k = 1 unprobed.
    #[test]
    fn a_thin_input_lowers_the_ladder_top() {
        let rule = rule(demo_cfg());
        // 40 rows seat only 25 probes at k = 16, below the floor of 36.
        let plan = selection_exec(memory_input(40), Some(16));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast(&optimized);
        let spec = &exec.certification().expect("spec should be attached").spec;

        assert!(
            *spec.ladder.first().unwrap() < 16,
            "ladder {:?} does not fit 40 rows",
            spec.ladder
        );
        assert!(spec.min_items() <= 40);
        assert_eq!(exec.certified_fixed(), None, "the prologue still runs");
        // k*_alpha is untouched; only the certified size may fall below it.
        assert_eq!(exec.optimal_batch_size(), Some(16));
    }

    /// Below the smallest seatable design there is nothing to measure, and
    /// buffering the input to discover that wastes a pass over it.
    #[test]
    fn an_input_too_small_for_any_rung_is_fixed_at_one() {
        let rule = rule(demo_cfg());
        let plan = selection_exec(memory_input(4), Some(16));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast(&optimized);

        assert_eq!(exec.certified_fixed(), Some(1));
        assert!(exec.certification().is_none(), "no prologue is attached");
    }

    #[test]
    fn cache_hit_resolves_fixed_and_clamps() {
        let cache = Arc::new(CertificateCache::default());
        let rule = CertifyBatchSizeRule::new(demo_cfg(), cache.clone());

        // Seed the cache by running the key derivation on an identical exec.
        let plan = selection_exec(memory_input(100), Some(16));
        let key = CertifyBatchSizeRule::cert_key_for(downcast(&plan));

        // Cached k below k*_alpha: taken as-is.
        cache.insert(key.clone(), certificate(Some(8)));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        let exec = downcast(&optimized);
        assert!(exec.certification().is_none());
        assert_eq!(exec.certified_fixed(), Some(8));
        assert_eq!(exec.optimal_batch_size(), Some(8));

        // Cached k above the current query's k*_alpha: clamped down.
        cache.insert(key.clone(), certificate(Some(32)));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert_eq!(downcast(&optimized).certified_fixed(), Some(16));

        // A cached failed ladder resolves to exact semantics.
        cache.insert(key, certificate(None));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        assert_eq!(downcast(&optimized).certified_fixed(), Some(1));
    }

    #[test]
    fn cardinality_gate_skips_small_inputs() {
        let cfg = QualityCertConfig {
            min_rows_for_certification: 1000,
            ..demo_cfg()
        };
        let rule = rule(cfg);

        // Memory input exposes exact statistics; 100 < 1000 => fixed k = 1.
        let optimized = rule
            .optimize(
                selection_exec(memory_input(100), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        let exec = downcast(&optimized);
        assert_eq!(exec.certified_fixed(), Some(1));
        assert_eq!(exec.optimal_batch_size(), Some(1));
        assert!(exec.certification().is_none());

        // 2000 >= 1000 => proceeds to certification.
        let optimized = rule
            .optimize(
                selection_exec(memory_input(2000), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        assert!(downcast(&optimized).certification().is_some());
    }

    #[test]
    fn trivial_and_foreign_nodes_untouched() {
        let rule = rule(demo_cfg());

        // k*_alpha = 1: nothing to certify.
        let plan = selection_exec(memory_input(100), Some(1));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&plan, &optimized));

        // Non-semantic plans pass through untouched.
        let mem = memory_input(100);
        let optimized = rule.optimize(mem.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&mem, &optimized));
    }

    #[test]
    fn rule_is_idempotent() {
        let rule = rule(demo_cfg());
        let once = rule
            .optimize(
                selection_exec(memory_input(1000), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        let twice = rule.optimize(once.clone(), &ConfigOptions::new()).unwrap();
        // Second pass is a no-op: same node, same shared OnceAsync identity.
        assert!(Arc::ptr_eq(&once, &twice));
        assert!(Arc::ptr_eq(
            &downcast(&once).certification().unwrap().once,
            &downcast(&twice).certification().unwrap().once,
        ));
    }

    #[test]
    fn query_level_budget_splits_across_instances() {
        let cfg = QualityCertConfig {
            query_level_budget: true,
            max_probes: 0,        // read the budget split, not the growth policy
            ..Default::default()  // eta = delta = 0.05
        };
        let rule = rule(cfg);
        // Two stacked selections => |O_q| = 2, delta' = 0.05/(3*2) = 1/120,
        // n = 94.
        let inner = selection_exec(memory_input(5000), Some(16));
        let outer = selection_exec(inner, Some(16));
        let optimized = rule.optimize(outer, &ConfigOptions::new()).unwrap();
        let spec = &downcast(&optimized).certification().unwrap().spec;
        assert!((spec.delta_prime - 1.0 / 160.0).abs() < 1e-12);
        assert_eq!(spec.n, 99);
        assert_eq!(spec.s_star, 99);
    }

    #[test]
    fn with_new_children_preserves_certification() {
        let rule = rule(demo_cfg());
        let optimized = rule
            .optimize(
                selection_exec(memory_input(1000), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        let before = downcast(&optimized);
        let once_before = before.certification().unwrap().once.clone();

        let rebuilt = rebuild(optimized.clone(), vec![memory_input(1000)]).unwrap();
        let after = downcast(&rebuilt);
        let rt = after
            .certification()
            .expect("certification must survive with_new_children");
        assert!(Arc::ptr_eq(&once_before, &rt.once));
        assert_eq!(after.certified_fixed(), None);

        // The fixed-provenance variant survives too.
        let fixed: Arc<dyn ExecutionPlan> = Arc::new(before.with_fixed_certified_size(8));
        let rebuilt = rebuild(fixed, vec![memory_input(1000)]).unwrap();
        assert_eq!(downcast(&rebuilt).certified_fixed(), Some(8));
    }

    #[test]
    fn extend_attaches_spec_with_demo_budget() {
        let rule = rule(demo_cfg());
        let plan = extend_exec(memory_input(1000), Some(16));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast_extend(&optimized);

        let spec = &exec.certification().expect("spec should be attached").spec;
        assert_eq!(spec.ladder, vec![16, 8, 4, 2]);
        assert_eq!(spec.n, 36);
        assert_eq!(spec.s_star, 36);
        assert_eq!(spec.cache_key.operator_kind, OperatorKind::Extend);
        assert_eq!(exec.certified_fixed(), None);
        assert_eq!(exec.optimal_batch_size(), Some(16));
    }

    #[test]
    fn extend_cache_hit_resolves_fixed_and_clamps() {
        let cache = Arc::new(CertificateCache::default());
        let rule = CertifyBatchSizeRule::new(demo_cfg(), cache.clone());

        let plan = extend_exec(memory_input(100), Some(16));
        let key = CertifyBatchSizeRule::cert_key_for_extend(downcast_extend(&plan));

        // Cached k below k*_alpha: taken as-is.
        cache.insert(key.clone(), certificate(Some(8)));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        let exec = downcast_extend(&optimized);
        assert!(exec.certification().is_none());
        assert_eq!(exec.certified_fixed(), Some(8));
        assert_eq!(exec.optimal_batch_size(), Some(8));

        // A cached failed ladder resolves to exact semantics.
        cache.insert(key, certificate(None));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        assert_eq!(downcast_extend(&optimized).certified_fixed(), Some(1));
    }

    #[test]
    fn extend_rule_is_idempotent() {
        let rule = rule(demo_cfg());
        let once = rule
            .optimize(
                extend_exec(memory_input(1000), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        let twice = rule.optimize(once.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&once, &twice));
        assert!(Arc::ptr_eq(
            &downcast_extend(&once).certification().unwrap().once,
            &downcast_extend(&twice).certification().unwrap().once,
        ));
    }

    #[test]
    fn extend_with_new_children_preserves_certification() {
        let rule = rule(demo_cfg());
        let optimized = rule
            .optimize(
                extend_exec(memory_input(1000), Some(16)),
                &ConfigOptions::new(),
            )
            .unwrap();
        let before = downcast_extend(&optimized);
        let once_before = before.certification().unwrap().once.clone();

        let rebuilt = rebuild(optimized.clone(), vec![memory_input(1000)]).unwrap();
        let after = downcast_extend(&rebuilt);
        let rt = after
            .certification()
            .expect("certification must survive with_new_children");
        assert!(Arc::ptr_eq(&once_before, &rt.once));
        assert_eq!(after.certified_fixed(), None);

        // The fixed-provenance variant survives too.
        let fixed: Arc<dyn ExecutionPlan> = Arc::new(before.with_fixed_certified_size(8));
        let rebuilt = rebuild(fixed, vec![memory_input(1000)]).unwrap();
        assert_eq!(downcast_extend(&rebuilt).certified_fixed(), Some(8));
    }

    #[test]
    fn extend_key_includes_derived_schema() {
        let utf8 = extend_exec_with_output(
            memory_input(10),
            Some(16),
            derived_schema_with(DataType::Utf8),
        );
        let int64 = extend_exec_with_output(
            memory_input(10),
            Some(16),
            derived_schema_with(DataType::Int64),
        );
        let k_utf8 = CertifyBatchSizeRule::cert_key_for_extend(downcast_extend(&utf8));
        let k_int64 = CertifyBatchSizeRule::cert_key_for_extend(downcast_extend(&int64));
        // Same instruction + input columns, different derived type => the
        // certificates must never cross-contaminate.
        assert_ne!(k_utf8, k_int64);

        // Determinism: an identically-shaped exec hashes identically.
        let utf8_again = extend_exec_with_output(
            memory_input(10),
            Some(16),
            derived_schema_with(DataType::Utf8),
        );
        assert_eq!(
            k_utf8,
            CertifyBatchSizeRule::cert_key_for_extend(downcast_extend(&utf8_again))
        );
    }

    #[test]
    fn unsupported_output_type_skipped() {
        let rule = rule(demo_cfg());
        // Date32 is outside the production array builders' supported set:
        // no spec, no cache entry, node untouched.
        let plan = extend_exec_with_output(
            memory_input(1000),
            Some(16),
            derived_schema_with(DataType::Date32),
        );
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&plan, &optimized));
    }

    #[test]
    fn mixed_selection_extend_query_budget() {
        let cfg = QualityCertConfig {
            query_level_budget: true,
            max_probes: 0,        // read the budget split, not the growth policy
            ..Default::default()  // eta = delta = 0.05
        };
        let rule = rule(cfg);
        // Extend stacked on selection => |O_q| = 2 for BOTH instances:
        // delta' = 0.05/(3*2) = 1/120, n = 94.
        let inner = selection_exec(memory_input(5000), Some(16));
        let outer = extend_exec(inner, Some(16));
        let optimized = rule.optimize(outer, &ConfigOptions::new()).unwrap();

        let outer_spec = &downcast_extend(&optimized).certification().unwrap().spec;
        assert!((outer_spec.delta_prime - 1.0 / 160.0).abs() < 1e-12);
        assert_eq!(outer_spec.n, 99);
        assert_eq!(outer_spec.s_star, 99);

        let inner_plan = optimized.children()[0];
        let inner_spec = &downcast(inner_plan).certification().unwrap().spec;
        assert!((inner_spec.delta_prime - 1.0 / 160.0).abs() < 1e-12);
        assert_eq!(inner_spec.n, 99);
    }

    fn right_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "rval",
            DataType::Int64,
            false,
        )]))
    }

    fn memory_input_with(schema: SchemaRef, rows: usize) -> Arc<dyn ExecutionPlan> {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn join_exec_over(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        left_bs: Option<usize>,
        right_bs: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let out_fields: Vec<Field> = left
            .schema()
            .fields()
            .iter()
            .chain(right.schema().fields().iter())
            .map(|f| f.as_ref().clone())
            .collect();
        Arc::new(
            SemanticJoinExec::try_new(
                left,
                right,
                vec!["val".to_string()],
                vec!["rval".to_string()],
                Arc::new(Schema::new(out_fields)),
                Arc::new(PlanTimeMockProvider),
                "test-model".to_string(),
                ModelCard::new("mock", 8192),
                "the values are related".to_string(),
                left_bs.zip(right_bs).map(|(l, r)| l * r),
                left_bs,
                right_bs,
            )
            .unwrap(),
        )
    }

    fn join_exec(
        left_rows: usize,
        right_rows: usize,
        left_bs: Option<usize>,
        right_bs: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        join_exec_over(
            memory_input(left_rows),
            memory_input_with(right_schema(), right_rows),
            left_bs,
            right_bs,
        )
    }

    fn downcast_join(plan: &Arc<dyn ExecutionPlan>) -> &SemanticJoinExec {
        plan.downcast_ref::<SemanticJoinExec>()
            .expect("expected SemanticJoinExec at plan root")
    }

    #[test]
    fn join_attaches_spec_with_demo_budget() {
        let rule = rule(demo_cfg());
        let plan = join_exec(1000, 24, Some(4), Some(4));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast_join(&optimized);

        let spec = &exec.certification().expect("spec should be attached").spec;
        assert_eq!(spec.ladder, vec![16, 8, 4, 2]);
        assert_eq!(spec.n, 36);
        assert_eq!(spec.s_star, 36);
        assert!((spec.delta_prime - 0.025).abs() < 1e-12);
        assert_eq!(spec.cache_key.operator_kind, OperatorKind::Join);
        assert_eq!(
            spec.cache_key.packing_policy,
            PackingPolicy::JoinTile { right_block: 4 },
            "the tile width the rungs will be drawn under is part of the key"
        );
        assert_eq!(exec.certified_fixed(), None);
        // The plan geometry is untouched until the prologue decides.
        assert_eq!(exec.planned_block_sizes(), Some((4, 4)));
    }

    #[test]
    fn join_cache_hit_decomposes_block_geometry() {
        let cache = Arc::new(CertificateCache::default());
        let rule = CertifyBatchSizeRule::new(demo_cfg(), cache.clone());

        let plan = join_exec(100, 24, Some(4), Some(4));
        let key = CertifyBatchSizeRule::cert_key_for_join(downcast_join(&plan));

        // Cached k below k*_alpha: keep-R decomposition (2, 4).
        cache.insert(key.clone(), certificate(Some(8)));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        let exec = downcast_join(&optimized);
        assert!(exec.certification().is_none());
        assert_eq!(exec.certified_fixed(), Some(8));
        assert_eq!(exec.planned_block_sizes(), Some((2, 4)));

        // Cached k above the current query's k*_alpha: clamped, geometry kept.
        cache.insert(key.clone(), certificate(Some(32)));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        let exec = downcast_join(&optimized);
        assert_eq!(exec.certified_fixed(), Some(16));
        assert_eq!(exec.planned_block_sizes(), Some((4, 4)));

        // A cached failed ladder resolves to per-pair semantics.
        cache.insert(key, certificate(None));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();
        let exec = downcast_join(&optimized);
        assert_eq!(exec.certified_fixed(), Some(1));
        assert_eq!(exec.planned_block_sizes(), Some((1, 1)));
    }

    #[test]
    fn join_skipped_without_plan_time_geometry() {
        let rule = rule(demo_cfg());
        // Neither block size resolved: runtime-sizing path, uncertified.
        let plan = join_exec(1000, 24, None, None);
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&plan, &optimized));

        // One-sided geometry is equally unusable for a plan-time spec.
        let plan = join_exec(1000, 24, Some(4), None);
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&plan, &optimized));

        // Degenerate product (1×1): phi(1) = 1, nothing to certify.
        let plan = join_exec(1000, 24, Some(1), Some(1));
        let optimized = rule.optimize(plan.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&plan, &optimized));
    }

    #[test]
    fn join_rule_is_idempotent() {
        let rule = rule(demo_cfg());
        let once = rule
            .optimize(join_exec(1000, 24, Some(4), Some(4)), &ConfigOptions::new())
            .unwrap();
        let twice = rule.optimize(once.clone(), &ConfigOptions::new()).unwrap();
        assert!(Arc::ptr_eq(&once, &twice));
        assert!(Arc::ptr_eq(
            &downcast_join(&once).certification().unwrap().once,
            &downcast_join(&twice).certification().unwrap().once,
        ));
    }

    #[test]
    fn join_with_new_children_preserves_certification() {
        let rule = rule(demo_cfg());
        let optimized = rule
            .optimize(join_exec(1000, 24, Some(4), Some(4)), &ConfigOptions::new())
            .unwrap();
        let before = downcast_join(&optimized);
        let once_before = before.certification().unwrap().once.clone();

        let rebuilt = rebuild(
            optimized.clone(),
            vec![memory_input(1000), memory_input_with(right_schema(), 24)],
        )
        .unwrap();
        let after = downcast_join(&rebuilt);
        let rt = after
            .certification()
            .expect("certification must survive with_new_children");
        assert!(Arc::ptr_eq(&once_before, &rt.once));
        assert_eq!(after.certified_fixed(), None);

        // The fixed-provenance variant survives too.
        let fixed: Arc<dyn ExecutionPlan> = Arc::new(before.with_fixed_certified_size(8));
        let rebuilt = rebuild(
            fixed,
            vec![memory_input(1000), memory_input_with(right_schema(), 24)],
        )
        .unwrap();
        assert_eq!(downcast_join(&rebuilt).certified_fixed(), Some(8));
    }

    #[test]
    fn join_key_separates_sides_and_condition() {
        use datafusion::arrow::array::ArrayRef;

        fn mem(schema: SchemaRef, rows: usize) -> Arc<dyn ExecutionPlan> {
            let cols: Vec<ArrayRef> = schema
                .fields()
                .iter()
                .map(|_| Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef)
                .collect();
            let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
        }

        fn join_with(
            left: Arc<dyn ExecutionPlan>,
            right: Arc<dyn ExecutionPlan>,
            left_cols: Vec<String>,
            right_cols: Vec<String>,
            condition: &str,
        ) -> Arc<dyn ExecutionPlan> {
            let out_fields: Vec<Field> = left
                .schema()
                .fields()
                .iter()
                .chain(right.schema().fields().iter())
                .map(|f| f.as_ref().clone())
                .collect();
            Arc::new(
                SemanticJoinExec::try_new(
                    left,
                    right,
                    left_cols,
                    right_cols,
                    Arc::new(Schema::new(out_fields)),
                    Arc::new(PlanTimeMockProvider),
                    "test-model".to_string(),
                    ModelCard::new("mock", 8192),
                    condition.to_string(),
                    Some(16),
                    Some(4),
                    Some(4),
                )
                .unwrap(),
            )
        }

        let ab = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let c = Arc::new(Schema::new(vec![Field::new("c", DataType::Int64, false)]));
        let a_only = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let bc = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ]));

        // Same flat column sequence (a, b, c) but the side boundary moves:
        // the side markers must keep the keys apart.
        let x = join_with(
            mem(ab.clone(), 10),
            mem(c.clone(), 10),
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
            "related",
        );
        let y = join_with(
            mem(a_only, 10),
            mem(bc, 10),
            vec!["a".to_string()],
            vec!["b".to_string(), "c".to_string()],
            "related",
        );
        let key_x = CertifyBatchSizeRule::cert_key_for_join(downcast_join(&x));
        let key_y = CertifyBatchSizeRule::cert_key_for_join(downcast_join(&y));
        assert_ne!(key_x, key_y);

        // The bound condition is part of the identity.
        let z = join_with(
            mem(ab.clone(), 10),
            mem(c.clone(), 10),
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
            "unrelated",
        );
        assert_ne!(
            key_x,
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&z))
        );

        // Determinism: an identically-shaped exec hashes identically.
        let x_again = join_with(
            mem(ab, 10),
            mem(c, 10),
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
            "related",
        );
        assert_eq!(
            key_x,
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&x_again))
        );

        // Same templates and columns, different tile width: the rungs are
        // drawn as sub-geometries of the tile, so `D_k` differs and the
        // certificate must not transfer.
        let wide = join_exec(10, 10, Some(4), Some(4));
        let narrow = join_exec(10, 10, Some(8), Some(2));
        assert_eq!(
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&wide)).template_hash,
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&narrow)).template_hash,
            "same bound template parameters"
        );
        assert_ne!(
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&wide)),
            CertifyBatchSizeRule::cert_key_for_join(downcast_join(&narrow))
        );
    }

    #[test]
    fn mixed_three_operator_query_budget() {
        let cfg = QualityCertConfig {
            query_level_budget: true,
            max_probes: 0,        // read the budget split, not the growth policy
            ..Default::default()  // eta = delta = 0.05
        };
        let rule = rule(cfg);
        // Join over (extend over selection) => |O_q| = 3 for ALL instances:
        // delta' = 0.05/(3*3) = 1/180, n = 102, s* = 102 (perfect-run-only).
        let inner = selection_exec(memory_input(5000), Some(16));
        let mid = extend_exec(inner, Some(16));
        let plan = join_exec_over(mid, memory_input_with(right_schema(), 24), Some(4), Some(4));
        let optimized = rule.optimize(plan, &ConfigOptions::new()).unwrap();

        let join_spec = &downcast_join(&optimized).certification().unwrap().spec;
        assert!((join_spec.delta_prime - 1.0 / 240.0).abs() < 1e-12);
        assert_eq!(join_spec.n, 107);
        assert_eq!(join_spec.s_star, 107);

        let mid_plan = optimized.children()[0];
        let mid_spec = &downcast_extend(mid_plan).certification().unwrap().spec;
        assert!((mid_spec.delta_prime - 1.0 / 240.0).abs() < 1e-12);
        assert_eq!(mid_spec.n, 107);

        let inner_plan = mid_plan.children()[0];
        let inner_spec = &downcast(inner_plan).certification().unwrap().spec;
        assert!((inner_spec.delta_prime - 1.0 / 240.0).abs() < 1e-12);
        assert_eq!(inner_spec.n, 107);
    }

    #[test]
    fn join_pair_gate_counts_pairs_not_rows() {
        let cfg = QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            min_rows_for_certification: 10_000,
            ..Default::default()
        };
        let rule = rule(cfg);

        // Only 50 left rows, a row-based gate would fire, but 50×400 =
        // 20000 pairs clear the pair gate => spec attached.
        let big_pairs = join_exec(50, 400, Some(4), Some(4));
        let optimized = rule.optimize(big_pairs, &ConfigOptions::new()).unwrap();
        assert!(downcast_join(&optimized).certification().is_some());

        // A genuinely small pair space runs exact: 30×30 = 900 < 10000.
        let small = join_exec(30, 30, Some(4), Some(4));
        let optimized = rule.optimize(small, &ConfigOptions::new()).unwrap();
        let exec = downcast_join(&optimized);
        assert_eq!(exec.certified_fixed(), Some(1));
        assert_eq!(exec.planned_block_sizes(), Some((1, 1)));
    }

    /// A different system prompt changes agreement rates, so a certificate
    /// earned under one reasoning mode must not be reused under another: all
    /// four modes hash to distinct keys, for every operator.
    #[test]
    fn certificate_keys_differ_across_reasoning_modes() {
        use nousdb_core::prompt::reasoning::ReasoningMode;

        let derived = derived_schema_with(DataType::Utf8);
        let full: Vec<Field> = test_schema()
            .fields()
            .iter()
            .chain(derived.fields().iter())
            .map(|f| f.as_ref().clone())
            .collect();
        let left = memory_input(64);
        let right = memory_input_with(right_schema(), 64);
        let join_out: Vec<Field> = left
            .schema()
            .fields()
            .iter()
            .chain(right.schema().fields().iter())
            .map(|f| f.as_ref().clone())
            .collect();

        let mut selection = Vec::new();
        let mut extend = Vec::new();
        let mut join = Vec::new();
        for mode in ReasoningMode::all() {
            selection.push(CertifyBatchSizeRule::cert_key_for(
                &SemanticSelectionExec::new(
                    memory_input(64),
                    "is positive".to_string(),
                    vec!["val".to_string()],
                    "test-model".to_string(),
                    Arc::new(PlanTimeMockProvider),
                    ModelCard::new("mock", 8192),
                    test_schema(),
                    Some(16),
                )
                .with_reasoning_mode(mode),
            ));

            extend.push(CertifyBatchSizeRule::cert_key_for_extend(
                &ExtendExec::new(
                    memory_input(64),
                    "classify the value".to_string(),
                    vec!["val".to_string()],
                    derived.clone(),
                    "test-model".to_string(),
                    Arc::new(PlanTimeMockProvider),
                    ModelCard::new("mock", 8192),
                    Arc::new(Schema::new(full.clone())),
                    Some(16),
                )
                .with_reasoning_mode(mode),
            ));

            join.push(CertifyBatchSizeRule::cert_key_for_join(
                &SemanticJoinExec::try_new(
                    left.clone(),
                    right.clone(),
                    vec!["val".to_string()],
                    vec!["rval".to_string()],
                    Arc::new(Schema::new(join_out.clone())),
                    Arc::new(PlanTimeMockProvider),
                    "test-model".to_string(),
                    ModelCard::new("mock", 8192),
                    "the values are related".to_string(),
                    Some(16),
                    Some(4),
                    Some(4),
                )
                .unwrap()
                .with_reasoning_mode(mode),
            ));
        }

        for (op, keys) in [
            ("selection", &selection),
            ("extend", &extend),
            ("join", &join),
        ] {
            let mut hashes: Vec<u64> = keys.iter().map(|k| k.template_hash).collect();
            hashes.sort_unstable();
            hashes.dedup();
            assert_eq!(
                hashes.len(),
                ReasoningMode::all().len(),
                "{op}: reasoning modes must not share a certificate key"
            );
        }
    }
}
