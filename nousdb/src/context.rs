//! [`NousDBContext`] and its builder: a DataFusion session with the NousDB
//! planner and optimizer rules installed.

use crate::dataframe::{IntoOutputSchema, NousDBDataFrame};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::context::SessionConfig;
use datafusion::execution::context::SessionContext;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::OptimizerRule;
use nousdb_core::certification::{CertificateCache, QualityCertConfig};
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::token_estimator::HistogramCatalog;
use nousdb_operators::generate::create_generate;
use nousdb_optimizer::{
    CertifyBatchSizeRule, SemanticOptimizerConfig, TokenRepartitionInsertionRule,
    semantic_rule_chain,
};
use nousdb_planner::query_planner::NousDBQueryPlanner;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

/// Builder for constructing a [`NousDBContext`] with fine-grained control.
///
/// # Example
///
/// ```rust,ignore
/// use nousdb::prelude::*;
/// use std::sync::Arc;
///
/// let provider = OllamaProvider::builder().build().await?;
/// let registry = Arc::new(ModelRegistry::new());
/// registry.register(provider)?;
///
/// let ctx = NousDBContextBuilder::new(registry)
///     .with_config(config)
///     .with_histogram_catalog("/data/histograms")
///     .build()?;
/// ```
pub struct NousDBContextBuilder {
    model_registry: Arc<ModelRegistry>,
    config: Option<SessionConfig>,
    /// When set, installed on the SessionState (memory pool and spill files).
    runtime_env: Option<Arc<RuntimeEnv>>,
    histogram_dir: Option<PathBuf>,
    /// When set, replaces ALL optimizer rules (DF defaults + NousDB).
    custom_rules: Option<Vec<Arc<dyn OptimizerRule + Send + Sync>>>,
    /// When set, configures the standard chain (see
    /// [`NousDBContextBuilder::with_semantic_optimizer`]); mutually exclusive
    /// with `custom_rules`. `None` means `SemanticOptimizerConfig::default()`.
    semantic_optimizer: Option<SemanticOptimizerConfig>,
    quality_cert: Option<QualityCertConfig>,
    /// `None` defers to `NOUSDB_REASONING_MODE`; see [`resolve_reasoning_mode`].
    reasoning: Option<ReasoningMode>,
    /// Whether the semantic SQL functions are installed; see
    /// [`NousDBContextBuilder::with_sql_functions`].
    sql_functions: bool,
}

impl NousDBContextBuilder {
    /// Start building a NousDBContext. The model registry is required.
    pub fn new(model_registry: Arc<ModelRegistry>) -> Self {
        Self {
            model_registry,
            config: None,
            runtime_env: None,
            histogram_dir: None,
            custom_rules: None,
            semantic_optimizer: None,
            quality_cert: None,
            reasoning: None,
            sql_functions: true,
        }
    }

    /// Chooses whether SQL in this session accepts the semantic functions
    /// `EXTEND`, `SEM_FILTER`, `SEM_JOIN` and `GENERATE` (the `nousdb-sql`
    /// crate). On by default.
    ///
    /// Turn it off for a session whose SQL must never reach a model: the
    /// functions are then unknown names and such a query fails to plan.
    pub fn with_sql_functions(mut self, enabled: bool) -> Self {
        self.sql_functions = enabled;
        self
    }

    /// Sets the reasoning mode every semantic operator in this session runs
    /// under. Overrides `NOUSDB_REASONING_MODE`.
    pub fn with_reasoning_mode(mut self, reasoning: ReasoningMode) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set a custom SessionConfig (batch size, parallelism, etc.).
    /// If not called, DataFusion defaults are used.
    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Install a custom RuntimeEnv (memory pool, disk manager for spilling).
    ///
    /// Without this, DataFusion's default RuntimeEnv applies, which has an
    /// unbounded memory pool. A `RuntimeEnvBuilder::with_memory_limit` env only
    /// takes effect if it is passed here.
    pub fn with_runtime_env(mut self, runtime_env: Arc<RuntimeEnv>) -> Self {
        self.runtime_env = Some(runtime_env);
        self
    }

    /// Enable histogram-guided optimization by loading histograms from `dir`.
    ///
    /// Adds `HistogramInjectionRule` between `SemanticRewritingRule` and
    /// `TokenRepartitionInsertionRule`. Ignored if a custom rules override is set.
    pub fn with_histogram_catalog(mut self, dir: impl Into<PathBuf>) -> Self {
        self.histogram_dir = Some(dir.into());
        self
    }

    /// Override ALL optimizer rules (both DataFusion defaults and NousDB rules).
    ///
    /// The caller is responsible for including DataFusion's default rules if
    /// desired, and for ordering the rules correctly. Use this when
    /// [`Self::with_semantic_optimizer`] does not give enough control.
    pub fn with_optimizer_rules(
        mut self,
        rules: Vec<Arc<dyn OptimizerRule + Send + Sync>>,
    ) -> Self {
        self.custom_rules = Some(rules);
        self
    }

    /// Configure the standard optimizer chain from plain data: algebraic
    /// rewrites on/off, token repartitioning auto/off/forced, batch sizing
    /// histogram-guided/fixed. The chain is DataFusion's rules followed by the
    /// NousDB rules this selects, in the usual order, built by
    /// [`nousdb_optimizer::semantic_rule_chain`]; not calling this is the same
    /// as passing `SemanticOptimizerConfig::default()`.
    ///
    /// Mutually exclusive with [`Self::with_optimizer_rules`], which replaces
    /// the whole chain: setting both makes [`Self::build`] return a
    /// `DataFusionError::Configuration`. Quality certification still removes
    /// token repartitioning, whatever the choice.
    pub fn with_semantic_optimizer(mut self, config: SemanticOptimizerConfig) -> Self {
        self.semantic_optimizer = Some(config);
        self
    }

    /// Enable quality-certified batch sizing.
    ///
    /// Registers `CertifyBatchSizeRule` as a physical optimizer rule (appended
    /// after DataFusion's defaults, so it runs after all plan-mutating
    /// rewrites) with a session-scoped certificate cache, available through
    /// [`NousDBContext::certificate_cache`]. The rule takes each semantic
    /// exec's resolved `optimal_batch_size` as the feasibility size `k*_alpha`.
    ///
    /// Enabling this also removes `TokenRepartitionInsertionRule` from the
    /// rule chain, including from a list supplied via
    /// [`Self::with_optimizer_rules`].
    pub fn with_quality_certification(mut self, config: QualityCertConfig) -> Self {
        self.quality_cert = Some(config);
        self
    }

    /// Assembles the session state and builds the [`NousDBContext`].
    ///
    /// Returns a configuration error if both [`Self::with_optimizer_rules`] and
    /// [`Self::with_semantic_optimizer`] were set.
    pub fn build(self) -> Result<NousDBContext> {
        // Both name the logical rule list, so combining them is ambiguous.
        if self.custom_rules.is_some() && self.semantic_optimizer.is_some() {
            return Err(DataFusionError::Configuration(
                "with_optimizer_rules replaces the whole logical rule chain, so it cannot be \
                 combined with with_semantic_optimizer, which configures the standard chain; \
                 call only one of them"
                    .to_string(),
            ));
        }

        // The only read of NOUSDB_REASONING_MODE. Resolving it deeper could let
        // a query render one mode's templates while the certificate key
        // records another.
        let reasoning = resolve_reasoning_mode(self.reasoning);

        let nousdb_query_planner: Arc<dyn QueryPlanner + Send + Sync> = Arc::new(
            NousDBQueryPlanner::with_reasoning_mode(self.model_registry.clone(), reasoning),
        );

        let histogram_catalog = match &self.histogram_dir {
            Some(dir) => Some(Arc::new(HistogramCatalog::load(dir)?)),
            None => None,
        };

        // Quality certification and token repartitioning are mutually
        // exclusive; see `strip_token_repartition`.
        let certifying = self.quality_cert.is_some();

        let rules = if let Some(custom) = self.custom_rules {
            if reasoning != ReasoningMode::Off {
                // A caller-supplied rule list is opaque (`dyn OptimizerRule`
                // exposes only `name()`), so the mode cannot be pushed into a
                // `HistogramInjectionRule` built there. Sizing would then measure
                // the baseline templates while execution sends the mode's.
                log::warn!(
                    "[NousDBContext] reasoning mode {} with a custom optimizer rule list: any \
                     HistogramInjectionRule in it must be built with \
                     .with_reasoning_mode(), or batch sizing will measure the wrong templates",
                    reasoning.label()
                );
            }
            if certifying {
                strip_token_repartition(custom)
            } else {
                custom
            }
        } else {
            // The standard chain. `reasoning` makes batch sizing measure the
            // templates execution sends, and `certifying` leaves out token
            // repartitioning.
            semantic_rule_chain(
                &self.semantic_optimizer.unwrap_or_default(),
                histogram_catalog.clone(),
                self.model_registry.clone(),
                reasoning,
                certifying,
            )
        };

        // `with_default_features` registers the built-in scalar, aggregate,
        // window and table functions, expression planners, file formats and
        // table factories; without it SQL using any built-in fails to plan.
        // It does not touch the optimizer or query planner, so `rules` still
        // replaces DataFusion's logical rules wholesale.
        let mut state_builder = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(nousdb_query_planner)
            .with_optimizer_rules(rules);

        // The semantic SQL functions: marker functions, the relation planner
        // and the analyzer rule that turns the markers into logical nodes.
        if self.sql_functions {
            state_builder = nousdb_sql::install(state_builder);
        }

        // Quality certification is a physical rule, appended after DataFusion's
        // default physical optimizers so it sees the final plan positions.
        let certificate_cache = match self.quality_cert {
            Some(cfg) => {
                let cache = Arc::new(CertificateCache::default());
                state_builder = state_builder.with_physical_optimizer_rule(Arc::new(
                    CertifyBatchSizeRule::new(cfg, cache.clone()),
                ));
                Some(cache)
            }
            None => None,
        };

        if let Some(config) = self.config {
            state_builder = state_builder.with_config(config);
        }

        if let Some(runtime_env) = self.runtime_env {
            state_builder = state_builder.with_runtime_env(runtime_env);
        }

        let session_ctx = SessionContext::new_with_state(state_builder.build());

        Ok(NousDBContext {
            session_ctx,
            model_registry: self.model_registry,
            histogram_catalog,
            certificate_cache,
            reasoning,
        })
    }
}

/// Environment variable naming the session's reasoning mode.
const REASONING_MODE_ENV: &str = "NOUSDB_REASONING_MODE";

/// Resolves the session's reasoning mode: an explicit
/// [`NousDBContextBuilder::with_reasoning_mode`] wins, then
/// `NOUSDB_REASONING_MODE`, then [`ReasoningMode::Off`].
///
/// An unrecognized label falls back to `Off` with a warning instead of failing
/// the build.
fn resolve_reasoning_mode(explicit: Option<ReasoningMode>) -> ReasoningMode {
    if let Some(mode) = explicit {
        return mode;
    }
    match std::env::var(REASONING_MODE_ENV) {
        Ok(raw) => match ReasoningMode::from_label(&raw) {
            Some(mode) => {
                log::info!("[NousDBContext] {REASONING_MODE_ENV}={raw} => reasoning mode {mode:?}");
                mode
            }
            None => {
                log::warn!(
                    "[NousDBContext] {REASONING_MODE_ENV}='{raw}' is not one of \
                     off/scaffold/rationale/think; falling back to Off"
                );
                ReasoningMode::Off
            }
        },
        Err(_) => ReasoningMode::Off,
    }
}

/// Drop `TokenRepartitionInsertionRule` from a rule list.
///
/// Certification samples each partition's leading rows, which token-ordered
/// partitions would bias toward a token-count range that correlates with
/// difficulty, so the two are not combined. Rules are matched by
/// [`OptimizerRule::name`].
fn strip_token_repartition(
    rules: Vec<Arc<dyn OptimizerRule + Send + Sync>>,
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let (kept, dropped): (Vec<_>, Vec<_>) = rules
        .into_iter()
        .partition(|rule| rule.name() != TokenRepartitionInsertionRule::default().name());
    if !dropped.is_empty() {
        log::info!(
            "[NousDBContext] quality certification is enabled; dropped {} \
             TokenRepartitionInsertionRule instance(s) from the optimizer chain \
             so the calibrated batch distribution matches what executes",
            dropped.len()
        );
    }
    kept
}

/// The main entry point for NousDB query execution.
///
/// NousDBContext wraps DataFusion's SessionContext and extends it with semantic
/// analytics capabilities. Use [`NousDBContextBuilder`] for full control, or
/// the convenience constructors below.
///
/// # Example
///
/// ```rust,ignore
/// use nousdb::prelude::*;
///
/// // Simple, with `registry: Arc<ModelRegistry>` holding the registered providers:
/// let ctx = NousDBContext::new(registry.clone())?;
///
/// // With config + histograms:
/// let ctx = NousDBContextBuilder::new(registry)
///     .with_config(config)
///     .with_histogram_catalog("/data/histograms")
///     .build()?;
/// ```
pub struct NousDBContext {
    session_ctx: SessionContext,
    model_registry: Arc<ModelRegistry>,
    histogram_catalog: Option<Arc<HistogramCatalog>>,
    /// Session-scoped certificate store; `Some` iff quality certification
    /// was enabled via [`NousDBContextBuilder::with_quality_certification`].
    certificate_cache: Option<Arc<CertificateCache>>,
    /// The reasoning mode resolved at build time.
    reasoning: ReasoningMode,
}

impl NousDBContext {
    /// Create a new NousDBContext with default configuration and standard rules.
    pub fn new(model_registry: Arc<ModelRegistry>) -> Result<Self> {
        NousDBContextBuilder::new(model_registry).build()
    }

    /// Create a new NousDBContext with custom SessionConfig and standard rules.
    pub fn with_config(config: SessionConfig, model_registry: Arc<ModelRegistry>) -> Result<Self> {
        NousDBContextBuilder::new(model_registry)
            .with_config(config)
            .build()
    }

    /// The model registry this context resolves models through.
    pub fn model_registry(&self) -> &Arc<ModelRegistry> {
        &self.model_registry
    }

    /// The underlying DataFusion [`SessionContext`].
    pub fn session_context(&self) -> &SessionContext {
        &self.session_ctx
    }

    /// The session's certificate cache, if quality certification is enabled.
    pub fn certificate_cache(&self) -> Option<&Arc<CertificateCache>> {
        self.certificate_cache.as_ref()
    }

    /// The reasoning mode every semantic operator in this session runs under.
    pub fn reasoning_mode(&self) -> ReasoningMode {
        self.reasoning
    }

    /// Create a NousDBDataFrame from a table name.
    pub async fn nousdb_table(&self, table_name: &str) -> Result<NousDBDataFrame> {
        let df = self.session_ctx.table(table_name).await?;
        Ok(NousDBDataFrame::new(df, self.session_ctx.state()))
    }

    /// Execute SQL and return a NousDBDataFrame.
    pub async fn nousdb_sql(&self, sql: &str) -> Result<NousDBDataFrame> {
        let df = self.session_ctx.sql(sql).await?;
        Ok(NousDBDataFrame::new(df, self.session_ctx.state()))
    }

    /// Convert a regular DataFrame to a NousDBDataFrame.
    pub fn to_nousdb(&self, df: DataFrame) -> NousDBDataFrame {
        NousDBDataFrame::new(df, self.session_ctx.state())
    }

    /// Generate a table with a model: `num_rows` rows of `output_schema`,
    /// composed following `instruction`.
    ///
    /// Generate is a source with no input relation. The result is an ordinary
    /// `NousDBDataFrame`, so it
    /// chains into `extend`, `semantic_select`, `semantic_join` and the
    /// relational methods. Field types are limited to Utf8, Int64, Float64 and
    /// Boolean; every field is made nullable.
    ///
    /// The model runs on every execution of the plan, so collecting the same
    /// frame twice generates (and pays for) the table twice.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let startups = ctx.generate(
    ///     vec![("name", DataType::Utf8), ("founded", DataType::Int64)],
    ///     "llama3.1:latest",
    ///     "Fictional early-stage technology startups",
    ///     12,
    /// )?;
    /// ```
    pub fn generate<S: IntoOutputSchema>(
        &self,
        output_schema: S,
        model: impl Into<String>,
        instruction: impl Into<String>,
        num_rows: usize,
    ) -> Result<NousDBDataFrame> {
        self.generate_with_examples(output_schema, model, instruction, num_rows, &[])
    }

    /// [`Self::generate`] steered by few-shot `examples`: rows whose columns
    /// match `output_schema` by name and type, shown to the model as
    /// illustrations of style and range. They are not part of the output.
    /// Rows from an existing table work too (`df.collect().await?`).
    pub fn generate_with_examples<S: IntoOutputSchema>(
        &self,
        output_schema: S,
        model: impl Into<String>,
        instruction: impl Into<String>,
        num_rows: usize,
        examples: &[RecordBatch],
    ) -> Result<NousDBDataFrame> {
        let plan = create_generate(
            output_schema.into_output_schema()?,
            model,
            instruction,
            num_rows,
            examples,
        )?;
        let state = self.session_ctx.state();
        Ok(NousDBDataFrame::new(
            DataFrame::new(state.clone(), plan),
            state,
        ))
    }
}

impl Deref for NousDBContext {
    type Target = SessionContext;

    fn deref(&self) -> &Self::Target {
        &self.session_ctx
    }
}

impl Clone for NousDBContext {
    fn clone(&self) -> Self {
        Self {
            session_ctx: self.session_ctx.clone(),
            model_registry: self.model_registry.clone(),
            histogram_catalog: self.histogram_catalog.clone(),
            certificate_cache: self.certificate_cache.clone(),
            reasoning: self.reasoning,
        }
    }
}

#[cfg(test)]
mod reasoning_mode_tests {
    use super::*;

    /// An explicit builder value is authoritative over the environment.
    #[test]
    fn an_explicit_mode_ignores_the_environment() {
        for mode in ReasoningMode::all() {
            assert_eq!(resolve_reasoning_mode(Some(mode)), mode);
        }
    }

    /// Exercises the environment read. All cases run inside one test because
    /// tests share a process and would race on `NOUSDB_REASONING_MODE`.
    #[test]
    fn the_environment_supplies_the_fallback() {
        let restore = std::env::var(REASONING_MODE_ENV).ok();

        for mode in ReasoningMode::all() {
            // SAFETY: single-threaded within this test, and the only test in
            // the crate that touches this variable.
            unsafe { std::env::set_var(REASONING_MODE_ENV, mode.label()) };
            assert_eq!(resolve_reasoning_mode(None), mode);
        }

        // An unrecognized label falls back to Off.
        unsafe { std::env::set_var(REASONING_MODE_ENV, "chain-of-thought") };
        assert_eq!(resolve_reasoning_mode(None), ReasoningMode::Off);

        unsafe { std::env::remove_var(REASONING_MODE_ENV) };
        assert_eq!(resolve_reasoning_mode(None), ReasoningMode::Off);

        if let Some(value) = restore {
            unsafe { std::env::set_var(REASONING_MODE_ENV, value) };
        }
    }
}
