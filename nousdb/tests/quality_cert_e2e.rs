//! End-to-end quality certification through the full NousDB stack:
//! `NousDBContextBuilder::with_quality_certification` registers the physical
//! rule; run 1 calibrates at execution time (n singleton reference calls,
//! rung trials, certificate written to the session cache); run 2 re-plans
//! the same operator instance, hits the certificate cache at PLAN time, and
//! executes with a fixed certified size, no prologue, zero new singletons.
//!
//! The `*_overrides_predicted_size` tests pin the precedence itself: with a
//! mock whose fidelity breaks at k = 16, `CertifyBatchSizeRule`'s k̂ = 8 is
//! what the production requests run at, never `BatchSizeOptimizationRule`'s
//! k*_alpha = 16, which the logical node still carries.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::context::SessionConfig;
use datafusion::logical_expr::{Extension, LogicalPlan};

use datafusion::physical_plan::ExecutionPlan;

use nousdb::context::NousDBContextBuilder;
use nousdb::{BatchSizingChoice, SemanticOptimizerConfig};
use nousdb_core::certification::QualityCertConfig;
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::providers::schemas::ModelOutput;
use nousdb_operators::extend::{ExtendExec, ExtendNode, create_extend};
use nousdb_operators::join::{SemanticJoinExec, SemanticJoinNode, create_semantic_join};
use nousdb_operators::selection::create_semantic_selection;
use nousdb_operators::selection::logical::SemanticSelectionNode;
use nousdb_operators::selection::physical::SemanticSelectionExec;

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

/// Find the first node of type `T` in a physical plan tree.
fn find_exec<T: ExecutionPlan>(plan: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    if let Some(found) = plan.downcast_ref::<T>() {
        return Some(found);
    }
    plan.children().into_iter().find_map(find_exec)
}

/// The `EXPLAIN` text of a logical plan, as one string.
async fn explain_text(ctx: &nousdb::context::NousDBContext, plan: LogicalPlan) -> String {
    let explain = DataFrame::new(ctx.state(), plan)
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut text = String::new();
    for b in &explain {
        for col in b.columns() {
            if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..strings.len() {
                    text.push_str(strings.value(i));
                    text.push('\n');
                }
            }
        }
    }
    text
}

/// Faithful mock (`val % 2 == 0`) that exposes its call trace and registers a
/// "mock-model" card so the planner can resolve it. With `parity_labels` set
/// it answers extend-style, a structured object with a `parity` string
/// instead of selection-style booleans. With `join_match` set it answers
/// join-style over the synthetic pair batch (`left_0` / `right_0` stringified
/// columns): true iff the parities agree.
///
/// `flip_at` inverts every answer of a request whose `batch_size` is listed,
/// which is how a test makes fidelity break at one specific `k`: the
/// calibration then certifies a `k̂` strictly below `k*_alpha`, so the two
/// sizes become distinguishable in the call trace.
///
/// `transport_first` makes the first that many calls transport failures (the
/// request never came back), after which the endpoint answers normally.
#[derive(Debug, Default)]
struct E2eMockProvider {
    calls: Mutex<Vec<usize>>,
    parity_labels: bool,
    join_match: bool,
    flip_at: HashSet<usize>,
    transport_first: usize,
}

impl E2eMockProvider {
    fn card() -> ModelCard {
        ModelCard::new("mock-model", 8192)
    }

    fn singleton_calls(&self) -> usize {
        self.count_of_size(1)
    }

    fn count_of_size(&self, size: usize) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|&&s| s == size)
            .count()
    }

    /// Every distinct request size seen so far, with its call count, the
    /// whole trace in one comparable value.
    fn size_histogram(&self) -> BTreeMap<usize, usize> {
        let mut hist = BTreeMap::new();
        for &s in self.calls.lock().unwrap().iter() {
            *hist.entry(s).or_insert(0) += 1;
        }
        hist
    }
}

/// Per-size call counts issued between two `size_histogram()` snapshots.
fn trace_delta(
    before: &BTreeMap<usize, usize>,
    after: &BTreeMap<usize, usize>,
) -> BTreeMap<usize, usize> {
    let mut delta = BTreeMap::new();
    for (&size, &count) in after {
        let issued = count - before.get(&size).copied().unwrap_or(0);
        if issued > 0 {
            delta.insert(size, issued);
        }
    }
    delta
}

#[async_trait]
impl ModelProvider for E2eMockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn models(&self) -> Vec<ModelCard> {
        vec![Self::card()]
    }

    fn model(&self, model_id: &str) -> Option<ModelCard> {
        (model_id == "mock-model").then(Self::card)
    }

    async fn predict(
        &self,
        _model_id: &str,
        _prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        _renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let ordinal = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(batch_size);
            calls.len() - 1
        };
        if ordinal < self.transport_first {
            return Ok(PredictResult::transport_failure(1));
        }
        let flip = self.flip_at.contains(&batch_size);
        if self.join_match {
            let lvals = batch
                .column_by_name("left_0")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let rvals = batch
                .column_by_name("right_0")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let outputs = (start_row..start_row + batch_size)
                .map(|i| {
                    let l: i64 = lvals.value(i).parse().unwrap();
                    let r: i64 = rvals.value(i).parse().unwrap();
                    let matched = l % 2 == r % 2;
                    ModelOutput::Structured(serde_json::json!({ "join_match": matched != flip }))
                })
                .collect();
            return Ok(PredictResult::new(
                outputs,
                InferenceMetrics::new(10, 5, 1_000),
                false,
            ));
        }
        let vals = batch
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let outputs = (start_row..start_row + batch_size)
            .map(|i| {
                let even = (vals.value(i) % 2 == 0) != flip;
                if self.parity_labels {
                    let label = if even { "even" } else { "odd" };
                    ModelOutput::Structured(serde_json::json!({ "parity": label }))
                } else {
                    ModelOutput::Boolean(even)
                }
            })
            .collect();
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
    }
}

fn table_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]))
}

/// Force k*_alpha = 16 on the logical node (simulated feasibility sizing),
/// using the same downcast-clone-rewrap idiom as the optimizer rules.
fn with_batch_size(plan: LogicalPlan, k: usize) -> LogicalPlan {
    let LogicalPlan::Extension(ext) = plan else {
        panic!("expected the semantic selection extension node");
    };
    let node = ext
        .node
        .as_any()
        .downcast_ref::<SemanticSelectionNode>()
        .expect("expected SemanticSelectionNode");
    let mut new_node = node.clone();
    new_node.optimal_batch_size = Some(k);
    LogicalPlan::Extension(Extension {
        node: Arc::new(new_node),
    })
}

/// Join twin of `with_batch_size`: pin the plan-time block geometry (the
/// certify rule gates on both sizes being present; their product is
/// k*_alpha). BatchSizeOptimizationRule no-ops when both are already set.
fn join_with_block_sizes(plan: LogicalPlan, left_bs: usize, right_bs: usize) -> LogicalPlan {
    let LogicalPlan::Extension(ext) = plan else {
        panic!("expected the semantic join extension node");
    };
    let node = ext
        .node
        .as_any()
        .downcast_ref::<SemanticJoinNode>()
        .expect("expected SemanticJoinNode");
    let mut new_node = node.clone();
    new_node.left_block_size = Some(left_bs);
    new_node.right_block_size = Some(right_bs);
    LogicalPlan::Extension(Extension {
        node: Arc::new(new_node),
    })
}

/// Extend-node twin of `with_batch_size`.
fn extend_with_batch_size(plan: LogicalPlan, k: usize) -> LogicalPlan {
    let LogicalPlan::Extension(ext) = plan else {
        panic!("expected the extend extension node");
    };
    let node = ext
        .node
        .as_any()
        .downcast_ref::<ExtendNode>()
        .expect("expected ExtendNode");
    let mut new_node = node.clone();
    new_node.optimal_batch_size = Some(k);
    LogicalPlan::Extension(Extension {
        node: Arc::new(new_node),
    })
}

fn collect_vals(batches: &[RecordBatch]) -> Vec<i64> {
    let mut vals = Vec::new();
    for b in batches {
        let col = b
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..col.len() {
            vals.push(col.value(i));
        }
    }
    vals
}

#[tokio::test(flavor = "multi_thread")]
async fn run_twice_certificate_cache_skips_prologue() -> Result<()> {
    // Registry with the mock provider (model id resolved via list_models).
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider::default());
    registry.register_as("mock", provider.clone())?;

    // Demo-grade certification budget: eta = delta = 0.10, full ladder
    // [16, 8, 4, 2] => delta' = 0.025, n = 36, prefix = 36 * 16 = 576 rows.
    // Singleton calls per calibration are 2n: the reference arm plus the
    // always-run k = 1 diagnostic rung.
    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    // 600-row table.
    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = with_batch_size(
        create_semantic_selection(
            &base_plan,
            "mock-model",
            vec!["val".to_string()],
            "keep even values",
        )?,
        16,
    );
    let expected: Vec<i64> = (0..600).filter(|v| v % 2 == 0).collect();

    // ---- Run 1: calibrates. The mock is perfectly faithful, so the top
    // rung (k = 16) certifies immediately.
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run1), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 1 must issue n = 36 reference calls + 36 for the k=1 rung"
    );

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(cache.len(), 1, "run 1 must write exactly one certificate");

    // ---- Run 2: re-plans the identical operator instance. The physical
    // rule hits the certificate cache at plan time: no prologue, no new
    // singleton calls, same results.
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run2), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 2 must not re-run the reference arm"
    );

    // EXPLAIN after the cache is warm shows the plan-time-resolved size.
    let explain = DataFrame::new(ctx.state(), plan)
        .explain(false, false)?
        .collect()
        .await?;
    let mut explain_text = String::new();
    for b in &explain {
        for col in b.columns() {
            if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..strings.len() {
                    explain_text.push_str(strings.value(i));
                    explain_text.push('\n');
                }
            }
        }
    }
    assert!(
        explain_text.contains("certified_batch_size=16(cache)"),
        "EXPLAIN should show the cache-resolved certified size; got:\n{explain_text}"
    );

    Ok(())
}

/// The endpoint is down for the first 40 requests of a session, then recovers.
/// Run 1's calibration gets no answer to any reference probe, so its ladder is
/// exhausted and the run falls back to k = 1. That outcome comes from request
/// errors rather than the model's answers, so nothing is cached. Run 2 plans
/// the same operator with no certificate in the cache, recalibrates on the
/// healthy endpoint and certifies 16; run 3 is a plain cache hit.
#[tokio::test(flavor = "multi_thread")]
async fn transient_failures_are_not_cached_and_the_next_run_recalibrates() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider {
        transport_first: 40,
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = with_batch_size(
        create_semantic_selection(
            &base_plan,
            "mock-model",
            vec!["val".to_string()],
            "keep even values",
        )?,
        16,
    );
    let expected: Vec<i64> = (0..600).filter(|v| v % 2 == 0).collect();
    let cache = ctx.certificate_cache().expect("certification enabled");

    // ---- Run 1: 36 references + one probe at each of 16, 8, 4, 2 = 40 calls,
    // all unanswered. k̂ = 1 for this run: the n diagnostic singletons and all
    // 600 rows solo, every answer correct.
    let before = provider.size_histogram();
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run1), expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(1, 2 * N + 600), (2, 1), (4, 1), (8, 1), (16, 1)])
    );
    assert!(
        cache.is_empty(),
        "a ladder exhausted by request errors must not be cached"
    );
    // So the next plan attaches a fresh calibration instead of a cached k = 1.
    let explain = explain_text(&ctx, plan.clone()).await;
    assert!(
        explain.contains("certify=[") && !explain.contains("(cache)"),
        "run 2 must be planned to calibrate; got:\n{explain}"
    );

    // ---- Run 2: the endpoint has recovered. A full calibration certifies the
    // top rung, and the 600 rows go out at 16 (37 requests + a tail of 8).
    let before = provider.size_histogram();
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run2), expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(1, 2 * N), (8, 1), (16, N + 37)])
    );
    let entries = cache.entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].1.certified_k, Some(16));

    // ---- Run 3: an ordinary cache hit, no prologue.
    let before = provider.size_histogram();
    let run3 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run3), expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(8, 1), (16, 37)])
    );
    assert!(
        explain_text(&ctx, plan)
            .await
            .contains("certified_batch_size=16(cache)")
    );

    Ok(())
}

/// `run_twice_certificate_cache_skips_prologue` with k*_alpha = 16 pinned by
/// the context-level configuration (`BatchSizingChoice::Fixed`) instead of the
/// `with_batch_size` downcast-clone-rewrap helper: the configured chain's
/// `FixedBatchSizeRule` stamps the size at optimization time, so the bare
/// `create_semantic_selection` plan certifies, caches and re-plans exactly as
/// the hand-pinned one does.
#[tokio::test(flavor = "multi_thread")]
async fn configured_fixed_batch_size_run_twice_certificate_cache_skips_prologue() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider::default());
    registry.register_as("mock", provider.clone())?;

    // Same budget as the hand-pinned twin: eta = delta = 0.10, ladder
    // [16, 8, 4, 2], n = 36, and `max_probes: 0` to pin `n` so the singleton
    // count (2n = 72) is exact.
    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_semantic_optimizer(SemanticOptimizerConfig {
            batch_sizing: BatchSizingChoice::Fixed {
                batch_size: 16,
                join_blocks: None,
            },
            ..Default::default()
        })
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    // No `with_batch_size`: the logical node starts with no size at all.
    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = create_semantic_selection(
        &base_plan,
        "mock-model",
        vec!["val".to_string()],
        "keep even values",
    )?;
    let expected: Vec<i64> = (0..600).filter(|v| v % 2 == 0).collect();

    // ---- Run 1: calibrates; the faithful mock certifies the top rung.
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run1), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 1 must issue n = 36 reference calls + 36 for the k=1 rung"
    );
    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(cache.len(), 1, "run 1 must write exactly one certificate");

    // ---- Run 2: plan-time cache hit, no prologue, no new singletons.
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run2), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 2 must not re-run the reference arm"
    );

    let text = explain_text(&ctx, plan).await;
    assert!(
        text.contains("certified_batch_size=16(cache)"),
        "EXPLAIN should show the cache-resolved certified size; got:\n{text}"
    );

    Ok(())
}

fn collect_parity(batches: &[RecordBatch]) -> Vec<(i64, String)> {
    let mut pairs = Vec::new();
    for b in batches {
        let vals = b
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let labels = b
            .column_by_name("parity")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..vals.len() {
            assert!(labels.is_valid(i), "no NULL labels expected in this test");
            pairs.push((vals.value(i), labels.value(i).to_string()));
        }
    }
    pairs
}

#[tokio::test(flavor = "multi_thread")]
async fn extend_run_twice_certificate_cache_skips_prologue() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider {
        parity_labels: true,
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    // Derived column MUST be nullable: the extend NULL-fill recovery path
    // materializes nulls on request failure.
    let derived = Arc::new(Schema::new(vec![Field::new(
        "parity",
        DataType::Utf8,
        true,
    )]));
    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = extend_with_batch_size(
        create_extend(
            base_plan,
            "label the parity of val",
            "mock-model",
            vec!["val".to_string()],
            derived,
        )?,
        16,
    );
    let expected: Vec<(i64, String)> = (0..600)
        .map(|v| (v, if v % 2 == 0 { "even" } else { "odd" }.to_string()))
        .collect();

    // ---- Run 1: calibrates. The mock is perfectly faithful, so the top
    // rung (k = 16) certifies immediately.
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_parity(&run1), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 1 must issue n = 36 reference calls + 36 for the k=1 rung"
    );

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(cache.len(), 1, "run 1 must write exactly one certificate");

    // ---- Run 2: plan-time certificate-cache hit, no prologue, no new
    // singleton calls, identical output.
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_parity(&run2), expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 2 must not re-run the reference arm"
    );

    // EXPLAIN after the cache is warm: the ExtendExec line itself carries
    // the plan-time-resolved certified size.
    let explain = DataFrame::new(ctx.state(), plan)
        .explain(false, false)?
        .collect()
        .await?;
    let mut extend_line = String::new();
    for b in &explain {
        for col in b.columns() {
            if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..strings.len() {
                    for line in strings.value(i).lines() {
                        if line.contains("ExtendExec") {
                            extend_line.push_str(line);
                            extend_line.push('\n');
                        }
                    }
                }
            }
        }
    }
    assert!(
        extend_line.contains("certified_batch_size=16(cache)"),
        "ExtendExec EXPLAIN line should show the cache-resolved size; got:\n{extend_line}"
    );

    Ok(())
}

fn collect_join_matches(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    for b in batches {
        let l = b
            .column_by_name("lval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let r = b
            .column_by_name("rval")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..l.len() {
            pairs.push((l.value(i), r.value(i)));
        }
    }
    pairs
}

#[tokio::test(flavor = "multi_thread")]
async fn join_run_twice_certificate_cache_skips_prologue() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider {
        join_match: true,
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    // 40 left rows × 24 right rows = 960 pairs; the calibration design
    // needs 36 × 16 = 576 of them (T = 24 buffered left rows).
    let left_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]));
    let right_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "rval",
        DataType::Int64,
        false,
    )]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..40)) as ArrayRef],
    )?;
    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..24)) as ArrayRef],
    )?;
    ctx.register_table(
        "l",
        Arc::new(MemTable::try_new(left_schema, vec![vec![left_batch]])?),
    )?;
    ctx.register_table(
        "r",
        Arc::new(MemTable::try_new(right_schema, vec![vec![right_batch]])?),
    )?;

    let left_plan = ctx.table("l").await?.into_optimized_plan()?;
    let right_plan = ctx.table("r").await?.into_optimized_plan()?;
    // Plan geometry (4, 4) => k*_alpha = 16.
    let plan = join_with_block_sizes(
        create_semantic_join(
            &left_plan,
            &right_plan,
            vec!["lval".to_string()],
            vec!["rval".to_string()],
            "left and right values have the same parity",
            "mock-model",
        )?,
        4,
        4,
    );

    // Parity matches of the full cross product (order-insensitive check
    // enumeration order is pinned by the operator-level suite).
    let mut expected: Vec<(i64, i64)> = (0..40i64)
        .flat_map(|l| {
            (0..24i64)
                .filter(move |r| l % 2 == r % 2)
                .map(move |r| (l, r))
        })
        .collect();
    expected.sort_unstable();

    // ---- Run 1: calibrates. The mock is perfectly faithful, so the top
    // rung (k = 16 pairs) certifies immediately and keep-R decomposition
    // leaves the (4, 4) geometry unchanged.
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    let mut got1 = collect_join_matches(&run1);
    got1.sort_unstable();
    assert_eq!(got1, expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 1 must issue n = 36 solo-pair references + 36 for the k=1 rung"
    );

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(cache.len(), 1, "run 1 must write exactly one certificate");

    // ---- Run 2: plan-time certificate-cache hit, no prologue, no new
    // solo-pair calls, identical match set.
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    let mut got2 = collect_join_matches(&run2);
    got2.sort_unstable();
    assert_eq!(got2, expected);
    assert_eq!(
        provider.singleton_calls(),
        72,
        "run 2 must not re-run the reference arm"
    );

    // EXPLAIN after the cache is warm: the SemanticJoinExec line carries the
    // plan-time-resolved certified pair budget.
    let explain = DataFrame::new(ctx.state(), plan)
        .explain(false, false)?
        .collect()
        .await?;
    let mut join_line = String::new();
    for b in &explain {
        for col in b.columns() {
            if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..strings.len() {
                    for line in strings.value(i).lines() {
                        if line.contains("SemanticJoinExec") {
                            join_line.push_str(line);
                            join_line.push('\n');
                        }
                    }
                }
            }
        }
    }
    assert!(
        join_line.contains("certified_batch_size=16(cache)"),
        "SemanticJoinExec EXPLAIN line should show the cache-resolved size; got:\n{join_line}"
    );

    Ok(())
}

// Certified size beats predicted size: the tests above run a perfectly faithful
// mock, so k̂ == k*_alpha == 16. Here fidelity breaks at k = 16, so the
// certificate lands on k̂ = 8 while the logical node still carries
// k*_alpha = 16. Every production request must go out at 8.

/// n = 36 for the demo budget (eta = delta = 0.10, ladder [16, 8, 4, 2])
/// the same operating point the tests above assert 2n = 72 singletons for.
const N: usize = 36;

#[tokio::test(flavor = "multi_thread")]
async fn certified_size_overrides_predicted_size() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    // Fidelity breaks at 16 => rung 16 fails, rung 8 certifies.
    let provider = Arc::new(E2eMockProvider {
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = with_batch_size(
        create_semantic_selection(
            &base_plan,
            "mock-model",
            vec!["val".to_string()],
            "keep even values",
        )?,
        16,
    );
    let expected: Vec<i64> = (0..600).filter(|v| v % 2 == 0).collect();

    // ---- Run 1: calibrates, then processes every row at k̂.
    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(
        collect_vals(&run1),
        expected,
        "k̂ = 8 is faithful, so the answer is exact"
    );

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(
        cache.entries().first().map(|(_, c)| c.certified_k),
        Some(Some(8)),
        "the certificate must land below k*_alpha for this test to mean anything"
    );

    // The ONLY request ever issued at the predicted size is the rung-16 trial,
    // which the futility rule curtails after its first disagreement. All 600
    // rows go out at k̂ = 8: 36 rung-8 probes + 600/8 = 75 production requests.
    assert_eq!(
        provider.count_of_size(16),
        1,
        "k*_alpha = 16 must appear only as a curtailed calibration rung, \
         never as a production request size"
    );
    assert_eq!(provider.count_of_size(8), N + 75);

    // ---- Plan-time provenance after the certificate is cached: the physical
    // operator's resolved size IS the certified one, even though the logical
    // node it was planned from still says 16.
    let logical = match &plan {
        LogicalPlan::Extension(ext) => {
            ext.node
                .as_any()
                .downcast_ref::<SemanticSelectionNode>()
                .unwrap()
                .optimal_batch_size
        }
        _ => unreachable!(),
    };
    assert_eq!(logical, Some(16), "the predicted size is still on the node");

    let physical = ctx.state().create_physical_plan(&plan).await?;
    let exec = find_exec::<SemanticSelectionExec>(&physical)
        .expect("SemanticSelectionExec in the physical plan");
    assert_eq!(exec.certified_fixed(), Some(8));
    assert_eq!(
        exec.optimal_batch_size(),
        Some(8),
        "the exec must execute at the certified size, not the predicted 16"
    );

    // ---- Run 2: the cache-resolved fixed size, no prologue. The whole trace
    // for this run is 75 requests of 8, nothing at 16, nothing singleton.
    let before = provider.size_histogram();
    let run2 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_vals(&run2), expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(8, 75)]),
        "a certified re-run issues production requests at k̂ only"
    );

    let text = explain_text(&ctx, plan).await;
    assert!(
        text.contains("optimal_batch_size=Some(8)")
            && text.contains("certified_batch_size=8(cache)"),
        "EXPLAIN must show the certified size in place of the predicted one; got:\n{text}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn extend_certified_size_overrides_predicted_size() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider {
        parity_labels: true,
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from_iter_values(0..600)) as ArrayRef],
    )?;
    let mem = MemTable::try_new(table_schema(), vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;

    let derived = Arc::new(Schema::new(vec![Field::new(
        "parity",
        DataType::Utf8,
        true,
    )]));
    let base_plan = ctx.table("t").await?.into_optimized_plan()?;
    let plan = extend_with_batch_size(
        create_extend(
            base_plan,
            "label the parity of val",
            "mock-model",
            vec!["val".to_string()],
            derived,
        )?,
        16,
    );
    let expected: Vec<(i64, String)> = (0..600)
        .map(|v| (v, if v % 2 == 0 { "even" } else { "odd" }.to_string()))
        .collect();

    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    assert_eq!(collect_parity(&run1), expected);

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(
        cache.entries().first().map(|(_, c)| c.certified_k),
        Some(Some(8))
    );
    assert_eq!(
        provider.count_of_size(16),
        1,
        "k*_alpha = 16 only as a curtailed rung"
    );
    assert_eq!(provider.count_of_size(8), N + 75);

    let physical = ctx.state().create_physical_plan(&plan).await?;
    let exec = find_exec::<ExtendExec>(&physical).expect("ExtendExec in the physical plan");
    assert_eq!(exec.certified_fixed(), Some(8));
    assert_eq!(exec.optimal_batch_size(), Some(8));

    let before = provider.size_histogram();
    let run2 = DataFrame::new(ctx.state(), plan).collect().await?;
    assert_eq!(collect_parity(&run2), expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(8, 75)])
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn join_certified_geometry_overrides_predicted_blocks() -> Result<()> {
    let registry = Arc::new(ModelRegistry::new());
    let provider = Arc::new(E2eMockProvider {
        join_match: true,
        flip_at: HashSet::from([16]),
        ..Default::default()
    });
    registry.register_as("mock", provider.clone())?;

    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_quality_certification(QualityCertConfig {
            eta: 0.10,
            delta: 0.10,
            // Pin `n` to the requested budget (N below), so every model call
            // can be accounted for exactly.
            max_probes: 0,
            ..Default::default()
        })
        .build()?;

    let left_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "lval",
        DataType::Int64,
        false,
    )]));
    let right_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "rval",
        DataType::Int64,
        false,
    )]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..40)) as ArrayRef],
    )?;
    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..24)) as ArrayRef],
    )?;
    ctx.register_table(
        "l",
        Arc::new(MemTable::try_new(left_schema, vec![vec![left_batch]])?),
    )?;
    ctx.register_table(
        "r",
        Arc::new(MemTable::try_new(right_schema, vec![vec![right_batch]])?),
    )?;

    let left_plan = ctx.table("l").await?.into_optimized_plan()?;
    let right_plan = ctx.table("r").await?.into_optimized_plan()?;
    // Predicted geometry (4, 4) => k*_alpha = 16 pairs per request.
    let plan = join_with_block_sizes(
        create_semantic_join(
            &left_plan,
            &right_plan,
            vec!["lval".to_string()],
            vec!["rval".to_string()],
            "left and right values have the same parity",
            "mock-model",
        )?,
        4,
        4,
    );

    let mut expected: Vec<(i64, i64)> = (0..40i64)
        .flat_map(|l| {
            (0..24i64)
                .filter(move |r| l % 2 == r % 2)
                .map(move |r| (l, r))
        })
        .collect();
    expected.sort_unstable();

    let run1 = DataFrame::new(ctx.state(), plan.clone()).collect().await?;
    let mut got1 = collect_join_matches(&run1);
    got1.sort_unstable();
    assert_eq!(got1, expected);

    let cache = ctx.certificate_cache().expect("certification enabled");
    assert_eq!(
        cache.entries().first().map(|(_, c)| c.certified_k),
        Some(Some(8))
    );
    // 960 pairs at 8 = 120 production requests; the predicted 16-pair budget
    // survives only as the curtailed rung.
    assert_eq!(provider.count_of_size(16), 1);
    assert_eq!(provider.count_of_size(8), N + 120);

    // The certified pair budget is decomposed back into block geometry:
    // keep the tile width (R = 4), shrink the left block to 2.
    let physical = ctx.state().create_physical_plan(&plan).await?;
    let exec = find_exec::<SemanticJoinExec>(&physical).expect("SemanticJoinExec");
    assert_eq!(exec.certified_fixed(), Some(8));
    assert_eq!(
        exec.planned_block_sizes(),
        Some((2, 4)),
        "the executed geometry must be the certified one, not the predicted (4, 4)"
    );

    let before = provider.size_histogram();
    let run2 = DataFrame::new(ctx.state(), plan).collect().await?;
    let mut got2 = collect_join_matches(&run2);
    got2.sort_unstable();
    assert_eq!(got2, expected);
    assert_eq!(
        trace_delta(&before, &provider.size_histogram()),
        BTreeMap::from([(8, 120)])
    );

    Ok(())
}
