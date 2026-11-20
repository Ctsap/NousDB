//! A session built by `NousDBContextBuilder` must carry DataFusion's default
//! features, built-in scalar / aggregate / window functions, expr planners,
//! file formats, table factories, exactly as a plain `SessionContext` does.
//!
//! The first half checks that SQL naming a built-in function plans, which
//! needs the default function registry (a bare `SessionStateBuilder::new()`
//! has none). The second half pins what installing the default features must
//! not disturb: the logical rule chain (`with_optimizer_rules` replaces all
//! rules, DataFusion's included) and the physical one (`CertifyBatchSizeRule`
//! appended after DataFusion's defaults).
//!
//! No model is involved: the registry is empty and no semantic operator is
//! executed.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::context::SessionConfig;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::optimizer::OptimizerRule;
use datafusion::physical_optimizer::PhysicalOptimizerRule;

use datafusion::error::DataFusionError;
use datafusion::logical_expr::LogicalPlan;

use nousdb::context::{NousDBContext, NousDBContextBuilder};
use nousdb::{BatchSizingChoice, SemanticOptimizerConfig, TokenRepartitionChoice};
use nousdb_core::certification::{CertificateCache, QualityCertConfig};
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_operators::join::logical::SemanticJoinNode;
use nousdb_operators::selection::logical::SemanticSelectionNode;
use nousdb_operators::token_repartition::config::PartitionStrategy;
use nousdb_optimizer::{
    BatchSizeOptimizationRule, CertifyBatchSizeRule, SemanticRewritingRule,
    TokenRepartitionInsertionRule,
};

/// Review lengths in characters, by row. Five fall under the 200-character
/// cut the query below uses, two do not.
const REVIEW_LENGTHS: [usize; 7] = [12, 40, 260, 199, 5, 200, 80];
const SHORT_REVIEWS: usize = 5;

fn builder() -> NousDBContextBuilder {
    NousDBContextBuilder::new(Arc::new(ModelRegistry::new()))
        .with_config(SessionConfig::new().with_target_partitions(1))
}

/// Register `t(title, review)` with one review per entry of [`REVIEW_LENGTHS`].
fn register_reviews(ctx: &NousDBContext) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::Utf8, false),
        Field::new("review", DataType::Utf8, false),
    ]));
    let titles: Vec<String> = (0..REVIEW_LENGTHS.len())
        .map(|i| format!("movie_{i}"))
        .collect();
    let reviews: Vec<String> = REVIEW_LENGTHS.iter().map(|&n| "x".repeat(n)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(titles)),
            Arc::new(StringArray::from(reviews)),
        ],
    )?;
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
    Ok(())
}

/// A session with `t` registered.
fn reviews_ctx() -> Result<NousDBContext> {
    let ctx = builder().build()?;
    register_reviews(&ctx)?;
    Ok(ctx)
}

fn optimizer_names(state: &SessionState) -> Vec<String> {
    state
        .optimizers()
        .iter()
        .map(|rule| rule.name().to_string())
        .collect()
}

fn analyzer_names(state: &SessionState) -> Vec<String> {
    state
        .analyzer()
        .rules
        .iter()
        .map(|rule| rule.name().to_string())
        .collect()
}

fn physical_optimizer_names(state: &SessionState) -> Vec<String> {
    state
        .physical_optimizers()
        .iter()
        .map(|rule| rule.name().to_string())
        .collect()
}

// SQL that names a built-in function

/// A query that names built-in functions.
#[tokio::test]
async fn sql_resolves_a_scalar_function() -> Result<()> {
    let ctx = reviews_ctx()?;

    let batches = ctx
        .nousdb_sql("SELECT title, review FROM t WHERE character_length(review) < 200 LIMIT 4")
        .await?
        .collect()
        .await?;

    let mut rows = 0;
    for batch in &batches {
        let reviews = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("review is Utf8");
        for i in 0..reviews.len() {
            assert!(reviews.value(i).len() < 200, "the predicate was applied");
            rows += 1;
        }
    }
    assert_eq!(rows, 4, "LIMIT 4 over {SHORT_REVIEWS} qualifying rows");
    Ok(())
}

#[tokio::test]
async fn sql_resolves_aggregate_functions() -> Result<()> {
    let ctx = reviews_ctx()?;

    let batches = ctx
        .nousdb_sql(
            "SELECT count(*) AS n, max(character_length(review)) AS longest \
             FROM t WHERE character_length(review) < 200",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let batch = batches.iter().find(|b| b.num_rows() == 1).unwrap();
    let n = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64");
    assert_eq!(n.value(0), SHORT_REVIEWS as i64);
    // `character_length` is Int32; widen through the display form rather than
    // pin DataFusion's choice of integer width.
    let longest = datafusion::arrow::util::display::array_value_to_string(batch.column(1), 0)?;
    assert_eq!(longest, "199");
    Ok(())
}

#[tokio::test]
async fn sql_resolves_a_window_function() -> Result<()> {
    let ctx = reviews_ctx()?;

    let batches = ctx
        .nousdb_sql("SELECT title, row_number() OVER (ORDER BY title) AS rn FROM t ORDER BY title")
        .await?
        .collect()
        .await?;

    let ranks: Vec<u64> = batches
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("row_number is UInt64")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ranks, (1..=REVIEW_LENGTHS.len() as u64).collect::<Vec<_>>());
    Ok(())
}

/// Syntax that never names a function still needs the default *expr planners*:
/// `substring(x FROM a FOR b)` and `EXTRACT(.. FROM ..)` are rewritten to
/// function calls by `CoreFunctionPlanner` / `DatetimeFunctionPlanner`.
#[tokio::test]
async fn sql_resolves_planner_backed_syntax() -> Result<()> {
    let ctx = reviews_ctx()?;

    let batches = ctx
        .nousdb_sql(
            "SELECT substring(title FROM 1 FOR 5) AS prefix, \
                    EXTRACT(YEAR FROM DATE '2026-10-12') AS y \
             FROM t LIMIT 1",
        )
        .await?
        .collect()
        .await?;

    let batch = batches.iter().find(|b| b.num_rows() == 1).expect("one row");
    let prefix = datafusion::arrow::util::display::array_value_to_string(batch.column(0), 0)?;
    let year = datafusion::arrow::util::display::array_value_to_string(batch.column(1), 0)?;
    assert_eq!(prefix, "movie");
    assert_eq!(year, "2026");
    Ok(())
}

/// A function-bearing SQL source still composes with a semantic operator: the
/// SQL filter sits below the `SemanticSelection` node in the optimized plan.
/// Planning stops at the logical level, so no model is resolved.
#[tokio::test]
async fn a_function_filtered_source_feeds_a_semantic_operator() -> Result<()> {
    let ctx = reviews_ctx()?;

    let plan = ctx
        .nousdb_sql("SELECT title, review FROM t WHERE character_length(review) < 200 LIMIT 4")
        .await?
        .semantic_select("no-such-model", "Keep the positive reviews.", ["review"])?
        .into_inner()
        .into_optimized_plan()?;

    let text = plan.display_indent().to_string();
    let selection = text
        .find("SemanticSelection")
        .unwrap_or_else(|| panic!("no semantic selection in:\n{text}"));
    let filter = text
        .find("character_length")
        .unwrap_or_else(|| panic!("no character_length filter in:\n{text}"));
    assert!(
        selection < filter,
        "the SQL filter must stay below the semantic selection:\n{text}"
    );
    Ok(())
}

// Parity with a plain DataFusion session

fn keys<V>(map: &std::collections::HashMap<String, V>) -> BTreeSet<String> {
    map.keys().cloned().collect()
}

/// Every registry `with_default_features()` fills must match a stock
/// `SessionContext`'s, name for name, not merely be non-empty.
#[test]
fn registries_match_a_plain_datafusion_session() -> Result<()> {
    let state = builder().build()?.state();
    let stock = SessionStateBuilder::new_with_default_features().build();

    assert!(!stock.scalar_functions().is_empty(), "reference is vacuous");
    // NousDB adds exactly its semantic marker functions to the stock set.
    let mut expected_scalars = keys(stock.scalar_functions());
    expected_scalars.extend(nousdb::sql::SCALAR_FUNCTION_NAMES.map(String::from));
    assert_eq!(keys(state.scalar_functions()), expected_scalars);
    assert_eq!(
        keys(state.aggregate_functions()),
        keys(stock.aggregate_functions())
    );
    assert_eq!(
        keys(state.window_functions()),
        keys(stock.window_functions())
    );
    assert_eq!(
        keys(state.higher_order_functions()),
        keys(stock.higher_order_functions())
    );
    // ...and the hidden table function that resolves `EXTEND(table_name, ...)`.
    let mut expected_table_functions = keys(stock.table_functions());
    expected_table_functions.extend(nousdb::sql::TABLE_FUNCTION_NAMES.map(String::from));
    assert_eq!(keys(state.table_functions()), expected_table_functions);
    assert_eq!(keys(state.table_factories()), keys(stock.table_factories()));

    assert!(!stock.expr_planners().is_empty(), "reference is vacuous");
    assert_eq!(state.expr_planners().len(), stock.expr_planners().len());

    for format in ["csv", "json", "parquet", "arrow"] {
        assert_eq!(
            state.get_file_format_factory(format).is_some(),
            stock.get_file_format_factory(format).is_some(),
            "file format '{format}'"
        );
    }
    assert!(state.get_file_format_factory("csv").is_some());

    // The analyzer chain is DataFusion's, then NousDB's rewrite rule.
    let mut expected_analyzers = analyzer_names(&stock);
    assert!(!expected_analyzers.is_empty(), "reference is vacuous");
    expected_analyzers.push(nousdb::sql::SemanticFunctionRewriteRule::NAME.to_string());
    assert_eq!(analyzer_names(&state), expected_analyzers);
    Ok(())
}

/// Without the SQL functions the session is a stock one: same scalar and
/// table functions, same analyzer rules.
#[test]
fn without_sql_functions_the_session_matches_a_plain_one() -> Result<()> {
    let state = builder().with_sql_functions(false).build()?.state();
    let stock = SessionStateBuilder::new_with_default_features().build();

    assert_eq!(
        keys(state.scalar_functions()),
        keys(stock.scalar_functions())
    );
    assert_eq!(keys(state.table_functions()), keys(stock.table_functions()));
    assert_eq!(analyzer_names(&state), analyzer_names(&stock));
    Ok(())
}

// What the default features must leave alone

/// Default path: DataFusion's logical rules, then NousDB's, in registration
/// order, nothing added, dropped or reordered by installing the features.
#[test]
fn the_default_logical_chain_is_datafusion_then_nousdb() -> Result<()> {
    let state = builder().build()?.state();

    let mut expected = optimizer_names(&SessionStateBuilder::new().build());
    assert!(!expected.is_empty(), "reference is vacuous");
    expected.push(SemanticRewritingRule.name().to_string());
    expected.push(TokenRepartitionInsertionRule::default().name().to_string());
    expected.push(BatchSizeOptimizationRule::default().name().to_string());

    assert_eq!(optimizer_names(&state), expected);
    Ok(())
}

/// `with_optimizer_rules` replaces ALL logical rules. The default features
/// must not smuggle DataFusion's back in, an empty list stays empty.
#[test]
fn custom_optimizer_rules_still_replace_every_logical_rule() -> Result<()> {
    let none = builder().with_optimizer_rules(vec![]).build()?.state();
    assert_eq!(optimizer_names(&none), Vec::<String>::new());

    let one = builder()
        .with_optimizer_rules(vec![Arc::new(SemanticRewritingRule)])
        .build()?
        .state();
    assert_eq!(
        optimizer_names(&one),
        vec![SemanticRewritingRule.name().to_string()]
    );

    // The functions are there all the same: rules and registries are
    // independent.
    assert!(!none.scalar_functions().is_empty());
    assert!(!one.aggregate_functions().is_empty());
    Ok(())
}

/// With certification on: `CertifyBatchSizeRule` is the LAST physical rule,
/// after DataFusion's untouched defaults, and token repartitioning is stripped
/// from the logical chain. Off: the physical chain is exactly DataFusion's.
#[test]
fn certification_appends_its_physical_rule_after_the_defaults() -> Result<()> {
    let stock_physical = physical_optimizer_names(&SessionStateBuilder::new().build());
    assert!(!stock_physical.is_empty(), "reference is vacuous");

    let plain = builder().build()?.state();
    assert_eq!(physical_optimizer_names(&plain), stock_physical);

    let certified = builder()
        .with_quality_certification(QualityCertConfig::default())
        .build()?
        .state();
    let certify_name = PhysicalOptimizerRule::name(&CertifyBatchSizeRule::new(
        QualityCertConfig::default(),
        Arc::new(CertificateCache::default()),
    ))
    .to_string();
    let mut expected = stock_physical;
    expected.push(certify_name);
    assert_eq!(physical_optimizer_names(&certified), expected);

    let repartition = TokenRepartitionInsertionRule::default().name().to_string();
    assert!(optimizer_names(&plain).contains(&repartition));
    assert!(!optimizer_names(&certified).contains(&repartition));
    assert!(!certified.scalar_functions().is_empty());
    Ok(())
}

// The configured chain: `with_semantic_optimizer`

/// Everything the NousDB part of the chain registered, after asserting that
/// DataFusion's own rules come first and untouched. Pinning the tail keeps
/// these tests independent of which rules a DataFusion upgrade adds.
fn nousdb_rule_names(state: &SessionState) -> Vec<String> {
    let defaults = optimizer_names(&SessionStateBuilder::new().build());
    assert!(!defaults.is_empty(), "reference is vacuous");
    let all = optimizer_names(state);
    assert_eq!(
        all[..defaults.len()],
        defaults[..],
        "DataFusion's rules must come first, untouched"
    );
    all[defaults.len()..].to_vec()
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn certification() -> QualityCertConfig {
    QualityCertConfig::default()
}

/// The expected default chains, hard-coded. Not calling `with_semantic_optimizer`,
/// and passing `SemanticOptimizerConfig::default()`, must both reproduce them
/// exactly.
#[test]
fn the_default_config_reproduces_the_pre_existing_chains() -> Result<()> {
    let catalog = tempfile::tempdir().expect("tempdir");
    let default_cfg = SemanticOptimizerConfig::default;

    // (a) no catalog
    let expected = names(&[
        "SemanticRewritingRule",
        "TokenRepartitionInsertionRule",
        "BatchSizeOptimizationRule",
    ]);
    assert_eq!(nousdb_rule_names(&builder().build()?.state()), expected);
    let configured = builder().with_semantic_optimizer(default_cfg()).build()?;
    assert_eq!(nousdb_rule_names(&configured.state()), expected);

    // (b) a catalog (an empty directory loads as an empty catalog; the rule is
    // still registered)
    let expected = names(&[
        "SemanticRewritingRule",
        "HistogramInjectionRule",
        "TokenRepartitionInsertionRule",
        "BatchSizeOptimizationRule",
    ]);
    let plain = builder().with_histogram_catalog(catalog.path()).build()?;
    assert_eq!(nousdb_rule_names(&plain.state()), expected);
    let configured = builder()
        .with_histogram_catalog(catalog.path())
        .with_semantic_optimizer(default_cfg())
        .build()?;
    assert_eq!(nousdb_rule_names(&configured.state()), expected);

    // (c) certification: no token repartitioning
    let expected = names(&["SemanticRewritingRule", "BatchSizeOptimizationRule"]);
    let plain = builder()
        .with_quality_certification(certification())
        .build()?;
    assert_eq!(nousdb_rule_names(&plain.state()), expected);
    let configured = builder()
        .with_quality_certification(certification())
        .with_semantic_optimizer(default_cfg())
        .build()?;
    assert_eq!(nousdb_rule_names(&configured.state()), expected);

    // (d) both: histogram injection stays, repartitioning goes
    let expected = names(&[
        "SemanticRewritingRule",
        "HistogramInjectionRule",
        "BatchSizeOptimizationRule",
    ]);
    let plain = builder()
        .with_histogram_catalog(catalog.path())
        .with_quality_certification(certification())
        .build()?;
    assert_eq!(nousdb_rule_names(&plain.state()), expected);
    let configured = builder()
        .with_histogram_catalog(catalog.path())
        .with_quality_certification(certification())
        .with_semantic_optimizer(default_cfg())
        .build()?;
    assert_eq!(nousdb_rule_names(&configured.state()), expected);
    Ok(())
}

/// `Off` / `Forced` / `Fixed` / `rewrites: false` each change exactly the entry
/// they name, in place.
#[test]
fn each_choice_changes_exactly_its_own_entry() -> Result<()> {
    let tail = |cfg: SemanticOptimizerConfig| -> Result<Vec<String>> {
        Ok(nousdb_rule_names(
            &builder().with_semantic_optimizer(cfg).build()?.state(),
        ))
    };

    let no_rewrites = SemanticOptimizerConfig {
        rewrites: false,
        ..Default::default()
    };
    assert_eq!(
        tail(no_rewrites)?,
        names(&["TokenRepartitionInsertionRule", "BatchSizeOptimizationRule"])
    );

    let repartition_off = SemanticOptimizerConfig {
        token_repartition: TokenRepartitionChoice::Off,
        ..Default::default()
    };
    assert_eq!(
        tail(repartition_off)?,
        names(&["SemanticRewritingRule", "BatchSizeOptimizationRule"])
    );

    let repartition_forced = SemanticOptimizerConfig {
        token_repartition: TokenRepartitionChoice::Forced {
            partitions: 4,
            strategy: PartitionStrategy::EqualRowCounts,
        },
        ..Default::default()
    };
    assert_eq!(
        tail(repartition_forced)?,
        names(&[
            "SemanticRewritingRule",
            "TokenRepartitionInsertionRule",
            "BatchSizeOptimizationRule"
        ])
    );

    let sizing_fixed = SemanticOptimizerConfig {
        batch_sizing: BatchSizingChoice::Fixed {
            batch_size: 16,
            join_blocks: None,
        },
        ..Default::default()
    };
    assert_eq!(
        tail(sizing_fixed)?,
        names(&[
            "SemanticRewritingRule",
            "TokenRepartitionInsertionRule",
            "FixedBatchSizeRule"
        ])
    );

    let sizing_alpha = SemanticOptimizerConfig {
        batch_sizing: BatchSizingChoice::Auto { alpha: Some(0.05) },
        ..Default::default()
    };
    assert_eq!(
        tail(sizing_alpha)?,
        names(&[
            "SemanticRewritingRule",
            "TokenRepartitionInsertionRule",
            "BatchSizeOptimizationRule"
        ])
    );

    // Nothing but fixed sizing.
    let bare = SemanticOptimizerConfig {
        rewrites: false,
        token_repartition: TokenRepartitionChoice::Off,
        batch_sizing: BatchSizingChoice::Fixed {
            batch_size: 5,
            join_blocks: Some((1, 5)),
        },
    };
    assert_eq!(tail(bare)?, names(&["FixedBatchSizeRule"]));
    Ok(())
}

/// Certification removes token repartitioning even when the config forces it,
/// while the rest of the configured chain survives.
#[test]
fn certification_overrides_a_forced_repartition_choice() -> Result<()> {
    let cfg = SemanticOptimizerConfig {
        token_repartition: TokenRepartitionChoice::Forced {
            partitions: 4,
            strategy: PartitionStrategy::OptimalTokenBalance,
        },
        batch_sizing: BatchSizingChoice::Fixed {
            batch_size: 8,
            join_blocks: None,
        },
        ..Default::default()
    };
    let ctx = builder()
        .with_quality_certification(certification())
        .with_semantic_optimizer(cfg)
        .build()?;
    assert_eq!(
        nousdb_rule_names(&ctx.state()),
        names(&["SemanticRewritingRule", "FixedBatchSizeRule"])
    );
    Ok(())
}

/// `with_optimizer_rules` replaces the whole chain and `with_semantic_optimizer`
/// configures it; asking for both is an error naming both, in either order.
#[test]
fn custom_rules_and_a_semantic_config_cannot_be_combined() {
    let custom_first = builder()
        .with_optimizer_rules(vec![Arc::new(SemanticRewritingRule)])
        .with_semantic_optimizer(SemanticOptimizerConfig::default())
        .build();
    let config_first = builder()
        .with_semantic_optimizer(SemanticOptimizerConfig::default())
        .with_optimizer_rules(vec![])
        .build();

    for result in [custom_first, config_first] {
        match result.err().expect("combining the two knobs must fail") {
            DataFusionError::Configuration(msg) => {
                assert!(msg.contains("with_optimizer_rules"), "{msg}");
                assert!(msg.contains("with_semantic_optimizer"), "{msg}");
            }
            other => panic!("expected a Configuration error, got {other}"),
        }
    }
}

fn find_node<T: Clone + 'static>(plan: &LogicalPlan) -> Option<T> {
    if let LogicalPlan::Extension(ext) = plan
        && let Some(node) = ext.node.as_any().downcast_ref::<T>()
    {
        return Some(node.clone());
    }
    plan.inputs().into_iter().find_map(find_node::<T>)
}

fn contains_token_repartition(plan: &LogicalPlan) -> bool {
    plan.display_indent()
        .to_string()
        .contains("TokenRepartition")
}

/// The sizes the configured chain stamps are the ones the optimized plan
/// carries: extend/selection get `batch_size`, joins the square default or the
/// explicit geometry.
#[tokio::test]
async fn fixed_sizing_reaches_the_optimized_plan() -> Result<()> {
    async fn optimized(cfg: SemanticOptimizerConfig, join: bool) -> Result<LogicalPlan> {
        let ctx = builder().with_semantic_optimizer(cfg).build()?;
        register_reviews(&ctx)?;
        let left = ctx.nousdb_table("t").await?;
        let df = if join {
            let right = ctx.nousdb_table("t").await?;
            left.semantic_join(right, ["review"], ["review"], "no-such-model", "Same?")?
        } else {
            left.semantic_select("no-such-model", "Keep positive reviews.", ["review"])?
        };
        df.into_inner().into_optimized_plan()
    }
    let fixed = |batch_size, join_blocks| SemanticOptimizerConfig {
        batch_sizing: BatchSizingChoice::Fixed {
            batch_size,
            join_blocks,
        },
        ..Default::default()
    };

    let plan = optimized(fixed(7, None), false).await?;
    let sel = find_node::<SemanticSelectionNode>(&plan).expect("a selection node");
    assert_eq!(sel.optimal_batch_size, Some(7));

    let plan = optimized(fixed(7, None), true).await?;
    let join = find_node::<SemanticJoinNode>(&plan).expect("a join node");
    assert_eq!(
        (join.left_block_size, join.right_block_size),
        (Some(7), Some(7))
    );

    let plan = optimized(fixed(7, Some((2, 3))), true).await?;
    let join = find_node::<SemanticJoinNode>(&plan).expect("a join node");
    assert_eq!(
        (join.left_block_size, join.right_block_size),
        (Some(2), Some(3))
    );
    Ok(())
}

/// `Forced` pins its own partition count: it inserts a repartition node in a
/// one-partition session, where `Auto` and `Off` insert none.
#[tokio::test]
async fn forced_repartition_inserts_where_auto_and_off_do_not() -> Result<()> {
    async fn optimized(choice: TokenRepartitionChoice) -> Result<LogicalPlan> {
        let ctx = builder()
            .with_semantic_optimizer(SemanticOptimizerConfig {
                token_repartition: choice,
                ..Default::default()
            })
            .build()?;
        register_reviews(&ctx)?;
        ctx.nousdb_table("t")
            .await?
            .semantic_select("no-such-model", "Keep positive reviews.", ["review"])?
            .into_inner()
            .into_optimized_plan()
    }

    let forced = optimized(TokenRepartitionChoice::Forced {
        partitions: 2,
        strategy: PartitionStrategy::OptimalTokenBalance,
    })
    .await?;
    assert!(
        contains_token_repartition(&forced),
        "{}",
        forced.display_indent()
    );
    assert!(!contains_token_repartition(
        &optimized(TokenRepartitionChoice::Auto).await?
    ));
    assert!(!contains_token_repartition(
        &optimized(TokenRepartitionChoice::Off).await?
    ));
    Ok(())
}
