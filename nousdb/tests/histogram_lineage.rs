//! Histogram injection through aliases, end to end through the default rule
//! chain: a semantic join whose columns are written `alias.column` (a `LIMIT`,
//! then `.alias(..)`, then `select_columns`) must be sized from the registered
//! table's histograms exactly as the same join written with plain column
//! names, instead of falling back to the model card.

use async_trait::async_trait;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::context::SessionConfig;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::ExecutionPlan;

use nousdb::context::{NousDBContext, NousDBContextBuilder};
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::PredictResult;
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::token_estimator::TokenHistogram;
use nousdb_operators::join::{SemanticJoinExec, SemanticJoinNode, create_semantic_join};

use std::sync::Arc;
use tempfile::TempDir;

/// Plans only: serves a card so sizing has real ceilings, never predicts.
#[derive(Debug)]
struct PlanOnlyProvider;

impl PlanOnlyProvider {
    fn card() -> ModelCard {
        ModelCard::new("mock-model", 8192)
            .with_max_output_tokens(2048)
            .with_max_batch_size(32)
    }
}

#[async_trait]
impl ModelProvider for PlanOnlyProvider {
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
        _batch: &RecordBatch,
        _start_row: usize,
        _batch_size: usize,
        _renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        unreachable!("these tests only plan")
    }
}

/// `movies/{title,review}.json`: short titles, long reviews (short titles, long reviews).
fn write_histograms(dir: &TempDir) {
    let table = dir.path().join("movies");
    std::fs::create_dir(&table).unwrap();
    let title: Vec<usize> = (0..200).map(|i| 4 + i % 8).collect();
    let review: Vec<usize> = (0..200).map(|i| 120 + (i * 7) % 240).collect();
    for (column, counts) in [("title", title), ("review", review)] {
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        std::fs::write(
            table.join(format!("{column}.json")),
            serde_json::to_string(&hist).unwrap(),
        )
        .unwrap();
    }
}

async fn context(histograms: Option<&TempDir>) -> Result<NousDBContext> {
    let registry = Arc::new(ModelRegistry::new());
    registry.register_as("mock", Arc::new(PlanOnlyProvider))?;
    let mut builder = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(4));
    if let Some(dir) = histograms {
        builder = builder.with_histogram_catalog(dir.path());
    }
    let ctx = builder.build()?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::Utf8, false),
        Field::new("review", DataType::Utf8, false),
    ]));
    let titles: Vec<String> = (0..40).map(|i| format!("movie {i}")).collect();
    let reviews: Vec<String> = (0..40).map(|i| format!("review of movie {i}")).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(titles)),
            Arc::new(StringArray::from(reviews)),
        ],
    )?;
    ctx.register_table(
        "movies",
        Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
    )?;
    Ok(ctx)
}

/// Two aliased `LIMIT`s of one table, projected, joined on `title` + `review`. `qualify` writes the join columns `alias.column`.
async fn demo_join(ctx: &NousDBContext, qualify: bool) -> Result<LogicalPlan> {
    let side = |alias: &'static str, rows: usize| async move {
        ctx.table("movies")
            .await?
            .limit(0, Some(rows))?
            .alias(alias)?
            .select_columns(&["title", "review"])
    };
    let drama = side("drama_movies", 20).await?;
    let comedy = side("comedy_movies", 10).await?;
    let columns = |alias: &str| -> Vec<String> {
        ["title", "review"]
            .iter()
            .map(|c| {
                if qualify {
                    format!("{alias}.{c}")
                } else {
                    c.to_string()
                }
            })
            .collect()
    };
    create_semantic_join(
        drama.logical_plan(),
        comedy.logical_plan(),
        columns("drama_movies"),
        columns("comedy_movies"),
        "The reviews discuss similar themes.",
        "mock-model",
    )
}

fn find_join_node(plan: &LogicalPlan) -> Option<SemanticJoinNode> {
    if let LogicalPlan::Extension(ext) = plan
        && let Some(join) = ext.node.as_any().downcast_ref::<SemanticJoinNode>()
    {
        return Some(join.clone());
    }
    plan.inputs().into_iter().find_map(find_join_node)
}

fn find_join_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<&SemanticJoinExec> {
    if let Some(exec) = plan.downcast_ref::<SemanticJoinExec>() {
        return Some(exec);
    }
    plan.children().into_iter().find_map(find_join_exec)
}

/// The optimized join node and the executed block geometry.
async fn plan_join(
    ctx: &NousDBContext,
    qualify: bool,
) -> Result<(SemanticJoinNode, Option<(usize, usize)>)> {
    let plan = demo_join(ctx, qualify).await?;
    let node = find_join_node(&ctx.state().optimize(&plan)?).expect("SemanticJoinNode");
    let physical = ctx.state().create_physical_plan(&plan).await?;
    let geometry = find_join_exec(&physical)
        .expect("SemanticJoinExec")
        .planned_block_sizes();
    Ok((node, geometry))
}

// `HistogramInjectionRule` resolves the card with `block_in_place`.
#[tokio::test(flavor = "multi_thread")]
async fn an_alias_qualified_join_is_sized_from_the_tables_histograms() -> Result<()> {
    let dir = TempDir::new().unwrap();
    write_histograms(&dir);
    let ctx = context(Some(&dir)).await?;

    let (qualified, qualified_geometry) = plan_join(&ctx, true).await?;
    assert!(
        qualified.token_histogram.is_some() && qualified.left_token_histogram.is_some(),
        "both sides' histograms must be found through the aliases"
    );
    assert!(qualified.join_batch_budget.is_some());
    let geometry = qualified_geometry.expect("the optimizer must plan block sizes");
    assert_eq!(
        (qualified.left_block_size, qualified.right_block_size),
        (Some(geometry.0), Some(geometry.1))
    );

    // Exactly what the same join gets when its columns are written bare.
    let (bare, bare_geometry) = plan_join(&ctx, false).await?;
    assert_eq!(bare_geometry, Some(geometry));
    let json = |h: &Option<TokenHistogram>| serde_json::to_string(h).unwrap();
    assert_eq!(
        json(&qualified.token_histogram),
        json(&bare.token_histogram)
    );
    assert_eq!(
        json(&qualified.left_token_histogram),
        json(&bare.left_token_histogram)
    );
    Ok(())
}

/// The control: without a catalog nothing is injected and the join keeps no
/// plan-time geometry (it sizes at runtime from the card).
#[tokio::test(flavor = "multi_thread")]
async fn without_a_catalog_the_join_has_no_plan_time_geometry() -> Result<()> {
    let ctx = context(None).await?;
    let (node, geometry) = plan_join(&ctx, true).await?;
    assert!(node.token_histogram.is_none());
    assert_eq!(geometry, None);
    Ok(())
}
