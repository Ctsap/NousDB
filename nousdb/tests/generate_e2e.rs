//! The Generate operator through the full NousDB stack: `NousDBContext::generate`
//! builds a leaf logical node, the physical planner resolves it to a
//! `GenerateExec` sized from the model card, DataFusion keeps relational
//! operators above it, and its rows feed the other semantic operators.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::context::SessionConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{col, lit};

use nousdb::context::{NousDBContext, NousDBContextBuilder};
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::providers::schemas::ModelOutput;
use nousdb_operators::generate::GenerateExec;

use serde_json::json;
use std::sync::{Arc, Mutex};

/// Answers by template: generate requests get `startup{i}` founded in
/// `2000 + i`, extend requests a `tagline` derived from `name`, selection
/// requests keep even founding years.
#[derive(Debug, Default)]
struct GenerateE2eMock {
    templates: Mutex<Vec<String>>,
}

impl GenerateE2eMock {
    /// 200 output tokens over 60 per generated (name, founded) row gives 3 rows
    /// per generate request.
    fn card() -> ModelCard {
        ModelCard::new("mock-model", 8192)
            .with_name("mock-model")
            .with_max_output_tokens(200)
            .with_max_batch_size(4096)
    }

    fn templates(&self) -> Vec<String> {
        self.templates.lock().unwrap().clone()
    }
}

#[async_trait]
impl ModelProvider for GenerateE2eMock {
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
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        prompt.render_batch(renderer, batch, start_row, batch_size)?;
        let template = prompt.template_name().to_string();
        self.templates.lock().unwrap().push(template.clone());
        let rows = start_row..start_row + batch_size;

        let outputs = match template.as_str() {
            "generate/user.md" => rows
                .map(|i| {
                    ModelOutput::Structured(json!({
                        "name": format!("startup{i}"),
                        "founded": 2000 + i,
                    }))
                })
                .collect(),
            "extend/user.md" => {
                let names = batch
                    .column_by_name("name")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                rows.map(|i| ModelOutput::Text(format!("tag-{}", names.value(i))))
                    .collect()
            }
            "selection/user.md" => {
                let founded = batch
                    .column_by_name("founded")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                rows.map(|i| ModelOutput::Boolean(founded.value(i) % 2 == 0))
                    .collect()
            }
            other => panic!("unexpected template {other}"),
        };
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(50, 10, 0),
            false,
        ))
    }
}

async fn context(mock: Arc<GenerateE2eMock>) -> Result<NousDBContext> {
    let registry = Arc::new(ModelRegistry::new());
    registry.register_as("mock", mock)?;
    NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build()
}

fn startup_schema() -> Vec<(&'static str, DataType)> {
    vec![("name", DataType::Utf8), ("founded", DataType::Int64)]
}

fn find_exec<T: ExecutionPlan>(plan: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    if let Some(found) = plan.downcast_ref::<T>() {
        return Some(found);
    }
    plan.children().into_iter().find_map(find_exec)
}

fn strings(batches: &[RecordBatch], column: &str) -> Vec<String> {
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column_by_name(column)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone();
            (0..col.len()).map(move |i| col.value(i).to_string())
        })
        .collect();
    out.sort();
    out
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

#[tokio::test(flavor = "multi_thread")]
async fn generate_plans_to_a_card_sized_leaf_exec() -> Result<()> {
    let mock = Arc::new(GenerateE2eMock::default());
    let ctx = context(mock.clone()).await?;
    let df = ctx.generate(startup_schema(), "mock-model", "startups", 12)?;

    let physical = df.clone().into_inner().create_physical_plan().await?;
    let exec = find_exec::<GenerateExec>(&physical).expect("a GenerateExec in the plan");
    assert_eq!(exec.num_rows(), 12);
    assert_eq!(exec.chunk_size(), 3);
    assert_eq!(exec.request_window(), 1);

    let explain = df.explain(false, false).await?.collect().await?;
    let text = format!(
        "{}",
        datafusion::arrow::util::pretty::pretty_format_batches(&explain)?
    );
    assert!(text.contains("NousDB_Generate"), "{text}");
    assert!(text.contains("GenerateExec"), "{text}");
    assert!(text.contains("chunk_size=3"), "{text}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn relational_operators_stay_above_the_leaf() -> Result<()> {
    let mock = Arc::new(GenerateE2eMock::default());
    let ctx = context(mock.clone()).await?;
    let df = ctx.generate(startup_schema(), "mock-model", "startups", 12)?;

    assert_eq!(df.clone().count().await?, 12);

    let recent = df
        .clone()
        .filter(col("founded").gt_eq(lit(2006i64)))?
        .collect()
        .await?;
    let mut expected: Vec<String> = (6..12).map(|i| format!("startup{i}")).collect();
    expected.sort();
    assert_eq!(strings(&recent, "name"), expected);

    let limited = df.limit(0, Some(5))?.collect().await?;
    assert_eq!(total_rows(&limited), 5);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn generated_rows_feed_extend_and_semantic_select() -> Result<()> {
    let mock = Arc::new(GenerateE2eMock::default());
    let ctx = context(mock.clone()).await?;

    let batches = ctx
        .generate(startup_schema(), "mock-model", "startups", 12)?
        .extend(
            ("tagline", DataType::Utf8),
            "mock-model",
            "write a tagline",
            ["name"],
        )?
        .semantic_select("mock-model", "founded in an even year", ["founded"])?
        .collect()
        .await?;

    let mut evens: Vec<usize> = (0..12).filter(|i| i % 2 == 0).collect();
    evens.sort_by_key(|i| format!("startup{i}"));
    assert_eq!(
        strings(&batches, "tagline"),
        evens
            .iter()
            .map(|i| format!("tag-startup{i}"))
            .collect::<Vec<_>>()
    );

    let templates = mock.templates();
    assert_eq!(
        templates
            .iter()
            .filter(|t| t.as_str() == "generate/user.md")
            .count(),
        4,
        "12 rows at 3 per request"
    );
    assert!(templates.iter().any(|t| t == "extend/user.md"));
    assert!(templates.iter().any(|t| t == "selection/user.md"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_generate_arguments_fail_when_the_frame_is_built() -> Result<()> {
    let ctx = context(Arc::new(GenerateE2eMock::default())).await?;

    assert!(
        ctx.generate(("n", DataType::Int32), "mock-model", "x", 3)
            .is_err()
    );
    assert!(
        ctx.generate(startup_schema(), "mock-model", "x", 0)
            .is_err()
    );

    let wrong_examples = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(vec!["Acme"]))],
    )?;
    assert!(
        ctx.generate_with_examples(startup_schema(), "mock-model", "x", 3, &[wrong_examples])
            .is_err()
    );
    Ok(())
}
