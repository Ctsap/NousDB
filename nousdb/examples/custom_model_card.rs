//! Describes a model with a `ModelCard`, in code and in TOML, and shows how
//! the card shapes the plan.
//!
//! ```sh
//! cargo run -p nousdb --example custom_model_card
//! ```

mod common;

use std::sync::Arc;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let model = common::model();

    // A card in code: a larger context window than Ollama's default, a firm
    // output budget, and at most 8 rows per request.
    let card = ModelCard::new(model.clone(), 16_384)
        .with_max_output_tokens(2_048)
        .with_quality_max_batch_size(8)
        .with_max_concurrent_requests(2);

    // The same kind of description in TOML, merged over the built-in catalog.
    let catalog = ModelCatalog::builtin().merge(ModelCatalog::from_toml_str(
        r#"
        [[models]]
        id = "my-org/my-finetune"
        context_window = 32768
        max_output_tokens = 4096
        pricing = { input_per_1k_tokens = 0.0002, output_per_1k_tokens = 0.0006 }
        "#,
    )?);
    let finetune = catalog.get("my-org/my-finetune").expect("just added");
    println!(
        "my-org/my-finetune: {} input tokens per request, ${:.4} for 10k in / 1k out",
        finetune.effective_input_budget(),
        finetune.estimate_cost(10_000, 1_000),
    );

    let registry = Arc::new(ModelRegistry::new());
    match std::env::var("NOUSDB_BASE_URL") {
        Ok(url) => registry.register(
            OpenAiCompatibleProvider::builder("endpoint")
                .base_url(url)
                .catalog(catalog)
                .model(card)
                .build()
                .await?,
        )?,
        Err(_) => registry.register(OllamaProvider::builder().model(card).build().await?)?,
    }

    let ctx = NousDBContext::new(registry)?;
    common::register_data(&ctx).await?;
    let df = ctx.nousdb_table("reviews").await?.semantic_select(
        model,
        "The review praises an actor's performance.",
        ["review"],
    )?;

    // The plan shows the batch size the card allowed.
    df.clone().explain(false, false).await?.show().await?;
    df.select_columns(&["title"])?.show().await
}
