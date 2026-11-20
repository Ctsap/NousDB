//! Setup shared by the examples.
//!
//! By default the examples use a local Ollama server. Set `NOUSDB_BASE_URL`
//! to an OpenAI-compatible endpoint (for example `http://localhost:8000/v1`
//! for vLLM) to use that instead, `NOUSDB_MODEL` to pick the model, and
//! `NOUSDB_API_KEY` if the endpoint needs a key.

#![allow(dead_code)]

use std::sync::Arc;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

/// The model the examples use.
pub fn model() -> String {
    std::env::var("NOUSDB_MODEL").unwrap_or_else(|_| "llama3.1:latest".to_string())
}

/// A registry with one provider, chosen from the environment.
pub async fn registry() -> Result<Arc<ModelRegistry>> {
    let registry = Arc::new(ModelRegistry::new());
    match std::env::var("NOUSDB_BASE_URL") {
        Ok(url) => {
            let mut builder = OpenAiCompatibleProvider::builder("endpoint").base_url(url);
            if let Ok(key) = std::env::var("NOUSDB_API_KEY") {
                builder = builder.api_key(key);
            }
            let model = model();
            if ModelCatalog::builtin().get(&model).is_none() {
                builder = builder.model(ModelCard::new(model, 8192).with_max_output_tokens(2048));
            }
            registry.register(builder.build().await?)?;
        }
        Err(_) => registry.register(OllamaProvider::builder().build().await?)?,
    }
    Ok(registry)
}

/// Registers the bundled `reviews` and `people` tables.
pub async fn register_data(ctx: &NousDBContext) -> Result<()> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data");
    ctx.register_csv(
        "reviews",
        &format!("{dir}/reviews.csv"),
        CsvReadOptions::new(),
    )
    .await?;
    ctx.register_csv(
        "people",
        &format!("{dir}/people.csv"),
        CsvReadOptions::new(),
    )
    .await?;
    Ok(())
}
