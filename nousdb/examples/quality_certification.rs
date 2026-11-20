//! Certifies the batch size of an extend before running it, and reports what
//! happened with `EXPLAIN ANALYZE`.
//!
//! ```sh
//! RUST_LOG=info cargo run -p nousdb --example quality_certification
//! ```

mod common;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = NousDBContextBuilder::new(common::registry().await?)
        .with_quality_certification(QualityCertConfig {
            eta: 0.2,
            delta: 0.1,
            tolerated_disagreements: Some(1),
            ..QualityCertConfig::default()
        })
        .build()?;
    common::register_data(&ctx).await?;

    let df = ctx.nousdb_table("reviews").await?.extend(
        ("sentiment", DataType::Utf8),
        common::model(),
        "Classify the sentiment of the movie review as exactly one of: \
         positive, negative, neutral.",
        ["review"],
    )?;

    df.explain(false, true).await?.show().await
}
