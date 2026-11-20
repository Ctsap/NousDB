//! Adds a model-derived column to a table, then filters on it.
//!
//! ```sh
//! cargo run -p nousdb --example quickstart
//! ```

mod common;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = NousDBContext::new(common::registry().await?)?;
    common::register_data(&ctx).await?;

    ctx.nousdb_table("reviews")
        .await?
        .extend(
            ("sentiment", DataType::Utf8),
            common::model(),
            "Classify the sentiment of the movie review as exactly one of: \
             positive, negative, neutral.",
            ["review"],
        )?
        .filter(col("sentiment").eq(lit("positive")))?
        .select_columns(&["title", "sentiment"])?
        .show()
        .await
}
