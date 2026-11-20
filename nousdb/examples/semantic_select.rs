//! Keeps the rows a model accepts, after an ordinary SQL filter.
//!
//! ```sh
//! cargo run -p nousdb --example semantic_select
//! ```

mod common;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = NousDBContext::new(common::registry().await?)?;
    common::register_data(&ctx).await?;

    let comedies = ctx
        .nousdb_sql("SELECT title, year, review FROM reviews WHERE year >= 2019")
        .await?
        .semantic_select(
            common::model(),
            "The review describes the film as a comedy or says it is funny.",
            ["title", "review"],
        )?
        .sort(vec![col("year").sort(true, false)])?;

    comedies.select_columns(&["title", "year"])?.show().await
}
