//! Joins people to the reviews that mention them.
//!
//! ```sh
//! cargo run -p nousdb --example semantic_join
//! ```

mod common;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = NousDBContext::new(common::registry().await?)?;
    common::register_data(&ctx).await?;

    // The model sees every (person, review) pair, so filter both sides first.
    let actors = ctx
        .nousdb_sql("SELECT name FROM people WHERE role = 'actor'")
        .await?;
    let recent = ctx
        .nousdb_sql("SELECT title, review FROM reviews WHERE year >= 2021")
        .await?;

    actors
        .semantic_join(
            recent,
            ["name"],
            ["review"],
            common::model(),
            "The review mentions this person by name.",
        )?
        .select_columns(&["name", "title"])?
        .show()
        .await
}
