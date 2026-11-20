//! Generates a small table from a description, then adds a column to it.
//!
//! ```sh
//! cargo run -p nousdb --example generate
//! ```

mod common;

use nousdb::datafusion::error::Result;
use nousdb::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = NousDBContext::new(common::registry().await?)?;
    let model = common::model();

    ctx.generate(
        vec![("name", DataType::Utf8), ("founded", DataType::Int64)],
        model.clone(),
        "Fictional early-stage technology startups founded between 2015 and 2024.",
        8,
    )?
    .extend(
        ("sector", DataType::Utf8),
        model,
        "The industry sector this company most likely belongs to, in one or two words.",
        ["name"],
    )?
    .sort(vec![col("founded").sort(true, false)])?
    .show()
    .await
}
