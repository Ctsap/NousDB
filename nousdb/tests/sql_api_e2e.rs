//! The SQL functions (`EXTEND`, `SEM_FILTER`, `SEM_JOIN`, `GENERATE`) through
//! the full NousDB stack: a real `NousDBContext` (default builder, so the
//! analyzer rule, the relation planner, the semantic optimizer rules and the
//! physical planner are all live), a rule-based mock model, real planning and
//! execution, no network.
//!
//! The mock answers by `prompt.template_name()`:
//! - selection: true iff the row text contains the keyword the instruction
//!   names after `contains:` (default `good`);
//! - extend: by output type (Utf8 `positive`/`negative` on `good`, Int64 the
//!   text length, Boolean contains the keyword, Float64 0.5);
//! - join: true iff the left text contains the right text;
//! - generate: `"<column><i>"` / `i` for the i-th generated row.
//!
//! "Row text" is the values of the operator's input columns joined by a space.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::error::Result;
use datafusion::execution::context::SessionConfig;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::displayable;

use nousdb::context::{NousDBContext, NousDBContextBuilder};
use nousdb::operators::token_repartition::config::PartitionStrategy;
use nousdb::{SemanticOptimizerConfig, TokenRepartitionChoice};
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::registry::ModelRegistry;
use nousdb_core::providers::schemas::ModelOutput;

use serde_json::{Map, Value, json};

const MODEL: &str = "mock-model";

// ---------------------------------------------------------------------------
// The mock model
// ---------------------------------------------------------------------------

/// One request the mock received.
#[derive(Debug, Clone)]
struct Call {
    template: String,
    /// Rows (pairs, for a join) in the request.
    rows: usize,
    /// The system and user prompt as rendered.
    rendered: String,
}

#[derive(Debug, Default)]
struct SqlMock {
    calls: Mutex<Vec<Call>>,
}

impl SqlMock {
    fn card() -> ModelCard {
        ModelCard::new(MODEL, 8192)
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    /// Total rows (pairs) the model saw in requests of `template`.
    fn rows_seen(&self, template: &str) -> usize {
        self.calls()
            .iter()
            .filter(|c| c.template == template)
            .map(|c| c.rows)
            .sum()
    }

    fn requests(&self, template: &str) -> usize {
        self.calls()
            .iter()
            .filter(|c| c.template == template)
            .count()
    }
}

/// The keyword an instruction names (`contains: movie`), else `good`.
fn keyword(instruction: &str) -> String {
    match instruction.find("contains:") {
        Some(i) => instruction[i + "contains:".len()..].trim().to_lowercase(),
        None => "good".to_string(),
    }
}

/// The text of each row of a selection / extend request: the values of its
/// input columns, via the renderer (the batch may carry more columns).
fn row_texts(
    renderer: &(dyn BatchRenderer + Send + Sync),
    batch: &RecordBatch,
    start_row: usize,
    batch_size: usize,
) -> Vec<String> {
    let context = renderer
        .prepare_context(batch, start_row, batch_size)
        .expect("render the batch");
    let rows = context.get("batch_data").expect("batch_data");
    rows.try_iter()
        .expect("batch_data is a sequence")
        .map(|row| {
            row.try_iter()
                .expect("a row is a map")
                .map(|key| {
                    let value = row.get_item(&key).expect("a value");
                    value
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| value.to_string())
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[async_trait]
impl ModelProvider for SqlMock {
    fn name(&self) -> &str {
        "mock"
    }

    fn models(&self) -> Vec<ModelCard> {
        vec![Self::card()]
    }

    fn model(&self, model_id: &str) -> Option<ModelCard> {
        (model_id == MODEL).then(Self::card)
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
        let rendered = format!(
            "{}\n{}",
            prompt.render_system()?.unwrap_or_default(),
            prompt.render_batch(renderer, batch, start_row, batch_size)?
        );
        let template = prompt.template_name().to_string();
        self.calls.lock().unwrap().push(Call {
            template: template.clone(),
            rows: batch_size,
            rendered,
        });

        let outputs: Vec<ModelOutput> = match template.as_str() {
            "selection/user.md" => {
                let kw = keyword(prompt.user_instruction());
                row_texts(renderer, batch, start_row, batch_size)
                    .iter()
                    .map(|t| ModelOutput::Boolean(t.to_lowercase().contains(&kw)))
                    .collect()
            }
            "extend/user.md" => {
                let kw = keyword(prompt.user_instruction());
                let schema = prompt.output_schema().clone();
                row_texts(renderer, batch, start_row, batch_size)
                    .iter()
                    .map(|text| {
                        let mut object = Map::new();
                        for field in schema.fields() {
                            let value = match field.data_type() {
                                DataType::Utf8 => {
                                    json!(if text.contains("good") {
                                        "positive"
                                    } else {
                                        "negative"
                                    })
                                }
                                DataType::Int64 => json!(text.len() as i64),
                                DataType::Boolean => json!(text.to_lowercase().contains(&kw)),
                                DataType::Float64 => json!(0.5),
                                other => panic!("the mock cannot answer {other}"),
                            };
                            object.insert(field.name().clone(), value);
                        }
                        ModelOutput::Structured(Value::Object(object))
                    })
                    .collect()
            }
            "join/user.md" => {
                // One key column per side: the pair batch is left columns
                // then right columns.
                assert_eq!(batch.num_columns(), 2, "the mock joins one column a side");
                (start_row..start_row + batch_size)
                    .map(|i| {
                        let left = array_value_to_string(batch.column(0), i).unwrap();
                        let right = array_value_to_string(batch.column(1), i).unwrap();
                        ModelOutput::Boolean(left.to_lowercase().contains(&right.to_lowercase()))
                    })
                    .collect()
            }
            "generate/user.md" => {
                let schema = prompt.output_schema().clone();
                (start_row..start_row + batch_size)
                    .map(|i| {
                        let mut object = Map::new();
                        for field in schema.fields() {
                            let value = match field.data_type() {
                                DataType::Utf8 => json!(format!("{}{i}", field.name())),
                                DataType::Int64 => json!(i as i64),
                                other => panic!("the mock cannot generate {other}"),
                            };
                            object.insert(field.name().clone(), value);
                        }
                        ModelOutput::Structured(Value::Object(object))
                    })
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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn session(mock: &Arc<SqlMock>, sql_functions: bool) -> Result<NousDBContext> {
    let registry = Arc::new(ModelRegistry::new());
    registry.register_as("mock", mock.clone())?;
    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_sql_functions(sql_functions)
        .build()?;

    // reviews(id, review)
    table(
        &ctx,
        "reviews",
        vec![
            ints("id", &[1, 2, 3, 4, 5, 6]),
            strs(
                "review",
                &[
                    "good movie",
                    "bad film",
                    "good plot",
                    "awful",
                    "very good indeed",
                    "meh",
                ],
            ),
        ],
    )?;
    // papers(id, year, title, abstract)
    table(
        &ctx,
        "papers",
        vec![
            ints("id", &[1, 2, 3, 4, 5, 6]),
            ints("year", &[2018, 2019, 2021, 2021, 2022, 2023]),
            strs(
                "title",
                &[
                    "paper one",
                    "paper two",
                    "paper three",
                    "paper four",
                    "paper five",
                    "paper six",
                ],
            ),
            strs(
                "abstract",
                &[
                    "database systems",
                    "sql database engine",
                    "database indexing",
                    "sql optimization",
                    "sql database tuning",
                    "graphics rendering",
                ],
            ),
        ],
    )?;
    // a(id, title), b(sku, name)
    table(
        &ctx,
        "a",
        vec![
            ints("id", &[1, 2, 3]),
            strs("title", &["iphone 12", "galaxy s21", "pixel 6"]),
        ],
    )?;
    table(
        &ctx,
        "b",
        vec![
            ints("sku", &[10, 20, 30, 40]),
            strs("name", &["iphone", "galaxy", "pixel", "nokia"]),
        ],
    )?;
    // l(k, v), r(k, v): the same column names on both sides
    table(
        &ctx,
        "l",
        vec![ints("k", &[1, 2]), strs("v", &["iphone 12", "pixel 6"])],
    )?;
    table(
        &ctx,
        "r",
        vec![ints("k", &[10, 20]), strs("v", &["iphone", "nokia"])],
    )?;
    // mentions(id, review), products(name)
    table(
        &ctx,
        "mentions",
        vec![
            ints("id", &[1, 2, 3, 4, 5, 6, 7, 8]),
            strs(
                "review",
                &[
                    "loved the iphone, good",
                    "the iphone is bad",
                    "pixel is good",
                    "nothing here",
                    "good galaxy phone",
                    "galaxy broke",
                    "good iphone and good galaxy phone",
                    "good iphone",
                ],
            ),
        ],
    )?;
    table(
        &ctx,
        "products",
        vec![strs("name", &["iphone", "galaxy", "pixel"])],
    )?;
    Ok(ctx)
}

fn ints(name: &str, values: &[i64]) -> (Field, ArrayRef) {
    (
        Field::new(name, DataType::Int64, true),
        Arc::new(Int64Array::from(values.to_vec())),
    )
}

fn strs(name: &str, values: &[&str]) -> (Field, ArrayRef) {
    (
        Field::new(name, DataType::Utf8, true),
        Arc::new(StringArray::from(values.to_vec())),
    )
}

fn table(ctx: &NousDBContext, name: &str, columns: Vec<(Field, ArrayRef)>) -> Result<()> {
    let (fields, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    ctx.register_batch(name, batch)?;
    Ok(())
}

/// A query result: the schema's (name, type) list and every row as strings,
/// sorted so a test does not depend on output order.
struct Out {
    fields: Vec<(String, DataType)>,
    rows: Vec<Vec<String>>,
}

impl Out {
    fn names(&self) -> Vec<&str> {
        self.fields.iter().map(|(n, _)| n.as_str()).collect()
    }

    fn types(&self) -> Vec<DataType> {
        self.fields.iter().map(|(_, t)| t.clone()).collect()
    }

    fn column(&self, name: &str) -> Vec<String> {
        let at = self
            .names()
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.names()));
        self.rows.iter().map(|r| r[at].clone()).collect()
    }
}

fn out_of(fields: Vec<(String, DataType)>, batches: &[RecordBatch]) -> Out {
    let mut rows = Vec::new();
    for batch in batches {
        for i in 0..batch.num_rows() {
            rows.push(
                (0..batch.num_columns())
                    .map(|c| array_value_to_string(batch.column(c), i).unwrap())
                    .collect::<Vec<_>>(),
            );
        }
    }
    rows.sort();
    Out { fields, rows }
}

/// Runs `sql` through the stock entry point (`SessionContext::sql`, reached
/// through `Deref`).
async fn run(ctx: &NousDBContext, sql: &str) -> Out {
    let df = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should plan: {e}"));
    let fields = df
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let batches = df
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should run: {e}"));
    out_of(fields, &batches)
}

/// Runs `sql` through `NousDBContext::nousdb_sql`.
async fn run_nousdb(ctx: &NousDBContext, sql: &str) -> Out {
    let df = ctx
        .nousdb_sql(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should plan: {e}"));
    let fields = df
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let batches = df
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should run: {e}"));
    out_of(fields, &batches)
}

/// The error of planning or running `sql`, which must fail somewhere.
async fn failure(ctx: &NousDBContext, sql: &str) -> String {
    match ctx.sql(sql).await {
        Err(e) => e.to_string(),
        Ok(df) => match df.collect().await {
            Err(e) => e.to_string(),
            Ok(batches) => panic!("`{sql}` should fail, but returned {batches:?}"),
        },
    }
}

fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();
    rows.sort();
    rows
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

/// The optimized logical plan of `sql`.
async fn optimized(ctx: &NousDBContext, sql: &str) -> LogicalPlan {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should plan: {e}"))
        .into_optimized_plan()
        .unwrap_or_else(|e| panic!("`{sql}` should optimize: {e}"))
}

fn lines(plan: &LogicalPlan) -> Vec<String> {
    plan.display_indent()
        .to_string()
        .lines()
        .map(str::to_string)
        .collect()
}

fn line_of(lines: &[String], needle: &str) -> usize {
    lines
        .iter()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line with {needle:?} in\n{}", lines.join("\n")))
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The text of `EXPLAIN <sql>` (logical and physical plan).
async fn explain(ctx: &NousDBContext, sql: &str) -> String {
    let batches = ctx
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap_or_else(|e| panic!("EXPLAIN `{sql}` should plan: {e}"))
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

async fn default_session() -> Result<(Arc<SqlMock>, NousDBContext)> {
    let mock = Arc::new(SqlMock::default());
    let ctx = session(&mock, true)?;
    Ok((mock, ctx))
}

// ---------------------------------------------------------------------------
// 1. Scalar EXTEND
// ---------------------------------------------------------------------------

const POSITIVE_ROWS: [&str; 6] = [
    "positive", "negative", "positive", "negative", "positive", "negative",
];

#[tokio::test(flavor = "multi_thread")]
async fn scalar_extend_utf8_default_in_any_case() -> Result<()> {
    let (_, ctx) = default_session().await?;
    for f in ["extend", "EXTEND", "Extend", "eXtEnD"] {
        let q = format!(
            "SELECT *, {f}('classify sentiment', 'mock-model', review) AS sentiment FROM reviews"
        );
        let out = run(&ctx, &q).await;
        assert_eq!(out.names(), ["id", "review", "sentiment"], "{q}");
        assert_eq!(
            out.types(),
            [DataType::Int64, DataType::Utf8, DataType::Utf8],
            "{q}"
        );
        assert_eq!(out.rows.len(), 6, "{q}");
        let by_id: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (r[0].clone(), r[2].clone()))
            .collect();
        let expect: Vec<(String, String)> = (1..=6)
            .map(|i| (i.to_string(), POSITIVE_ROWS[i - 1].to_string()))
            .collect();
        assert_eq!(by_id, expect, "{q}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scalar_extend_typed_named_and_boolean() -> Result<()> {
    let (_, ctx) = default_session().await?;

    // A bare type, named by the alias.
    let out = run(
        &ctx,
        "SELECT id, EXTEND('rate 1 to 5', 'mock-model', review, 'Int64') AS stars FROM reviews",
    )
    .await;
    assert_eq!(out.names(), ["id", "stars"]);
    assert_eq!(out.types(), [DataType::Int64, DataType::Int64]);
    // The mock answers the text length.
    assert_eq!(
        out.rows,
        expected(&[
            &["1", "10"],
            &["2", "8"],
            &["3", "9"],
            &["4", "5"],
            &["5", "16"],
            &["6", "3"],
        ])
    );

    // `name: type`, no alias: the spec's name names the output column.
    let out = run(
        &ctx,
        "SELECT id, EXTEND('rate 1 to 5', 'mock-model', review, 'stars: Int64') FROM reviews",
    )
    .await;
    assert_eq!(out.names()[0], "id");
    assert_eq!(out.names()[1], "stars");
    assert_eq!(out.types(), [DataType::Int64, DataType::Int64]);
    assert_eq!(out.rows.len(), 6);
    assert!(out.rows.contains(&strings(&["1", "10"])));

    // Boolean, in Mixed case: contains the keyword.
    let out = run(
        &ctx,
        "SELECT id, Extend('contains: movie', 'mock-model', review, 'Boolean') AS is_movie FROM reviews",
    )
    .await;
    assert_eq!(out.names(), ["id", "is_movie"]);
    assert_eq!(out.types(), [DataType::Int64, DataType::Boolean]);
    assert_eq!(
        out.rows,
        expected(&[
            &["1", "true"],
            &["2", "false"],
            &["3", "false"],
            &["4", "false"],
            &["5", "false"],
            &["6", "false"],
        ])
    );

    // Named Boolean over the unnamed default column: `name: Boolean`.
    let out = run(
        &ctx,
        "select extend('contains: plot', 'mock-model', review, 'has_plot: Boolean') from reviews",
    )
    .await;
    // Unaliased: named by the spec, as above.
    assert_eq!(out.names()[0], "has_plot");
    assert_eq!(out.types(), [DataType::Boolean]);
    assert_eq!(out.rows.iter().filter(|r| r[0] == "true").count(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scalar_extend_through_nousdb_sql() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let q =
        "SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS sentiment FROM reviews";
    let out = run_nousdb(&ctx, q).await;
    assert_eq!(out.names(), ["id", "review", "sentiment"]);
    assert_eq!(out.rows, run(&ctx, q).await.rows);
    assert_eq!(
        out.column("sentiment")
            .iter()
            .filter(|s| *s == "positive")
            .count(),
        3
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Relation EXTEND
// ---------------------------------------------------------------------------

const TWO_OUTPUTS: &str = "'sentiment: Utf8, n: Int64'";

#[tokio::test(flavor = "multi_thread")]
async fn relation_extend_over_a_table_name() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    // The table is named only as the first argument of EXTEND: DataFusion
    // does not load it for the statement, so the hidden table resolver finds it.
    let q = format!(
        "SELECT * FROM EXTEND(reviews, 'extract both', 'mock-model', review, {TWO_OUTPUTS})"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.names(), ["id", "review", "sentiment", "n"]);
    assert_eq!(
        out.types(),
        [
            DataType::Int64,
            DataType::Utf8,
            DataType::Utf8,
            DataType::Int64
        ]
    );
    assert_eq!(out.rows.len(), 6);
    assert!(
        out.rows
            .contains(&strings(&["1", "good movie", "positive", "10"]))
    );
    assert!(out.rows.contains(&strings(&["6", "meh", "negative", "3"])));
    assert_eq!(mock.rows_seen("extend/user.md"), 6);

    // Lowercase and UPPERCASE name the same function; nousdb_sql too.
    for f in ["extend", "EXTEND"] {
        let q = format!("SELECT * FROM {f}(reviews, 'x', 'mock-model', review, {TWO_OUTPUTS})");
        assert_eq!(run_nousdb(&ctx, &q).await.rows, out.rows, "{q}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn relation_extend_over_a_subquery_with_limit() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM EXTEND((SELECT * FROM reviews ORDER BY id LIMIT 3), 'extract both', \
         'mock-model', review, {TWO_OUTPUTS})"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.names(), ["id", "review", "sentiment", "n"]);
    assert_eq!(
        out.rows,
        expected(&[
            &["1", "good movie", "positive", "10"],
            &["2", "bad film", "negative", "8"],
            &["3", "good plot", "positive", "9"],
        ])
    );
    assert_eq!(mock.rows_seen("extend/user.md"), 3, "only the limited rows");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn relation_extend_over_a_cte() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "WITH r AS (SELECT * FROM reviews WHERE id > 4) \
         SELECT * FROM EXTEND(r, 'extract both', 'mock-model', review, {TWO_OUTPUTS})"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.names(), ["id", "review", "sentiment", "n"]);
    assert_eq!(
        out.rows,
        expected(&[
            &["5", "very good indeed", "positive", "16"],
            &["6", "meh", "negative", "3"],
        ])
    );
    assert_eq!(mock.rows_seen("extend/user.md"), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn where_on_a_derived_column() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let q = format!(
        "SELECT id FROM EXTEND(reviews, 'extract both', 'mock-model', review, {TWO_OUTPUTS}) \
         WHERE sentiment = 'positive' AND n > 9"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.rows, expected(&[&["1"], &["5"]]));

    // The scalar form, filtered on its alias in an outer query.
    let q = "SELECT id FROM (SELECT *, EXTEND('s', 'mock-model', review) AS sentiment FROM reviews) \
             WHERE sentiment = 'negative'";
    assert_eq!(run(&ctx, q).await.rows, expected(&[&["2"], &["4"], &["6"]]));
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. SEM_FILTER
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn sem_filter_alone_in_any_case() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    for f in ["sem_filter", "SEM_FILTER", "Sem_Filter"] {
        let q =
            format!("SELECT * FROM papers WHERE {f}('contains: database', 'mock-model', abstract)");
        let out = run(&ctx, &q).await;
        assert_eq!(out.names(), ["id", "year", "title", "abstract"], "{q}");
        assert_eq!(out.column("id"), strings(&["1", "2", "3", "5"]), "{q}");
    }
    assert_eq!(
        mock.rows_seen("selection/user.md"),
        18,
        "6 rows, three runs"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sem_filter_sees_only_the_rows_a_relational_conjunct_keeps() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    // years 2021, 2021, 2022, 2023 pass `year > 2020`: 4 of 6 rows.
    for q in [
        "SELECT * FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', abstract) AND year > 2020",
        "SELECT * FROM papers WHERE year > 2020 AND SEM_FILTER('contains: database', 'mock-model', abstract)",
    ] {
        let before = mock.rows_seen("selection/user.md");
        let out = run(&ctx, q).await;
        assert_eq!(out.column("id"), strings(&["3", "5"]), "{q}");
        assert_eq!(
            mock.rows_seen("selection/user.md") - before,
            4,
            "the model saw only the rows that passed `year > 2020`: {q}"
        );
    }
    // Through nousdb_sql too.
    let out = run_nousdb(
        &ctx,
        "SELECT id FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', abstract) AND year > 2020",
    )
    .await;
    assert_eq!(out.rows, expected(&[&["3"], &["5"]]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn two_sem_filters_compose() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT id FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', abstract) \
             AND SEM_FILTER('contains: sql', 'mock-model', abstract)";
    let out = run(&ctx, q).await;
    assert_eq!(out.rows, expected(&[&["2"], &["5"]]));
    // The second filter sees only what the first kept (4 or 3 of 6 rows), so
    // the model saw fewer rows than two full passes.
    let seen = mock.rows_seen("selection/user.md");
    assert!((9..12).contains(&seen), "rows seen: {seen}");
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. SEM_JOIN
// ---------------------------------------------------------------------------

const PAIRS: [[&str; 2]; 3] = [["1", "10"], ["2", "20"], ["3", "30"]];

#[tokio::test(flavor = "multi_thread")]
async fn sem_join_in_on_and_in_the_comma_form_give_the_same_rows() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let want = expected(&[&PAIRS[0], &PAIRS[1], &PAIRS[2]]);
    for f in ["sem_join", "SEM_JOIN", "Sem_Join"] {
        let on = format!(
            "SELECT a.id, b.sku FROM a JOIN b ON {f}('same product?', 'mock-model', a.title, b.name)"
        );
        let comma = format!(
            "SELECT a.id, b.sku FROM a, b WHERE {f}('same product?', 'mock-model', a.title, b.name)"
        );
        assert_eq!(run(&ctx, &on).await.rows, want, "{on}");
        assert_eq!(run(&ctx, &comma).await.rows, want, "{comma}");
    }
    // The model judged every pair: 3 x 4 per run.
    assert_eq!(mock.rows_seen("join/user.md"), 6 * 12);

    // SELECT * keeps both tables' columns, in order.
    let out = run_nousdb(
        &ctx,
        "SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'mock-model', a.title, b.name)",
    )
    .await;
    assert_eq!(out.names(), ["id", "title", "sku", "name"]);
    assert!(
        out.rows
            .contains(&strings(&["2", "galaxy s21", "20", "galaxy"]))
    );
    assert_eq!(out.rows.len(), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sem_join_with_table_aliases_and_qualified_columns() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let want = expected(&[&PAIRS[0], &PAIRS[1], &PAIRS[2]]);
    let on = "SELECT x.id, y.sku FROM a AS x JOIN b AS y \
              ON SEM_JOIN('same product?', 'mock-model', x.title, y.name)";
    let comma = "SELECT x.id, y.sku FROM a x, b y \
                 WHERE SEM_JOIN('same product?', 'mock-model', x.title, y.name)";
    assert_eq!(run(&ctx, on).await.rows, want);
    assert_eq!(run(&ctx, comma).await.rows, want);

    // A relational conjunct on one side goes below the join.
    let q = "SELECT x.id, y.sku FROM a AS x JOIN b AS y \
             ON SEM_JOIN('same product?', 'mock-model', x.title, y.name) AND y.sku > 10";
    assert_eq!(run(&ctx, q).await.rows, expected(&[&PAIRS[1], &PAIRS[2]]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sem_join_over_tables_sharing_a_column_name_does_not_exchange_values() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let on = "SELECT * FROM l JOIN r ON SEM_JOIN('same product?', 'mock-model', l.v, r.v)";
    let comma = "SELECT * FROM l, r WHERE SEM_JOIN('same product?', 'mock-model', l.v, r.v)";
    for q in [on, comma] {
        let out = run(&ctx, q).await;
        // `k` and `v` twice: left's, then right's.
        assert_eq!(out.names(), ["k", "v", "k", "v"], "{q}");
        assert_eq!(
            out.rows,
            expected(&[&["1", "iphone 12", "10", "iphone"]]),
            "{q}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sem_join_next_to_an_exists_on_one_side() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    // r.k is 10 or 20: the EXISTS keeps b's skus 10 and 20.
    let exists = "EXISTS (SELECT 1 FROM r WHERE r.k = b.sku)";
    // The same match written relationally (the mock's rule: title contains name).
    let relational =
        format!("SELECT a.id, b.sku FROM a, b WHERE strpos(a.title, b.name) > 0 AND {exists}");
    let want = run(&ctx, &relational).await.rows;
    assert_eq!(want, expected(&[&PAIRS[0], &PAIRS[1]]));
    for q in [
        format!(
            "SELECT a.id, b.sku FROM a, b \
             WHERE SEM_JOIN('same product?', 'mock-model', a.title, b.name) AND {exists}"
        ),
        format!(
            "SELECT a.id, b.sku FROM a JOIN b \
             ON SEM_JOIN('same product?', 'mock-model', a.title, b.name) AND {exists}"
        ),
    ] {
        let before = mock.rows_seen("join/user.md");
        assert_eq!(run(&ctx, &q).await.rows, want, "{q}");
        // The EXISTS filtered b before the join: 3 x 2 pairs, not 3 x 4.
        assert_eq!(mock.rows_seen("join/user.md") - before, 6, "{q}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sem_filter_inside_a_correlated_subquery() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let q = "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.sku = a.id * 10 \
             AND SEM_FILTER('contains: iphone', 'mock-model', b.name))";
    let relational = "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.sku = a.id * 10 \
                      AND b.name LIKE '%iphone%')";
    let want = run(&ctx, relational).await.rows;
    assert_eq!(want, expected(&[&["1"]]));
    assert_eq!(run(&ctx, q).await.rows, want);
    // NOT EXISTS and IN take the same path.
    let q = "SELECT id FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.sku = a.id * 10 \
             AND SEM_FILTER('contains: iphone', 'mock-model', b.name))";
    assert_eq!(run(&ctx, q).await.rows, expected(&[&["2"], &["3"]]));
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. GENERATE
// ---------------------------------------------------------------------------

const STARTUPS: &str = "'name: Utf8, founded: Int64'";

#[tokio::test(flavor = "multi_thread")]
async fn generate_plain_and_with_examples() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    for f in ["generate", "GENERATE", "Generate"] {
        let q = format!("SELECT * FROM {f}('fictional startups', 'mock-model', {STARTUPS}, 5)");
        let out = run(&ctx, &q).await;
        assert_eq!(out.names(), ["name", "founded"], "{q}");
        assert_eq!(out.types(), [DataType::Utf8, DataType::Int64], "{q}");
        assert_eq!(
            out.rows,
            expected(&[
                &["name0", "0"],
                &["name1", "1"],
                &["name2", "2"],
                &["name3", "3"],
                &["name4", "4"],
            ]),
            "{q}"
        );
    }
    assert!(
        mock.calls().iter().all(|c| !c.rendered.contains("Acme")),
        "no examples were given"
    );

    // Few-shot examples: one JSON object per line.
    let q = format!(
        "SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 5, \
         '{{\"name\": \"Acme\", \"founded\": 1999}}\n{{\"name\": \"Zeta\"}}')"
    );
    let out = run_nousdb(&ctx, &q).await;
    assert_eq!(out.rows.len(), 5);
    let with_examples: Vec<Call> = mock
        .calls()
        .into_iter()
        .filter(|c| c.rendered.contains("Acme"))
        .collect();
    assert!(!with_examples.is_empty(), "the examples reach the prompt");
    assert!(with_examples.iter().all(|c| c.rendered.contains("Zeta")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn generate_with_an_alias() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let q = format!(
        "SELECT g.name FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 6) AS g \
         WHERE g.founded >= 4"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.names(), ["name"]);
    assert_eq!(out.rows, expected(&[&["name4"], &["name5"]]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn generate_feeds_extend_and_sem_filter_in_one_statement() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM (SELECT *, EXTEND('tag it', 'mock-model', name) AS tag \
         FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 6)) \
         WHERE SEM_FILTER('contains: 3', 'mock-model', name)"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.names(), ["name", "founded", "tag"]);
    assert_eq!(out.rows, expected(&[&["name3", "3", "negative"]]));
    assert!(mock.requests("generate/user.md") >= 1);
    // The selection was pushed below the extend, so only one row was tagged.
    assert_eq!(mock.rows_seen("extend/user.md"), 1);
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. A combined pipeline
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn extend_filter_join_and_aggregate_in_one_query() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT p.name, COUNT(*) AS n \
             FROM (SELECT * FROM \
                     (SELECT *, EXTEND('sentiment', 'mock-model', review) AS sentiment FROM mentions) \
                   WHERE sentiment = 'positive' \
                     AND SEM_FILTER('contains: phone', 'mock-model', review)) m \
             JOIN products p ON SEM_JOIN('mentions the product', 'mock-model', m.review, p.name) \
             GROUP BY p.name ORDER BY p.name";
    let out = run(&ctx, q).await;
    assert_eq!(out.names(), ["name", "n"]);
    assert_eq!(out.types(), [DataType::Utf8, DataType::Int64]);
    // Positive and about a phone: reviews 1, 5, 7, 8; iphone in 1, 7, 8 and
    // galaxy in 5, 7.
    assert_eq!(out.rows, expected(&[&["galaxy", "2"], &["iphone", "3"]]));
    assert_eq!(mock.rows_seen("extend/user.md"), 8);
    assert_eq!(mock.rows_seen("selection/user.md"), 5, "the positive rows");
    assert_eq!(
        mock.rows_seen("join/user.md"),
        4 * 3,
        "4 reviews x 3 products"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. Equivalence with the DataFrame API
// ---------------------------------------------------------------------------

fn assert_same_plan(what: &str, sql: &LogicalPlan, api: &LogicalPlan) {
    assert!(
        sql == api,
        "{what}: the optimized plans differ\nSQL:\n{}\nDataFrame API:\n{}",
        sql.display_indent_schema(),
        api.display_indent_schema()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_and_dataframe_extend_are_the_same_plan_and_rows() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let sql =
        "SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS sentiment FROM reviews";
    let api = ctx.nousdb_table("reviews").await?.extend(
        ("sentiment", DataType::Utf8),
        MODEL,
        "classify sentiment",
        ["review"],
    )?;
    let sql_plan = optimized(&ctx, sql).await;
    let api_plan = api.clone().into_inner().into_optimized_plan()?;
    assert_same_plan("extend", &sql_plan, &api_plan);

    let sql_rows = run(&ctx, sql).await;
    let api_batches = api.collect().await?;
    let api_rows = out_of(sql_rows.fields.clone(), &api_batches);
    assert_eq!(sql_rows.rows, api_rows.rows);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_and_dataframe_semantic_select_are_the_same_plan_and_rows() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let sql = "SELECT * FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', abstract)";
    let api = ctx.nousdb_table("papers").await?.semantic_select(
        MODEL,
        "contains: database",
        ["abstract"],
    )?;
    let sql_plan = optimized(&ctx, sql).await;
    let api_plan = api.clone().into_inner().into_optimized_plan()?;
    assert_same_plan("semantic_select", &sql_plan, &api_plan);

    let sql_rows = run(&ctx, sql).await;
    let api_rows = out_of(sql_rows.fields.clone(), &api.collect().await?);
    assert_eq!(sql_rows.rows, api_rows.rows);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_and_dataframe_semantic_join_are_the_same_plan_and_rows() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let sql = "SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'mock-model', a.title, b.name)";
    let api = ctx.nousdb_table("a").await?.semantic_join(
        ctx.nousdb_table("b").await?,
        ["title"],
        ["name"],
        MODEL,
        "same product?",
    )?;
    let sql_plan = optimized(&ctx, sql).await;
    let api_plan = api.clone().into_inner().into_optimized_plan()?;
    assert_same_plan("semantic_join", &sql_plan, &api_plan);

    let sql_rows = run(&ctx, sql).await;
    let api_rows = out_of(sql_rows.fields.clone(), &api.collect().await?);
    assert_eq!(sql_rows.rows, api_rows.rows);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_and_dataframe_extend_then_select_are_the_same_plan_and_rows() -> Result<()> {
    let (_, ctx) = default_session().await?;
    // The filter reads the derived column, so neither pushdown (A) nor fusion
    // (E, the projection keeps the derived column) applies to either plan.
    let sql = "SELECT * FROM (SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS sentiment \
               FROM reviews) WHERE SEM_FILTER('contains: positive', 'mock-model', sentiment)";
    let api = ctx
        .nousdb_table("reviews")
        .await?
        .extend(
            ("sentiment", DataType::Utf8),
            MODEL,
            "classify sentiment",
            ["review"],
        )?
        .semantic_select(MODEL, "contains: positive", ["sentiment"])?;
    let sql_plan = optimized(&ctx, sql).await;
    let api_plan = api.clone().into_inner().into_optimized_plan()?;
    assert_same_plan("extend -> select", &sql_plan, &api_plan);

    let sql_rows = run(&ctx, sql).await;
    let api_rows = out_of(sql_rows.fields.clone(), &api.collect().await?);
    assert_eq!(sql_rows.rows, api_rows.rows);
    assert_eq!(sql_rows.column("id"), strings(&["1", "3", "5"]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_and_dataframe_generate_are_the_same_plan_and_rows() -> Result<()> {
    let (_, ctx) = default_session().await?;
    let sql = format!("SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 4)");
    let api = ctx.generate(
        vec![("name", DataType::Utf8), ("founded", DataType::Int64)],
        MODEL,
        "fictional startups",
        4,
    )?;
    let sql_plan = optimized(&ctx, &sql).await;
    let api_plan = api.clone().into_inner().into_optimized_plan()?;
    assert_same_plan("generate", &sql_plan, &api_plan);

    let sql_rows = run(&ctx, &sql).await;
    let api_rows = out_of(sql_rows.fields.clone(), &api.collect().await?);
    assert_eq!(sql_rows.rows, api_rows.rows);
    Ok(())
}

// ---------------------------------------------------------------------------
// 8. The optimizer's rewrites fire on SQL-written plans
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_a_pushes_a_selection_below_an_extend() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT * FROM (SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS s FROM reviews) \
             WHERE SEM_FILTER('contains: movie', 'mock-model', review)";
    let plan = optimized(&ctx, q).await;
    let l = lines(&plan);
    assert!(
        line_of(&l, "NousDB_Extend") < line_of(&l, "NousDB_SemanticSelection"),
        "the selection should be below the extend:\n{}",
        l.join("\n")
    );
    let out = run(&ctx, q).await;
    assert_eq!(out.names(), ["id", "review", "s"]);
    assert_eq!(out.rows, expected(&[&["1", "good movie", "positive"]]));
    assert_eq!(mock.rows_seen("selection/user.md"), 6);
    assert_eq!(
        mock.rows_seen("extend/user.md"),
        1,
        "only the kept row is extended"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_b_pushes_a_selection_into_the_joins_left_input() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'mock-model', a.title, b.name) \
             WHERE SEM_FILTER('contains: s21', 'mock-model', a.title)";
    let plan = optimized(&ctx, q).await;
    let l = lines(&plan);
    let join = line_of(&l, "NousDB_SemanticJoin");
    let selection = line_of(&l, "NousDB_SemanticSelection");
    assert!(
        selection > join,
        "the selection is below the join:\n{}",
        l.join("\n")
    );
    assert!(
        indent(&l[selection]) > indent(&l[join]),
        "the selection is a child of the join:\n{}",
        l.join("\n")
    );
    assert!(
        selection < line_of(&l, "TableScan: b"),
        "...and sits in the left input, before b's scan:\n{}",
        l.join("\n")
    );
    let out = run(&ctx, q).await;
    assert_eq!(out.rows, expected(&[&["2", "galaxy s21", "20", "galaxy"]]));
    assert_eq!(mock.rows_seen("selection/user.md"), 3, "a's rows");
    assert_eq!(
        mock.rows_seen("join/user.md"),
        4,
        "1 surviving left row x 4 right rows"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_c_runs_the_selection_that_reads_fewer_columns_first() -> Result<()> {
    let (_, ctx) = default_session().await?;
    // As written, the two-column selection is the inner one (it runs first);
    // the rule swaps them so the one-column selection is innermost.
    let q = "SELECT * FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', title, abstract) \
             AND SEM_FILTER('contains: sql', 'mock-model', abstract)";
    let plan = optimized(&ctx, q).await;
    let l = lines(&plan);
    let first = l
        .iter()
        .position(|s| s.contains("NousDB_SemanticSelection"))
        .unwrap();
    let second = l
        .iter()
        .enumerate()
        .skip(first + 1)
        .find(|(_, s)| s.contains("NousDB_SemanticSelection"))
        .map(|(i, _)| i)
        .expect("two selections");
    assert!(
        l[first].contains("\"title\", \"abstract\""),
        "outer: {}",
        l[first]
    );
    assert!(
        l[second].contains("[\"abstract\"]") && !l[second].contains("title"),
        "inner: {}",
        l[second]
    );
    let out = run(&ctx, q).await;
    assert_eq!(out.column("id"), strings(&["2", "5"]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_e_fuses_an_extend_into_the_selection_that_reads_it() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT review FROM (SELECT *, EXTEND('classify topic', 'mock-model', review) AS topic FROM reviews) \
             WHERE SEM_FILTER('contains: film', 'mock-model', topic)";
    let plan = optimized(&ctx, q).await;
    let text = plan.display_indent().to_string();
    assert!(
        !text.contains("NousDB_Extend"),
        "no extend is left:\n{text}"
    );
    assert!(text.contains("NousDB_SemanticSelection"), "{text}");
    // The fused operator is a selection over the base columns; the mock
    // answers it by the keyword on the review text.
    let out = run(&ctx, q).await;
    assert_eq!(out.names(), ["review"]);
    assert_eq!(out.rows, expected(&[&["bad film"]]));
    assert_eq!(
        mock.requests("extend/user.md"),
        0,
        "no separate extend request"
    );
    assert_eq!(mock.rows_seen("selection/user.md"), 6);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_f_fuses_an_extend_into_the_join_that_reads_it() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = "SELECT review, name FROM \
             (SELECT *, EXTEND('who is mentioned', 'mock-model', review) AS who FROM mentions) \
             JOIN products ON SEM_JOIN('same product', 'mock-model', who, name)";
    let plan = optimized(&ctx, q).await;
    let text = plan.display_indent().to_string();
    assert!(
        !text.contains("NousDB_Extend"),
        "no extend is left:\n{text}"
    );
    assert!(text.contains("NousDB_SemanticJoin"), "{text}");
    let out = run(&ctx, q).await;
    assert_eq!(out.names(), ["review", "name"]);
    // The fused join compares the review text itself with the product name.
    assert_eq!(
        out.rows.len(),
        8,
        "iphone in 1, 2, 7, 8; galaxy in 5, 6, 7; pixel in 3"
    );
    assert_eq!(
        mock.requests("extend/user.md"),
        0,
        "no separate extend request"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_aliased_inner_subquery_stops_the_rewrite_but_not_the_results() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    // The alias `x` puts a SubqueryAlias between the selection and the
    // extend, and the rules match a selection directly over an extend, so rewrite A
    // does not fire. The query is still correct, just not optimized.
    let q = "SELECT * FROM (SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS s FROM reviews) x \
             WHERE SEM_FILTER('contains: movie', 'mock-model', x.review)";
    let plan = optimized(&ctx, q).await;
    let l = lines(&plan);
    let selection = line_of(&l, "NousDB_SemanticSelection");
    let extend = line_of(&l, "NousDB_Extend");
    assert!(
        selection < extend,
        "the selection stays above the extend:\n{}",
        l.join("\n")
    );
    assert!(
        l[selection + 1..extend]
            .iter()
            .any(|s| s.contains("SubqueryAlias")),
        "...with the alias between them:\n{}",
        l.join("\n")
    );
    let out = run(&ctx, q).await;
    assert_eq!(out.rows, expected(&[&["1", "good movie", "positive"]]));
    assert_eq!(
        mock.rows_seen("extend/user.md"),
        6,
        "every row is extended first"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 9. EXPLAIN
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn explain_shows_the_physical_operators() -> Result<()> {
    let (mock, ctx) = default_session().await?;

    let text = explain(
        &ctx,
        "SELECT *, EXTEND('classify sentiment', 'mock-model', review) AS s FROM reviews",
    )
    .await;
    assert!(text.contains("ExtendExec"), "{text}");

    let text = explain(
        &ctx,
        "SELECT * FROM papers WHERE SEM_FILTER('contains: database', 'mock-model', abstract)",
    )
    .await;
    assert!(text.contains("SemanticSelectionExec"), "{text}");

    let text = explain(
        &ctx,
        "SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'mock-model', a.title, b.name)",
    )
    .await;
    assert!(text.contains("SemanticJoinExec"), "{text}");

    let text = explain(
        &ctx,
        &format!("SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 5)"),
    )
    .await;
    assert!(text.contains("GenerateExec"), "{text}");

    let text = explain(
        &ctx,
        &format!("SELECT * FROM EXTEND(reviews, 'x', 'mock-model', review, {TWO_OUTPUTS})"),
    )
    .await;
    assert!(text.contains("ExtendExec"), "{text}");

    assert!(mock.calls().is_empty(), "EXPLAIN runs no model");
    Ok(())
}

// ---------------------------------------------------------------------------
// 10. with_sql_functions(false)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn without_the_sql_functions_each_one_fails_to_plan() -> Result<()> {
    let mock = Arc::new(SqlMock::default());
    let ctx = session(&mock, false)?;
    let scalar = [
        (
            "extend",
            "SELECT *, EXTEND('x', 'mock-model', review) AS s FROM reviews",
        ),
        (
            "sem_filter",
            "SELECT * FROM papers WHERE SEM_FILTER('x', 'mock-model', abstract)",
        ),
        (
            "sem_join",
            "SELECT * FROM a JOIN b ON SEM_JOIN('x', 'mock-model', a.title, b.name)",
        ),
    ];
    for (name, q) in scalar {
        let error = failure(&ctx, q).await;
        assert!(
            error.to_lowercase().contains("invalid function")
                && error.to_lowercase().contains(name),
            "{q}: {error}"
        );
    }
    // The relation forms are unknown table functions.
    for (name, q) in [
        (
            "generate",
            format!("SELECT * FROM GENERATE('x', 'mock-model', {STARTUPS}, 3)"),
        ),
        (
            "extend",
            format!("SELECT * FROM EXTEND(reviews, 'x', 'mock-model', review, {TWO_OUTPUTS})"),
        ),
    ] {
        let error = failure(&ctx, &q).await;
        assert!(error.to_lowercase().contains(name), "{q}: {error}");
    }
    assert!(mock.calls().is_empty(), "nothing reached the model");
    Ok(())
}

// ---------------------------------------------------------------------------
// 11. Errors reach the user intact
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_shapes_report_the_rewrite_errors() -> Result<()> {
    let (mock, ctx) = default_session().await?;

    let error = failure(
        &ctx,
        "SELECT * FROM a LEFT JOIN b ON SEM_JOIN('same product?', 'mock-model', a.title, b.name)",
    )
    .await;
    assert!(
        error.contains("SEM_JOIN supports only INNER joins"),
        "{error}"
    );

    let error = failure(
        &ctx,
        "SELECT * FROM papers WHERE SEM_FILTER('contains: sql', 'mock-model', abstract) OR year > 2020",
    )
    .await;
    assert!(
        error.contains("SEM_FILTER") && error.contains("OR"),
        "{error}"
    );

    assert!(mock.calls().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_model_fails_at_physical_planning_like_the_dataframe_api() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    // Planning the SQL succeeds: the model is not checked until the physical plan.
    let df = ctx
        .sql("SELECT *, EXTEND('x', 'no-such-model', review) AS s FROM reviews")
        .await?;
    let sql_error = df.collect().await.unwrap_err().to_string();
    assert!(
        sql_error.contains("no-such-model") && sql_error.contains("no registered provider"),
        "{sql_error}"
    );

    let api_error = ctx
        .nousdb_table("reviews")
        .await?
        .extend(("s", DataType::Utf8), "no-such-model", "x", ["review"])?
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(
        api_error.contains("no-such-model") && api_error.contains("no registered provider"),
        "{api_error}"
    );
    assert!(mock.calls().is_empty());
    Ok(())
}

// ---------------------------------------------------------------------------
// 12. LIMIT above an extend
// ---------------------------------------------------------------------------

/// A plain `LIMIT` above an extend is also applied below it (the extend
/// declares DataFusion's limit pushdown, as a logical node and as an exec), so
/// the model is called for the limited rows only: SQL (scalar and relation
/// form), the DataFrame API, and an OFFSET (the skipped rows are not sent).
#[tokio::test(flavor = "multi_thread")]
async fn a_limit_above_an_extend_bounds_the_rows_sent_to_the_model() -> Result<()> {
    // Scalar form, identity projection: Limit directly over the extend.
    let (mock, ctx) = default_session().await?;
    let q = "SELECT *, EXTEND('classify', 'mock-model', review) AS s FROM reviews LIMIT 2";
    let plan = lines(&optimized(&ctx, q).await);
    let extend = line_of(&plan, "NousDB_Extend");
    let below: Vec<&String> = plan[extend + 1..]
        .iter()
        .filter(|l| l.contains("Limit:"))
        .collect();
    assert_eq!(
        below.len(),
        1,
        "one limit below the extend:\n{}",
        plan.join("\n")
    );
    assert!(below[0].contains("fetch=2"), "{}", plan.join("\n"));
    let out = run(&ctx, q).await;
    assert_eq!(out.rows.len(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 2, "only the limited rows");

    // A projection between the limit and the extend.
    let (mock, ctx) = default_session().await?;
    let out = run(
        &ctx,
        "SELECT id, EXTEND('classify', 'mock-model', review) AS s FROM reviews LIMIT 3",
    )
    .await;
    assert_eq!(out.rows.len(), 3);
    assert_eq!(mock.rows_seen("extend/user.md"), 3);

    // Relation form.
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM EXTEND(reviews, 'extract both', 'mock-model', review, {TWO_OUTPUTS}) LIMIT 2"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.rows.len(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 2);

    // OFFSET: the logical plan keeps skip + fetch rows under the extend; the
    // physical limit pushdown then moves the skip below `ExtendExec` too.
    let (mock, ctx) = default_session().await?;
    let out = run(
        &ctx,
        "SELECT *, EXTEND('classify', 'mock-model', review) AS s FROM reviews LIMIT 2 OFFSET 1",
    )
    .await;
    assert_eq!(out.rows.len(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 2);

    // The DataFrame API gets the same pushdown.
    let (mock, ctx) = default_session().await?;
    let batches = ctx
        .nousdb_table("reviews")
        .await?
        .extend(("s", DataType::Utf8), MODEL, "classify", ["review"])?
        .limit(0, Some(2))?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 2);
    Ok(())
}

/// The physical plan: `ExtendExec` declares DataFusion's physical limit
/// pushdown, so no limit is left above it and the scan carries the fetch. That
/// hook works on its own too, in a session with no logical optimizer rule.
#[tokio::test(flavor = "multi_thread")]
async fn the_physical_plan_has_the_limit_below_the_extend() -> Result<()> {
    let q = "SELECT *, EXTEND('classify', 'mock-model', review) AS s FROM reviews LIMIT 2";

    let (_mock, ctx) = default_session().await?;
    let physical = ctx.sql(q).await?.create_physical_plan().await?;
    let text = displayable(physical.as_ref()).indent(false).to_string();
    let plan: Vec<&str> = text.lines().collect();
    assert!(
        plan[0].starts_with("ExtendExec"),
        "nothing above the extend:\n{text}"
    );
    assert!(
        plan[1..].iter().any(|l| l.contains("fetch=2")),
        "the fetch is below the extend:\n{text}"
    );

    // No logical rules at all: only the physical hook is left to push the limit.
    let mock = Arc::new(SqlMock::default());
    let registry = Arc::new(ModelRegistry::new());
    registry.register_as("mock", mock.clone())?;
    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_optimizer_rules(vec![])
        .build()?;
    table(
        &ctx,
        "reviews",
        vec![
            ints("id", &[1, 2, 3, 4, 5, 6]),
            strs("review", &["a", "b", "c", "d", "e", "f"]),
        ],
    )?;
    let out = run(&ctx, q).await;
    assert_eq!(out.rows.len(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 2);
    Ok(())
}

/// What is NOT pushed: a limit with an `ORDER BY`. The sort needs every row,
/// extended or not, so every row is still read.
#[tokio::test(flavor = "multi_thread")]
async fn an_ordered_limit_still_reads_every_row() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let out = run(
        &ctx,
        "SELECT *, EXTEND('classify', 'mock-model', review) AS s FROM reviews \
         ORDER BY id DESC LIMIT 2",
    )
    .await;
    assert_eq!(out.rows.len(), 2);
    assert_eq!(mock.rows_seen("extend/user.md"), 6);

    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 6) \
         ORDER BY founded DESC LIMIT 2"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.column("founded"), ["4", "5"]);
    assert_eq!(mock.rows_seen("generate/user.md"), 6);
    Ok(())
}

// ---------------------------------------------------------------------------
// 13. LIMIT above a selection and above GENERATE (the operators' fetch)
// ---------------------------------------------------------------------------

/// A limit above a semantic selection is not pushed to its input (a filter is
/// not one to one). As with its own filter, DataFusion hands the row count to
/// the operator, which stops once that many rows passed.
#[tokio::test(flavor = "multi_thread")]
async fn a_limit_above_a_selection_becomes_its_fetch() -> Result<()> {
    // Enough rows for several requests; every other one passes the filter.
    let many = |ctx: &NousDBContext| {
        let reviews: Vec<&str> = (0..2000)
            .map(|i| if i % 2 == 0 { "good" } else { "bad" })
            .collect();
        table(
            ctx,
            "many",
            vec![
                ints("id", &(0..2000).collect::<Vec<i64>>()),
                strs("review", &reviews),
            ],
        )
    };
    let all = "SELECT * FROM many WHERE SEM_FILTER('contains: good', 'mock-model', review)";
    let q = format!("{all} LIMIT 3");

    let (mock, ctx) = default_session().await?;
    many(&ctx)?;
    let physical = ctx.sql(&q).await?.create_physical_plan().await?;
    let text = displayable(physical.as_ref()).indent(false).to_string();
    let selection = text
        .lines()
        .find(|l| l.contains("SemanticSelectionExec"))
        .unwrap_or_else(|| panic!("no selection in\n{text}"));
    assert!(
        selection.ends_with("fetch=3"),
        "the selection has the fetch:\n{text}"
    );
    assert!(
        !text.contains("LimitExec"),
        "no limit operator is left:\n{text}"
    );
    assert!(
        text.lines()
            .last()
            .is_some_and(|l| !l.contains("fetch") && !l.contains("limit")),
        "the scan is not limited:\n{text}"
    );

    let unlimited = run(&ctx, all).await;
    assert_eq!(unlimited.rows.len(), 1000);
    let all_requests = mock.requests("selection/user.md");
    assert_eq!(mock.rows_seen("selection/user.md"), 2000);
    assert!(all_requests > 1, "the table needs several requests");

    // Three accepted rows are in the first request: one call, not all of them.
    let out = run(&ctx, &q).await;
    assert_eq!(out.column("id"), ["0", "2", "4"]);
    assert_eq!(mock.requests("selection/user.md") - all_requests, 1);
    assert!(mock.rows_seen("selection/user.md") - 2000 < 2000);

    // The DataFrame API takes the same path.
    let (mock, ctx) = default_session().await?;
    many(&ctx)?;
    let batches = ctx
        .nousdb_table("many")
        .await?
        .semantic_select(MODEL, "contains: good", ["review"])?
        .limit(0, Some(3))?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert_eq!(mock.requests("selection/user.md"), 1);
    Ok(())
}

/// `GENERATE` is a leaf: like a scan, it takes the limit as a fetch and asks
/// the model for those rows only. The prompt still describes the whole table.
#[tokio::test(flavor = "multi_thread")]
async fn a_limit_above_generate_bounds_the_rows_asked_for() -> Result<()> {
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 100) LIMIT 5"
    );
    let physical = ctx.sql(&q).await?.create_physical_plan().await?;
    let text = displayable(physical.as_ref()).indent(false).to_string();
    assert!(text.contains("num_rows=100"), "{text}");
    assert!(text.trim_end().ends_with("fetch=5"), "{text}");
    assert!(
        !text.contains("LimitExec"),
        "no limit operator is left:\n{text}"
    );

    let out = run(&ctx, &q).await;
    assert_eq!(out.column("founded"), ["0", "1", "2", "3", "4"]);
    assert_eq!(mock.rows_seen("generate/user.md"), 5);
    assert!(
        mock.calls()
            .iter()
            .all(|c| c.rendered.contains("**100** rows")),
        "the table is still described as 100 rows"
    );

    // An OFFSET is applied above; the leaf is asked for skip + fetch rows.
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT * FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 100) \
         LIMIT 3 OFFSET 2"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.column("founded"), ["2", "3", "4"]);
    assert_eq!(mock.rows_seen("generate/user.md"), 5);

    // An extend over the generated rows passes the limit down to the leaf.
    let (mock, ctx) = default_session().await?;
    let q = format!(
        "SELECT name, EXTEND('classify', 'mock-model', name) AS s \
         FROM GENERATE('fictional startups', 'mock-model', {STARTUPS}, 100) LIMIT 4"
    );
    let out = run(&ctx, &q).await;
    assert_eq!(out.rows.len(), 4);
    assert_eq!(mock.rows_seen("generate/user.md"), 4);
    assert_eq!(mock.rows_seen("extend/user.md"), 4);
    Ok(())
}

/// With token repartitioning forced, the pushed limit must end up BELOW the
/// repartition node, and only once: a limit between the extend and its
/// repartitioning would collapse the balanced partitions into one, and the
/// insertion rule would then repartition a second time.
#[tokio::test(flavor = "multi_thread")]
async fn a_pushed_limit_sits_below_token_repartitioning() -> Result<()> {
    let mock = Arc::new(SqlMock::default());
    let registry = Arc::new(ModelRegistry::new());
    registry.register_as("mock", mock.clone())?;
    let ctx = NousDBContextBuilder::new(registry)
        .with_config(SessionConfig::new().with_target_partitions(2))
        .with_semantic_optimizer(SemanticOptimizerConfig {
            token_repartition: TokenRepartitionChoice::Forced {
                partitions: 2,
                strategy: PartitionStrategy::OptimalTokenBalance,
            },
            ..Default::default()
        })
        .build()?;
    table(
        &ctx,
        "reviews",
        vec![
            ints("id", &[1, 2, 3, 4, 5, 6]),
            strs("review", &["a", "b", "c", "d", "e", "f"]),
        ],
    )?;
    let q = "SELECT *, EXTEND('classify', 'mock-model', review) AS s FROM reviews LIMIT 4";
    let plan = lines(&optimized(&ctx, q).await);
    let text = plan.join("\n");
    let extend = line_of(&plan, "NousDB_Extend");
    let repartition = line_of(&plan, "TokenRepartition");
    assert_eq!(
        repartition,
        extend + 1,
        "repartition directly under the extend:\n{text}"
    );
    assert!(
        plan[repartition + 1..]
            .iter()
            .any(|l| l.contains("Limit:") && l.contains("fetch=4")),
        "the limit is below the repartition node:\n{text}"
    );
    assert_eq!(
        plan.iter()
            .filter(|l| l.contains("TokenRepartition"))
            .count(),
        1,
        "one repartition node:\n{text}"
    );
    assert_eq!(
        plan.iter().filter(|l| l.contains("Limit:")).count(),
        2,
        "one limit above the extend and one below the repartition node:\n{text}"
    );
    let out = run(&ctx, q).await;
    assert_eq!(out.rows.len(), 4);
    assert_eq!(mock.rows_seen("extend/user.md"), 4);
    Ok(())
}
