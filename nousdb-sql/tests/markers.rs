//! The scalar marker functions plan through plain SQL.
//!
//! `ctx.sql(..)` only builds the logical plan (the analyzer runs later), so
//! these tests look at the plan as the SQL planner produced it: the marker is
//! still in it, with the type `return_field` gave it.

use std::sync::Arc;

use datafusion::arrow::array::StringArray;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{DataFrame, SessionContext};

use nousdb_sql::markers::{MarkerKind, marker_kind, scalar_functions};

fn table(ctx: &SessionContext, name: &str, columns: &[&str]) {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|c| Field::new(*c, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    ));
    let arrays = columns
        .iter()
        .map(|_| Arc::new(StringArray::from(vec!["x", "y"])) as _)
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    ctx.register_table(
        name,
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()),
    )
    .unwrap();
}

fn session() -> SessionContext {
    let ctx = SessionContext::new();
    nousdb_sql::register(&ctx).unwrap();
    table(&ctx, "reviews", &["review", "title"]);
    table(&ctx, "papers", &["abstract"]);
    table(&ctx, "a", &["title"]);
    table(&ctx, "b", &["name"]);
    ctx
}

/// Every marker kind found in any expression of the plan.
fn markers_in(plan: &LogicalPlan) -> Vec<MarkerKind> {
    let mut found = Vec::new();
    plan.apply(|node| {
        for expr in node.expressions() {
            expr.apply(|e| {
                found.extend(marker_kind(e));
                Ok(TreeNodeRecursion::Continue)
            })?;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

async fn plan(ctx: &SessionContext, sql: &str) -> DataFrame {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` should plan: {e}"))
}

/// The statements, each with a `{f}` placeholder for the function name.
const EXTEND_FORMS: [&str; 4] = [
    "SELECT *, {f}('classify sentiment', 'm', review) AS sentiment FROM reviews",
    "SELECT *, {f}('rate 1 to 5', 'm', review, 'Int64') AS stars FROM reviews",
    "SELECT *, {f}('rate 1 to 5', 'm', review, title, 'stars: Int64') FROM reviews",
    "SELECT *, {f}('rate 1 to 5', 'm', review, title) AS r FROM reviews",
];

#[tokio::test]
async fn scalar_extend_plans_in_any_case() {
    let ctx = session();
    for name in ["extend", "EXTEND", "Extend", "eXtEnD"] {
        for form in EXTEND_FORMS {
            let df = plan(&ctx, &form.replace("{f}", name)).await;
            assert_eq!(
                markers_in(df.logical_plan()),
                vec![MarkerKind::Extend],
                "{name}: {form}"
            );
        }
    }
}

#[tokio::test]
async fn sem_filter_plans_in_any_case_and_is_boolean() {
    let ctx = session();
    for name in ["sem_filter", "SEM_FILTER", "Sem_Filter"] {
        let sql = format!("SELECT * FROM papers WHERE {name}('is about databases', 'm', abstract)");
        let df = plan(&ctx, &sql).await;
        assert_eq!(markers_in(df.logical_plan()), vec![MarkerKind::SemFilter]);
        // The filter keeps the table's schema, the predicate itself is Boolean.
        assert_eq!(df.schema().field(0).name(), "abstract");
    }
    let df = plan(
        &ctx,
        "SELECT SEM_FILTER('p', 'm', abstract, abstract) AS keep FROM papers",
    )
    .await;
    // The same column twice plans (the rule rejects it), the type is Boolean.
    assert_eq!(df.schema().field(0).data_type(), &DataType::Boolean);
}

#[tokio::test]
async fn sem_join_plans_in_any_case() {
    let ctx = session();
    for name in ["sem_join", "SEM_JOIN", "Sem_Join"] {
        let sql =
            format!("SELECT * FROM a JOIN b ON {name}('same product?', 'm', a.title, b.name)");
        let df = plan(&ctx, &sql).await;
        assert_eq!(markers_in(df.logical_plan()), vec![MarkerKind::SemJoin]);
        assert_eq!(df.schema().fields().len(), 2);
    }
}

#[tokio::test]
async fn output_types_follow_the_trailing_literal() {
    let ctx = session();

    let df = plan(
        &ctx,
        "SELECT EXTEND('classify sentiment', 'm', review) AS sentiment FROM reviews",
    )
    .await;
    let field = df.schema().field(0).clone();
    assert_eq!(field.name(), "sentiment");
    assert_eq!(field.data_type(), &DataType::Utf8);

    let df = plan(
        &ctx,
        "SELECT EXTEND('rate', 'm', review, 'Int64') AS stars FROM reviews",
    )
    .await;
    let field = df.schema().field(0).clone();
    assert_eq!(field.name(), "stars");
    assert_eq!(field.data_type(), &DataType::Int64);

    // `'stars: Int64'` names the column, as an alias would (the marker's
    // `schema_name`), and types it.
    let df = plan(
        &ctx,
        "SELECT EXTEND('rate', 'm', review, 'stars: Int64') FROM reviews",
    )
    .await;
    let field = df.schema().field(0).clone();
    assert_eq!(field.name(), "stars");
    assert_eq!(field.data_type(), &DataType::Int64);
}

#[tokio::test]
async fn relation_forms_do_not_disturb_scalar_planning() {
    // The relation planner (a stub in this step) must hand the scalar call back
    // untouched, and unrelated relations still plan.
    let ctx = session();
    plan(&ctx, "SELECT * FROM (SELECT * FROM reviews) AS r").await;
}

async fn plan_error(ctx: &SessionContext, sql: &str) -> String {
    match ctx.sql(sql).await {
        Ok(_) => panic!("`{sql}` should not plan"),
        Err(e) => e.to_string(),
    }
}

#[tokio::test]
async fn planning_errors_name_the_function_and_its_usage() {
    let ctx = session();

    let msg = plan_error(&ctx, "SELECT EXTEND('i', 'm') FROM reviews").await;
    assert!(
        msg.contains("EXTEND") && msg.contains("Usage: EXTEND(instruction"),
        "{msg}"
    );

    let msg = plan_error(&ctx, "SELECT EXTEND('i', 'm', 'Int64') FROM reviews").await;
    assert!(msg.contains("inputs must be columns"), "{msg}");

    let msg = plan_error(&ctx, "SELECT EXTEND('i', 'm', review, 'int8') FROM reviews").await;
    assert!(msg.contains("'int8'") && msg.contains("Int64"), "{msg}");

    let msg = plan_error(
        &ctx,
        "SELECT EXTEND('i', 'm', review, 'a: Utf8, b: Utf8') FROM reviews",
    )
    .await;
    assert!(msg.contains("FROM EXTEND("), "{msg}");

    let msg = plan_error(&ctx, "SELECT EXTEND(title, 'm', review) FROM reviews").await;
    assert!(
        msg.contains("instruction") && msg.contains("string literal"),
        "{msg}"
    );

    // In the select list the type is resolved at planning time, so a malformed
    // call fails there.
    let msg = plan_error(&ctx, "SELECT SEM_FILTER('i', 'm') FROM papers").await;
    assert!(msg.contains("SEM_FILTER"), "{msg}");

    let msg = plan_error(&ctx, "SELECT SEM_JOIN('i', 'm', a.title) FROM a").await;
    assert!(msg.contains("SEM_JOIN"), "{msg}");
}

/// DataFusion resolves a `WHERE` / `ON` predicate's type on a best-effort basis
/// and ignores a failure, so a malformed call there is only rejected later, by
/// the analyzer rule (which re-validates with `args::*Call::parse`).
#[tokio::test]
async fn malformed_calls_in_predicates_are_left_to_the_rule() {
    let ctx = session();
    let df = plan(&ctx, "SELECT * FROM papers WHERE SEM_FILTER('i', 'm')").await;
    assert_eq!(markers_in(df.logical_plan()), vec![MarkerKind::SemFilter]);
    let df = plan(
        &ctx,
        "SELECT * FROM a JOIN b ON SEM_JOIN('i', 'm', a.title)",
    )
    .await;
    assert_eq!(markers_in(df.logical_plan()), vec![MarkerKind::SemJoin]);
}

#[tokio::test]
async fn invoking_a_marker_fails_with_the_must_be_rewritten_message() {
    // Only the UDFs, no analyzer rule: the marker survives to execution.
    let ctx = SessionContext::new();
    for f in scalar_functions() {
        ctx.register_udf(f.as_ref().clone());
    }
    table(&ctx, "reviews", &["review"]);

    let df = ctx
        .sql("SELECT EXTEND('classify', 'm', review) AS s FROM reviews")
        .await
        .expect("it plans");
    let msg = df.collect().await.expect_err("it must not run").to_string();
    assert!(msg.contains("EXTEND(...)"), "{msg}");
    assert!(msg.contains("must be rewritten"), "{msg}");
    assert!(msg.contains("NousDBContextBuilder"), "{msg}");
}
