//! The analyzer rule on `EXTEND` in a SELECT list.

#[path = "common/rewrite.rs"]
mod fixtures;

use datafusion::arrow::datatypes::DataType;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::AnalyzerRule;
use nousdb_operators::{ExtendNode, SemanticSelectionNode};
use nousdb_sql::SemanticFunctionRewriteRule;

use fixtures::*;

fn derived(node: &ExtendNode) -> Vec<(String, DataType, bool)> {
    node.derived_schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
        .collect()
}

#[tokio::test]
async fn every_case_gives_the_same_plan() {
    let ctx = session();
    let forms = [
        "SELECT *, {f}('classify sentiment', 'm', review) AS sentiment FROM reviews",
        "SELECT *, {f}('rate 1 to 5', 'm', review, 'Int64') AS stars FROM reviews",
        "SELECT title, {f}('rate', 'm', review, title, 'stars: Int64') FROM reviews",
    ];
    for form in forms {
        let reference = text(&analyze(&ctx, &form.replace("{f}", "extend")).await);
        for name in ["EXTEND", "Extend", "eXtEnD"] {
            let plan = analyze(&ctx, &form.replace("{f}", name)).await;
            assert_eq!(text(&plan), reference, "{name}: {form}");
        }
    }
}

#[tokio::test]
async fn star_and_extend_is_the_node_alone() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('classify sentiment', 'my-model', review) AS sentiment FROM reviews",
    )
    .await;
    let node = extend(&plan);
    assert_eq!(node.prompt, "classify sentiment");
    assert_eq!(node.model, "my-model");
    assert_eq!(node.input_schema, vec!["review".to_string()]);
    assert_eq!(
        derived(node),
        vec![("sentiment".into(), DataType::Utf8, true)]
    );
    assert!(matches!(node.input, LogicalPlan::TableScan(_)));
    assert!(text(&plan).starts_with("NousDB_Extend:"), "{}", text(&plan));
    assert_eq!(
        field_names(&plan),
        [
            "reviews.review",
            "reviews.title",
            "reviews.year",
            "sentiment"
        ]
    );
}

#[tokio::test]
async fn a_table_alias_keeps_its_qualifier() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT r.*, EXTEND('classify', 'm', r.review, r.title) AS s FROM reviews r",
    )
    .await;
    let node = extend(&plan);
    assert_eq!(node.input_schema, vec!["review", "title"]);
    assert!(matches!(node.input, LogicalPlan::SubqueryAlias(_)));
    assert_eq!(field_names(&plan), ["r.review", "r.title", "r.year", "s"]);
}

#[tokio::test]
async fn a_partial_select_list_keeps_a_projection() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT title, EXTEND('classify', 'm', review) AS s FROM reviews",
    )
    .await;
    assert!(matches!(plan, LogicalPlan::Projection(_)));
    let node = extend(input(&plan));
    assert_eq!(derived(node)[0].0, "s");
    assert_eq!(field_names(&plan), ["reviews.title", "s"]);
}

#[tokio::test]
async fn an_unaliased_call_gets_a_generated_name_under_its_output_name() {
    let ctx = session();
    let sql = "SELECT *, EXTEND('classify', 'm', review) FROM reviews";
    let plan = analyze(&ctx, sql).await;
    let node = extend(input(&plan));
    assert_eq!(derived(node)[0].0, "result");
    let out = field_names(&plan);
    assert!(out[3].starts_with("extend("), "{out:?}");
}

#[tokio::test]
async fn typed_and_named_specs() {
    let ctx = session();
    // A type and an alias: the alias names the column.
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('rate 1 to 5', 'm', review, 'Int64') AS stars FROM reviews",
    )
    .await;
    assert_eq!(
        derived(extend(&plan)),
        vec![("stars".into(), DataType::Int64, true)]
    );

    // A named spec without an alias: the spec names the column, as an alias
    // would, so nothing sits above the node here either.
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('rate 1 to 5', 'm', review, 'stars: Int64') FROM reviews",
    )
    .await;
    assert_eq!(
        derived(extend(&plan)),
        vec![("stars".into(), DataType::Int64, true)]
    );
    assert_eq!(field_names(&plan)[3], "stars");
    assert_eq!(plan.schema().field(3).data_type(), &DataType::Int64);

    // A named spec and an alias: the spec names the model's column.
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('rate', 'm', review, 'stars: Float64') AS score FROM reviews",
    )
    .await;
    assert_eq!(derived(extend(input(&plan)))[0].0, "stars");
    assert_eq!(field_names(&plan)[3], "score");
}

#[tokio::test]
async fn two_calls_stack_in_written_order() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('first', 'm', review) AS one, EXTEND('second', 'm', title, 'Boolean') AS two \
         FROM reviews",
    )
    .await;
    let outer = extend(&plan);
    let inner = extend(&outer.input);
    assert_eq!(inner.prompt, "first");
    assert_eq!(outer.prompt, "second");
    assert_eq!(derived(inner)[0].0, "one");
    assert_eq!(
        derived(outer),
        vec![("two".into(), DataType::Boolean, true)]
    );
    assert!(matches!(inner.input, LogicalPlan::TableScan(_)));
}

#[tokio::test]
async fn the_same_call_twice_is_one_node() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT EXTEND('i', 'm', review) AS a, upper(EXTEND('i', 'm', review)) AS b FROM reviews",
    )
    .await;
    assert_eq!(count::<ExtendNode>(&plan), 1);
    let node = extend(input(&plan));
    assert_eq!(derived(node)[0].0, "a");
    assert_eq!(field_names(&plan), ["a", "b"]);
}

#[tokio::test]
async fn a_nested_call() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT title, upper(EXTEND('i', 'm', review)) AS u FROM reviews",
    )
    .await;
    let node = extend(input(&plan));
    assert_eq!(derived(node)[0].0, "result");
    assert_eq!(field_names(&plan), ["reviews.title", "u"]);
    assert!(
        text(&plan).contains("upper(result) AS u"),
        "{}",
        text(&plan)
    );
}

#[tokio::test]
async fn an_alias_that_collides_with_an_input_column_is_renamed() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT title, EXTEND('i', 'm', review) AS review FROM reviews",
    )
    .await;
    let node = extend(input(&plan));
    assert_eq!(derived(node)[0].0, "review_1");
    assert_eq!(field_names(&plan), ["reviews.title", "review"]);

    // Two generated names.
    let plan = analyze(
        &ctx,
        "SELECT EXTEND('i', 'm', review), EXTEND('j', 'm', review) FROM reviews",
    )
    .await;
    let outer = extend(input(&plan));
    assert_eq!(derived(outer)[0].0, "result_1");
    assert_eq!(derived(extend(&outer.input))[0].0, "result");
}

#[tokio::test]
async fn extend_over_a_semantic_selection() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT *, EXTEND('i', 'm', abstract) AS s FROM papers WHERE SEM_FILTER('db?', 'm', abstract)",
    )
    .await;
    let node = extend(&plan);
    assert!(as_node::<SemanticSelectionNode>(&node.input).is_some());
}

#[tokio::test]
async fn extend_over_a_group_by() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT year, EXTEND('describe', 'm', year) AS d FROM papers GROUP BY year",
    )
    .await;
    let node = find(&plan, |p| as_node::<ExtendNode>(p).is_some()).unwrap();
    assert!(matches!(input(&node), LogicalPlan::Aggregate(_)));
}

#[tokio::test]
async fn markers_in_a_cte_and_a_subquery_are_rewritten() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "WITH x AS (SELECT *, EXTEND('i', 'm', review) AS s FROM reviews) SELECT s FROM x",
    )
    .await;
    assert_eq!(count::<ExtendNode>(&plan), 1);

    let plan = analyze(
        &ctx,
        "SELECT * FROM papers WHERE abstract IN \
         (SELECT review FROM reviews WHERE SEM_FILTER('i', 'm', review))",
    )
    .await;
    assert_eq!(count::<SemanticSelectionNode>(&plan), 1);

    let plan = analyze(
        &ctx,
        "SELECT * FROM (SELECT title, EXTEND('i', 'm', review) AS s FROM reviews) t WHERE s = 'x'",
    )
    .await;
    assert_eq!(count::<ExtendNode>(&plan), 1);
}

#[tokio::test]
async fn a_query_without_markers_is_returned_unchanged() {
    let ctx = session();
    for sql in [
        "SELECT title, upper(review) AS u FROM reviews WHERE year > 2000 ORDER BY title",
        "SELECT * FROM a JOIN b ON a.title = b.title",
    ] {
        let df = ctx.sql(sql).await.unwrap();
        let plan = df.logical_plan().clone();
        let out = SemanticFunctionRewriteRule::new()
            .analyze(plan.clone(), &ConfigOptions::default())
            .unwrap();
        assert_eq!(out, plan);
    }
}
