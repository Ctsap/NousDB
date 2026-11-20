//! `GENERATE(...)` and relation-form `EXTEND(...)` in `FROM`.
//!
//! The relation planner runs at SQL-planning time, so `ctx.sql(..)` already
//! holds NousDB's nodes. A plain `SessionContext` cannot physically plan them,
//! so nothing here is collected.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{DataFrame, SessionContext};

use nousdb_operators::{ExtendNode, GenerateNode};

fn table(ctx: &SessionContext, name: &str, columns: &[(&str, DataType)]) {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(c, t)| Field::new(*c, t.clone(), true))
            .collect::<Vec<_>>(),
    ));
    let arrays = columns
        .iter()
        .map(|(_, t)| match t {
            DataType::Int64 => Arc::new(Int64Array::from(vec![1, 2])) as _,
            _ => Arc::new(StringArray::from(vec!["x", "y"])) as _,
        })
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
    // Until `register` adds it itself: the lookup for a table named only as
    // EXTEND's first argument (registering it twice is harmless).
    ctx.register_udtf(
        nousdb_sql::relation::TABLE_RESOLVER_NAME,
        nousdb_sql::relation::table_resolver(),
    );
    let utf8 = DataType::Utf8;
    table(
        &ctx,
        "reviews",
        &[
            ("id", DataType::Int64),
            ("review", utf8.clone()),
            ("title", utf8.clone()),
        ],
    );
    table(
        &ctx,
        "movies",
        &[("id", DataType::Int64), ("name", utf8.clone())],
    );
    table(
        &ctx,
        "a",
        &[("id", DataType::Int64), ("review", utf8.clone())],
    );
    table(
        &ctx,
        "b",
        &[("id", DataType::Int64), ("other", utf8.clone())],
    );
    // A real table literally named `extend`.
    table(&ctx, "extend", &[("v", utf8)]);
    ctx
}

async fn plan(q: &str) -> DataFrame {
    session()
        .sql(q)
        .await
        .unwrap_or_else(|e| panic!("{q}\nfailed: {e}"))
}

async fn error(q: &str) -> String {
    match session().sql(q).await {
        Ok(df) => panic!(
            "{q}\nshould fail, planned:\n{}",
            df.logical_plan().display_indent()
        ),
        Err(e) => {
            let msg = e.to_string();
            eprintln!("{q}\n  -> {msg}");
            msg
        }
    }
}

fn find<T: 'static + Clone>(plan: &LogicalPlan) -> Vec<T> {
    let mut found = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::Extension(ext) = node
            && let Some(n) = ext.node.as_any().downcast_ref::<T>()
        {
            found.push(n.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

fn generate_node(df: &DataFrame) -> GenerateNode {
    let nodes = find::<GenerateNode>(df.logical_plan());
    assert_eq!(nodes.len(), 1, "{}", df.logical_plan().display_indent());
    nodes.into_iter().next().unwrap()
}

fn extend_nodes(df: &DataFrame) -> Vec<ExtendNode> {
    find::<ExtendNode>(df.logical_plan())
}

fn extend_node(df: &DataFrame) -> ExtendNode {
    let nodes = extend_nodes(df);
    assert_eq!(nodes.len(), 1, "{}", df.logical_plan().display_indent());
    nodes.into_iter().next().unwrap()
}

fn has_alias(plan: &LogicalPlan, alias: &str) -> bool {
    let mut found = false;
    plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(a) = node
            && a.alias.table() == alias
        {
            found = true;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

fn any_alias(plan: &LogicalPlan) -> bool {
    let mut found = false;
    plan.apply(|node| {
        found |= matches!(node, LogicalPlan::SubqueryAlias(_));
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

fn names(df: &DataFrame) -> Vec<String> {
    df.schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn derived_names(node: &ExtendNode) -> Vec<String> {
    node.derived_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

// ---------------------------------------------------------------- GENERATE

#[tokio::test]
async fn generate_in_any_case() {
    for f in ["generate", "GENERATE", "Generate"] {
        let q = format!(
            "SELECT * FROM {f}('20 fictional startups', 'm', 'name: Utf8, pitch: Utf8, founded: Int64', 20)"
        );
        let df = plan(&q).await;
        let node = generate_node(&df);
        assert_eq!(node.instruction, "20 fictional startups");
        assert_eq!(node.model, "m");
        assert_eq!(node.num_rows, 20);
        assert!(node.examples.is_empty());
        let fields: Vec<_> = node
            .schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
            .collect();
        assert_eq!(
            fields,
            vec![
                ("name".into(), DataType::Utf8, true),
                ("pitch".into(), DataType::Utf8, true),
                ("founded".into(), DataType::Int64, true),
            ]
        );
        assert_eq!(names(&df), vec!["name", "pitch", "founded"]);
        assert!(!any_alias(df.logical_plan()), "{q}");
        assert!(
            df.logical_plan()
                .display_indent()
                .to_string()
                .contains("NousDB_Generate:")
        );
    }
}

#[tokio::test]
async fn generate_is_the_dataframe_node() {
    let df = plan("SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 3)").await;
    let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
    let expected = nousdb_operators::create_generate(schema, "m", "s", 3, &[]).unwrap();
    let LogicalPlan::Extension(ext) = &expected else {
        panic!()
    };
    let expected = ext.node.as_any().downcast_ref::<GenerateNode>().unwrap();
    assert_eq!(&generate_node(&df), expected);
}

#[tokio::test]
async fn generate_with_alias_filter_and_projection() {
    let df = plan("SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 5) AS g").await;
    assert!(has_alias(df.logical_plan(), "g"));
    let df =
        plan("SELECT g.name FROM GENERATE('s', 'm', 'name: Utf8, n: Int64', 5) AS g WHERE g.n > 3")
            .await;
    assert_eq!(names(&df), vec!["name"]);
    let df = plan(
        "SELECT name FROM GENERATE('s', 'm', 'name: Utf8, n: Int64', 5) WHERE n > 3 ORDER BY name",
    )
    .await;
    assert_eq!(names(&df), vec!["name"]);
    generate_node(&df);
}

#[tokio::test]
async fn generate_folds_constant_arguments() {
    let df = plan("SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 10 + 10)").await;
    assert_eq!(generate_node(&df).num_rows, 20);
    let df = plan("SELECT * FROM GENERATE('s' || 'x', 'm', 'name: Utf8', CAST(4 AS INT))").await;
    let node = generate_node(&df);
    assert_eq!((node.instruction.as_str(), node.num_rows), ("sx", 4));
}

#[tokio::test]
async fn generate_decodes_examples() {
    let q = r#"SELECT * FROM GENERATE('fictional startups', 'm', 'name: Utf8, pitch: Utf8, n: Int64', 6,
        '{"name": "Acme", "pitch": "rockets", "n": 1}

{"pitch": "energy", "name": "Globex"}') AS g"#;
    let df = plan(q).await;
    let node = generate_node(&df);
    assert_eq!(node.examples.len(), 2, "{:?}", node.examples);
    assert!(node.examples[0].contains("Acme"));
    // Columns in schema order; the missing key is NULL.
    assert!(
        node.examples[1].starts_with(r#"{"name":"Globex""#),
        "{}",
        node.examples[1]
    );
    assert!(
        node.examples[1].contains(r#""n":null"#),
        "{}",
        node.examples[1]
    );
}

#[tokio::test]
async fn generate_errors() {
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8')",
            &["GENERATE", "4 or 5"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 1, '', 'x')",
            &["GENERATE", "4 or 5"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: date', 3)",
            &["GENERATE", "invalid schema", "'date'"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 0)",
            &["at least 1"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', -3)",
            &["at least 1"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2.5)",
            &["integer literal"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 'ten')",
            &["integer literal"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2, '')",
            &["GENERATE", "examples", "empty"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2, '{\"name\": \"a\"}\n[1, 2]')",
            &["GENERATE", "line 2", "not a JSON object"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2, '{\"name\": \"a\"')",
            &["GENERATE", "line 1"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2, '{\"name\": \"a\", \"zzz\": 1}')",
            &["GENERATE", "line 1", "zzz"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'n: Int64', 2, '{\"n\": \"many\"}')",
            &["GENERATE", "line 1"],
        ),
        (
            "SELECT * FROM GENERATE(instruction => 's', 'm', 'name: Utf8', 2)",
            &["GENERATE", "named argument"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', 2) WITH ORDINALITY",
            &["GENERATE", "WITH ORDINALITY"],
        ),
        (
            "SELECT * FROM GENERATE('s', 'm', 'name: Utf8', nope)",
            &["GENERATE", "argument 4"],
        ),
    ];
    for (q, needles) in cases {
        let msg = error(q).await;
        for n in *needles {
            assert!(msg.contains(n), "{q}\n  error lacks {n:?}: {msg}");
        }
    }
}

// ------------------------------------------------------------ EXTEND (FROM)

#[tokio::test]
async fn extend_over_a_table_in_any_case() {
    for f in ["extend", "EXTEND", "Extend"] {
        let q = format!(
            "SELECT * FROM {f}(reviews, 'extract both', 'm', review, 'sentiment: Utf8, genre: Utf8')"
        );
        let df = plan(&q).await;
        let node = extend_node(&df);
        assert_eq!(node.prompt, "extract both");
        assert_eq!(node.model, "m");
        assert_eq!(node.input_schema, vec!["review"]);
        assert_eq!(derived_names(&node), vec!["sentiment", "genre"]);
        assert!(node.derived_schema.fields().iter().all(|f| f.is_nullable()));
        assert_eq!(
            names(&df),
            vec!["id", "review", "title", "sentiment", "genre"]
        );
        assert!(!any_alias(df.logical_plan()), "{q}");
        assert!(
            df.logical_plan()
                .display_indent()
                .to_string()
                .contains("NousDB_Extend:")
        );
    }
}

#[tokio::test]
async fn extend_is_the_dataframe_node() {
    let ctx = session();
    let df = ctx
        .sql("SELECT * FROM EXTEND(reviews, 'rate', 'm', title, review, 'stars: Int64')")
        .await
        .unwrap();
    let input = ctx.table("reviews").await.unwrap().into_unoptimized_plan();
    let derived = Arc::new(Schema::new(vec![Field::new(
        "stars",
        DataType::Int64,
        true,
    )]));
    let expected = nousdb_operators::create_extend(
        input,
        "rate",
        "m",
        vec!["title".into(), "review".into()],
        derived,
    )
    .unwrap();
    let LogicalPlan::Extension(ext) = &expected else {
        panic!()
    };
    let expected = ext.node.as_any().downcast_ref::<ExtendNode>().unwrap();
    let got = extend_node(&df);
    assert_eq!(got.prompt, expected.prompt);
    assert_eq!(got.model, expected.model);
    assert_eq!(got.input_schema, expected.input_schema);
    assert_eq!(got.derived_schema, expected.derived_schema);
    assert_eq!(got.schema, expected.schema);
}

#[tokio::test]
async fn extend_keeps_input_column_order_and_qualifiers() {
    let df =
        plan("SELECT * FROM EXTEND(reviews, 'i', 'm', title, reviews.review, id, 'label: Utf8')")
            .await;
    let node = extend_node(&df);
    assert_eq!(node.input_schema, vec!["title", "review", "id"]);
    assert_eq!(derived_names(&node), vec!["label"]);
    // Input fields stay qualified; derived ones are unqualified.
    let quals: Vec<_> = df
        .schema()
        .iter()
        .map(|(q, f)| (q.map(|q| q.to_string()), f.name().clone()))
        .collect();
    assert_eq!(quals[0], (Some("reviews".into()), "id".into()));
    assert_eq!(quals[3], (None, "label".into()));
}

#[tokio::test]
async fn extend_over_a_schema_qualified_name() {
    let df = plan("SELECT * FROM EXTEND(public.reviews, 'i', 'm', review, 'label: Utf8')").await;
    assert_eq!(extend_node(&df).input_schema, vec!["review"]);
}

#[tokio::test]
async fn extend_over_subqueries_and_ctes() {
    for q in [
        "SELECT * FROM EXTEND((SELECT id, review FROM reviews), 'i', 'm', review, 'label: Utf8')",
        "SELECT * FROM EXTEND((SELECT * FROM reviews LIMIT 100), 'i', 'm', review, 'label: Utf8')",
        "SELECT * FROM EXTEND(((SELECT id, review FROM reviews)), 'i', 'm', review, 'label: Utf8')",
        "WITH r AS (SELECT id, review FROM reviews WHERE id > 1) \
         SELECT * FROM EXTEND(r, 'i', 'm', review, 'label: Utf8')",
    ] {
        let df = plan(q).await;
        assert_eq!(extend_node(&df).input_schema, vec!["review"], "{q}");
        assert_eq!(names(&df).last().unwrap(), "label", "{q}");
    }
    let df = plan(
        "SELECT * FROM EXTEND((SELECT * FROM reviews LIMIT 100), 'i', 'm', review, 'label: Utf8')",
    )
    .await;
    let text = df.logical_plan().display_indent().to_string();
    assert!(text.contains("Limit"), "{text}");
}

#[tokio::test]
async fn extend_derived_columns_filter_and_sort() {
    let df = plan(
        "SELECT title, sentiment FROM EXTEND((SELECT * FROM reviews LIMIT 100), 'extract both', 'm', \
         review, title, 'sentiment: Utf8, genre: Utf8') WHERE sentiment = 'positive' ORDER BY genre",
    )
    .await;
    assert_eq!(names(&df), vec!["title", "sentiment"]);
    assert_eq!(extend_node(&df).input_schema, vec!["review", "title"]);
}

#[tokio::test]
async fn extend_with_alias() {
    let df = plan(
        "SELECT e.sentiment, e.review FROM EXTEND(reviews, 'i', 'm', review, 'sentiment: Utf8') AS e \
         WHERE e.sentiment = 'x'",
    )
    .await;
    assert!(has_alias(df.logical_plan(), "e"));
    assert_eq!(names(&df), vec!["sentiment", "review"]);
}

#[tokio::test]
async fn nested_extend_through_a_subquery() {
    let df = plan(
        "SELECT * FROM EXTEND((SELECT * FROM EXTEND(reviews, 'a', 'm', review, 'sentiment: Utf8')), \
         'b', 'm', sentiment, title, 'summary: Utf8')",
    )
    .await;
    let nodes = extend_nodes(&df);
    assert_eq!(nodes.len(), 2);
    let outer = nodes.iter().find(|n| n.prompt == "b").unwrap();
    assert_eq!(outer.input_schema, vec!["sentiment", "title"]);
    assert_eq!(
        names(&df),
        vec!["id", "review", "title", "sentiment", "summary"]
    );
}

#[tokio::test]
async fn extend_as_a_join_side() {
    let df = plan(
        "SELECT m.name, e.label FROM movies m \
         JOIN EXTEND(reviews, 'i', 'm', review, 'label: Utf8') AS e ON m.id = e.id",
    )
    .await;
    assert_eq!(names(&df), vec!["name", "label"]);
    extend_node(&df);
    let df = plan(
        "SELECT * FROM movies JOIN EXTEND(reviews, 'i', 'm', review, 'label: Utf8') ON movies.id = reviews.id",
    )
    .await;
    assert_eq!(names(&df).last().unwrap(), "label");
}

#[tokio::test]
async fn extend_errors() {
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT * FROM EXTEND(reviews, 'i', 'm', review)",
            &["EXTEND", "output schema"],
        ),
        (
            "SELECT * FROM EXTEND(reviews, 'i', 'm', upper(review), 'label: Utf8')",
            &["EXTEND", "inputs must be columns"],
        ),
        (
            "SELECT * FROM EXTEND(reviews, 'i', 'm', nope, 'label: Utf8')",
            &["EXTEND", "argument 4", "nope"],
        ),
        (
            "SELECT * FROM EXTEND((SELECT a.id, b.id, a.review FROM a JOIN b ON a.id = b.id), \
             'i', 'm', a.id, 'label: Utf8')",
            &["EXTEND", "ambiguous", "id"],
        ),
        (
            "SELECT * FROM EXTEND(reviews, 'i', 'm', review, 'title: Utf8')",
            &["EXTEND", "'title'", "already exists"],
        ),
        (
            "SELECT * FROM EXTEND('reviews', 'i', 'm', review, 'label: Utf8')",
            &["first argument of EXTEND in FROM is a table name or a parenthesized subquery"],
        ),
        (
            "SELECT * FROM EXTEND(upper(x), 'i', 'm', review, 'label: Utf8')",
            &["first argument of EXTEND"],
        ),
        (
            "SELECT * FROM EXTEND(no_such_table, 'i', 'm', review, 'label: Utf8')",
            &["no_such_table"],
        ),
        (
            "SELECT * FROM EXTEND(reviews, instruction => 'i', 'm', review, 'label: Utf8')",
            &["EXTEND", "named argument"],
        ),
        ("SELECT * FROM EXTEND()", &["EXTEND"]),
        (
            "SELECT * FROM EXTEND(reviews, 'i', 'm', review, 'label: Utf8') TABLESAMPLE (10 ROWS)",
            &["EXTEND", "TABLESAMPLE"],
        ),
    ];
    for (q, needles) in cases {
        let msg = error(q).await;
        for n in *needles {
            assert!(msg.contains(n), "{q}\n  error lacks {n:?}: {msg}");
        }
    }
}

// ------------------------------------------------------------ not ours

#[tokio::test]
async fn a_table_named_extend_is_untouched() {
    let df = plan("SELECT * FROM extend").await;
    assert_eq!(names(&df), vec!["v"]);
    assert!(extend_nodes(&df).is_empty());
    let df = plan("SELECT e.v FROM extend AS e").await;
    assert_eq!(names(&df), vec!["v"]);
}

#[tokio::test]
async fn other_table_functions_are_untouched() {
    let df = plan("SELECT * FROM generate_series(1, 3)").await;
    assert_eq!(names(&df), vec!["value"]);
    assert!(find::<GenerateNode>(df.logical_plan()).is_empty());
}

#[tokio::test]
async fn a_session_without_the_resolver_gets_the_hint() {
    // The relation planner alone: `EXTEND(table, ...)` cannot look the table up.
    let ctx = SessionContext::new();
    ctx.register_relation_planner(Arc::new(nousdb_sql::SemanticRelationPlanner::new()))
        .unwrap();
    table(
        &ctx,
        "reviews",
        &[("id", DataType::Int64), ("review", DataType::Utf8)],
    );
    let msg = ctx
        .sql("SELECT * FROM EXTEND(reviews, 'i', 'm', review, 'label: Utf8')")
        .await
        .map(|_| ())
        .expect_err("should fail")
        .to_string();
    assert!(
        msg.contains("has no __nousdb_resolve_table table function"),
        "{msg}"
    );
    // With the resolver, an unknown table is the resolver's own error, not
    // the hint.
    let msg = error("SELECT * FROM EXTEND(no_such_table, 'i', 'm', review, 'label: Utf8')").await;
    assert!(!msg.contains("has no __nousdb_resolve_table"), "{msg}");
}
