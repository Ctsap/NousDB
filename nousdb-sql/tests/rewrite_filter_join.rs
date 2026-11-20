//! The analyzer rule on `SEM_FILTER` in WHERE / HAVING and `SEM_JOIN` in ON /
//! WHERE.

#[path = "common/rewrite.rs"]
mod fixtures;

use datafusion::logical_expr::LogicalPlan;
use nousdb_operators::{SemanticJoinNode, SemanticSelectionNode};

use fixtures::*;

/// The first semantic selection of `plan`.
fn first_selection(plan: &LogicalPlan) -> LogicalPlan {
    find(plan, |p| as_node::<SemanticSelectionNode>(p).is_some())
        .unwrap_or_else(|| panic!("no selection in\n{}", text(plan)))
}

/// The first semantic join of `plan`.
fn first_join(plan: &LogicalPlan) -> LogicalPlan {
    find(plan, |p| as_node::<SemanticJoinNode>(p).is_some())
        .unwrap_or_else(|| panic!("no semantic join in\n{}", text(plan)))
}

#[tokio::test]
async fn every_case_gives_the_same_plan() {
    let ctx = session();
    let forms = [
        "SELECT * FROM papers WHERE {f}('is about databases', 'm', abstract) AND year > 2020",
        "SELECT * FROM a JOIN c ON {j}('same product?', 'm', a.title, c.name)",
        "SELECT * FROM a, c WHERE {j}('same product?', 'm', a.title, c.name)",
    ];
    for form in forms {
        let lower = form.replace("{f}", "sem_filter").replace("{j}", "sem_join");
        let reference = text(&analyze(&ctx, &lower).await);
        for (f, j) in [("SEM_FILTER", "SEM_JOIN"), ("Sem_Filter", "Sem_Join")] {
            let sql = form.replace("{f}", f).replace("{j}", j);
            assert_eq!(text(&analyze(&ctx, &sql).await), reference, "{sql}");
        }
    }
}

#[tokio::test]
async fn relational_conjuncts_go_below_the_selection() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM papers WHERE SEM_FILTER('is about databases', 'my-model', abstract) \
         AND year > 2020 AND abstract <> ''",
    )
    .await;
    let sel = first_selection(&plan);
    let node = selection(&sel);
    assert_eq!(node.prompt, "is about databases");
    assert_eq!(node.model, "my-model");
    assert_eq!(node.input_schema, vec!["abstract"]);
    let LogicalPlan::Filter(filter) = &node.input else {
        panic!("expected a Filter below the selection:\n{}", text(&plan));
    };
    assert!(matches!(filter.input.as_ref(), LogicalPlan::TableScan(_)));
    let predicate = filter.predicate.to_string();
    assert!(
        predicate.contains("year") && predicate.contains("abstract"),
        "{predicate}"
    );
    // Relational conjuncts written before the marker go below it too.
    let plan = analyze(
        &ctx,
        "SELECT * FROM papers WHERE year > 2020 AND SEM_FILTER('db?', 'm', abstract)",
    )
    .await;
    let sel = first_selection(&plan);
    assert!(matches!(selection(&sel).input, LogicalPlan::Filter(_)));
}

#[tokio::test]
async fn only_semantic_conjuncts_leave_no_filter() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM papers WHERE SEM_FILTER('db?', 'm', abstract)",
    )
    .await;
    let sel = first_selection(&plan);
    assert!(matches!(selection(&sel).input, LogicalPlan::TableScan(_)));
    assert!(find(&plan, |p| matches!(p, LogicalPlan::Filter(_))).is_none());
}

#[tokio::test]
async fn two_filters_keep_written_order() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM papers WHERE SEM_FILTER('first', 'm', abstract) \
         AND SEM_FILTER('second', 'm', abstract, year)",
    )
    .await;
    let outer = first_selection(&plan);
    let outer = selection(&outer);
    let inner = selection(&outer.input);
    assert_eq!(outer.prompt, "second");
    assert_eq!(outer.input_schema, vec!["abstract", "year"]);
    assert_eq!(inner.prompt, "first");
}

#[tokio::test]
async fn a_qualified_column_with_a_table_alias() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT p.abstract FROM papers p WHERE SEM_FILTER('db?', 'm', p.abstract)",
    )
    .await;
    let sel = first_selection(&plan);
    assert_eq!(selection(&sel).input_schema, vec!["abstract"]);
    assert!(matches!(
        selection(&sel).input,
        LogicalPlan::SubqueryAlias(_)
    ));
}

#[tokio::test]
async fn having_goes_through_the_filter_path() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT year, count(*) AS n FROM papers GROUP BY year HAVING SEM_FILTER('recent?', 'm', year)",
    )
    .await;
    let sel = first_selection(&plan);
    assert!(matches!(selection(&sel).input, LogicalPlan::Aggregate(_)));
}

#[tokio::test]
async fn join_on_binds_columns_by_side_in_written_order() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON SEM_JOIN('same product?', 'my-model', a.title, c.name)",
    )
    .await;
    let j = first_join(&plan);
    let node = join(&j);
    assert_eq!(node.left_join_attrs, vec!["title"]);
    assert_eq!(node.right_join_attrs, vec!["name"]);
    assert_eq!(node.prompt, "same product?");
    assert_eq!(node.model, "my-model");
    assert!(text(&plan).contains("NousDB_SemanticJoin:"));

    // Written right side first, and interleaved: each side keeps its order.
    let plan = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', c.kind, a.title, c.name, a.price)",
    )
    .await;
    let j = first_join(&plan);
    let node = join(&j);
    assert_eq!(node.left_join_attrs, vec!["title", "price"]);
    assert_eq!(node.right_join_attrs, vec!["kind", "name"]);
}

#[tokio::test]
async fn join_with_table_aliases() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT x.title, y.name FROM a x JOIN c y ON SEM_JOIN('same?', 'm', y.name, x.title)",
    )
    .await;
    let j = first_join(&plan);
    let node = join(&j);
    assert_eq!(node.left_join_attrs, vec!["title"]);
    assert_eq!(node.right_join_attrs, vec!["name"]);
    assert!(matches!(node.left_input, LogicalPlan::SubqueryAlias(_)));
}

#[tokio::test]
async fn one_sided_conditions_become_side_filters() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON a.price > 10 AND SEM_JOIN('same?', 'm', a.title, c.name) \
         AND c.kind = 'k'",
    )
    .await;
    let j = first_join(&plan);
    let node = join(&j);
    let LogicalPlan::Filter(left) = &node.left_input else {
        panic!("expected a left Filter:\n{}", text(&plan));
    };
    assert!(left.predicate.to_string().contains("price"));
    let LogicalPlan::Filter(right) = &node.right_input else {
        panic!("expected a right Filter:\n{}", text(&plan));
    };
    assert!(right.predicate.to_string().contains("kind"));
}

#[tokio::test]
async fn where_over_a_comma_or_cross_join_equals_the_on_form() {
    let ctx = session();
    let on = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) AND a.price > 10",
    )
    .await;
    let comma = analyze(
        &ctx,
        "SELECT * FROM a, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) AND a.price > 10",
    )
    .await;
    let cross = analyze(
        &ctx,
        "SELECT * FROM a CROSS JOIN c WHERE a.price > 10 AND SEM_JOIN('same?', 'm', a.title, c.name)",
    )
    .await;
    assert_eq!(text(&comma), text(&on));
    assert_eq!(text(&cross), text(&on));
}

#[tokio::test]
async fn no_collision_means_nothing_above_the_node() {
    let ctx = session();
    // `SELECT *` keeps the planner's projection above, so look at what the
    // join node replaced: the projection's input.
    let plan = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name)",
    )
    .await;
    assert!(
        as_node::<SemanticJoinNode>(input(&plan)).is_some(),
        "{}",
        text(&plan)
    );
}

#[tokio::test]
async fn a_colliding_name_gets_one_fix_up_projection() {
    let ctx = session();
    let sql = "SELECT * FROM a JOIN b ON SEM_JOIN('same?', 'm', a.title, b.name)";
    let plan = analyze(&ctx, sql).await;
    assert_eq!(
        field_names(&plan),
        ["a.title", "a.price", "b.name", "b.title"]
    );
    let fix_up = input(&plan);
    assert!(
        matches!(fix_up, LogicalPlan::Projection(_)),
        "{}",
        text(&plan)
    );
    let j = input(fix_up);
    assert_eq!(
        field_names(j),
        ["a.title", "a.price", "b.name", "b.title_1"]
    );
    assert_eq!(field_names(fix_up), field_names(&plan));
    assert_eq!(join(j).right_join_attrs, vec!["name"]);

    // The WHERE form does the same.
    let comma = analyze(
        &ctx,
        "SELECT * FROM a, b WHERE SEM_JOIN('same?', 'm', a.title, b.name)",
    )
    .await;
    assert_eq!(text(&comma), text(&plan));
}

#[tokio::test]
async fn a_selection_above_a_join_sits_on_the_node() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) \
         WHERE SEM_FILTER('cheap?', 'm', a.price, c.kind)",
    )
    .await;
    let sel = first_selection(&plan);
    let node = selection(&sel);
    assert_eq!(node.input_schema, vec!["price", "kind"]);
    assert!(
        as_node::<SemanticJoinNode>(&node.input).is_some(),
        "{}",
        text(&plan)
    );

    // And in the WHERE form.
    let plan = analyze(
        &ctx,
        "SELECT * FROM a, c WHERE SEM_FILTER('cheap?', 'm', a.price) \
         AND SEM_JOIN('same?', 'm', a.title, c.name)",
    )
    .await;
    let sel = first_selection(&plan);
    assert!(as_node::<SemanticJoinNode>(&selection(&sel).input).is_some());
}

// ------------------------------------------------- conjunct placement

/// The `Filter` directly above the first semantic join (or above its fix-up
/// projection), if any.
fn filter_above_join(plan: &LogicalPlan) -> Option<LogicalPlan> {
    find(plan, |p| match p {
        LogicalPlan::Filter(f) => {
            let below = match f.input.as_ref() {
                LogicalPlan::Projection(fix_up) => fix_up.input.as_ref(),
                other => other,
            };
            as_node::<SemanticJoinNode>(below).is_some()
                || as_node::<SemanticJoinNode>(f.input.as_ref()).is_some()
        }
        _ => false,
    })
}

fn predicate(plan: &LogicalPlan) -> String {
    match plan {
        LogicalPlan::Filter(f) => f.predicate.to_string(),
        other => panic!("expected a Filter, found\n{}", text(other)),
    }
}

#[tokio::test]
async fn a_subquery_reading_one_side_filters_that_side() {
    let ctx = session();
    for (sql, side) in [
        (
            "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) \
             AND EXISTS (SELECT 1 FROM papers p WHERE p.abstract = c.kind)",
            "right",
        ),
        (
            "SELECT * FROM a, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
             AND EXISTS (SELECT 1 FROM papers p WHERE p.abstract = c.kind)",
            "right",
        ),
        (
            "SELECT * FROM a, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
             AND a.price IN (SELECT year FROM papers p WHERE p.abstract = a.title)",
            "left",
        ),
        (
            "SELECT * FROM a, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
             AND (SELECT max(year) FROM papers p WHERE p.abstract = c.kind) > 2000",
            "right",
        ),
    ] {
        let plan = analyze(&ctx, sql).await;
        let j = first_join(&plan);
        let node = join(&j);
        let (filtered, other) = match side {
            "left" => (&node.left_input, &node.right_input),
            _ => (&node.right_input, &node.left_input),
        };
        assert!(
            matches!(filtered, LogicalPlan::Filter(_)),
            "{sql}: expected a {side} Filter:\n{}",
            text(&plan)
        );
        assert!(!matches!(other, LogicalPlan::Filter(_)), "{sql}");
        assert!(filter_above_join(&plan).is_none(), "{sql}\n{}", text(&plan));
    }
}

#[tokio::test]
async fn a_subquery_reading_both_sides_stays_above_the_join() {
    let ctx = session();
    for sql in [
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) \
         AND EXISTS (SELECT 1 FROM papers p WHERE p.abstract = c.kind AND p.abstract = a.title)",
        "SELECT * FROM a, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
         AND EXISTS (SELECT 1 FROM papers p WHERE p.abstract = c.kind AND p.abstract = a.title)",
    ] {
        let plan = analyze(&ctx, sql).await;
        let above = filter_above_join(&plan)
            .unwrap_or_else(|| panic!("{sql}: no Filter above the join:\n{}", text(&plan)));
        assert!(predicate(&above).contains("EXISTS"), "{sql}");
        let j = first_join(&plan);
        let node = join(&j);
        assert!(!matches!(node.left_input, LogicalPlan::Filter(_)), "{sql}");
        assert!(!matches!(node.right_input, LogicalPlan::Filter(_)), "{sql}");
    }
}

#[tokio::test]
async fn volatile_and_column_free_conditions_stay_above_the_join() {
    let ctx = session();
    for (condition, needle) in [
        ("random() < 0.5", "random()"),
        ("a.price + random() > 1", "random()"),
        ("1 = 1", "Int64(1) = Int64(1)"),
        ("EXISTS (SELECT 1 FROM papers)", "EXISTS"),
    ] {
        for sql in [
            format!(
                "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) AND {condition}"
            ),
            format!(
                "SELECT * FROM a, c WHERE {condition} AND SEM_JOIN('same?', 'm', a.title, c.name)"
            ),
        ] {
            let plan = analyze(&ctx, &sql).await;
            let above = filter_above_join(&plan)
                .unwrap_or_else(|| panic!("{sql}: no Filter above the join:\n{}", text(&plan)));
            assert!(predicate(&above).contains(needle), "{sql}\n{}", text(&plan));
            let j = first_join(&plan);
            let node = join(&j);
            assert!(!matches!(node.left_input, LogicalPlan::Filter(_)), "{sql}");
            assert!(!matches!(node.right_input, LogicalPlan::Filter(_)), "{sql}");
        }
    }
}

#[tokio::test]
async fn a_filter_above_the_join_sits_above_the_fix_up_projection() {
    let ctx = session();
    // `b.title` collides with `a.title`, so the node renames it `title_1` and
    // a projection restores it; `b.title` above must still resolve.
    for sql in [
        "SELECT * FROM a JOIN b ON SEM_JOIN('same?', 'm', a.title, b.name) AND random() < 0.5",
        "SELECT * FROM a, b WHERE SEM_JOIN('same?', 'm', a.title, b.name) \
         AND EXISTS (SELECT 1 FROM papers p WHERE p.abstract = b.title AND p.abstract = a.title)",
    ] {
        let plan = analyze(&ctx, sql).await;
        let above = filter_above_join(&plan)
            .unwrap_or_else(|| panic!("{sql}: no Filter above the join:\n{}", text(&plan)));
        let fix_up = input(&above);
        assert!(
            matches!(fix_up, LogicalPlan::Projection(_)),
            "{sql}\n{}",
            text(&plan)
        );
        assert_eq!(
            field_names(fix_up),
            ["a.title", "a.price", "b.name", "b.title"]
        );
        assert!(
            as_node::<SemanticJoinNode>(input(fix_up)).is_some(),
            "{sql}"
        );
    }
}

#[tokio::test]
async fn a_correlated_conjunct_stays_above_the_selection() {
    let ctx = session();
    let plan = analyze(
        &ctx,
        "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM c WHERE c.kind = a.title \
         AND c.name <> '' AND SEM_FILTER('db?', 'm', c.name))",
    )
    .await;
    // Inside the subquery: Filter(correlated) -> selection -> Filter(c.name <> '').
    let top = find(&plan, |p| match p {
        LogicalPlan::Filter(f) => as_node::<SemanticSelectionNode>(&f.input).is_some(),
        _ => false,
    })
    .unwrap_or_else(|| panic!("no Filter above the selection:\n{}", text(&plan)));
    let correlated = predicate(&top);
    assert!(correlated.contains("outer_ref(a.title)"), "{correlated}");
    assert!(!correlated.contains("c.name"), "{correlated}");
    let sel = input(&top);
    let below = predicate(&selection(sel).input);
    assert!(
        below.contains("c.name") && !below.contains("outer_ref"),
        "{below}"
    );
}

#[tokio::test]
async fn a_correlated_conjunct_stays_above_a_semantic_join() {
    let ctx = session();
    // `c` and `b` share `name`: the join renames it and a fix-up projection
    // restores it; the correlated filter sits above everything.
    let plan = analyze(
        &ctx,
        "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM c, b \
         WHERE SEM_JOIN('same?', 'm', c.name, b.name) AND SEM_FILTER('x', 'm', c.kind) \
         AND c.kind = a.title AND b.title <> '')",
    )
    .await;
    let top = find(&plan, |p| match p {
        LogicalPlan::Filter(f) => as_node::<SemanticSelectionNode>(&f.input).is_some(),
        _ => false,
    })
    .unwrap_or_else(|| panic!("no Filter above the selection:\n{}", text(&plan)));
    assert!(predicate(&top).contains("outer_ref(a.title)"));
    let sel = input(&top);
    let fix_up = &selection(sel).input;
    assert!(
        matches!(fix_up, LogicalPlan::Projection(_)),
        "{}",
        text(&plan)
    );
    let j = input(fix_up);
    // The uncorrelated one-sided conjunct still filters its side.
    assert!(predicate(&join(j).right_input).contains("b.title"));

    // The ON form: the correlated conjunct goes above the fix-up projection.
    let plan = analyze(
        &ctx,
        "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM c JOIN b \
         ON SEM_JOIN('same?', 'm', c.name, b.name) AND c.kind = a.title)",
    )
    .await;
    let above = filter_above_join(&plan)
        .unwrap_or_else(|| panic!("no Filter above the join:\n{}", text(&plan)));
    assert!(predicate(&above).contains("outer_ref(a.title)"));
    assert!(matches!(input(&above), LogicalPlan::Projection(_)));
}
