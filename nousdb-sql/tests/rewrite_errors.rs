//! The analyzer rule's errors: markers in unsupported positions and malformed
//! calls.

#[path = "common/rewrite.rs"]
mod fixtures;

use fixtures::*;

async fn expect(sql: &str, parts: &[&str]) {
    let ctx = session();
    let msg = analyze_err(&ctx, sql).await;
    assert_contains(&msg, parts);
}

#[tokio::test]
async fn boolean_markers_in_a_select_list() {
    expect(
        "SELECT SEM_FILTER('db?', 'm', abstract) AS keep FROM papers",
        &["SEM_FILTER is not supported in a SELECT list", "'Boolean')"],
    )
    .await;
    expect(
        "SELECT SEM_JOIN('same?', 'm', a.title, a.price) FROM a",
        &["SEM_JOIN is not supported in a SELECT list", "EXTEND("],
    )
    .await;
}

#[tokio::test]
async fn markers_under_or_not_or_inside_an_expression() {
    for sql in [
        "SELECT * FROM papers WHERE SEM_FILTER('db?', 'm', abstract) OR year > 2020",
        "SELECT * FROM papers WHERE NOT SEM_FILTER('db?', 'm', abstract)",
        "SELECT * FROM papers WHERE SEM_FILTER('db?', 'm', abstract) = true",
    ] {
        expect(sql, &["SEM_FILTER must be a whole condition", "OR or NOT"]).await;
    }
    expect(
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) OR a.price > 1",
        &["SEM_JOIN must be a whole condition"],
    )
    .await;
}

#[tokio::test]
async fn extend_in_a_predicate() {
    expect(
        "SELECT * FROM papers WHERE EXTEND('topic', 'm', abstract) = 'db'",
        &[
            "EXTEND is not supported in a predicate",
            "subquery's SELECT",
        ],
    )
    .await;
    expect(
        "SELECT * FROM a JOIN c ON EXTEND('x', 'm', a.title) = c.name",
        &["EXTEND is not supported in a predicate"],
    )
    .await;
}

#[tokio::test]
async fn sem_join_in_where_needs_a_comma_join() {
    expect(
        "SELECT * FROM a WHERE SEM_JOIN('same?', 'm', a.title, a.price)",
        &["comma or CROSS join", "JOIN ... ON SEM_JOIN"],
    )
    .await;
    expect(
        "SELECT * FROM a JOIN c ON a.title = c.name WHERE SEM_JOIN('same?', 'm', a.title, c.name)",
        &["comma or CROSS join"],
    )
    .await;
}

#[tokio::test]
async fn sem_join_supports_only_inner_joins() {
    expect(
        "SELECT * FROM a LEFT JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name)",
        &["SEM_JOIN supports only INNER joins; found Left JOIN"],
    )
    .await;
}

#[tokio::test]
async fn a_two_sided_condition_with_sem_join() {
    expect(
        "SELECT * FROM a JOIN c ON a.title = c.name AND SEM_JOIN('same?', 'm', a.title, c.name)",
        &["over both sides of the join", "SEM_FILTER("],
    )
    .await;
    expect(
        "SELECT * FROM a, c WHERE a.title = c.name AND SEM_JOIN('same?', 'm', a.title, c.name)",
        &["over both sides of the join"],
    )
    .await;
}

#[tokio::test]
async fn sem_filter_in_on() {
    expect(
        "SELECT * FROM a JOIN c ON SEM_JOIN('same?', 'm', a.title, c.name) \
         AND SEM_FILTER('cheap?', 'm', a.price)",
        &[
            "SEM_FILTER is not supported in a JOIN's ON clause",
            "move it to WHERE",
        ],
    )
    .await;
}

#[tokio::test]
async fn sem_join_shape_errors() {
    expect(
        "SELECT * FROM a JOIN c ON SEM_JOIN('x', 'm', a.title, c.name) \
         AND SEM_JOIN('y', 'm', a.title, c.kind)",
        &["only one SEM_JOIN per join"],
    )
    .await;
    expect(
        "SELECT * FROM a JOIN c ON SEM_JOIN('x', 'm', a.title, a.price)",
        &["at least one column from each side", "right side"],
    )
    .await;
}

#[tokio::test]
async fn markers_left_in_other_nodes() {
    expect(
        "SELECT count(EXTEND('i', 'm', review)) FROM reviews",
        &[
            "EXTEND(...) is not supported in an Aggregate",
            "SELECT list",
        ],
    )
    .await;
    expect(
        "SELECT count(*) FROM reviews GROUP BY EXTEND('i', 'm', review)",
        &["EXTEND(...) is not supported in an Aggregate"],
    )
    .await;
    expect(
        "SELECT review FROM reviews ORDER BY EXTEND('i', 'm', review)",
        &["EXTEND(...) is not supported in a Sort (ORDER BY)"],
    )
    .await;
    expect(
        "SELECT row_number() OVER (ORDER BY EXTEND('i', 'm', review)) FROM reviews",
        &["EXTEND(...) is not supported in a Window"],
    )
    .await;
    expect(
        "SELECT * FROM a, c, b WHERE SEM_JOIN('same?', 'm', a.title, c.name)",
        &["at least one column from each side", "per pair"],
    )
    .await;
}

#[tokio::test]
async fn malformed_calls_are_reported() {
    // Predicates are not type-checked at SQL planning time.
    expect(
        "SELECT * FROM papers WHERE SEM_FILTER('i', 'm')",
        &["SEM_FILTER", "Usage: SEM_FILTER("],
    )
    .await;
    expect(
        "SELECT * FROM a JOIN c ON SEM_JOIN('i', 'm', a.title)",
        &["SEM_JOIN", "Usage: SEM_JOIN("],
    )
    .await;
    // A placeholder passes the planning-time check; the rule rejects it.
    expect(
        "SELECT * FROM papers WHERE SEM_FILTER($1, 'm', abstract)",
        &["SEM_FILTER", "string literal"],
    )
    .await;
    expect(
        "SELECT EXTEND('i', 'm', review, review) FROM reviews",
        &["EXTEND", "twice"],
    )
    .await;
}

#[tokio::test]
async fn an_ambiguous_bare_name_is_rejected() {
    // After a join that renamed `b.title`, the stock schema has two `title`s.
    expect(
        "SELECT * FROM a JOIN b ON SEM_JOIN('same?', 'm', a.title, b.name) \
         WHERE SEM_FILTER('x', 'm', b.title)",
        &["SEM_FILTER", "ambiguous"],
    )
    .await;
}

#[tokio::test]
async fn a_dotted_sem_join_input_name_is_rejected() {
    expect(
        "SELECT * FROM a JOIN (SELECT name AS \"user.name\" FROM c) u \
         ON SEM_JOIN('same?', 'm', a.title, u.\"user.name\")",
        &[
            "SEM_JOIN",
            "\"user.name\"",
            "'.' in its name",
            "rename it in a subquery",
        ],
    )
    .await;
    expect(
        "SELECT * FROM (SELECT title AS \"t.x\" FROM a) u, c \
         WHERE SEM_JOIN('same?', 'm', u.\"t.x\", c.name)",
        &["the left input column \"t.x\"", "rename it in a subquery"],
    )
    .await;
}

#[tokio::test]
async fn sem_join_in_where_over_three_relations_names_the_from_order() {
    // SEM_JOIN compares a and c, but the last comma join is (a, c) x papers.
    expect(
        "SELECT * FROM a, c, papers WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
         AND a.price = papers.year",
        &[
            "none is from the right side",
            "the two sides of the last comma/CROSS join in FROM",
            "not reordered",
            "JOIN ... ON SEM_JOIN(...)",
        ],
    )
    .await;
    // Listing one of the two last works.
    let ctx = session();
    analyze(
        &ctx,
        "SELECT * FROM a, papers, c WHERE SEM_JOIN('same?', 'm', a.title, c.name) \
         AND a.price = papers.year",
    )
    .await;
}
