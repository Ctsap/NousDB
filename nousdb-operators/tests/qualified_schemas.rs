//! The extend and semantic-join output schemas keep column qualifiers, so a
//! qualified reference (`t.review`, as SQL plans it) resolves above them,
//! while the Arrow schema and bare-name resolution stay what they were.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Column, DFSchema, DFSchemaRef, TableReference};
use datafusion::logical_expr::{EmptyRelation, LogicalPlan};
use nousdb_operators::join::logical::build_join_schema;
use nousdb_operators::{create_extend, create_semantic_join};

fn relation(qualifier: Option<&str>, fields: &[(&str, DataType)]) -> LogicalPlan {
    let schema = Schema::new(
        fields
            .iter()
            .map(|(name, dt)| Field::new(*name, dt.clone(), true))
            .collect::<Vec<_>>(),
    );
    let df_schema = match qualifier {
        Some(q) => DFSchema::try_from_qualified_schema(q, &schema).unwrap(),
        None => DFSchema::try_from(schema).unwrap(),
    };
    LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema: Arc::new(df_schema),
    })
}

fn qualifiers(schema: &DFSchemaRef) -> Vec<(Option<String>, String)> {
    schema
        .iter()
        .map(|(q, f)| (q.map(|q| q.to_string()), f.name().clone()))
        .collect()
}

fn q(qualifier: &str, name: &str) -> (Option<String>, String) {
    (Some(qualifier.to_string()), name.to_string())
}

fn bare(name: &str) -> (Option<String>, String) {
    (None, name.to_string())
}

fn derived() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "sentiment",
        DataType::Utf8,
        true,
    )]))
}

#[test]
fn extend_keeps_input_qualifiers_and_leaves_derived_columns_unqualified() {
    let input = relation(
        Some("t"),
        &[("review", DataType::Utf8), ("id", DataType::Int64)],
    );
    let plan = create_extend(
        input.clone(),
        "Classify",
        "m",
        vec!["review".to_string()],
        derived(),
    )
    .unwrap();
    let schema = plan.schema();

    assert_eq!(
        qualifiers(schema),
        vec![q("t", "review"), q("t", "id"), bare("sentiment")]
    );

    // The Arrow schema is exactly what the unqualified version produced: the
    // input's fields followed by the derived ones, no metadata.
    let mut fields = input.schema().fields().to_vec();
    fields.extend(derived().fields().iter().cloned());
    let expected = Schema::new(fields);
    assert_eq!(schema.as_arrow(), &expected);
}

#[test]
fn qualified_and_bare_columns_resolve_against_the_extend_schema() {
    let input = relation(
        Some("t"),
        &[("review", DataType::Utf8), ("id", DataType::Int64)],
    );
    let plan = create_extend(
        input,
        "Classify",
        "m",
        vec!["review".to_string()],
        derived(),
    )
    .unwrap();
    let schema = plan.schema();

    assert_eq!(
        schema
            .index_of_column(&Column::new(Some("t"), "review"))
            .unwrap(),
        0
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new_unqualified("review"))
            .unwrap(),
        0
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new_unqualified("sentiment"))
            .unwrap(),
        2
    );
    assert!(schema.field_with_unqualified_name("id").is_ok());
    // A wrong qualifier still does not resolve.
    assert!(
        schema
            .index_of_column(&Column::new(Some("u"), "review"))
            .is_err()
    );
}

#[test]
fn extend_over_an_unqualified_input_is_unchanged() {
    let input = relation(None, &[("review", DataType::Utf8)]);
    let plan = create_extend(
        input,
        "Classify",
        "m",
        vec!["review".to_string()],
        derived(),
    )
    .unwrap();
    assert_eq!(
        qualifiers(plan.schema()),
        vec![bare("review"), bare("sentiment")]
    );
}

#[test]
fn extend_still_rejects_a_derived_name_that_collides_with_a_qualified_input() {
    let input = relation(Some("t"), &[("sentiment", DataType::Utf8)]);
    assert!(
        create_extend(
            input,
            "Classify",
            "m",
            vec!["sentiment".to_string()],
            derived()
        )
        .is_err()
    );
}

#[test]
fn join_schema_keeps_both_sides_qualifiers() {
    let left = relation(
        Some("l"),
        &[("title", DataType::Utf8), ("id", DataType::Int64)],
    );
    let right = relation(Some("r"), &[("name", DataType::Utf8)]);
    let schema = build_join_schema(left.schema(), right.schema()).unwrap();
    assert_eq!(
        qualifiers(&schema),
        vec![q("l", "title"), q("l", "id"), q("r", "name")]
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new(Some("r"), "name"))
            .unwrap(),
        2
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new_unqualified("title"))
            .unwrap(),
        0
    );
    assert_eq!(schema.metadata().len(), 0);
}

#[test]
fn a_renamed_right_column_keeps_the_right_qualifier() {
    let left = relation(
        Some("l"),
        &[("title", DataType::Utf8), ("id", DataType::Int64)],
    );
    let right = relation(
        Some("r"),
        &[("title", DataType::Utf8), ("title_1", DataType::Utf8)],
    );
    let schema = build_join_schema(left.schema(), right.schema()).unwrap();
    // Renaming is by bare name, exactly as before.
    assert_eq!(
        qualifiers(&schema),
        vec![
            q("l", "title"),
            q("l", "id"),
            q("r", "title_1"),
            q("r", "title_1_1")
        ]
    );
    // The bare name resolves to the left column (the right one was renamed).
    assert_eq!(
        schema
            .index_of_column(&Column::new_unqualified("title"))
            .unwrap(),
        0
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new(Some("l"), "title"))
            .unwrap(),
        0
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new(Some("r"), "title_1"))
            .unwrap(),
        2
    );
    let names: Vec<_> = schema
        .as_arrow()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["title", "id", "title_1", "title_1_1"]);
}

#[test]
fn create_semantic_join_output_resolves_qualified_columns() {
    let left = relation(Some("t"), &[("review", DataType::Utf8)]);
    let right = relation(
        Some("r"),
        &[("review", DataType::Utf8), ("title", DataType::Utf8)],
    );
    let plan = create_semantic_join(
        &left,
        &right,
        vec!["t.review".to_string()],
        vec!["r.title".to_string()],
        "matches",
        "m",
    )
    .unwrap();
    let schema = plan.schema();
    assert_eq!(
        qualifiers(schema),
        vec![q("t", "review"), q("r", "review_1"), q("r", "title")]
    );
    assert_eq!(
        schema
            .index_of_column(&Column::new(Some(TableReference::bare("r")), "title"))
            .unwrap(),
        2
    );
}

#[test]
fn unqualified_join_inputs_give_the_unqualified_schema_as_before() {
    let left = relation(None, &[("a", DataType::Utf8)]);
    let right = relation(None, &[("a", DataType::Utf8)]);
    let schema = build_join_schema(left.schema(), right.schema()).unwrap();
    assert_eq!(qualifiers(&schema), vec![bare("a"), bare("a_1")]);
}
