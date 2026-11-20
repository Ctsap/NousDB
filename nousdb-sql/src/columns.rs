//! How a semantic function's input column becomes the name an operator binds.

use datafusion::common::{Column, DFSchema, Result, plan_err};

/// Returns the bare name of `column`, checked against the schema of the
/// relation it is read from.
///
/// The semantic operators bind their inputs by bare column name, so the name
/// must identify exactly one column of `schema`. `function` is the SQL
/// function's name, used in the error.
pub(crate) fn bare_input_name(
    column: &Column,
    schema: &DFSchema,
    function: &str,
) -> Result<String> {
    if !schema.has_column(column) {
        return plan_err!(
            "{function}: column {} is not a column of its input (available: {})",
            column.flat_name(),
            schema.field_names().join(", ")
        );
    }
    let same_name = schema.fields_with_unqualified_name(&column.name).len();
    if same_name > 1 {
        return plan_err!(
            "{function}: column name '{}' is ambiguous: {same_name} columns of its input share it, \
             and semantic operators bind inputs by bare name; rename one in a subquery",
            column.name
        );
    }
    Ok(column.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::TableReference;
    use std::sync::Arc;

    fn schema() -> DFSchema {
        let fields = Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("review", DataType::Utf8, true),
            Field::new("id", DataType::Int64, true),
        ]);
        let qualifiers = vec![
            Some(TableReference::bare("a")),
            Some(TableReference::bare("a")),
            Some(TableReference::bare("b")),
        ];
        DFSchema::from_field_specific_qualified_schema(qualifiers, &Arc::new(fields)).unwrap()
    }

    #[test]
    fn a_unique_name_resolves_qualified_or_not() {
        let schema = schema();
        let bare = Column::new_unqualified("review");
        let qualified = Column::new(Some("a"), "review");
        assert_eq!(bare_input_name(&bare, &schema, "EXTEND").unwrap(), "review");
        assert_eq!(
            bare_input_name(&qualified, &schema, "EXTEND").unwrap(),
            "review"
        );
    }

    #[test]
    fn a_shared_name_is_rejected_even_when_qualified() {
        let error = bare_input_name(&Column::new(Some("a"), "id"), &schema(), "SEM_FILTER")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("SEM_FILTER") && error.contains("ambiguous"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_column_is_rejected() {
        let error = bare_input_name(&Column::new(Some("c"), "review"), &schema(), "EXTEND")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a column of its input"), "{error}");
    }
}
