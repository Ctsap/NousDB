//! Column-name validation for the semantic join, with descriptive plan errors.

use datafusion::common::{Column, DFSchemaRef};
use datafusion::error::{DataFusionError, Result};

/// Resolves `column_name` (`column` or `table.column`) against `schema`.
///
/// Returns a plan error listing the available columns when it is not found.
pub fn validate_column_exists(
    column_name: &str,
    schema: &DFSchemaRef,
    relation_description: &str,
) -> Result<Column> {
    let column = if column_name.contains('.') {
        parse_qualified_column(column_name)?
    } else {
        Column::from_name(column_name)
    };

    match schema.index_of_column(&column) {
        Ok(_index) => Ok(column),
        Err(_) => {
            generate_column_not_found_error(column_name, &column, schema, relation_description)
        }
    }
}

fn parse_qualified_column(qualified_name: &str) -> Result<Column> {
    let parts: Vec<&str> = qualified_name.split('.').collect();

    match parts.len() {
        2 => Ok(Column::new(Some(parts[0]), parts[1])),
        1 => Ok(Column::from_name(parts[0])),
        _ => Err(DataFusionError::Plan(format!(
            "Invalid column name '{}': expected format 'table.column' or 'column', \
                 found {} components separated by dots",
            qualified_name,
            parts.len()
        ))),
    }
}

fn generate_column_not_found_error(
    original_name: &str,
    parsed_column: &Column,
    schema: &DFSchemaRef,
    relation_description: &str,
) -> Result<Column> {
    let available_columns = get_available_columns_description(schema);

    if let Some(qualifier) = &parsed_column.relation {
        // A qualifier was given but did not match: check whether the bare name
        // exists under a different qualifier.
        let unqualified_column = Column::from_name(&parsed_column.name);

        if schema.index_of_column(&unqualified_column).is_ok() {
            Err(DataFusionError::Plan(format!(
                "Column '{}' not found in {} schema with qualifier '{}'. \
                 \nThe column '{}' exists but with a different table qualifier. \
                 \nAvailable columns: {} \
                 \n\nTip: Try using just '{}' without the qualifier, or check if you used the correct alias name.",
                original_name,
                relation_description,
                qualifier,
                parsed_column.name,
                available_columns,
                parsed_column.name,
            )))
        } else {
            Err(DataFusionError::Plan(format!(
                "Column '{}' not found in {} schema. \
                 \nAvailable columns: {} \
                 \n\nTip: Check for typos or use one of the available column names.",
                original_name, relation_description, available_columns,
            )))
        }
    } else {
        Err(DataFusionError::Plan(format!(
            "Column '{}' not found in {} schema. \
             \nAvailable columns: {} \
             \n\nTip: Check for typos. If the column exists with a table qualifier, \
             you may need to use the qualified name (e.g., 'table.{}').",
            original_name, relation_description, available_columns, original_name,
        )))
    }
}

fn get_available_columns_description(schema: &DFSchemaRef) -> String {
    let column_names: Vec<String> = schema
        .fields()
        .iter()
        .map(|field| {
            if let Some(qualifier) = field.metadata().get("qualifier") {
                format!("{}.{}", qualifier, field.name())
            } else {
                field.name().to_string()
            }
        })
        .collect();

    format!("[{}]", column_names.join(", "))
}
