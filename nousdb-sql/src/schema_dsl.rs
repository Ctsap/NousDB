//! The output-schema literal: `'Int64'`, `'label: Int64'`, `'a: Utf8, b: Utf8'`.
//!
//! A schema is a comma-separated list of `name: Type` fields. Types are
//! case-insensitive; an unquoted name is lowercased like a SQL identifier and a
//! `"Quoted Name"` is kept verbatim (`""` inside is an escaped quote). Commas
//! and colons inside a quoted name do not split.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};

/// Human-readable list of the supported types, used in error messages.
pub const SUPPORTED_TYPES: &str = "Utf8 (text, string, varchar), Int64 (int, integer, bigint), \
     Float64 (float, double, real), Boolean (bool)";

fn plan_error<T>(msg: String) -> Result<T> {
    Err(DataFusionError::Plan(msg))
}

/// Parses a type name (case-insensitive, surrounding whitespace ignored).
///
/// `utf8 | text | string | varchar`, `int64 | int | integer | bigint`,
/// `float64 | float | double | real`, `boolean | bool`. Anything else is a plan
/// error naming the type and listing the supported ones.
pub fn parse_type(s: &str) -> Result<DataType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "utf8" | "text" | "string" | "varchar" => Ok(DataType::Utf8),
        "int64" | "int" | "integer" | "bigint" => Ok(DataType::Int64),
        "float64" | "float" | "double" | "real" => Ok(DataType::Float64),
        "boolean" | "bool" => Ok(DataType::Boolean),
        _ => plan_error(format!(
            "unsupported type '{}'; supported types: {SUPPORTED_TYPES}",
            s.trim()
        )),
    }
}

/// The trailing literal of scalar EXTEND: `'Int64'` (type only) or
/// `'label: Int64'` (name and type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSpec {
    /// The column name, when the literal gave one.
    pub name: Option<String>,
    /// The column type.
    pub data_type: DataType,
}

/// Parses the trailing literal of scalar EXTEND.
///
/// A literal naming more than one field is a plan error that points to the
/// relation form `SELECT ... FROM EXTEND(<table or subquery>, ...)`.
pub fn parse_output_spec(s: &str) -> Result<OutputSpec> {
    let parts = split_top_level(s, ',', s)?;
    if parts.len() > 1 {
        return plan_error(format!(
            "'{s}' names {} columns, but scalar EXTEND produces one; to add several columns \
             use the relation form: SELECT ... FROM EXTEND(<table or subquery>, ...)",
            parts.len()
        ));
    }
    let field = parts.into_iter().next().unwrap_or_default();
    if field.trim().is_empty() {
        return plan_error(format!(
            "empty output type '{s}'; expected a type such as 'Int64' or 'label: Int64' \
             (supported types: {SUPPORTED_TYPES})"
        ));
    }
    match find_top_level(&field, ':')? {
        None => Ok(OutputSpec {
            name: None,
            data_type: parse_type(&field)?,
        }),
        Some(at) => {
            let (name, data_type) = name_and_type(&field, at)?;
            Ok(OutputSpec {
                name: Some(name),
                data_type,
            })
        }
    }
}

/// Parses `'a: Utf8, b: Int64'` into a schema: fields in order, all nullable,
/// at least one.
pub fn parse_schema(s: &str) -> Result<SchemaRef> {
    if s.trim().is_empty() {
        return plan_error(format!(
            "empty schema '{s}'; expected 'name: Type[, name: Type ...]' \
             (supported types: {SUPPORTED_TYPES})"
        ));
    }
    let mut seen = HashSet::new();
    let mut fields = Vec::new();
    for part in split_top_level(s, ',', s)? {
        if part.trim().is_empty() {
            return plan_error(format!(
                "empty field in schema '{s}' (a doubled or trailing comma?)"
            ));
        }
        let Some(at) = find_top_level(&part, ':')? else {
            return plan_error(format!(
                "field '{}' in schema '{s}' has no type; expected 'name: Type'",
                part.trim()
            ));
        };
        let (name, data_type) = name_and_type(&part, at)?;
        if !seen.insert(name.clone()) {
            return plan_error(format!("duplicate field name '{name}' in schema '{s}'"));
        }
        fields.push(Field::new(name, data_type, true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Splits `field` at the colon at byte offset `at` into a normalized name and a
/// parsed type.
fn name_and_type(field: &str, at: usize) -> Result<(String, DataType)> {
    let name = normalize_name(&field[..at], field)?;
    let data_type = parse_type(&field[at + 1..])?;
    Ok((name, data_type))
}

/// An unquoted name is lowercased; a double-quoted one is kept verbatim with
/// `""` read as a quote.
fn normalize_name(raw: &str, field: &str) -> Result<String> {
    let raw = raw.trim();
    let name = if let Some(rest) = raw.strip_prefix('"') {
        let Some(inner) = rest.strip_suffix('"') else {
            return plan_error(format!(
                "unterminated quoted name in field '{}'",
                field.trim()
            ));
        };
        // `strip_suffix` on a lone `"` leaves the opening quote's remainder
        // empty only when raw == "\"", which has no closing quote.
        if raw.len() < 2 {
            return plan_error(format!(
                "unterminated quoted name in field '{}'",
                field.trim()
            ));
        }
        inner.replace("\"\"", "\"")
    } else if raw.contains('"') {
        return plan_error(format!(
            "misplaced quote in name '{raw}' (field '{}'); quote the whole name",
            field.trim()
        ));
    } else {
        raw.to_lowercase()
    };
    if name.is_empty() {
        return plan_error(format!("empty column name in field '{}'", field.trim()));
    }
    Ok(name)
}

/// Splits on `sep` outside double quotes. `whole` is echoed in errors.
fn split_top_level(s: &str, sep: char, whole: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_quote && chars.peek() == Some(&'"') {
                current.push_str("\"\"");
                chars.next();
                continue;
            }
            in_quote = !in_quote;
            current.push(c);
        } else if c == sep && !in_quote {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    if in_quote {
        return plan_error(format!("unterminated quote in '{whole}'"));
    }
    parts.push(current);
    Ok(parts)
}

/// Byte offset of the first `sep` outside double quotes.
fn find_top_level(s: &str, sep: char) -> Result<Option<usize>> {
    let mut in_quote = false;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '"' {
            if in_quote && matches!(chars.peek(), Some((_, '"'))) {
                chars.next();
                continue;
            }
            in_quote = !in_quote;
        } else if c == sep && !in_quote {
            return Ok(Some(i));
        }
    }
    if in_quote {
        return plan_error(format!("unterminated quote in '{s}'"));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(r: Result<impl std::fmt::Debug>) -> String {
        r.expect_err("expected an error").to_string()
    }

    #[test]
    fn types_and_their_synonyms() {
        for s in ["utf8", "TEXT", " String ", "varchar", "Utf8"] {
            assert_eq!(parse_type(s).unwrap(), DataType::Utf8, "{s}");
        }
        for s in ["int64", "INT", "integer", "BigInt"] {
            assert_eq!(parse_type(s).unwrap(), DataType::Int64, "{s}");
        }
        for s in ["float64", "float", "DOUBLE", "real"] {
            assert_eq!(parse_type(s).unwrap(), DataType::Float64, "{s}");
        }
        for s in ["boolean", "BOOL"] {
            assert_eq!(parse_type(s).unwrap(), DataType::Boolean, "{s}");
        }
    }

    #[test]
    fn unsupported_types_name_themselves_and_list_the_supported() {
        for bad in ["int8", "utf8view", "date", ""] {
            let msg = err(parse_type(bad));
            assert!(msg.contains(&format!("'{bad}'")), "{msg}");
            assert!(msg.contains("Int64") && msg.contains("Boolean"), "{msg}");
        }
    }

    #[test]
    fn output_spec_type_only_and_named() {
        assert_eq!(
            parse_output_spec("Int64").unwrap(),
            OutputSpec {
                name: None,
                data_type: DataType::Int64
            }
        );
        assert_eq!(
            parse_output_spec("  Stars : bigint ").unwrap(),
            OutputSpec {
                name: Some("stars".into()),
                data_type: DataType::Int64
            }
        );
        assert_eq!(
            parse_output_spec("\"My, Label: 1\": text").unwrap(),
            OutputSpec {
                name: Some("My, Label: 1".into()),
                data_type: DataType::Utf8
            }
        );
    }

    #[test]
    fn output_spec_rejects_several_fields_pointing_to_the_relation_form() {
        let msg = err(parse_output_spec("a: Utf8, b: Int64"));
        assert!(msg.contains("FROM EXTEND("), "{msg}");
        assert!(msg.contains("a: Utf8, b: Int64"), "{msg}");
        // A comma inside a quoted name is one field.
        assert!(parse_output_spec("\"a,b\": Utf8").is_ok());
    }

    #[test]
    fn output_spec_errors() {
        assert!(err(parse_output_spec("")).contains("empty"));
        assert!(err(parse_output_spec("x: date")).contains("'date'"));
        assert!(err(parse_output_spec(": Int64")).contains("empty column name"));
        assert!(err(parse_output_spec("\"x: Int64")).contains("unterminated"));
    }

    #[test]
    fn schema_in_order_all_nullable() {
        let s = parse_schema("name: Utf8, Score: float, ok: bool").unwrap();
        let fields: Vec<_> = s
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type().clone(), f.is_nullable()))
            .collect();
        assert_eq!(
            fields,
            vec![
                ("name", DataType::Utf8, true),
                ("score", DataType::Float64, true),
                ("ok", DataType::Boolean, true),
            ]
        );
    }

    #[test]
    fn schema_quoted_names_are_verbatim() {
        let s = parse_schema(r#""Full Name": Utf8, "a, b: c": Int64, "say ""hi""": text"#).unwrap();
        let names: Vec<_> = s.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, vec!["Full Name", "a, b: c", "say \"hi\""]);
    }

    #[test]
    fn schema_duplicates_are_detected_after_normalization() {
        let msg = err(parse_schema("A: Utf8, a: Int64"));
        assert!(msg.contains("duplicate") && msg.contains("'a'"), "{msg}");
        // Quoted names are case-sensitive, so these differ.
        assert!(parse_schema(r#""A": Utf8, "a": Int64"#).is_ok());
        assert!(parse_schema(r#""A": Utf8, a: Int64, A: Utf8"#).is_err());
    }

    #[test]
    fn schema_errors_echo_the_offending_text() {
        assert!(err(parse_schema("")).contains("empty schema"));
        assert!(err(parse_schema("   ")).contains("empty schema"));
        let msg = err(parse_schema("a: Utf8,"));
        assert!(
            msg.contains("empty field") && msg.contains("a: Utf8,"),
            "{msg}"
        );
        assert!(err(parse_schema("a: Utf8,, b: Int64")).contains("empty field"));
        let msg = err(parse_schema("a: Utf8, b"));
        assert!(msg.contains("'b'") && msg.contains("no type"), "{msg}");
        assert!(err(parse_schema("Int64")).contains("no type"));
        assert!(err(parse_schema(": Utf8")).contains("empty column name"));
        assert!(err(parse_schema(r#""": Utf8"#)).contains("empty column name"));
        assert!(err(parse_schema("a: date")).contains("'date'"));
        assert!(err(parse_schema(r#"a"b": Utf8"#)).contains("misplaced quote"));
        assert!(err(parse_schema(r#""a: Utf8"#)).contains("unterminated"));
    }
}
