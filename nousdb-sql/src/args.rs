//! Argument positions of the semantic SQL functions, and the typed calls they
//! parse into. This is the only module that knows the order of arguments.
//!
//! Argument order is always instruction, model, then the rest. The scalar
//! functions are parsed from logical [`Expr`]s (by the analyzer rule); the
//! SQL-planning-time check is [`return_field`].

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, FieldRef, SchemaRef};
use datafusion::common::{Column, ScalarValue};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, ReturnFieldArgs};

use crate::markers::MarkerKind;
use crate::schema_dsl::{OutputSpec, parse_output_spec, parse_schema};

/// Position of the instruction literal.
pub const INSTRUCTION: usize = 0;
/// Position of the model literal.
pub const MODEL: usize = 1;
/// Position of the first input column (scalar `EXTEND`, `SEM_FILTER`, `SEM_JOIN`).
pub const FIRST_INPUT: usize = 2;
/// `GENERATE`: position of the schema literal.
pub const SCHEMA: usize = 2;
/// `GENERATE`: position of the row-count literal.
pub const NUM_ROWS: usize = 3;
/// `GENERATE`: position of the optional few-shot examples literal.
pub const EXAMPLES: usize = 4;

/// Positions of the relation form `EXTEND(<relation>, instruction, model, ...)`,
/// which is the scalar layout shifted by one for the leading relation.
/// [`RelationExtendCall::parse`] takes the arguments after the relation, so it
/// uses the unshifted constants above.
pub mod relation_extend {
    /// Position of the input relation (table or subquery).
    pub const INPUT_RELATION: usize = 0;
    /// Position of the instruction literal.
    pub const INSTRUCTION: usize = super::INSTRUCTION + 1;
    /// Position of the model literal.
    pub const MODEL: usize = super::MODEL + 1;
    /// Position of the first input column.
    pub const FIRST_INPUT: usize = super::FIRST_INPUT + 1;
}

const EXTEND_USAGE: &str =
    "EXTEND(instruction, model, column [, column ...] [, 'type' | 'name: type'])";
const SEM_FILTER_USAGE: &str = "SEM_FILTER(instruction, model, column [, column ...])";
const SEM_JOIN_USAGE: &str = "SEM_JOIN(instruction, model, column, column [, column ...])";
const GENERATE_USAGE: &str =
    "GENERATE(instruction, model, 'name: type[, ...]', num_rows [, examples])";
const RELATION_EXTEND_USAGE: &str = "EXTEND(<table or subquery>, instruction, model, \
     column [, column ...], 'name: type[, ...]')";

/// Which function a parse is for: its uppercase name and usage line.
#[derive(Clone, Copy)]
struct Func {
    name: &'static str,
    usage: &'static str,
    /// The position in the SQL call (1-based) of the first argument `parse`
    /// receives: 2 for relation-form `EXTEND`, whose input relation comes
    /// first, 1 for the others.
    first: usize,
}

const EXTEND: Func = Func {
    name: "EXTEND",
    usage: EXTEND_USAGE,
    first: 1,
};
const SEM_FILTER: Func = Func {
    name: "SEM_FILTER",
    usage: SEM_FILTER_USAGE,
    first: 1,
};
const SEM_JOIN: Func = Func {
    name: "SEM_JOIN",
    usage: SEM_JOIN_USAGE,
    first: 1,
};
const GENERATE: Func = Func {
    name: "GENERATE",
    usage: GENERATE_USAGE,
    first: 1,
};
const RELATION_EXTEND: Func = Func {
    name: "EXTEND",
    usage: RELATION_EXTEND_USAGE,
    first: 2,
};

impl Func {
    fn of(kind: MarkerKind) -> Self {
        match kind {
            MarkerKind::Extend => EXTEND,
            MarkerKind::SemFilter => SEM_FILTER,
            MarkerKind::SemJoin => SEM_JOIN,
        }
    }

    fn error<T>(&self, msg: impl std::fmt::Display) -> Result<T> {
        Err(DataFusionError::Plan(format!(
            "{}: {msg}. Usage: {}",
            self.name, self.usage
        )))
    }

    /// Re-wraps an error from `schema_dsl` with the function and its usage.
    fn wrap<T>(&self, what: &str, e: DataFusionError) -> Result<T> {
        let msg = match e {
            DataFusionError::Plan(m) => m,
            other => other.to_string(),
        };
        self.error(format!("invalid {what}: {msg}"))
    }
}

/// A call to scalar `EXTEND`: one derived column over `inputs`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendCall {
    /// The natural-language instruction.
    pub instruction: String,
    /// The model id.
    pub model: String,
    /// The input columns, in call order.
    pub inputs: Vec<Column>,
    /// The output column: `data_type` defaults to `Utf8`, `name` is `None`
    /// when no trailing literal gave one.
    pub output: OutputSpec,
}

/// A call to `SEM_FILTER`: a Boolean marker over one or more columns.
#[derive(Debug, Clone, PartialEq)]
pub struct SemFilterCall {
    /// The natural-language predicate.
    pub instruction: String,
    /// The model id.
    pub model: String,
    /// The input columns (at least one).
    pub inputs: Vec<Column>,
}

/// A call to `SEM_JOIN`: a Boolean marker over columns of both join sides.
#[derive(Debug, Clone, PartialEq)]
pub struct SemJoinCall {
    /// The natural-language join condition.
    pub instruction: String,
    /// The model id.
    pub model: String,
    /// The input columns (at least two); the rewrite rule splits them by side.
    pub inputs: Vec<Column>,
}

/// A call to the `GENERATE` table function.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateCall {
    /// What to generate.
    pub instruction: String,
    /// The model id.
    pub model: String,
    /// The declared output schema (all fields nullable).
    pub schema: SchemaRef,
    /// The number of rows to generate (at least 1).
    pub num_rows: usize,
    /// Optional few-shot examples: JSON objects, one per line.
    pub examples: Option<String>,
}

/// A call to the relation form of `EXTEND`, parsed from the arguments after the
/// input relation.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationExtendCall {
    /// The natural-language instruction.
    pub instruction: String,
    /// The model id.
    pub model: String,
    /// The input columns (at least one).
    pub inputs: Vec<Column>,
    /// The derived columns' schema (all fields nullable); the literal is required.
    pub schema: SchemaRef,
}

/// The string held by a non-empty string literal, if `expr` is one.
fn string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Literal(
            ScalarValue::Utf8(Some(s))
            | ScalarValue::LargeUtf8(Some(s))
            | ScalarValue::Utf8View(Some(s)),
            _,
        ) => Some(s),
        _ => None,
    }
}

fn is_string_scalar(sv: &ScalarValue) -> Option<&str> {
    match sv {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Some(s),
        _ => None,
    }
}

fn required_string(f: Func, args: &[Expr], at: usize, what: &str) -> Result<String> {
    match args.get(at) {
        None => f.error(format!("missing {what} (argument {})", at + f.first)),
        Some(e) => match string_literal(e) {
            Some(s) if !s.trim().is_empty() => Ok(s.to_string()),
            Some(_) => f.error(format!(
                "{what} (argument {}) must not be empty",
                at + f.first
            )),
            None => f.error(format!(
                "{what} (argument {}) must be a string literal, found {e}",
                at + f.first
            )),
        },
    }
}

/// Parses `args[from..to]` as input columns: plain columns, no repeats, at
/// least `min` of them.
fn columns(f: Func, args: &[Expr], from: usize, to: usize, min: usize) -> Result<Vec<Column>> {
    let mut out: Vec<Column> = Vec::new();
    for (i, e) in args.iter().enumerate().take(to).skip(from) {
        match e {
            Expr::Column(c) => {
                if out.contains(c) {
                    return f.error(format!(
                        "column {c} is given twice (argument {}); each input column may appear once",
                        i + f.first
                    ));
                }
                out.push(c.clone());
            }
            Expr::OuterReferenceColumn(..) => {
                return f.error(format!(
                    "argument {} is a correlated (outer) column; correlated columns are not supported, \
                     inputs must be columns of the relation the function is applied to",
                    i + f.first
                ));
            }
            other => {
                let hint = if string_literal(other).is_some() {
                    " (a trailing type literal must come after at least one input column)"
                } else {
                    ""
                };
                return f.error(format!(
                    "argument {} is {other}, but inputs must be columns{hint}",
                    i + f.first
                ));
            }
        }
    }
    if out.len() < min {
        return f.error(format!(
            "expected at least {min} input column{}, found {}",
            if min == 1 { "" } else { "s" },
            out.len()
        ));
    }
    Ok(out)
}

impl ExtendCall {
    /// Parses the arguments of scalar `EXTEND(instruction, model, column ... [, spec])`.
    pub fn parse(args: &[Expr]) -> Result<Self> {
        let f = EXTEND;
        let instruction = required_string(f, args, INSTRUCTION, "instruction")?;
        let model = required_string(f, args, MODEL, "model")?;
        if args.len() <= FIRST_INPUT {
            return f.error("expected at least one input column");
        }
        let last = args.len() - 1;
        let has_spec = last > FIRST_INPUT && string_literal(&args[last]).is_some();
        let end = if has_spec { last } else { args.len() };
        let inputs = columns(f, args, FIRST_INPUT, end, 1)?;
        let output = if has_spec {
            let text = string_literal(&args[last]).unwrap_or_default();
            parse_output_spec(text).or_else(|e| f.wrap("output type", e))?
        } else {
            default_output()
        };
        Ok(Self {
            instruction,
            model,
            inputs,
            output,
        })
    }
}

fn default_output() -> OutputSpec {
    OutputSpec {
        name: None,
        data_type: DataType::Utf8,
    }
}

impl SemFilterCall {
    /// Parses the arguments of `SEM_FILTER(instruction, model, column [, column ...])`.
    pub fn parse(args: &[Expr]) -> Result<Self> {
        let f = SEM_FILTER;
        let instruction = required_string(f, args, INSTRUCTION, "instruction")?;
        let model = required_string(f, args, MODEL, "model")?;
        let inputs = columns(f, args, FIRST_INPUT, args.len(), 1)?;
        Ok(Self {
            instruction,
            model,
            inputs,
        })
    }
}

impl SemJoinCall {
    /// Parses the arguments of `SEM_JOIN(instruction, model, column, column [, column ...])`.
    pub fn parse(args: &[Expr]) -> Result<Self> {
        let f = SEM_JOIN;
        let instruction = required_string(f, args, INSTRUCTION, "instruction")?;
        let model = required_string(f, args, MODEL, "model")?;
        let inputs = columns(f, args, FIRST_INPUT, args.len(), 2)?;
        Ok(Self {
            instruction,
            model,
            inputs,
        })
    }
}

impl GenerateCall {
    /// Parses the arguments of
    /// `GENERATE(instruction, model, 'schema', num_rows [, examples])`.
    pub fn parse(args: &[Expr]) -> Result<Self> {
        let f = GENERATE;
        if args.len() < NUM_ROWS + 1 || args.len() > EXAMPLES + 1 {
            return f.error(format!("expected 4 or 5 arguments, found {}", args.len()));
        }
        let instruction = required_string(f, args, INSTRUCTION, "instruction")?;
        let model = required_string(f, args, MODEL, "model")?;
        let schema_text = required_string(f, args, SCHEMA, "schema")?;
        let schema = parse_schema(&schema_text).or_else(|e| f.wrap("schema", e))?;
        let num_rows = match &args[NUM_ROWS] {
            Expr::Literal(sv, _) => match integer_value(sv) {
                Some(n) if n >= 1 => n as usize,
                Some(n) => {
                    return f.error(format!(
                        "num_rows (argument 4) must be at least 1, found {n}"
                    ));
                }
                None => {
                    return f.error(format!(
                        "num_rows (argument 4) must be an integer literal, found {sv}"
                    ));
                }
            },
            other => {
                return f.error(format!(
                    "num_rows (argument 4) must be an integer literal, found {other}"
                ));
            }
        };
        let examples = match args.get(EXAMPLES) {
            None => None,
            Some(e) => match string_literal(e) {
                Some(s) => Some(s.to_string()),
                None => {
                    return f.error(format!(
                        "examples (argument 5) must be a string literal, found {e}"
                    ));
                }
            },
        };
        Ok(Self {
            instruction,
            model,
            schema,
            num_rows,
            examples,
        })
    }
}

/// An integer literal's value (`i128`, so every unsigned width fits).
fn integer_value(sv: &ScalarValue) -> Option<i128> {
    Some(match sv {
        ScalarValue::Int8(Some(v)) => *v as i128,
        ScalarValue::Int16(Some(v)) => *v as i128,
        ScalarValue::Int32(Some(v)) => *v as i128,
        ScalarValue::Int64(Some(v)) => *v as i128,
        ScalarValue::UInt8(Some(v)) => *v as i128,
        ScalarValue::UInt16(Some(v)) => *v as i128,
        ScalarValue::UInt32(Some(v)) => *v as i128,
        ScalarValue::UInt64(Some(v)) => *v as i128,
        _ => return None,
    })
}

impl RelationExtendCall {
    /// Parses the arguments AFTER the input relation:
    /// `instruction, model, column [, column ...], 'schema'`.
    pub fn parse(args: &[Expr]) -> Result<Self> {
        let f = RELATION_EXTEND;
        let instruction = required_string(f, args, INSTRUCTION, "instruction")?;
        let model = required_string(f, args, MODEL, "model")?;
        if args.len() <= FIRST_INPUT + 1 {
            return f
                .error("expected at least one input column followed by the output schema literal");
        }
        let last = args.len() - 1;
        let Some(schema_text) = string_literal(&args[last]) else {
            return f.error(format!(
                "the last argument must be the output schema literal, found {}",
                args[last]
            ));
        };
        let schema = parse_schema(schema_text).or_else(|e| f.wrap("schema", e))?;
        let inputs = columns(f, args, FIRST_INPUT, last, 1)?;
        Ok(Self {
            instruction,
            model,
            inputs,
            schema,
        })
    }
}

/// The marker's output field at SQL-planning time.
///
/// Runs with `ReturnFieldArgs`, where literals are `Some` and everything else
/// `None`. Validates what can be validated there: arity, a malformed output
/// spec, and an instruction or model that is neither a string literal nor
/// `Null`-typed (an untyped `$1` placeholder; the rewrite rule re-validates
/// with the `parse` functions).
pub fn return_field(kind: MarkerKind, args: &ReturnFieldArgs) -> Result<FieldRef> {
    let f = Func::of(kind);
    let n = args.scalar_arguments.len();
    let min = match kind {
        MarkerKind::Extend | MarkerKind::SemFilter => FIRST_INPUT + 1,
        MarkerKind::SemJoin => FIRST_INPUT + 2,
    };
    if n < min {
        return f.error(format!("expected at least {min} arguments, found {n}"));
    }
    for (at, what) in [(INSTRUCTION, "instruction"), (MODEL, "model")] {
        match args.scalar_arguments[at] {
            Some(sv) => match is_string_scalar(sv) {
                Some(s) if s.trim().is_empty() => {
                    return f.error(format!("{what} (argument {}) must not be empty", at + 1));
                }
                Some(_) => {}
                None if matches!(sv, ScalarValue::Null) => {}
                None => {
                    return f.error(format!(
                        "{what} (argument {}) must be a string literal, found {sv}",
                        at + 1
                    ));
                }
            },
            None => {
                let placeholder = args
                    .arg_fields
                    .get(at)
                    .is_some_and(|fld| fld.data_type() == &DataType::Null);
                if !placeholder {
                    return f.error(format!(
                        "{what} (argument {}) must be a string literal, not a column or expression",
                        at + 1
                    ));
                }
            }
        }
    }
    let last = n - 1;
    let spec_literal = (kind == MarkerKind::Extend && last > FIRST_INPUT)
        .then(|| args.scalar_arguments[last].and_then(is_string_scalar))
        .flatten();
    let input_end = if spec_literal.is_some() { last } else { n };
    let inputs = args.scalar_arguments.iter().enumerate();
    for (at, &arg) in inputs.take(input_end).skip(FIRST_INPUT) {
        if let Some(sv) = arg {
            let hint = if is_string_scalar(sv).is_some() {
                " (a trailing type literal must come after at least one input column)"
            } else {
                ""
            };
            return f.error(format!(
                "argument {} is the literal {sv}, but inputs must be columns{hint}",
                at + 1
            ));
        }
    }
    let data_type = match spec_literal {
        Some(text) => {
            parse_output_spec(text)
                .or_else(|e| f.wrap("output type", e))?
                .data_type
        }
        None => DataType::Utf8,
    };
    Ok(match kind {
        MarkerKind::Extend => Arc::new(Field::new(kind.name(), data_type, true)),
        MarkerKind::SemFilter | MarkerKind::SemJoin => {
            Arc::new(Field::new(kind.name(), DataType::Boolean, true))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};

    fn err(r: Result<impl std::fmt::Debug>) -> String {
        r.expect_err("expected an error").to_string()
    }

    fn qcol(rel: &str, name: &str) -> Expr {
        Expr::Column(Column::new(Some(rel), name))
    }

    #[test]
    fn extend_defaults_to_utf8_without_a_name() {
        let call = ExtendCall::parse(&[lit("rate"), lit("m"), col("review")]).unwrap();
        assert_eq!(call.instruction, "rate");
        assert_eq!(call.model, "m");
        assert_eq!(call.inputs, vec![Column::from_name("review")]);
        assert_eq!(call.output, default_output());
    }

    #[test]
    fn extend_trailing_spec_with_several_inputs() {
        let call = ExtendCall::parse(&[
            lit("rate"),
            lit("m"),
            col("review"),
            qcol("t", "title"),
            lit("Stars: Int64"),
        ])
        .unwrap();
        assert_eq!(call.inputs.len(), 2);
        assert_eq!(call.inputs[1], Column::new(Some("t"), "title"));
        assert_eq!(
            call.output,
            OutputSpec {
                name: Some("stars".into()),
                data_type: DataType::Int64
            }
        );
        let typed = ExtendCall::parse(&[lit("i"), lit("m"), col("a"), lit("float")]).unwrap();
        assert_eq!(typed.output.name, None);
        assert_eq!(typed.output.data_type, DataType::Float64);
    }

    #[test]
    fn extend_errors() {
        let usage = "EXTEND(instruction, model, column";
        for args in [
            vec![],
            vec![lit("i")],
            vec![lit("i"), lit("m")],
            vec![lit("i"), lit("m"), lit("Int64")],
        ] {
            let msg = err(ExtendCall::parse(&args));
            assert!(msg.starts_with("Error during planning: EXTEND:"), "{msg}");
            assert!(msg.contains(usage), "{msg}");
        }
        assert!(
            err(ExtendCall::parse(&[lit("i"), lit("m"), lit("Int64")]))
                .contains("inputs must be columns")
        );
        // bad spec, several fields in the spec, empty literal, non-literal instruction
        assert!(
            err(ExtendCall::parse(&[
                lit("i"),
                lit("m"),
                col("a"),
                lit("date")
            ]))
            .contains("'date'")
        );
        assert!(
            err(ExtendCall::parse(&[
                lit("i"),
                lit("m"),
                col("a"),
                lit("a: Utf8, b: Utf8")
            ]))
            .contains("FROM EXTEND(")
        );
        assert!(
            err(ExtendCall::parse(&[lit(""), lit("m"), col("a")])).contains("must not be empty")
        );
        assert!(err(ExtendCall::parse(&[col("i"), lit("m"), col("a")])).contains("string literal"));
        assert!(err(ExtendCall::parse(&[lit("i"), lit(1_i64), col("a")])).contains("model"));
        // non-column inputs, repeated column, outer reference
        assert!(
            err(ExtendCall::parse(&[
                lit("i"),
                lit("m"),
                col("a") + lit(1_i64)
            ]))
            .contains("inputs must be columns")
        );
        assert!(
            err(ExtendCall::parse(&[lit("i"), lit("m"), col("a"), col("a")])).contains("twice")
        );
        let outer = Expr::OuterReferenceColumn(
            Arc::new(Field::new("a", DataType::Utf8, true)),
            Column::from_name("a"),
        );
        assert!(err(ExtendCall::parse(&[lit("i"), lit("m"), outer])).contains("correlated"));
    }

    #[test]
    fn same_name_in_different_relations_is_not_a_repeat() {
        let call =
            SemJoinCall::parse(&[lit("i"), lit("m"), qcol("a", "x"), qcol("b", "x")]).unwrap();
        assert_eq!(call.inputs.len(), 2);
    }

    #[test]
    fn sem_filter_needs_one_column() {
        let call = SemFilterCall::parse(&[lit("i"), lit("m"), col("a"), col("b")]).unwrap();
        assert_eq!(call.inputs.len(), 2);
        let msg = err(SemFilterCall::parse(&[lit("i"), lit("m")]));
        assert!(
            msg.contains("SEM_FILTER") && msg.contains("at least 1 input column"),
            "{msg}"
        );
        assert!(
            err(SemFilterCall::parse(&[lit("i"), lit("m"), lit("x")]))
                .contains("inputs must be columns")
        );
    }

    #[test]
    fn sem_join_needs_two_columns() {
        let msg = err(SemJoinCall::parse(&[lit("i"), lit("m"), col("a")]));
        assert!(
            msg.contains("SEM_JOIN") && msg.contains("at least 2"),
            "{msg}"
        );
        assert!(SemJoinCall::parse(&[lit("i"), lit("m"), col("a"), col("b"), col("c")]).is_ok());
    }

    #[test]
    fn generate_with_and_without_examples() {
        let call = GenerateCall::parse(&[
            lit("startups"),
            lit("m"),
            lit("name: Utf8, N: int"),
            lit(20_i64),
        ])
        .unwrap();
        assert_eq!(call.num_rows, 20);
        assert_eq!(call.examples, None);
        let names: Vec<_> = call
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, vec!["name", "n"]);
        let call = GenerateCall::parse(&[
            lit("startups"),
            lit("m"),
            lit("name: Utf8"),
            Expr::Literal(ScalarValue::UInt8(Some(3)), None),
            lit("{\"name\": \"A\"}"),
        ])
        .unwrap();
        assert_eq!(call.num_rows, 3);
        assert_eq!(call.examples.as_deref(), Some("{\"name\": \"A\"}"));
    }

    #[test]
    fn generate_errors() {
        let ok = |n: Expr| vec![lit("i"), lit("m"), lit("a: Utf8"), n];
        assert!(err(GenerateCall::parse(&ok(lit(0_i64)))).contains("at least 1"));
        assert!(err(GenerateCall::parse(&ok(lit(-5_i64)))).contains("at least 1"));
        assert!(err(GenerateCall::parse(&ok(lit("ten")))).contains("integer literal"));
        assert!(err(GenerateCall::parse(&ok(col("n")))).contains("integer literal"));
        assert!(err(GenerateCall::parse(&[lit("i"), lit("m"), lit("a: Utf8")])).contains("4 or 5"));
        assert!(
            err(GenerateCall::parse(&[
                lit("i"),
                lit("m"),
                lit("a: Utf8"),
                lit(1_i64),
                lit("e"),
                lit("x")
            ]))
            .contains("4 or 5")
        );
        let msg = err(GenerateCall::parse(&[
            lit("i"),
            lit("m"),
            lit("a: date"),
            lit(1_i64),
        ]));
        assert!(
            msg.contains("invalid schema") && msg.contains("'date'") && msg.contains("GENERATE("),
            "{msg}"
        );
        assert!(
            err(GenerateCall::parse(&[
                lit("i"),
                lit("m"),
                col("s"),
                lit(1_i64)
            ]))
            .contains("schema")
        );
        assert!(
            err(GenerateCall::parse(&[
                lit("i"),
                lit("m"),
                lit("a: Utf8"),
                lit(1_i64),
                lit(2_i64)
            ]))
            .contains("examples")
        );
    }

    #[test]
    fn relation_extend_requires_the_schema() {
        let call = RelationExtendCall::parse(&[
            lit("both"),
            lit("m"),
            col("review"),
            lit("sentiment: Utf8, genre: Utf8"),
        ])
        .unwrap();
        assert_eq!(call.inputs, vec![Column::from_name("review")]);
        assert_eq!(call.schema.fields().len(), 2);

        let msg = err(RelationExtendCall::parse(&[
            lit("i"),
            lit("m"),
            col("review"),
        ]));
        assert!(msg.contains("output schema"), "{msg}");
        assert!(
            err(RelationExtendCall::parse(&[
                lit("i"),
                lit("m"),
                lit("a: Utf8")
            ]))
            .contains("at least one input")
        );
        assert!(
            err(RelationExtendCall::parse(&[
                lit("i"),
                lit("m"),
                col("a"),
                lit("Utf8")
            ]))
            .contains("invalid schema")
        );
        assert!(
            err(RelationExtendCall::parse(&[
                lit("i"),
                lit("m"),
                col("a"),
                col("a"),
                lit("x: Utf8")
            ]))
            .contains("twice")
        );
    }

    fn fields(types: &[DataType]) -> Vec<FieldRef> {
        types
            .iter()
            .map(|t| Arc::new(Field::new("a", t.clone(), true)))
            .collect()
    }

    fn rf(
        kind: MarkerKind,
        scalars: &[Option<ScalarValue>],
        types: &[DataType],
    ) -> Result<FieldRef> {
        let arg_fields = fields(types);
        let refs: Vec<Option<&ScalarValue>> = scalars.iter().map(|s| s.as_ref()).collect();
        return_field(
            kind,
            &ReturnFieldArgs {
                arg_fields: &arg_fields,
                scalar_arguments: &refs,
            },
        )
    }

    fn s(v: &str) -> Option<ScalarValue> {
        Some(ScalarValue::Utf8(Some(v.to_string())))
    }

    #[test]
    fn return_field_types() {
        let t = [const { DataType::Utf8 }; 3];
        let f = rf(MarkerKind::Extend, &[s("i"), s("m"), None], &t).unwrap();
        assert_eq!(
            (f.name().as_str(), f.data_type()),
            ("extend", &DataType::Utf8)
        );
        let t4 = [const { DataType::Utf8 }; 4];
        let f = rf(
            MarkerKind::Extend,
            &[s("i"), s("m"), None, s("stars: Int64")],
            &t4,
        )
        .unwrap();
        assert_eq!(f.data_type(), &DataType::Int64);
        assert!(f.is_nullable());
        let f = rf(MarkerKind::SemFilter, &[s("i"), s("m"), None], &t).unwrap();
        assert_eq!(
            (f.name().as_str(), f.data_type()),
            ("sem_filter", &DataType::Boolean)
        );
        let f = rf(MarkerKind::SemJoin, &[s("i"), s("m"), None, None], &t4).unwrap();
        assert_eq!(
            (f.name().as_str(), f.data_type()),
            ("sem_join", &DataType::Boolean)
        );
    }

    #[test]
    fn return_field_errors() {
        let t3 = [const { DataType::Utf8 }; 3];
        let t4 = [const { DataType::Utf8 }; 4];
        // arity
        assert!(err(rf(MarkerKind::Extend, &[s("i"), s("m")], &t3[..2])).contains("at least 3"));
        assert!(err(rf(MarkerKind::SemFilter, &[s("i")], &t3[..1])).contains("SEM_FILTER"));
        assert!(err(rf(MarkerKind::SemJoin, &[s("i"), s("m"), None], &t3)).contains("at least 4"));
        // malformed spec
        assert!(
            err(rf(
                MarkerKind::Extend,
                &[s("i"), s("m"), None, s("date")],
                &t4
            ))
            .contains("'date'")
        );
        assert!(
            err(rf(
                MarkerKind::Extend,
                &[s("i"), s("m"), None, s("a: Utf8, b: Utf8")],
                &t4
            ))
            .contains("FROM EXTEND(")
        );
        // spec literal without an input column
        assert!(
            err(rf(MarkerKind::Extend, &[s("i"), s("m"), s("Int64")], &t3))
                .contains("inputs must be columns")
        );
        // non-literal instruction / model; Null-typed placeholder accepted
        assert!(
            err(rf(MarkerKind::SemFilter, &[None, s("m"), None], &t3)).contains("string literal")
        );
        assert!(
            err(rf(
                MarkerKind::SemFilter,
                &[s("i"), Some(ScalarValue::Int64(Some(1))), None],
                &t3
            ))
            .contains("model")
        );
        assert!(
            err(rf(MarkerKind::SemFilter, &[s(""), s("m"), None], &t3))
                .contains("must not be empty")
        );
        let placeholders = [DataType::Null, DataType::Null, DataType::Utf8];
        assert!(rf(MarkerKind::SemFilter, &[None, None, None], &placeholders).is_ok());
    }
}
