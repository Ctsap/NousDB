//! Shared helpers for [`BatchRenderer`](super::renderer::BatchRenderer)
//! implementations: converting Arrow values to prompt text, and measuring the
//! static token overhead of the prompt templates.

use datafusion::arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::error::{DataFusionError, Result as DFResult};
use minijinja::value::Value as MValue;
use std::collections::HashMap;

use crate::arrow_util::StringColumn;
use crate::prompt::environment::ENV;
use crate::prompt::prompt::json_format_instructions;
use crate::prompt::reasoning::ReasoningMode;
use crate::token_estimator::estimator::TokenEstimator;

/// Tokens of chat/request framing that surround the rendered prompt (role
/// markers, message separators) and are not part of either template.
const CHAT_FRAMING_TOKENS: usize = 20;

/// Renders a template through the shared [`ENV`], which uses
/// `UndefinedBehavior::Strict`: a variable the template references and `ctx`
/// omits is an error, not an empty string.
fn render_template(name: &str, ctx: minijinja::Value) -> DFResult<String> {
    let env = ENV.read().map_err(|e| {
        DataFusionError::Execution(format!("Failed to acquire environment lock: {}", e))
    })?;
    env.get_template(name)
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to load template '{}': {}", name, e))
        })?
        .render(ctx)
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to render template '{}': {}", name, e))
        })
}

/// Measures the static token overhead of the join prompt: everything constant
/// across requests, i.e. the system template plus the user template rendered
/// with the actual join condition but zero rows and pairs.
///
/// Rendering the real templates keeps the figure accurate when templates are
/// edited. The templates are named by the caller, as in
/// [`flat_static_prompt_tokens`], because the reasoning-mode variants differ
/// substantially in size.
///
/// The reasoning mode is recovered from `system_template` with
/// [`ReasoningMode::from_system_template`], because the user template's
/// output-format example varies with it.
pub fn join_static_prompt_tokens(
    system_template: &str,
    user_template: &str,
    join_condition: &str,
    estimator: &TokenEstimator,
) -> DFResult<usize> {
    let system = render_template(system_template, minijinja::context! {})?;
    // Empty loops render to nothing, so this is exactly the shared prefix of
    // every join request plus the (constant-size) data-section headers.
    let user = render_template(
        user_template,
        minijinja::context! {
            join_condition => join_condition,
            pair_count => 0,
            left_rows => Vec::<MValue>::new(),
            right_rows => Vec::<MValue>::new(),
            pairs => Vec::<MValue>::new(),
            reasoning_mode => ReasoningMode::from_system_template(system_template).label(),
        },
    )?;

    Ok(estimator.estimate_text_tokens(&system)
        + estimator.estimate_text_tokens(&user)
        + CHAT_FRAMING_TOKENS)
}

/// The flat (extend/selection) counterpart of [`join_static_prompt_tokens`]:
/// the token cost of everything constant across requests, i.e. the system
/// template plus the user template rendered with the real instruction but zero
/// rows.
///
/// The templates are named by the caller rather than derived from an operator,
/// so a reasoning-mode variant is measured by passing its names.
///
/// `output_schema` is `Some` for extend, whose user template interpolates
/// `{{ output_schema }}` with `json_format_instructions`; selection passes
/// `None`. A template that references `{{ output_schema }}` while the caller
/// passes `None` fails the render, since silently rendering an empty schema
/// block would undercount the overhead.
///
/// The column-name lists (`input_columns`, `output_columns`) render empty, as
/// `batch_data` does, so any per-column text a template derives from them is
/// not counted.
///
/// The reasoning mode is recovered from `system_template` with
/// [`ReasoningMode::from_system_template`], because selection's output-format
/// example varies with it.
pub fn flat_static_prompt_tokens(
    system_template: &str,
    user_template: &str,
    instruction: &str,
    output_schema: Option<&SchemaRef>,
    estimator: &TokenEstimator,
) -> DFResult<usize> {
    // `Prompt::render_system` renders the system template with an empty
    // context; mirroring that keeps the two byte-identical.
    let system = render_template(system_template, minijinja::context! {})?;

    let mut ctx: HashMap<String, MValue> = HashMap::new();
    // `instruction` (extend) and `expression` (selection) name the same string;
    // supplying both keeps the function template-agnostic.
    ctx.insert("instruction".to_string(), MValue::from(instruction));
    ctx.insert("expression".to_string(), MValue::from(instruction));
    ctx.insert("row_count".to_string(), MValue::from(0usize));
    ctx.insert("batch_data".to_string(), MValue::from(Vec::<MValue>::new()));
    ctx.insert(
        "input_columns".to_string(),
        MValue::from(Vec::<MValue>::new()),
    );
    ctx.insert(
        "output_columns".to_string(),
        MValue::from(Vec::<MValue>::new()),
    );
    ctx.insert(
        "reasoning_mode".to_string(),
        MValue::from(ReasoningMode::from_system_template(system_template).label()),
    );
    if let Some(schema) = output_schema {
        ctx.insert(
            "output_schema".to_string(),
            MValue::from(json_format_instructions(schema)),
        );
    }

    let user = render_template(user_template, MValue::from(ctx))?;

    Ok(estimator.estimate_text_tokens(&system)
        + estimator.estimate_text_tokens(&user)
        + CHAT_FRAMING_TOKENS)
}

/// Renders the value at `index` of an Arrow array as prompt text.
///
/// # Rendering rule
///
/// In order:
///
/// 1. A null (`array.is_null(index)`) renders as the literal `NULL`, for every
///    type.
/// 2. Strings in any encoding render as themselves: `Utf8`, `LargeUtf8` and
///    `Utf8View` all go through [`StringColumn`], so the encoding a plan
///    produces never changes a prompt byte.
/// 3. `Int32`, `Int64`, `Float64` and `Boolean` use Rust's `to_string`. Their
///    prompt bytes are pinned, so they must not move to Arrow's formatter.
/// 4. Every other type (`Int8`/`Int16`/`UInt*`, `Float32`, `Decimal*`,
///    `Date32`/`Date64`, `Time*`, `Timestamp*`, `Duration`, `Interval`,
///    `Binary*`, dictionary-encoded strings, lists, structs, ...) renders with
///    Arrow's display formatter ([`ArrayFormatter`] with default
///    [`FormatOptions`], except that a null inside a nested value spells
///    `NULL` instead of the empty string). Examples: `Date32` renders as
///    `2024-01-15`, `Timestamp(Second, None)` as `2024-01-15T10:30:00`,
///    `Decimal128(10, 2)` as `12.34`.
///
/// Arrow's formatter and Rust's `to_string` disagree on a few float spellings
/// (Arrow writes `1.0` where `Float64` renders `1`). This keeps the four
/// pinned types byte-stable.
///
/// # Errors
///
/// Returns an error if a downcast fails (schema mismatch) or Arrow cannot
/// format the value (for example an out-of-range timestamp). A formatting
/// failure is never written into the prompt.
///
/// # Example
///
/// ```
/// use datafusion::arrow::array::{ArrayRef, Date32Array, Int64Array, StringArray};
/// use nousdb_core::prompt::utils::extract_value_from_array;
/// use std::sync::Arc;
///
/// // String array
/// let strings: ArrayRef = Arc::new(StringArray::from(vec!["hello", "world"]));
/// let value = extract_value_from_array(&strings, 0)?;
/// assert_eq!(value, "hello");
///
/// // Integer array
/// let ints: ArrayRef = Arc::new(Int64Array::from(vec![42, 100]));
/// let value = extract_value_from_array(&ints, 1)?;
/// assert_eq!(value, "100");
///
/// // Any other type renders through Arrow's display formatter.
/// let dates: ArrayRef = Arc::new(Date32Array::from(vec![19737]));
/// let value = extract_value_from_array(&dates, 0)?;
/// assert_eq!(value, "2024-01-15");
/// # Ok::<(), datafusion::error::DataFusionError>(())
/// ```
pub fn extract_value_from_array(array: &ArrayRef, index: usize) -> DFResult<String> {
    // Check for null first: Arrow tracks nulls in a separate bitmap, and the
    // value slot of a null is not meaningful.
    if array.is_null(index) {
        return Ok("NULL".to_string());
    }

    // Strings first, in whichever encoding they arrive: one borrowed read with
    // no formatter allocation, giving the same bytes for `Utf8`, `LargeUtf8`
    // and `Utf8View` by construction.
    if let Some(strings) = StringColumn::try_new(array.as_ref()) {
        return Ok(strings.value(index).to_string());
    }

    match array.data_type() {
        DataType::Int32 => {
            let int_array = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                DataFusionError::Execution(
                    "Failed to downcast to Int32Array. Schema mismatch?".to_string(),
                )
            })?;
            Ok(int_array.value(index).to_string())
        }
        DataType::Int64 => {
            let int_array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                DataFusionError::Execution(
                    "Failed to downcast to Int64Array. Schema mismatch?".to_string(),
                )
            })?;
            Ok(int_array.value(index).to_string())
        }
        DataType::Float64 => {
            let float_array = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "Failed to downcast to Float64Array. Schema mismatch?".to_string(),
                    )
                })?;
            Ok(float_array.value(index).to_string())
        }
        DataType::Boolean => {
            let bool_array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "Failed to downcast to BooleanArray. Schema mismatch?".to_string(),
                    )
                })?;
            Ok(bool_array.value(index).to_string())
        }
        // Every other type: Arrow's display formatting. `try_to_string`
        // (unlike `Display`) reports a formatting failure instead of writing
        // an `ERROR` marker into the prompt, and `with_null` makes a null
        // nested inside a value read `NULL`, matching the top-level rendering.
        _ => {
            let options = FormatOptions::default().with_null("NULL");
            let formatter = ArrayFormatter::try_new(array.as_ref(), &options)?;
            Ok(formatter.value(index).try_to_string()?)
        }
    }
}

/// Extracts the named columns of row `row_idx` as `(name, value)` pairs.
///
/// Values are returned in `columns` order: templates emit `key: value` pairs in
/// iteration order, and prompt bytes must be identical across renders of the
/// same row (prefix caching, reproducibility).
pub fn extract_row_values(
    batch: &RecordBatch,
    row_idx: usize,
    columns: &[String],
) -> DFResult<Vec<(String, String)>> {
    let mut row_data = Vec::with_capacity(columns.len());
    let schema = batch.schema();

    for col_name in columns {
        let col_idx = schema.index_of(col_name).map_err(|_| {
            DataFusionError::Execution(format!(
                "Column '{}' not found in batch. Available columns: {:?}",
                col_name,
                schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
            ))
        })?;

        let column = batch.column(col_idx);
        let value = extract_value_from_array(column, row_idx)?;

        row_data.push((col_name.clone(), value));
    }

    Ok(row_data)
}

/// Converts extracted rows into MiniJinja map values.
///
/// Insertion order survives into template iteration only because minijinja is
/// built with the `preserve_order` feature (see this crate's `Cargo.toml`).
pub fn rows_to_minijinja(rows: Vec<Vec<(String, String)>>) -> Vec<MValue> {
    rows.into_iter()
        .map(|row| MValue::from_iter(row.into_iter().map(|(k, v)| (k, MValue::from(v)))))
        .collect()
}

#[cfg(test)]
mod static_overhead_tests {
    use super::*;
    use datafusion::arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    /// The static overhead is measured from the rendered join templates, so it
    /// is at least the size of the system template.
    #[test]
    fn join_static_overhead_is_measured_not_guessed() {
        let est = TokenEstimator::default_estimator().unwrap();
        let tokens = join_static_prompt_tokens(
            "join/system.md",
            "join/user.md",
            "The review and the name refer to the same actor",
            &est,
        )
        .unwrap();
        assert!(
            tokens >= 400,
            "static join prompt should measure at least 400 tokens, got {tokens}"
        );
        assert!(tokens <= 2000, "sanity upper bound, got {tokens}");
    }

    const INSTRUCTION: &str = "the review expresses a positive opinion about the movie";

    fn extend_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]))
    }

    fn selection_overhead() -> usize {
        let est = TokenEstimator::default_estimator().unwrap();
        flat_static_prompt_tokens(
            "selection/system.md",
            "selection/user.md",
            INSTRUCTION,
            None,
            &est,
        )
        .unwrap()
    }

    /// The extend/selection counterpart of
    /// `join_static_overhead_is_measured_not_guessed`: the overhead exceeds a
    /// guess based on the instruction length alone.
    #[test]
    fn flat_static_overhead_is_measured_not_guessed() {
        let est = TokenEstimator::default_estimator().unwrap();
        let schema = extend_schema();
        let extend = flat_static_prompt_tokens(
            "extend/system.md",
            "extend/user.md",
            INSTRUCTION,
            Some(&schema),
            &est,
        )
        .unwrap();
        let selection = selection_overhead();

        let guessed = INSTRUCTION.len() / 3 + 100;
        for (operator, tokens) in [("extend", extend), ("selection", selection)] {
            assert!(
                tokens >= 350,
                "static {operator} prompt should measure at least 350 tokens, got {tokens}"
            );
            assert!(
                tokens > guessed,
                "measured {operator} overhead {tokens} must exceed the length-based guess {guessed}"
            );
            assert!(
                tokens <= 2000,
                "sanity upper bound for {operator}, got {tokens}"
            );
        }
    }

    /// Growing the system prompt (a worked example, say) must raise the
    /// overhead on both the flat and the join path. The template is registered
    /// on the shared environment under a test-only name rather than added to
    /// the templates directory.
    #[test]
    fn a_longer_system_template_raises_the_measured_overhead() {
        const NAME: &str = "__test_only__/system_with_worked_example.md";
        let long_system = format!(
            "{}\n\n## Worked Example\n\n{}",
            "Follow user instructions for every input row.",
            "Given the row `text: a dazzling, generous film`, the answer is \
             `{\"sentiment\": \"positive\"}`. Note that the verdict is about the \
             film and not about the reviewer's prose. "
                .repeat(8)
        );
        {
            let mut env = ENV.write().unwrap();
            env.add_template_owned(NAME, long_system).unwrap();
        }

        let est = TokenEstimator::default_estimator().unwrap();

        let flat_baseline = selection_overhead();
        let flat_longer =
            flat_static_prompt_tokens(NAME, "selection/user.md", INSTRUCTION, None, &est).unwrap();
        assert!(
            flat_longer > flat_baseline,
            "a longer system template must raise the flat overhead: {flat_longer} vs \
             {flat_baseline}"
        );

        let condition = "The review and the name refer to the same actor";
        let join_baseline =
            join_static_prompt_tokens("join/system.md", "join/user.md", condition, &est).unwrap();
        let join_longer = join_static_prompt_tokens(NAME, "join/user.md", condition, &est).unwrap();
        assert!(
            join_longer > join_baseline,
            "a longer system template must raise the join overhead: {join_longer} vs \
             {join_baseline}"
        );
    }
}

#[cfg(test)]
mod extract_value_tests {
    use super::*;
    use datafusion::arrow::array::{
        Array, BinaryArray, Date32Array, Date64Array, Decimal128Array, DictionaryArray,
        DurationSecondArray, Float32Array, Int8Array, Int16Array, LargeStringArray, ListArray,
        NullArray, StringArray, StringViewArray, Time64MicrosecondArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
        UInt64Array,
    };
    use datafusion::arrow::datatypes::{Field, Int32Type, Schema};
    use std::sync::Arc;

    /// 2024-01-15 10:30:00 UTC, in seconds since the epoch.
    const TS_SECONDS: i64 = 1_705_314_600;
    /// 2024-01-15, in days since the epoch.
    const DATE_DAYS: i32 = 19_737;

    fn render(array: impl Array + 'static, index: usize) -> String {
        let array: ArrayRef = Arc::new(array);
        extract_value_from_array(&array, index).unwrap()
    }

    /// Strings must read the same in every encoding. The values straddle
    /// `Utf8View`'s 12-byte inline limit, because longer ones live in a
    /// separate data buffer and take a different read path.
    #[test]
    fn strings_render_identically_in_every_encoding() {
        let values = [
            "",
            "short",
            "exactly12byt",
            "a considerably longer review that does not fit in a 12-byte view",
            "multi\nline \"quoted\" text with unicode: caf\u{e9} \u{1f3ac}",
        ];
        let utf8: ArrayRef = Arc::new(StringArray::from(values.to_vec()));
        let large: ArrayRef = Arc::new(LargeStringArray::from(values.to_vec()));
        let view: ArrayRef = Arc::new(StringViewArray::from(values.to_vec()));

        for (i, expected) in values.iter().enumerate() {
            for (name, array) in [("Utf8", &utf8), ("LargeUtf8", &large), ("Utf8View", &view)] {
                assert_eq!(
                    extract_value_from_array(array, i).unwrap(),
                    *expected,
                    "{name} row {i}"
                );
            }
        }
    }

    /// `Int32`, `Int64`, `Float64` and `Boolean` keep Rust's `to_string`
    /// spelling, not Arrow's (`Float64` 1.0 is `1`, where Arrow's formatter
    /// writes `1.0`).
    #[test]
    fn legacy_types_keep_their_original_spelling() {
        assert_eq!(render(Int32Array::from(vec![-7]), 0), "-7");
        assert_eq!(
            render(Int64Array::from(vec![i64::MIN]), 0),
            i64::MIN.to_string()
        );
        assert_eq!(render(BooleanArray::from(vec![true, false]), 1), "false");
        assert_eq!(render(Float64Array::from(vec![1.0]), 0), "1");
        assert_eq!(
            render(Float64Array::from(vec![0.1 + 0.2]), 0),
            "0.30000000000000004"
        );
        assert_eq!(
            render(Float64Array::from(vec![1e21]), 0),
            "1000000000000000000000"
        );
        assert_eq!(render(Float64Array::from(vec![f64::NAN]), 0), "NaN");
        assert_eq!(
            render(Float64Array::from(vec![f64::NEG_INFINITY]), 0),
            "-inf"
        );
    }

    #[test]
    fn small_and_unsigned_integers_render_as_digits() {
        assert_eq!(render(Int8Array::from(vec![-128]), 0), "-128");
        assert_eq!(render(Int16Array::from(vec![32_000]), 0), "32000");
        assert_eq!(
            render(UInt32Array::from(vec![4_000_000_000]), 0),
            "4000000000"
        );
        assert_eq!(
            render(UInt64Array::from(vec![u64::MAX]), 0),
            u64::MAX.to_string()
        );
    }

    #[test]
    fn float32_renders_through_the_arrow_formatter() {
        assert_eq!(render(Float32Array::from(vec![1.5]), 0), "1.5");
        assert_eq!(render(Float32Array::from(vec![0.1]), 0), "0.1");
        assert_eq!(render(Float32Array::from(vec![-2.0]), 0), "-2.0");
    }

    #[test]
    fn decimals_keep_their_scale() {
        let d = Decimal128Array::from(vec![1234, -5, 0])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let d: ArrayRef = Arc::new(d);
        assert_eq!(extract_value_from_array(&d, 0).unwrap(), "12.34");
        assert_eq!(extract_value_from_array(&d, 1).unwrap(), "-0.05");
        assert_eq!(extract_value_from_array(&d, 2).unwrap(), "0.00");
    }

    #[test]
    fn dates_render_as_iso_8601() {
        assert_eq!(render(Date32Array::from(vec![DATE_DAYS]), 0), "2024-01-15");
        assert_eq!(render(Date32Array::from(vec![0]), 0), "1970-01-01");
        // `Date64` is a millisecond instant, and Arrow spells it as a datetime.
        assert_eq!(
            render(
                Date64Array::from(vec![i64::from(DATE_DAYS) * 86_400_000]),
                0
            ),
            "2024-01-15T00:00:00"
        );
    }

    #[test]
    fn timestamps_render_as_iso_8601_at_their_own_precision() {
        assert_eq!(
            render(TimestampSecondArray::from(vec![TS_SECONDS]), 0),
            "2024-01-15T10:30:00"
        );
        assert_eq!(
            render(
                TimestampMillisecondArray::from(vec![TS_SECONDS * 1_000 + 123]),
                0
            ),
            "2024-01-15T10:30:00.123"
        );
        assert_eq!(
            render(
                TimestampMicrosecondArray::from(vec![TS_SECONDS * 1_000_000 + 123_456]),
                0
            ),
            "2024-01-15T10:30:00.123456"
        );
        assert_eq!(
            render(
                TimestampNanosecondArray::from(vec![TS_SECONDS * 1_000_000_000 + 123_456_789]),
                0
            ),
            "2024-01-15T10:30:00.123456789"
        );
    }

    #[test]
    fn timestamps_with_a_fixed_offset_keep_the_offset() {
        let ts = TimestampSecondArray::from(vec![TS_SECONDS]).with_timezone("+02:00");
        assert_eq!(render(ts, 0), "2024-01-15T12:30:00+02:00");
    }

    /// IANA zone names (`UTC`, `Europe/Athens`) are formatted by Arrow only
    /// with its `chrono-tz` feature. A build without it fails the render here
    /// rather than dropping the zone.
    #[test]
    fn timestamps_with_a_named_timezone_render_in_that_zone() {
        let utc = TimestampSecondArray::from(vec![TS_SECONDS]).with_timezone("UTC");
        assert_eq!(render(utc, 0), "2024-01-15T10:30:00Z");
        let athens = TimestampSecondArray::from(vec![TS_SECONDS]).with_timezone("Europe/Athens");
        assert_eq!(render(athens, 0), "2024-01-15T12:30:00+02:00");
    }

    #[test]
    fn times_durations_and_binary_render() {
        assert_eq!(
            render(Time64MicrosecondArray::from(vec![37_800_000_000]), 0),
            "10:30:00"
        );
        assert_eq!(render(DurationSecondArray::from(vec![90]), 0), "PT90S");
        assert_eq!(
            render(BinaryArray::from_vec(vec![b"hi".as_slice()]), 0),
            "6869"
        );
    }

    #[test]
    fn dictionary_encoded_strings_render_their_value() {
        let dict: DictionaryArray<Int32Type> =
            vec![Some("positive"), Some("negative"), None, Some("positive")]
                .into_iter()
                .collect();
        let dict: ArrayRef = Arc::new(dict);
        assert_eq!(extract_value_from_array(&dict, 0).unwrap(), "positive");
        assert_eq!(extract_value_from_array(&dict, 1).unwrap(), "negative");
        assert_eq!(extract_value_from_array(&dict, 2).unwrap(), "NULL");
        assert_eq!(extract_value_from_array(&dict, 3).unwrap(), "positive");
    }

    #[test]
    fn a_null_nested_in_a_list_reads_null_not_empty() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(1),
            None,
            Some(3),
        ])]);
        assert_eq!(render(list, 0), "[1, NULL, 3]");
    }

    /// A null is `NULL` whatever the type, including the types that take the
    /// formatter path, where Arrow's own default for a null is the empty string.
    #[test]
    fn nulls_render_as_null_for_every_type() {
        let arrays: Vec<(&str, ArrayRef)> = vec![
            ("Utf8", Arc::new(StringArray::from(vec![Some("x"), None]))),
            (
                "LargeUtf8",
                Arc::new(LargeStringArray::from(vec![Some("x"), None])),
            ),
            (
                "Utf8View",
                Arc::new(StringViewArray::from(vec![Some("x"), None])),
            ),
            ("Int8", Arc::new(Int8Array::from(vec![Some(1), None]))),
            ("Int32", Arc::new(Int32Array::from(vec![Some(1), None]))),
            ("Int64", Arc::new(Int64Array::from(vec![Some(1), None]))),
            ("UInt32", Arc::new(UInt32Array::from(vec![Some(1), None]))),
            (
                "Float32",
                Arc::new(Float32Array::from(vec![Some(1.0), None])),
            ),
            (
                "Float64",
                Arc::new(Float64Array::from(vec![Some(1.0), None])),
            ),
            (
                "Boolean",
                Arc::new(BooleanArray::from(vec![Some(true), None])),
            ),
            (
                "Date32",
                Arc::new(Date32Array::from(vec![Some(DATE_DAYS), None])),
            ),
            (
                "Timestamp",
                Arc::new(TimestampSecondArray::from(vec![Some(TS_SECONDS), None])),
            ),
            (
                "Decimal128",
                Arc::new(
                    Decimal128Array::from(vec![Some(1234), None])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ),
        ];
        for (name, array) in &arrays {
            assert_eq!(
                extract_value_from_array(array, 1).unwrap(),
                "NULL",
                "{name} null"
            );
            assert_ne!(
                extract_value_from_array(array, 0).unwrap(),
                "NULL",
                "{name} non-null"
            );
        }
    }

    /// A column typed `Null` (`SELECT NULL AS x`) has no validity buffer, so
    /// `is_null` is false for it; the formatter must still say `NULL`.
    #[test]
    fn a_null_typed_column_renders_null() {
        assert_eq!(render(NullArray::new(2), 1), "NULL");
    }

    /// A value Arrow cannot represent is an error, not an `ERROR` marker typed
    /// into the prompt.
    #[test]
    fn an_unformattable_value_is_an_error_not_prompt_text() {
        let ts: ArrayRef = Arc::new(TimestampSecondArray::from(vec![i64::MAX]));
        assert!(extract_value_from_array(&ts, 0).is_err());
    }

    /// `extract_row_values` handles a mixed-type batch in `columns` order.
    #[test]
    fn row_values_cover_a_mixed_type_batch_in_column_order() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("when", DataType::Date32, true),
            Field::new("body", DataType::Utf8View, true),
            Field::new("score", DataType::Float32, true),
            Field::new("n", DataType::Int8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Date32Array::from(vec![Some(DATE_DAYS), None])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![
                    Some("great film"),
                    Some("dull"),
                ])),
                Arc::new(Float32Array::from(vec![Some(4.5), Some(1.0)])),
                Arc::new(Int8Array::from(vec![Some(7), None])),
            ],
        )
        .unwrap();
        let columns: Vec<String> = ["n", "body", "when", "score"].map(String::from).to_vec();

        let row0 = extract_row_values(&batch, 0, &columns).unwrap();
        assert_eq!(
            row0,
            vec![
                ("n".to_string(), "7".to_string()),
                ("body".to_string(), "great film".to_string()),
                ("when".to_string(), "2024-01-15".to_string()),
                ("score".to_string(), "4.5".to_string()),
            ]
        );
        let row1 = extract_row_values(&batch, 1, &columns).unwrap();
        assert_eq!(row1[0].1, "NULL");
        assert_eq!(row1[1].1, "dull");
        assert_eq!(row1[2].1, "NULL");
        assert_eq!(row1[3].1, "1.0");
    }
}
