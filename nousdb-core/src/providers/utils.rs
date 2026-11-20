//! Shared utilities for model providers: response-schema construction and
//! response parsing, used by every backend so behaviour is consistent.

use datafusion::arrow::datatypes::{DataType, FieldRef};
use serde_json::Value;

use crate::prompt::prompt::Prompt;
use crate::prompt::reasoning::ReasoningMode;
use crate::providers::schemas::ModelOutput;

/// Result of parsing a JSON model response into `ModelOutput` values.
///
/// On any failure (malformed JSON or row-count mismatch), `outputs` is filled
/// with null-equivalent values matching `expected_count` and `row_count_mismatch`
/// is set to `true`. The provider propagates the flag through `PredictResult`
/// so operator streams can activate their recovery paths.
pub struct ParseResult {
    /// One output per requested item.
    pub outputs: Vec<ModelOutput>,
    /// Whether parsing failed or the item count was wrong.
    pub row_count_mismatch: bool,
}

/// Field the prediction objects live under when the schema root has to be an
/// object. See [`wrap_batch_schema_in_object`].
pub const BATCH_RESULTS_KEY: &str = "results";

/// Builds a JSON Schema string describing an array of prediction objects.
///
/// The schema is derived from the `Prompt`'s fields. Each field maps to a JSON
/// Schema type (`string`, `boolean`, `integer`, `number`, `array`). Providers
/// use it to request structured JSON output from models.
pub fn construct_output_schema_from_prompt(prompt: &Prompt, batch_size: usize) -> String {
    build_batch_array_schema(prompt, batch_size).to_string()
}

/// The array-rooted batch schema: exactly `batch_size` prediction objects.
///
/// Most backends (for example vLLM and Ollama) take an array at the root, and
/// this shape matches the array the prompt templates ask for.
///
/// Built from [`Prompt::wire_fields`], not from the output schema:
/// `additionalProperties: false` would otherwise forbid the model from emitting
/// a prompt-only field the system prompt demands. The wire fields lead with the
/// prompt-only ones, and `serde_json`'s `preserve_order` keeps schema
/// properties in declaration order in the serialized grammar, which determines
/// the order the model emits fields.
pub fn build_batch_array_schema(prompt: &Prompt, batch_size: usize) -> Value {
    let fields = prompt.wire_fields();
    // Only the output-schema fields carry the abstention convention; the
    // prompt-only fields are always answerable, so widening their type would
    // let the model skip the reasoning it is asked for.
    let abstainable_from = prompt.prompt_only_fields().len();

    let mut properties = serde_json::Map::new();
    let mut required = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let abstainable = i >= abstainable_from;
        properties.insert(
            field.name().clone(),
            serde_json::json!({ "type": json_type_of(field, prompt.reasoning_mode(), abstainable) }),
        );
        required.push(Value::String(field.name().clone()));
    }

    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BatchPrediction",
        "description": "Predictions for a batch of inputs",
        "type": "array",
        "minItems": batch_size,
        "maxItems": batch_size,
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        },
    })
}

/// The JSON Schema `type` for one wire field.
///
/// A nullable field becomes `["<type>", "null"]` so the grammar permits the
/// abstention the reasoning system prompts ask for ("answer `null` when the row
/// does not contain the requested information"). This applies only when the
/// mode is not [`ReasoningMode::Off`]: under `Off` nothing asks for an
/// abstention, and the baseline wire bytes must stay unchanged.
fn json_type_of(field: &FieldRef, mode: ReasoningMode, abstainable: bool) -> Value {
    let base = match field.data_type() {
        DataType::Utf8 => "string",
        DataType::Boolean => "boolean",
        DataType::Int64 => "integer",
        DataType::Float64 => "number",
        DataType::List(_) => "array",
        _ => "string",
    };
    if mode != ReasoningMode::Off && abstainable && field.is_nullable() {
        serde_json::json!([base, "null"])
    } else {
        Value::String(base.to_string())
    }
}

/// Puts the array from [`build_batch_array_schema`] inside a single-field
/// object.
///
/// Azure's Structured Outputs rejects an array at the root, so the wrapper is
/// the only way to get a grammar there. Running without a grammar is not an
/// option, since an unconstrained answer silently becomes an unfiltered
/// pass-through or a NULL-fill.
///
/// The wrapper is transport-only. Prompt templates still ask for a JSON array
/// and stay byte-identical across backends, because the grammar decides the
/// shape, not the prose; this keeps prefix-cache invariants intact.
/// [`parse_json_to_model_outputs`] removes the wrapper when reading the
/// response.
///
/// The nested `minItems`/`maxItems` pin is kept: it keeps the answer aligned
/// with the input rows.
pub fn wrap_batch_schema_in_object(mut array: Value) -> Value {
    // The annotations belong to the root document, not to the inner array.
    if let Some(obj) = array.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
        obj.remove("description");
    }

    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BatchPrediction",
        "description": "Predictions for a batch of inputs",
        "type": "object",
        "properties": { BATCH_RESULTS_KEY: array },
        "required": [BATCH_RESULTS_KEY],
        "additionalProperties": false,
    })
}

/// Strips common model artifacts from response text:
///
/// - `<think>...</think>` blocks emitted by reasoning models
/// - Markdown code fences (` ```json ... ``` `) that some models wrap JSON in
/// - Leading and trailing whitespace
pub fn clean_model_response(text: &str) -> String {
    // Strip all <think>...</think> blocks
    let mut cleaned = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        cleaned.push_str(&remaining[..start]);
        match remaining[start..].find("</think>") {
            Some(end) => remaining = &remaining[start + end + "</think>".len()..],
            None => {
                // Unclosed <think> tag: drop everything after it.
                remaining = "";
            }
        }
    }
    cleaned.push_str(remaining);

    let cleaned = cleaned.trim();

    // Strip markdown code fences
    if cleaned.starts_with("```") {
        cleaned
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        cleaned.to_string()
    }
}

/// Parses a JSON array response from a model into `Vec<ModelOutput>`.
///
/// The response is expected to be a JSON array whose length equals
/// `expected_count`. Each element is mapped to a `ModelOutput` variant based on
/// the prompt's output schema:
///
/// * Single-field schema → scalar `ModelOutput` per element
/// * Multi-field schema  → `ModelOutput::Structured` per element
///
/// Numeric values are mapped to `ModelOutput::Float(f64)` when possible,
/// falling back to `ModelOutput::Text`.
///
/// A single-row request whose response is a bare object rather than a
/// one-element array is accepted (see `parse_batch_array`).
pub fn parse_json_to_model_outputs(
    json_response: &str,
    prompt: &Prompt,
    expected_count: usize,
) -> ParseResult {
    // Peeling the `{"results": [...]}` wrapper is safe only when `results` is
    // not itself one of the model's output fields: a single List-typed column
    // named `results` would otherwise be mistaken for the transport wrapper.
    let allow_unwrap = !prompt
        .output_schema()
        .fields()
        .iter()
        .any(|f| f.name() == BATCH_RESULTS_KEY);

    let parsed: Vec<Value> = match parse_batch_array(json_response, expected_count, allow_unwrap) {
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "Failed to parse JSON from model (returning {} null outputs): {}. Response was: {}",
                expected_count,
                e,
                json_response
            );
            return ParseResult {
                outputs: null_outputs(expected_count, prompt),
                row_count_mismatch: true,
            };
        }
    };

    if parsed.len() != expected_count {
        log::warn!(
            "Row count mismatch: expected {} results but got {} (returning null outputs). Response: {}",
            expected_count,
            parsed.len(),
            json_response
        );
        return ParseResult {
            outputs: null_outputs(expected_count, prompt),
            row_count_mismatch: true,
        };
    }

    let schema = prompt.output_schema();
    let mut outputs = Vec::with_capacity(parsed.len());

    for json_obj in parsed {
        if schema.fields().len() == 1 {
            let field = schema.field(0);
            let field_name = field.name();
            let value = &json_obj[field_name];

            let output = match value {
                Value::String(s) => ModelOutput::Text(s.clone()),
                Value::Bool(b) => ModelOutput::Boolean(*b),
                Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        ModelOutput::Float(f)
                    } else {
                        ModelOutput::Text(n.to_string())
                    }
                }
                Value::Object(_) | Value::Array(_) => ModelOutput::Structured(value.clone()),
                // Not `Text("")`: the reasoning system prompts instruct the
                // model to answer `null` when a row lacks the information, and
                // an empty string is a value the model declined to give.
                Value::Null => ModelOutput::Null,
            };

            outputs.push(output);
        } else {
            // Multi-field schema: ensure json_obj is an object
            if json_obj.is_object() {
                outputs.push(ModelOutput::Structured(json_obj.clone()));
            } else {
                // Non-object element (e.g. null): create a null-valued object.
                let mut map = serde_json::Map::new();
                for field in schema.fields() {
                    map.insert(field.name().clone(), Value::Null);
                }
                outputs.push(ModelOutput::Structured(Value::Object(map)));
            }
        }
    }

    ParseResult {
        outputs,
        row_count_mismatch: false,
    }
}

/// Parses a batch response into its array of per-item objects.
///
/// The contract is a top-level JSON array of `expected_count` items. One
/// tolerance is added: a single-row request answered with a bare object
/// (`{"matches_filter": true}` instead of `[{"matches_filter": true}]`) is
/// accepted, since there is exactly one item. Rejecting it would turn a
/// `k = 1` request (the certification reference and the fallback batch size)
/// into a null-filled one, which selection recovers from by passing its rows
/// through unfiltered.
///
/// The tolerance is narrow: a bare object for `expected_count > 1` is a genuine
/// mismatch and still fails.
///
/// The other non-array top level is the `{"results": [...]}` object produced by
/// [`wrap_batch_schema_in_object`]; it is unwrapped when `allow_unwrap` is set.
/// That check must come first: at `expected_count == 1` the wrapper is also a
/// bare object, so the tolerance would otherwise accept it as the row itself,
/// miss the field lookup, and decode a default value with
/// `row_count_mismatch: false`, a wrong answer no recovery path can see.
fn parse_batch_array(
    json_response: &str,
    expected_count: usize,
    allow_unwrap: bool,
) -> serde_json::Result<Vec<Value>> {
    match serde_json::from_str::<Vec<Value>>(json_response) {
        Ok(items) => Ok(items),
        Err(array_err) => {
            let Ok(value) = serde_json::from_str::<Value>(json_response) else {
                return Err(array_err);
            };

            if allow_unwrap && let Some(items) = unwrap_batch_payload(&value) {
                log::debug!(
                    "Unwrapped the object-rooted batch payload ({} items)",
                    items.len()
                );
                return Ok(items);
            }

            if expected_count == 1 && value.is_object() {
                log::debug!(
                    "Model answered a 1-row request with a bare object; accepting it as a \
                     one-element array"
                );
                return Ok(vec![value]);
            }

            Err(array_err)
        }
    }
}

/// Removes the single-field object wrapper from a batch payload.
///
/// Matches only the exact shape the wrapped schema can produce: an object with
/// exactly one key, [`BATCH_RESULTS_KEY`], holding an array. Anything else
/// (extra keys, a different key, a non-array value) is a malformed answer, and
/// turning it into a batch would hide a failure that the `row_count_mismatch`
/// path counts.
fn unwrap_batch_payload(value: &Value) -> Option<Vec<Value>> {
    let obj = value.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    obj.get(BATCH_RESULTS_KEY)?.as_array().cloned()
}

/// Builds null-equivalent `ModelOutput` values for the prompt's schema.
fn null_outputs(count: usize, prompt: &Prompt) -> Vec<ModelOutput> {
    let schema = prompt.output_schema();
    let null_value = if schema.fields().len() == 1 {
        match schema.field(0).data_type() {
            DataType::Boolean => ModelOutput::Boolean(false),
            DataType::Int64 => ModelOutput::Float(0.0),
            DataType::Float64 => ModelOutput::Float(0.0),
            _ => ModelOutput::Text("".to_string()),
        }
    } else {
        let mut map = serde_json::Map::new();
        for field in schema.fields() {
            map.insert(field.name().clone(), Value::Null);
        }
        ModelOutput::Structured(Value::Object(map))
    };
    vec![null_value; count]
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn selection_prompt() -> Prompt {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )]));
        Prompt::new("does this match?", None::<String>, "selection", schema)
    }

    /// The shape an unconstrained model may return for a one-row request.
    #[test]
    fn bare_object_is_accepted_for_a_single_row_request() {
        let result =
            parse_json_to_model_outputs(r#"{"matches_filter": true}"#, &selection_prompt(), 1);
        assert!(!result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 1);
        assert!(matches!(result.outputs[0], ModelOutput::Boolean(true)));
    }

    /// One object when many rows were asked for is a genuine mismatch: the model
    /// answered a single row. Accepting it would silently drop the rest.
    #[test]
    fn bare_object_is_rejected_when_more_than_one_row_was_requested() {
        let result =
            parse_json_to_model_outputs(r#"{"matches_filter": true}"#, &selection_prompt(), 4);
        assert!(result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 4);
    }

    /// The contract shape still works, and is still preferred.
    #[test]
    fn one_element_array_is_unaffected() {
        let result =
            parse_json_to_model_outputs(r#"[{"matches_filter": false}]"#, &selection_prompt(), 1);
        assert!(!result.row_count_mismatch);
        assert!(matches!(result.outputs[0], ModelOutput::Boolean(false)));
    }

    /// Tolerance is for a *bare object*, not for any scalar: a naked `true`
    /// carries no field name and must not be guessed at.
    #[test]
    fn a_bare_scalar_is_still_a_parse_failure() {
        for body in ["true", "\"positive\"", "null", "42"] {
            let result = parse_json_to_model_outputs(body, &selection_prompt(), 1);
            assert!(
                result.row_count_mismatch,
                "{body} must not be accepted as a one-element array"
            );
        }
    }

    /// Genuine garbage still fails, at every batch size.
    #[test]
    fn malformed_json_still_fails() {
        for k in [1usize, 8] {
            let result = parse_json_to_model_outputs("not json at all", &selection_prompt(), k);
            assert!(result.row_count_mismatch);
            assert_eq!(result.outputs.len(), k);
        }
    }

    /// The shape the wrapped schema produces (see `wrap_batch_schema_in_object`)
    /// reads back as an ordinary batch.
    #[test]
    fn azure_wrapper_is_unwrapped() {
        let body = r#"{"results": [{"matches_filter": true}, {"matches_filter": false},
                       {"matches_filter": true}]}"#;
        let result = parse_json_to_model_outputs(body, &selection_prompt(), 3);
        assert!(!result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 3);
        assert!(matches!(result.outputs[0], ModelOutput::Boolean(true)));
        assert!(matches!(result.outputs[1], ModelOutput::Boolean(false)));
    }

    /// At `k = 1` the wrapper is also a bare object, so the single-row
    /// tolerance would accept it as the row itself: the field lookup would miss
    /// and the answer would decode `false` with no mismatch flagged. Unwrapping
    /// must therefore be tried first.
    #[test]
    fn wrapper_at_k1_is_not_swallowed_by_the_bare_object_tolerance() {
        let result = parse_json_to_model_outputs(
            r#"{"results": [{"matches_filter": true}]}"#,
            &selection_prompt(),
            1,
        );
        assert!(!result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 1);
        assert!(
            matches!(result.outputs[0], ModelOutput::Boolean(true)),
            "the wrapper must be peeled, not read as the row: got {:?}",
            result.outputs[0]
        );
    }

    /// A wrapped array of the wrong length is still a mismatch; unwrapping
    /// must not hide a short answer.
    #[test]
    fn wrapped_array_of_the_wrong_length_is_still_a_mismatch() {
        let result = parse_json_to_model_outputs(
            r#"{"results": [{"matches_filter": true}]}"#,
            &selection_prompt(),
            4,
        );
        assert!(result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 4);
    }

    /// Only the exact one-key wrapper is peeled. An object carrying `results`
    /// alongside other fields is a malformed answer, not transport packaging.
    #[test]
    fn multi_key_object_is_not_unwrapped() {
        let result = parse_json_to_model_outputs(
            r#"{"results": [{"matches_filter": true}], "note": "hi"}"#,
            &selection_prompt(),
            1,
        );
        // Falls through to the bare-object tolerance, which finds no
        // `matches_filter`: a visible empty answer, not a fabricated batch.
        assert!(!matches!(result.outputs[0], ModelOutput::Boolean(true)));
    }

    /// `results` can legitimately be the model's own output column. A
    /// single-row answer for a List-typed `results` field is a bare object that
    /// looks exactly like the wrapper, so the prompt schema has to veto the peel.
    #[test]
    fn results_output_field_is_not_unwrapped() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "results",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        )]));
        let prompt = Prompt::new("list them", None::<String>, "extend", schema);

        let result = parse_json_to_model_outputs(r#"{"results": ["a", "b"]}"#, &prompt, 1);
        assert!(!result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 1);
        assert!(
            matches!(
                &result.outputs[0],
                ModelOutput::Structured(Value::Array(items)) if items.len() == 2
            ),
            "the column value must survive intact: got {:?}",
            result.outputs[0]
        );
    }

    /// The grammar the server enforces decides the order the model emits its
    /// fields in, so `properties` must follow the declaration order, not
    /// `serde_json`'s alphabetical fallback. A per-item `rationale` declared
    /// before a decision field would otherwise be re-sorted behind it,
    /// turning reason-then-decide into decide-then-narrate.
    ///
    /// This holds only while `serde_json` is built with `preserve_order`, which
    /// this crate's `Cargo.toml` requires explicitly.
    #[test]
    fn schema_properties_follow_declaration_order_not_the_alphabet() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("zebra", DataType::Utf8, false),
            Field::new("alpha", DataType::Boolean, false),
        ]));
        let prompt = Prompt::new("classify", None::<String>, "extend", schema);

        let value = build_batch_array_schema(&prompt, 3);
        let keys: Vec<&str> = value["items"]["properties"]
            .as_object()
            .expect("properties must be an object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["zebra", "alpha"]);

        // The wire form is what the server's grammar compiler reads.
        let serialized = value.to_string();
        let zebra = serialized.find("\"zebra\"").unwrap();
        let alpha = serialized.find("\"alpha\"").unwrap();
        assert!(
            zebra < alpha,
            "serialized schema re-sorted the fields alphabetically: {serialized}"
        );
    }

    /// A wrong-length array is a mismatch and is null-filled to the requested
    /// length, so operators must consult `row_count_mismatch` rather than
    /// comparing lengths.
    #[test]
    fn wrong_length_array_is_null_filled_to_the_request() {
        let result = parse_json_to_model_outputs(
            r#"[{"matches_filter": true}, {"matches_filter": true}]"#,
            &selection_prompt(),
            3,
        );
        assert!(result.row_count_mismatch);
        assert_eq!(result.outputs.len(), 3);
    }
}
