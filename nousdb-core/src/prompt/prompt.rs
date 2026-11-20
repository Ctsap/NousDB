//! The [`Prompt`] type: template rendering and response parsing.

use super::message::Message;
use crate::prompt::environment::ENV;
use crate::prompt::reasoning::ReasoningMode;
use crate::prompt::renderer::BatchRenderer;
use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::array::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DFResult};
use minijinja::context;
use minijinja::value::Value as MValue;
use serde_json::Value;
use std::sync::Arc;

/// A prompt for one operator: user instruction, optional system message,
/// template name, output schema and reasoning mode.
#[derive(Debug, Clone)]
pub struct Prompt {
    user_instruction: Message,
    system_message: Option<Message>,
    template_name: String,
    output_schema: SchemaRef,
    /// Fields the model must emit that are not operator output columns.
    /// They lead [`Prompt::wire_fields`] and therefore the response grammar,
    /// so the model writes them before it commits to an answer. No Arrow array
    /// is built for them ([`Prompt::parse_response`] iterates the output schema
    /// alone).
    prompt_only_fields: Vec<Field>,
    /// Reasoning mode of this prompt, read by the response-schema builder.
    /// `Off` produces the same wire bytes as a prompt without reasoning modes.
    reasoning: ReasoningMode,
}

impl Prompt {
    /// Creates a prompt with reasoning off and no prompt-only fields.
    pub fn new(
        user_instruction: impl Into<String>,
        system_message: Option<impl Into<String>>,
        template_name: impl Into<String>,
        output_schema: SchemaRef,
    ) -> Self {
        let user_msg = Message::User(user_instruction.into());
        let system_msg = system_message.map(|s| Message::System(s.into()));
        Self {
            user_instruction: user_msg,
            system_message: system_msg,
            template_name: template_name.into(),
            output_schema,
            prompt_only_fields: Vec::new(),
            reasoning: ReasoningMode::Off,
        }
    }

    /// Applies a reasoning mode: its wire-only fields plus the mode itself,
    /// which the response-schema builder consults. Setting both together keeps
    /// them consistent.
    pub fn with_reasoning(mut self, mode: ReasoningMode) -> Self {
        self.prompt_only_fields = mode.prompt_only_fields();
        self.reasoning = mode;
        self
    }

    /// Declares wire-only fields directly, for callers that build a response
    /// contract without a reasoning mode. [`Self::with_reasoning`] is the usual
    /// path.
    pub fn with_prompt_only_fields(mut self, fields: Vec<Field>) -> Self {
        self.prompt_only_fields = fields;
        self
    }

    /// The reasoning mode of this prompt.
    pub fn reasoning_mode(&self) -> ReasoningMode {
        self.reasoning
    }

    /// Fields the model emits that are not operator output columns.
    pub fn prompt_only_fields(&self) -> &[Field] {
        &self.prompt_only_fields
    }

    /// Every field the model must emit, prompt-only fields first.
    ///
    /// The order matters: the grammar the server compiles decides emission
    /// order, so a `rationale` placed after the decision field would turn
    /// reason-then-decide into decide-then-narrate.
    pub fn wire_fields(&self) -> Vec<FieldRef> {
        self.prompt_only_fields
            .iter()
            .cloned()
            .map(Arc::new)
            .chain(self.output_schema.fields().iter().cloned())
            .collect()
    }

    /// Renders the system message template, or returns `None` when the prompt
    /// has no system message.
    pub fn render_system(&self) -> DFResult<Option<String>> {
        let env = ENV.read().map_err(|e| {
            DataFusionError::Execution(format!("Failed to acquire environment lock: {}", e))
        })?;

        if let Some(_system_msg) = &self.system_message {
            let system_msg_str = self
                .system_message
                .as_ref()
                .ok_or_else(|| DataFusionError::Execution("System message is missing".to_string()))?
                .as_str();

            let template = env.get_template(system_msg_str).map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to load template '{}': {}",
                    system_msg_str, e
                ))
            })?;
            let ctx = context! {};
            let rendered = template.render(ctx).map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to render template '{}': {}",
                    self.template_name, e
                ))
            })?;

            Ok(Some(rendered))
        } else {
            Ok(None)
        }
    }

    /// Renders the user template for rows `start_row..start_row + batch_size`
    /// of `batch` (clamped to the batch length).
    ///
    /// Returns an error if the slice is empty or the template fails to load or
    /// render.
    pub fn render_batch<R: BatchRenderer + ?Sized>(
        &self,
        renderer: &R,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> DFResult<String> {
        // Validate the slice is within bounds
        let end_row = (start_row + batch_size).min(batch.num_rows());
        let actual_batch_size = end_row - start_row;

        if actual_batch_size == 0 {
            return Err(DataFusionError::Execution(
                "Batch size is zero or out of bounds".to_string(),
            ));
        }

        let mut context = renderer.prepare_context(batch, start_row, batch_size)?;

        // Standard context variables available to all templates.
        context.insert(
            "output_schema".to_string(),
            MValue::from(self.build_json_format_instructions()),
        );
        // The active mode's `ReasoningMode::label`, so a user template can show
        // an output example matching the grammar the answer is decoded against.
        // It is injected here rather than in a `BatchRenderer` so the mode has
        // a single source of truth that cannot drift from `wire_fields()`.
        // It is inserted unconditionally: the environment uses
        // `UndefinedBehavior::Strict`, so a key present only in some modes
        // would fail the render in the others.
        context.insert(
            "reasoning_mode".to_string(),
            MValue::from(self.reasoning.label()),
        );

        // Load the template
        let env = ENV.read().map_err(|e| {
            DataFusionError::Execution(format!("Failed to acquire environment lock: {}", e))
        })?;

        let template = env.get_template(&self.template_name).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to load template '{}': {}",
                self.template_name, e
            ))
        })?;

        // The `context!` macro expects static keys, so build the context from
        // the dynamic map directly.
        let ctx = minijinja::Value::from(context);

        // Render the template with the prepared context
        let rendered = template.render(ctx).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to render template '{}': {}",
                self.template_name, e
            ))
        })?;

        Ok(rendered)
    }

    /// Instructions telling the model which JSON structure to return. They
    /// describe the wire fields: prose that omits a field the grammar requires
    /// would contradict the grammar the answer is decoded against.
    fn build_json_format_instructions(&self) -> String {
        json_format_instructions_for(&self.wire_fields())
    }

    /// Estimates the number of tokens in a rendered batch, for cost modelling
    /// and batch sizing, by rendering the batch and counting words.
    pub fn estimate_tokens<R: BatchRenderer>(
        &self,
        renderer: &R,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
    ) -> DFResult<usize> {
        let rendered = self.render_batch(renderer, batch, start_row, batch_size)?;

        // Approximation: 1 token is about 0.75 words, so tokens = words * 1.33.
        let word_count = rendered.split_whitespace().count();
        Ok((word_count as f64 * 1.33) as usize)
    }

    /// The user instruction.
    pub fn user_instruction(&self) -> &str {
        self.user_instruction.as_str()
    }

    /// The name of the user template.
    pub fn template_name(&self) -> &str {
        self.template_name.as_str()
    }

    /// The operator output schema.
    pub fn output_schema(&self) -> &Arc<Schema> {
        &self.output_schema
    }

    /// Parses a JSON array response into one Arrow array per output-schema
    /// field.
    ///
    /// The response must be valid JSON with exactly `expected_count` objects;
    /// otherwise an error is returned.
    pub fn parse_response(&self, response: &str, expected_count: usize) -> DFResult<Vec<ArrayRef>> {
        // Parse the JSON directly; the response is assumed to be valid JSON.
        let parsed: Vec<Value> = serde_json::from_str(response)
            .map_err(|e| DataFusionError::Execution(format!("JSON parse error: {}", e)))?;

        // Verify we got the expected number of results
        if parsed.len() != expected_count {
            return Err(DataFusionError::Execution(format!(
                "Expected {} results, got {}",
                expected_count,
                parsed.len()
            )));
        }

        // Build an Arrow array for each field in the output schema
        let mut arrays = Vec::with_capacity(self.output_schema.fields().len());

        for field in self.output_schema.fields() {
            let array = build_array(field, &parsed)?;
            arrays.push(array);
        }

        Ok(arrays)
    }
}

/// The string `render_batch` interpolates into `{{ output_schema }}`.
///
/// Shared with `prompt::utils::flat_static_prompt_tokens` so plan-time batch
/// sizing measures the same bytes that are sent at run time.
pub(crate) fn json_format_instructions(output_schema: &SchemaRef) -> String {
    json_format_instructions_for(output_schema.fields())
}

/// [`json_format_instructions`] over an explicit field list, so a prompt can
/// describe wire-only fields the output schema does not contain.
pub(crate) fn json_format_instructions_for(fields: &[FieldRef]) -> String {
    let field_names: Vec<String> = fields.iter().map(|f| f.name().clone()).collect();

    let example_fields = field_names
        .iter()
        .map(|name| format!("\"{}\": \"value\"", name))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "Return your answer as a JSON array with one object per input item.\n\
         Each object must have these exact fields: {}\n\n\
         Example format:\n[\n  {{ {} }},\n  {{ {} }}\n]\n\n\
         IMPORTANT: Return ONLY the JSON array, no explanatory text before or after.",
        field_names.join(", "),
        example_fields,
        example_fields
    )
}

/// Builds an Arrow array for a field from JSON values, dispatching on the
/// field's data type.
fn build_array(field: &Field, json_values: &[Value]) -> DFResult<ArrayRef> {
    match field.data_type() {
        DataType::Boolean => build_bool_array(field.name(), json_values),
        DataType::Int64 => build_int64_array(field.name(), json_values),
        DataType::Float64 => build_float64_array(field.name(), json_values),
        DataType::Utf8 => build_string_array(field.name(), json_values),
        other => Err(DataFusionError::Execution(format!(
            "Unsupported type: {:?}",
            other
        ))),
    }
}

/// Build a BooleanArray from JSON values
fn build_bool_array(field_name: &str, values: &[Value]) -> DFResult<ArrayRef> {
    let mut builder = BooleanBuilder::new();

    for value in values {
        match &value[field_name] {
            Value::Bool(b) => builder.append_value(*b),
            Value::Null => builder.append_null(),
            other => {
                return Err(DataFusionError::Execution(format!(
                    "Expected boolean, got: {:?}",
                    other
                )));
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Build an Int64Array from JSON values
///
/// An integral value spelled with a decimal point (`5.0`) is accepted as `5`.
/// The `response_format` schema asks for `"type": "integer"`, but some servers
/// return `5.0` anyway, and `serde_json`'s `as_i64()` rejects it. The Int64
/// null-fill recovery value (`ModelOutput::Float(0.0)` in `providers/utils.rs`)
/// serializes to `0.0` and takes the same path.
///
/// A genuinely fractional value is still an error: truncating `4.7` to `4`
/// would silently corrupt a score instead of surfacing that the model ignored
/// the schema.
fn build_int64_array(field_name: &str, values: &[Value]) -> DFResult<ArrayRef> {
    let mut builder = Int64Builder::new();

    for value in values {
        match &value[field_name] {
            Value::Number(n) => {
                let i = n.as_i64().or_else(|| {
                    n.as_f64()
                        .filter(|f| {
                            f.fract() == 0.0 && *f >= i64::MIN as f64 && *f <= i64::MAX as f64
                        })
                        .map(|f| f as i64)
                });
                let i =
                    i.ok_or_else(|| DataFusionError::Execution(format!("Not an i64: {}", n)))?;
                builder.append_value(i);
            }
            Value::Null => builder.append_null(),
            other => {
                return Err(DataFusionError::Execution(format!(
                    "Expected number, got: {:?}",
                    other
                )));
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Build a Float64Array from JSON values
fn build_float64_array(field_name: &str, values: &[Value]) -> DFResult<ArrayRef> {
    let mut builder = Float64Builder::new();

    for value in values {
        match &value[field_name] {
            Value::Number(n) => {
                let f = n
                    .as_f64()
                    .ok_or_else(|| DataFusionError::Execution(format!("Not an f64: {}", n)))?;
                builder.append_value(f);
            }
            Value::Null => builder.append_null(),
            other => {
                return Err(DataFusionError::Execution(format!(
                    "Expected number, got: {:?}",
                    other
                )));
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Build a StringArray from JSON values
fn build_string_array(field_name: &str, values: &[Value]) -> DFResult<ArrayRef> {
    let mut builder = StringBuilder::new();
    for value in values {
        match &value[field_name] {
            Value::String(s) => builder.append_value(s),
            Value::Bool(b) => builder.append_value(b.to_string()),
            Value::Number(n) => builder.append_value(n.to_string()),
            Value::Null => builder.append_null(),
            other => {
                return Err(DataFusionError::Execution(format!(
                    "Expected string, got: {:?}",
                    other
                )));
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, Int64Array};
    use serde_json::json;

    fn int64_values(values: &[Value]) -> DFResult<Vec<Option<i64>>> {
        let arr = build_int64_array("score", values)?;
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .collect())
    }

    /// Some servers return `5.0` for a `"type": "integer"` field; it is
    /// accepted as `5`.
    #[test]
    fn integral_floats_are_accepted_as_i64() {
        let values = vec![
            json!({"score": 5}),
            json!({"score": 5.0}),
            json!({"score": -3.0}),
            json!({"score": 0.0}),
            json!({"score": null}),
        ];
        assert_eq!(
            int64_values(&values).unwrap(),
            vec![Some(5), Some(5), Some(-3), Some(0), None]
        );
    }

    /// A fractional value means the model ignored the schema, so it stays an
    /// error.
    #[test]
    fn fractional_values_are_still_rejected() {
        let err = int64_values(&[json!({"score": 4.7})]).unwrap_err();
        assert!(err.to_string().contains("Not an i64"), "got: {err}");
    }
}
