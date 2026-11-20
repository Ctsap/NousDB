//! Reasoning-mode plumbing: which system template each operator renders, what
//! the response contract asks for, and what must not change when the mode is
//! [`ReasoningMode::Off`].
//!
//! `prompt_prefix.rs` pins the user templates' layout; this file pins the
//! system prompts and the wire schema.

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

use nousdb_core::certification::OperatorKind;
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::schemas::ModelOutput;
use nousdb_core::providers::utils::{
    construct_output_schema_from_prompt, parse_json_to_model_outputs,
};

use nousdb_operators::extend::stream::ExtendStream;
use nousdb_operators::metrics::{SemanticOperatorMetrics, WallClockTimer};

use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;

const OPERATORS: [OperatorKind; 3] = [
    OperatorKind::Selection,
    OperatorKind::Extend,
    OperatorKind::Join,
];

fn prompt_for(kind: OperatorKind, mode: ReasoningMode) -> Prompt {
    let output_schema: SchemaRef = match kind {
        OperatorKind::Selection => Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )])),
        OperatorKind::Join => Arc::new(Schema::new(vec![Field::new(
            "join_match",
            DataType::Boolean,
            false,
        )])),
        OperatorKind::Extend => {
            Arc::new(Schema::new(vec![Field::new("topic", DataType::Utf8, true)]))
        }
    };
    Prompt::new(
        "the instruction".to_string(),
        Some(mode.system_template(kind)),
        mode.user_template(kind),
        output_schema,
    )
    .with_reasoning(mode)
}

// System prompts.

/// Every (operator, mode) pair must name a template that exists, renders
/// without leaving Jinja behind, and renders to the same bytes twice: the
/// system prompt is the head of the shared prefix a provider caches.
#[test]
fn every_operator_and_mode_renders_a_stable_system_prompt() {
    for kind in OPERATORS {
        for mode in ReasoningMode::all() {
            let prompt = prompt_for(kind, mode);
            let rendered = prompt
                .render_system()
                .unwrap_or_else(|e| panic!("{kind:?}/{mode:?}: {e}"))
                .unwrap_or_else(|| panic!("{kind:?}/{mode:?}: no system message"));

            assert!(
                !rendered.trim().is_empty(),
                "{kind:?}/{mode:?} rendered an empty system prompt"
            );
            assert!(
                !rendered.contains("{{") && !rendered.contains("{%"),
                "{kind:?}/{mode:?} left Jinja unrendered:\n{rendered}"
            );
            assert_eq!(
                rendered,
                prompt.render_system().unwrap().unwrap(),
                "{kind:?}/{mode:?} system prompt is not byte-stable"
            );
        }
    }
}

/// The reasoning modes must differ from the default and from each other.
#[test]
fn each_arm_renders_a_distinct_system_prompt() {
    for kind in OPERATORS {
        let mut rendered: Vec<String> = ReasoningMode::all()
            .into_iter()
            .map(|mode| prompt_for(kind, mode).render_system().unwrap().unwrap())
            .collect();
        let total = rendered.len();
        rendered.sort();
        rendered.dedup();
        assert_eq!(rendered.len(), total, "{kind:?}: two arms render alike");
    }
}

/// Selection must render selection's own rationale prompt, not extend's. The
/// two system prompts are byte-identical under `Off`, so only the reasoning
/// modes distinguish them.
#[test]
fn selection_rationale_is_selections_own_template_not_extends() {
    let selection = prompt_for(OperatorKind::Selection, ReasoningMode::Rationale)
        .render_system()
        .unwrap()
        .unwrap();
    let extend = prompt_for(OperatorKind::Extend, ReasoningMode::Rationale)
        .render_system()
        .unwrap()
        .unwrap();

    assert_ne!(
        selection, extend,
        "selection must not render extend's rationale scaffold"
    );
    assert!(
        selection.contains("matches_filter"),
        "selection's rationale prompt must speak about matches_filter:\n{selection}"
    );
    assert!(
        !selection.contains("output schema"),
        "extend's schema-driven wording leaked into selection:\n{selection}"
    );
}

/// Under `Off` the extend and selection system prompts are the same bytes. If
/// this fails, `Off` has become operator-specific.
#[test]
fn off_mode_selection_and_extend_system_prompts_are_still_identical() {
    assert_eq!(
        prompt_for(OperatorKind::Selection, ReasoningMode::Off)
            .render_system()
            .unwrap(),
        prompt_for(OperatorKind::Extend, ReasoningMode::Off)
            .render_system()
            .unwrap(),
    );
}

#[test]
fn off_mode_resolves_the_pre_reasoning_template_names() {
    assert_eq!(
        ReasoningMode::Off.system_template(OperatorKind::Selection),
        "selection/system.md"
    );
    assert_eq!(
        ReasoningMode::Off.system_template(OperatorKind::Extend),
        "extend/system.md"
    );
    assert_eq!(
        ReasoningMode::Off.system_template(OperatorKind::Join),
        "join/system.md"
    );
    for kind in OPERATORS {
        assert_eq!(
            ReasoningMode::Off.user_template(kind),
            ReasoningMode::Rationale.user_template(kind),
            "the user template must not vary with the arm"
        );
    }
}

// Wire schema.

/// In `Rationale` mode the model must write the reason before the decision.
/// The grammar the server compiles decides emission order, so `rationale` has
/// to lead `properties` and `required` in the serialized schema, not merely be
/// present in it.
#[test]
fn rationale_leads_the_wire_schema() {
    let prompt = prompt_for(OperatorKind::Selection, ReasoningMode::Rationale);

    let fields: Vec<String> = prompt
        .wire_fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(fields, vec!["rationale", "matches_filter"]);

    let serialized = construct_output_schema_from_prompt(&prompt, 4);
    let rationale = serialized.find("\"rationale\"").expect("rationale present");
    let decision = serialized
        .find("\"matches_filter\"")
        .expect("decision present");
    assert!(
        rationale < decision,
        "the model would decide then narrate: {serialized}"
    );

    // `additionalProperties: false` would forbid the key the system prompt
    // just demanded unless `required` carries it too.
    let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        value["items"]["required"],
        json!(["rationale", "matches_filter"])
    );
    assert_eq!(value["items"]["additionalProperties"], json!(false));

    // The abstention widening is for derived columns only. `rationale` is
    // never an Arrow column, and letting the grammar accept `"rationale": null`
    // would let the model skip the reasoning this mode exists to obtain.
    assert_eq!(
        value["items"]["properties"]["rationale"]["type"],
        json!("string")
    );
}

/// `Off` must put no extra key on the wire and use a plain `"boolean"` rather
/// than a nullable union. Pinned as an exact string.
#[test]
fn off_mode_wire_schema_is_byte_identical_to_the_baseline() {
    const BASELINE: &str = concat!(
        r#"{"$schema":"http://json-schema.org/draft-07/schema#","title":"BatchPrediction","#,
        r#""description":"Predictions for a batch of inputs","type":"array","minItems":4,"#,
        r#""maxItems":4,"items":{"type":"object","properties":{"matches_filter":"#,
        r#"{"type":"boolean"}},"required":["matches_filter"],"additionalProperties":false}}"#
    );

    let explicit = prompt_for(OperatorKind::Selection, ReasoningMode::Off);
    assert_eq!(construct_output_schema_from_prompt(&explicit, 4), BASELINE);

    // A prompt that never mentions a mode at all must agree with `Off`.
    let untouched = Prompt::new(
        "the instruction".to_string(),
        Some("selection/system.md".to_string()),
        "selection/user.md",
        Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )])),
    );
    assert_eq!(construct_output_schema_from_prompt(&untouched, 4), BASELINE);
}

/// The reasoning system prompts instruct the model to answer `null` when a row
/// lacks the information. The grammar has to permit that, but only in the modes
/// that ask for it, so the default mode's bytes stay unchanged.
#[test]
fn nullable_derived_fields_widen_only_outside_off_mode() {
    let json_type = |mode: ReasoningMode| {
        let prompt = prompt_for(OperatorKind::Extend, mode);
        let value: serde_json::Value =
            serde_json::from_str(&construct_output_schema_from_prompt(&prompt, 2)).unwrap();
        value["items"]["properties"]["topic"]["type"].clone()
    };

    assert_eq!(json_type(ReasoningMode::Off), json!("string"));
    for mode in [
        ReasoningMode::Scaffold,
        ReasoningMode::Rationale,
        ReasoningMode::Think,
    ] {
        assert_eq!(
            json_type(mode),
            json!(["string", "null"]),
            "{mode:?} asks the model to abstain but forbids it in the grammar"
        );
    }
}

/// `HistogramInjectionRule` sizes batches from `flat_static_prompt_tokens` /
/// `join_static_prompt_tokens` over the mode's own templates. The reasoning
/// system prompts are several hundred tokens longer than the default, so a
/// hardcoded template name would hand the model batches that no longer fit.
/// This pins the gap the rule has to see.
#[test]
fn reasoning_arms_cost_measurably_more_static_prompt() {
    use nousdb_core::prompt::utils::{flat_static_prompt_tokens, join_static_prompt_tokens};
    use nousdb_core::token_estimator::TokenEstimator;

    let est = TokenEstimator::default_estimator().unwrap();
    let derived: SchemaRef = Arc::new(Schema::new(vec![Field::new("topic", DataType::Utf8, true)]));

    let flat = |kind: OperatorKind, mode: ReasoningMode| {
        let schema = matches!(kind, OperatorKind::Extend).then(|| derived.clone());
        flat_static_prompt_tokens(
            &mode.system_template(kind),
            mode.user_template(kind),
            "the instruction",
            schema.as_ref(),
            &est,
        )
        .unwrap()
    };
    let join = |mode: ReasoningMode| {
        join_static_prompt_tokens(
            &mode.system_template(OperatorKind::Join),
            mode.user_template(OperatorKind::Join),
            "the values are related",
            &est,
        )
        .unwrap()
    };

    for kind in [OperatorKind::Selection, OperatorKind::Extend] {
        let baseline = flat(kind, ReasoningMode::Off);
        for mode in [
            ReasoningMode::Scaffold,
            ReasoningMode::Rationale,
            ReasoningMode::Think,
        ] {
            let measured = flat(kind, mode);
            assert!(
                measured > baseline + 200,
                "{kind:?}/{mode:?}: {measured} vs baseline {baseline} — sizing is \
                 measuring the wrong template"
            );
        }
    }

    let baseline = join(ReasoningMode::Off);
    for mode in [
        ReasoningMode::Scaffold,
        ReasoningMode::Rationale,
        ReasoningMode::Think,
    ] {
        assert!(
            join(mode) > baseline + 200,
            "join/{mode:?}: {} vs baseline {baseline}",
            join(mode)
        );
    }
}

/// A non-nullable decision field is never widened: selection and join have no
/// abstention, and `["boolean","null"]` would let the model dodge the filter.
#[test]
fn non_nullable_decision_fields_are_never_widened() {
    for (kind, name) in [
        (OperatorKind::Selection, "matches_filter"),
        (OperatorKind::Join, "join_match"),
    ] {
        for mode in ReasoningMode::all() {
            let prompt = prompt_for(kind, mode);
            let value: serde_json::Value =
                serde_json::from_str(&construct_output_schema_from_prompt(&prompt, 2)).unwrap();
            assert_eq!(
                value["items"]["properties"][name]["type"],
                json!("boolean"),
                "{kind:?}/{mode:?}"
            );
        }
    }
}

/// The prompt prose and the grammar are decoded against each other, so the
/// `{{ output_schema }}` block must name the wire fields.
#[test]
fn the_format_instructions_describe_the_wire_fields() {
    let renderer = nousdb_operators::extend::renderer::ExtendBatchRenderer::new(
        "give the topic".to_string(),
        vec!["note".to_string()],
        vec!["topic".to_string()],
    );
    let schema = Arc::new(Schema::new(vec![Field::new("note", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["a note"])) as ArrayRef],
    )
    .unwrap();

    let rendered = prompt_for(OperatorKind::Extend, ReasoningMode::Rationale)
        .render_batch(&renderer, &batch, 0, 1)
        .unwrap();
    assert!(
        rendered.contains("rationale, topic"),
        "the format block must list the wire fields in order:\n{rendered}"
    );

    let baseline = prompt_for(OperatorKind::Extend, ReasoningMode::Off)
        .render_batch(&renderer, &batch, 0, 1)
        .unwrap();
    assert!(!baseline.contains("rationale"));
}

// The user templates' hand-written output examples.

/// Extend interpolates `{{ output_schema }}`, which is built from
/// `wire_fields()` and so follows the mode. Selection and join hand-write their
/// example (`{{ output_schema }}` renders every field as `"name": "value"`,
/// which contradicts a boolean), so they must branch on the mode themselves or
/// show the model a shape the grammar will reject.
fn render_user_prompt(kind: OperatorKind, mode: ReasoningMode) -> String {
    let prompt = prompt_for(kind, mode);
    match kind {
        OperatorKind::Selection => {
            let renderer = nousdb_operators::selection::renderer::SemanticSelectionRenderer::new(
                "the note describes an outdoor activity".to_string(),
                vec!["note".to_string()],
            );
            let schema = Arc::new(Schema::new(vec![Field::new("note", DataType::Utf8, false)]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(vec!["a note", "another"])) as ArrayRef],
            )
            .unwrap();
            prompt.render_batch(&renderer, &batch, 0, 2).unwrap()
        }
        OperatorKind::Join => {
            let renderer = nousdb_operators::join::renderer::SemanticJoinRenderer::new(
                "the two entries name the same city".to_string(),
                vec!["city".to_string()],
                vec!["location".to_string()],
            );
            let schema = Arc::new(Schema::new(vec![
                Field::new("left_0", DataType::Utf8, false),
                Field::new("right_0", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["Munich", "Munich"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["München", "Marseille"])) as ArrayRef,
                ],
            )
            .unwrap();
            prompt.render_batch(&renderer, &batch, 0, 2).unwrap()
        }
        OperatorKind::Extend => {
            let renderer = nousdb_operators::extend::renderer::ExtendBatchRenderer::new(
                "give the topic".to_string(),
                vec!["note".to_string()],
                vec!["topic".to_string()],
            );
            let schema = Arc::new(Schema::new(vec![Field::new("note", DataType::Utf8, false)]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(vec!["a note"])) as ArrayRef],
            )
            .unwrap();
            prompt.render_batch(&renderer, &batch, 0, 1).unwrap()
        }
    }
}

/// The `## Required Output Format` section: everything from its heading to the
/// next one. Both templates keep the row data strictly below it.
fn output_format_section(rendered: &str) -> &str {
    let start = rendered
        .find("## Required Output Format")
        .unwrap_or_else(|| panic!("output format section present:\n{rendered}"));
    let rest = &rendered[start..];
    let end = rest[3..]
        .find("\n## ")
        .map(|i| i + 3 + 1)
        .unwrap_or(rest.len());
    &rest[..end]
}

/// Every fenced ```json block inside the output-format section, parsed.
/// Objects are returned flattened, so a one-object example and a two-object
/// array example are checked the same way.
fn output_example_objects(section: &str) -> Vec<serde_json::Map<String, serde_json::Value>> {
    let mut objects = Vec::new();
    let mut fences = section.split("```json").skip(1).peekable();
    assert!(
        fences.peek().is_some(),
        "fenced example present:\n{section}"
    );
    for fence in fences {
        let body = fence.split("```").next().unwrap().trim();
        let parsed: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("example must be valid JSON ({e}):\n{body}"));
        match parsed {
            serde_json::Value::Array(items) => {
                for item in items {
                    objects.push(item.as_object().expect("array of objects").clone());
                }
            }
            serde_json::Value::Object(map) => objects.push(map),
            other => panic!("example must be an object or an array of them: {other}"),
        }
    }
    objects
}

/// Under `Rationale` the grammar requires a `rationale` before the decision
/// field, so an example showing only the decision contradicts the schema the
/// answer is decoded against. The response would still parse, so the damage
/// would show up as perfunctory rationales rather than an error.
#[test]
fn selection_and_join_examples_carry_a_rationale_only_under_that_arm() {
    for kind in [OperatorKind::Selection, OperatorKind::Join] {
        let section = |mode| output_format_section(&render_user_prompt(kind, mode)).to_string();

        let rationale = section(ReasoningMode::Rationale);
        assert!(
            rationale.contains("rationale"),
            "{kind:?}: the Rationale arm's example omits the field the grammar \
             demands first:\n{rationale}"
        );

        for mode in [
            ReasoningMode::Off,
            ReasoningMode::Scaffold,
            ReasoningMode::Think,
        ] {
            let other = section(mode);
            assert!(
                !other.contains("rationale"),
                "{kind:?}/{mode:?}: this arm declares no wire-only field, so its \
                 example must not invent one:\n{other}"
            );
        }
    }
}

/// Whatever the example shows must be exactly `Prompt::wire_fields()`, in
/// order, so prose and grammar drift fails regardless of how the template
/// spells its branch.
#[test]
fn every_rendered_output_example_matches_the_wire_schema() {
    for kind in OPERATORS {
        for mode in ReasoningMode::all() {
            let expected: Vec<String> = prompt_for(kind, mode)
                .wire_fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();

            let rendered = render_user_prompt(kind, mode);
            let section = output_format_section(&rendered);

            // Extend describes its wire fields in prose (`{{ output_schema }}`
            // lists them by name) rather than in a fenced block.
            if matches!(kind, OperatorKind::Extend) {
                assert!(
                    section.contains(&expected.join(", ")),
                    "{kind:?}/{mode:?}: the format block must list {expected:?}:\n{section}"
                );
                continue;
            }

            let objects = output_example_objects(section);
            assert!(!objects.is_empty(), "{kind:?}/{mode:?}: no example objects");
            for object in &objects {
                let mut keys: Vec<String> = object.keys().cloned().collect();
                let mut want = expected.clone();
                keys.sort();
                want.sort();
                assert_eq!(
                    keys, want,
                    "{kind:?}/{mode:?}: the in-prompt example and the response \
                     grammar disagree"
                );
            }

            // Order is checked on the rendered bytes rather than on a parsed
            // map, so the assertion does not depend on serde_json's
            // `preserve_order` feature. The grammar emits wire fields in this
            // order; an example showing the decision first would model
            // decide-then-narrate.
            let positions: Vec<usize> = expected
                .iter()
                .map(|name| {
                    section
                        .find(&format!("\"{name}\""))
                        .unwrap_or_else(|| panic!("{kind:?}/{mode:?}: '{name}' absent:\n{section}"))
                })
                .collect();
            assert!(
                positions.windows(2).all(|w| w[0] < w[1]),
                "{kind:?}/{mode:?}: the example orders {expected:?} as {positions:?}"
            );
        }
    }
}

/// The output example must not grow with the batch, for selection as well as
/// for join, and in every mode. `prompt_prefix.rs` pins the join half against
/// pair count; this pins that the section is identical across batch sizes,
/// which is what a provider's prefix cache keys on.
#[test]
fn the_output_format_section_is_constant_size_within_an_arm() {
    let renderer = nousdb_operators::selection::renderer::SemanticSelectionRenderer::new(
        "the note describes an outdoor activity".to_string(),
        vec!["note".to_string()],
    );
    let schema = Arc::new(Schema::new(vec![Field::new("note", DataType::Utf8, false)]));
    let values: Vec<String> = (0..9).map(|i| format!("note {i}")).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(values)) as ArrayRef],
    )
    .unwrap();

    for mode in ReasoningMode::all() {
        let prompt = prompt_for(OperatorKind::Selection, mode);
        let one = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();
        let nine = prompt.render_batch(&renderer, &batch, 0, 9).unwrap();
        assert_eq!(
            output_format_section(&one),
            output_format_section(&nine),
            "{mode:?}: the output example was expanded once per row"
        );
    }
}

// Decoding.

/// An extra wire key must not disturb the single-field decode path: selection
/// and join index their one field by name and stay on the `Boolean` branch.
#[test]
fn a_two_key_response_still_decodes_to_boolean() {
    for (kind, body) in [
        (
            OperatorKind::Selection,
            r#"[{"rationale":"a hike is outdoors","matches_filter":true},
                {"rationale":"desk work","matches_filter":false}]"#,
        ),
        (
            OperatorKind::Join,
            r#"[{"rationale":"same actor","join_match":true},
                {"rationale":"different people","join_match":false}]"#,
        ),
    ] {
        let prompt = prompt_for(kind, ReasoningMode::Rationale);
        let result = parse_json_to_model_outputs(body, &prompt, 2);
        assert!(!result.row_count_mismatch, "{kind:?}");
        assert!(
            matches!(result.outputs[0], ModelOutput::Boolean(true)),
            "{kind:?} decoded {:?}",
            result.outputs[0]
        );
        assert!(
            matches!(result.outputs[1], ModelOutput::Boolean(false)),
            "{kind:?} decoded {:?}",
            result.outputs[1]
        );
    }
}

/// A single-field extend answering `null` must decode as an abstention, not as
/// the empty string, as the reasoning templates ask.
#[test]
fn a_null_answer_decodes_as_an_abstention() {
    let prompt = prompt_for(OperatorKind::Extend, ReasoningMode::Rationale);
    let result = parse_json_to_model_outputs(
        r#"[{"rationale":"the note is absent","topic":null}]"#,
        &prompt,
        1,
    );
    assert!(!result.row_count_mismatch);
    assert!(
        matches!(result.outputs[0], ModelOutput::Null),
        "got {:?}",
        result.outputs[0]
    );
}

// Extend has no rationale column.

/// Returns whatever `outputs` it was built with, ignoring the request.
#[derive(Debug)]
struct FixedOutputMock {
    row: serde_json::Value,
}

#[async_trait]
impl ModelProvider for FixedOutputMock {
    fn name(&self) -> &str {
        "mock"
    }
    fn models(&self) -> Vec<ModelCard> {
        vec![]
    }
    fn model(&self, _model_id: &str) -> Option<ModelCard> {
        None
    }
    async fn predict(
        &self,
        _model_id: &str,
        _prompt: &Prompt,
        _batch: &RecordBatch,
        _start_row: usize,
        batch_size: usize,
        _renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        Ok(PredictResult::new(
            vec![ModelOutput::Structured(self.row.clone()); batch_size],
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
    }
}

/// The rationale reaches the model but must never reach Arrow. Extend's probe
/// comparator requires every derived field to agree, and two independent
/// free-text generations never do, so a `rationale` column would drive every
/// certified batch size to 1.
#[tokio::test]
async fn extend_under_rationale_emits_no_rationale_column() {
    let input_schema: SchemaRef =
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));
    let derived: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("label", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]));
    let full: SchemaRef = Arc::new(Schema::new(
        input_schema
            .fields()
            .iter()
            .chain(derived.fields().iter())
            .cloned()
            .collect::<Vec<_>>(),
    ));

    let batch = RecordBatch::try_new(
        input_schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
    )
    .unwrap();
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        input_schema,
        futures::stream::iter(vec![Ok(batch)]),
    ));

    let provider = Arc::new(FixedOutputMock {
        row: json!({"rationale": "even values score high", "label": "even", "score": 0.9}),
    });

    let stream = ExtendStream::new(
        input,
        "label the parity".to_string(),
        vec!["val".to_string()],
        derived.clone(),
        "mock-model".to_string(),
        provider,
        ModelCard::new("mock", 8192),
        full.clone(),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        Some(8),
        1,
        8192,
        ReasoningMode::Rationale,
        None,
    );

    let out: Vec<RecordBatch> = stream.map(|r| r.unwrap()).collect().await;
    let batch = &out[0];

    assert_eq!(
        batch.num_columns(),
        1 + derived.fields().len(),
        "output columns: {:?}",
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    );
    assert!(batch.column_by_name("rationale").is_none());
    let labels = batch
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(labels.value(0), "even");
    assert!(!labels.is_null(0));
}
