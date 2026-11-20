//! Prompt-layout invariants for prefix caching and correctness:
//! the constant preamble (instructions + output format) must be byte-identical
//! across renders, with row data as the only varying suffix.

use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use std::sync::Arc;

use nousdb_core::certification::OperatorKind;
use nousdb_core::prompt::environment::ENV;
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_operators::extend::renderer::ExtendBatchRenderer;
use nousdb_operators::join::renderer::SemanticJoinRenderer;
use nousdb_operators::selection::renderer::SemanticSelectionRenderer;

/// Everything a template renders before its row data. Must be byte-identical
/// across every request of one operator.
fn preamble<'a>(rendered: &'a str, data_heading: &str) -> &'a str {
    let pos = rendered
        .find(data_heading)
        .unwrap_or_else(|| panic!("'{data_heading}' present:\n{rendered}"));
    &rendered[..pos]
}

fn test_batch() -> RecordBatch {
    // Non-alphabetical column order on purpose, to check that column order is kept.
    let schema = Arc::new(Schema::new(vec![
        Field::new("zeta", DataType::Utf8, false),
        Field::new("alpha", DataType::Int64, false),
    ]));
    let zeta: ArrayRef = Arc::new(StringArray::from(vec!["first row text", "second row text"]));
    let alpha: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
    RecordBatch::try_new(schema, vec![zeta, alpha]).unwrap()
}

fn extend_prompt_and_renderer() -> (Prompt, ExtendBatchRenderer) {
    let output_schema = Arc::new(Schema::new(vec![Field::new(
        "sentiment",
        DataType::Utf8,
        false,
    )]));
    let renderer = ExtendBatchRenderer::new(
        "Classify the sentiment of the review".to_string(),
        vec!["zeta".to_string(), "alpha".to_string()],
        vec!["sentiment".to_string()],
    );
    let mode = ReasoningMode::Off;
    let prompt = Prompt::new(
        "Classify the sentiment of the review".to_string(),
        Some(mode.system_template(OperatorKind::Extend)),
        mode.user_template(OperatorKind::Extend),
        output_schema,
    )
    .with_reasoning(mode);
    (prompt, renderer)
}

#[test]
fn extend_prompt_contains_instruction() {
    let (prompt, renderer) = extend_prompt_and_renderer();
    let rendered = prompt.render_batch(&renderer, &test_batch(), 0, 1).unwrap();
    assert!(
        rendered.contains("Classify the sentiment of the review"),
        "instruction must reach the model:\n{rendered}"
    );
}

#[test]
fn per_row_prompts_share_constant_preamble() {
    let (prompt, renderer) = extend_prompt_and_renderer();
    let batch = test_batch();
    let a = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();
    let b = prompt.render_batch(&renderer, &batch, 1, 1).unwrap();

    let lcp = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    let data_pos = a.find("## Data to Process").expect("data section present");
    assert!(
        lcp > data_pos,
        "prompts diverged before the data section (lcp={lcp}, data section at {data_pos})"
    );
    let preamble = &a[..data_pos];
    assert!(preamble.contains("Required Output Format"));
    assert!(preamble.contains("Output Schema"));
}

#[test]
fn row_rendering_is_deterministic_and_schema_ordered() {
    let (prompt, renderer) = extend_prompt_and_renderer();
    let batch = test_batch();
    let a = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();
    let b = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();
    assert_eq!(a, b, "same row must render to identical bytes");

    let zeta_pos = a.find("zeta:").expect("zeta column rendered");
    let alpha_pos = a.find("alpha:").expect("alpha column rendered");
    assert!(
        zeta_pos < alpha_pos,
        "columns must render in input_columns order, not alphabetical"
    );
}

fn selection_prompt_and_renderer() -> (Prompt, SemanticSelectionRenderer) {
    selection_prompt_and_renderer_for(ReasoningMode::Off)
}

fn selection_prompt_and_renderer_for(mode: ReasoningMode) -> (Prompt, SemanticSelectionRenderer) {
    let output_schema = Arc::new(Schema::new(vec![Field::new(
        "matches_filter",
        DataType::Boolean,
        false,
    )]));
    let renderer = SemanticSelectionRenderer::new(
        "the review is positive".to_string(),
        vec!["zeta".to_string()],
    );
    let prompt = Prompt::new(
        "the review is positive".to_string(),
        Some(mode.system_template(OperatorKind::Selection)),
        mode.user_template(OperatorKind::Selection),
        output_schema,
    )
    .with_reasoning(mode);
    (prompt, renderer)
}

#[test]
fn selection_prompt_uses_matches_filter() {
    let (prompt, renderer) = selection_prompt_and_renderer();
    let rendered = prompt.render_batch(&renderer, &test_batch(), 0, 1).unwrap();
    assert!(rendered.contains("matches_filter"));
    assert!(
        !rendered.contains("semantic_filter"),
        "stale field name must be gone"
    );
    let data_pos = rendered.find("## Data to Evaluate").unwrap();
    assert!(rendered[..data_pos].contains("Required Output Format"));
}

/// The row count belongs with the row data, not in the preamble: a partition's
/// last batch is short, and a `{{row_count}}` above the output-format block
/// truncates the shared prefix for every one of those requests.
#[test]
fn extend_preamble_stable_across_batch_sizes() {
    let (prompt, renderer) = extend_prompt_and_renderer();
    let batch = test_batch();
    let full = prompt.render_batch(&renderer, &batch, 0, 2).unwrap();
    let tail = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();

    assert_eq!(
        preamble(&full, "## Data to Process"),
        preamble(&tail, "## Data to Process"),
        "batches of different sizes must share the whole preamble"
    );
}

/// Every reasoning mode, not just the default: the mode-dependent output example
/// lives in the preamble, so a branch that read `row_count` would truncate the
/// shared prefix on the short tail batches.
#[test]
fn selection_preamble_stable_across_batch_sizes() {
    for mode in ReasoningMode::all() {
        let (prompt, renderer) = selection_prompt_and_renderer_for(mode);
        let batch = test_batch();
        let full = prompt.render_batch(&renderer, &batch, 0, 2).unwrap();
        let tail = prompt.render_batch(&renderer, &batch, 0, 1).unwrap();

        assert_eq!(
            preamble(&full, "## Data to Evaluate"),
            preamble(&tail, "## Data to Evaluate"),
            "{mode:?}: batches of different sizes must share the whole preamble"
        );
    }
}

/// The synthetic pair batch `SemanticJoinStream::process_pair_batch` feeds the
/// renderer: left join columns first, then right, all Utf8, one row per pair in
/// the row-major order `next_pair_set` enumerates a left block × right tile.
fn join_pair_batch(lefts: &[&str], rights: &[&str]) -> RecordBatch {
    let mut left_vals = Vec::new();
    let mut right_vals = Vec::new();
    for lv in lefts {
        for rv in rights {
            left_vals.push(*lv);
            right_vals.push(*rv);
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_0", DataType::Utf8, false),
        Field::new("right_0", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(left_vals)) as ArrayRef,
            Arc::new(StringArray::from(right_vals)) as ArrayRef,
        ],
    )
    .unwrap()
}

fn join_prompt_and_renderer_for(mode: ReasoningMode) -> (Prompt, SemanticJoinRenderer) {
    let condition = "the two companies compete".to_string();
    let output_schema = Arc::new(Schema::new(vec![Field::new(
        "join_match",
        DataType::Boolean,
        false,
    )]));
    let renderer = SemanticJoinRenderer::new(
        condition.clone(),
        vec!["company".to_string()],
        vec!["rival".to_string()],
    );
    let prompt = Prompt::new(
        condition,
        Some(mode.system_template(OperatorKind::Join)),
        mode.user_template(OperatorKind::Join),
        output_schema,
    )
    .with_reasoning(mode);
    (prompt, renderer)
}

fn render_join_as(mode: ReasoningMode, batch: &RecordBatch) -> String {
    let (prompt, renderer) = join_prompt_and_renderer_for(mode);
    prompt
        .render_batch(&renderer, batch, 0, batch.num_rows())
        .unwrap()
}

fn render_join(batch: &RecordBatch) -> String {
    render_join_as(ReasoningMode::Off, batch)
}

/// The `## Required Output Format` section, which must not scale with pairs.
fn join_output_format(rendered: &str) -> &str {
    let start = rendered
        .find("## Required Output Format")
        .unwrap_or_else(|| panic!("output format section present:\n{rendered}"));
    let rest = &rendered[start..];
    let end = rest
        .find("## Left Rows")
        .expect("left rows must follow the output format");
    &rest[..end]
}

/// Join's block nested loop holds the left block fixed while scanning the right
/// relation tile by tile, so consecutive requests share the preamble *and* the
/// whole left reference table.
#[test]
fn join_preamble_stable_across_right_tiles() {
    let lefts = ["Acme", "Globex"];
    let a = render_join(&join_pair_batch(&lefts, &["Initech", "Umbrella"]));
    let b = render_join(&join_pair_batch(&lefts, &["Soylent", "Tyrell"]));

    let head = preamble(&a, "## Right Rows");
    assert_eq!(
        head,
        preamble(&b, "## Right Rows"),
        "everything up to the right rows must be byte-identical across tiles"
    );
    assert!(
        head.contains("## Required Output Format"),
        "output format must sit in the shared prefix, not after the row data:\n{head}"
    );
    assert!(head.contains("## Left Rows"));
    assert!(head.contains("Acme") && head.contains("Globex"));
}

/// The last right tile of a scan is short whenever the right relation does not
/// divide evenly, so `pair_count` must never appear above the row data.
#[test]
fn join_preamble_stable_across_pair_counts() {
    for mode in ReasoningMode::all() {
        let lefts = ["Acme", "Globex"];
        let full = render_join_as(
            mode,
            &join_pair_batch(&lefts, &["Initech", "Umbrella", "Soylent"]),
        );
        let tail = render_join_as(mode, &join_pair_batch(&lefts, &["Initech"]));

        assert_eq!(
            preamble(&full, "## Right Rows"),
            preamble(&tail, "## Right Rows"),
            "{mode:?}: a short tail tile must not truncate the shared prefix"
        );
    }
}

/// The example depends on the reasoning mode (a `Rationale` response carries an
/// extra key), so the constant-size guarantee must hold within every mode.
#[test]
fn join_output_format_does_not_scale_with_pair_count() {
    for mode in ReasoningMode::all() {
        let small = render_join_as(mode, &join_pair_batch(&["Acme"], &["Initech", "Umbrella"]));
        let large = render_join_as(
            mode,
            &join_pair_batch(
                &["Acme", "Globex", "Hooli", "Vehement"],
                &["Initech", "Umbrella", "Soylent", "Tyrell", "Wayne"],
            ),
        );

        assert_eq!(
            join_output_format(&small),
            join_output_format(&large),
            "{mode:?}: the output example must be a fixed size, not expanded once per pair"
        );
    }
}

/// `response_format` pins the response shape, but the in-prompt example still
/// has to be copyable, so it must be valid JSON (no `true | false` placeholder).
#[test]
fn join_output_example_is_valid_json() {
    for mode in ReasoningMode::all() {
        let rendered = render_join_as(mode, &join_pair_batch(&["Acme"], &["Initech", "Umbrella"]));
        let section = join_output_format(&rendered);
        let example = section
            .split("```json")
            .nth(1)
            .expect("fenced json example present")
            .split("```")
            .next()
            .unwrap();

        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(example.trim()).unwrap_or_else(|e| {
                panic!("{mode:?}: output example must be valid JSON ({e}):\n{example}")
            });
        assert!(
            parsed
                .iter()
                .all(|v| v.get("join_match").is_some_and(|b| b.is_boolean())),
            "{mode:?}: example objects must carry a boolean join_match: {parsed:?}"
        );
    }
}

/// A calibration probe input is a pair batch plus a trailing row-identity
/// column, and the join renderer reads its columns positionally. The rendered
/// request must therefore be byte-identical to the production request for the
/// same pairs; otherwise the certificate would describe a prompt the query
/// never sends, and probe requests would miss the provider's prefix cache.
#[test]
fn probe_id_column_does_not_change_a_rendered_join_request() {
    use datafusion::arrow::array::UInt32Array;

    let production = join_pair_batch(&["Acme", "Globex"], &["Initech", "Umbrella"]);

    let mut fields = production.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "__nousdb_probe_id",
        DataType::UInt32,
        false,
    )));
    let mut columns = production.columns().to_vec();
    columns.push(Arc::new(UInt32Array::from_iter_values(
        0..production.num_rows() as u32,
    )) as ArrayRef);
    let probe = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

    assert_eq!(render_join(&probe), render_join(&production));
}

#[test]
fn undefined_template_variables_error_loudly() {
    let env = ENV.read().unwrap();
    let result = env.render_str("{{ this_variable_does_not_exist }}", minijinja::context! {});
    assert!(
        result.is_err(),
        "Strict mode must reject undefined variables"
    );
}

// Input encodings and scalar types: plans built elsewhere (parquet,
// `CAST .. AS VARCHAR`, other DataFrame libraries) can hand the operators
// `Utf8View` / `LargeUtf8` text and Date / Timestamp / Float32 / small-int
// columns. Row text is the last section of every prompt, so it must be the
// same bytes whatever the encoding.

/// `test_batch()` with its text column re-encoded. Everything else is the same
/// rows, so any difference in the rendered prompt is the encoding's doing.
fn test_batch_with_text_as(encoding: DataType) -> RecordBatch {
    let base = test_batch();
    let schema = Arc::new(Schema::new(vec![
        Field::new("zeta", encoding.clone(), false),
        Field::new("alpha", DataType::Int64, false),
    ]));
    let zeta = datafusion::arrow::compute::cast(base.column(0), &encoding).unwrap();
    assert_eq!(zeta.data_type(), &encoding);
    RecordBatch::try_new(schema, vec![zeta, base.column(1).clone()]).unwrap()
}

#[test]
fn extend_prompt_bytes_do_not_depend_on_the_string_encoding() {
    let (prompt, renderer) = extend_prompt_and_renderer();
    let utf8 = test_batch();

    for (start, size) in [(0, 2), (1, 1)] {
        let expected = prompt.render_batch(&renderer, &utf8, start, size).unwrap();
        for encoding in [DataType::LargeUtf8, DataType::Utf8View] {
            let batch = test_batch_with_text_as(encoding.clone());
            let rendered = prompt.render_batch(&renderer, &batch, start, size).unwrap();
            assert_eq!(
                rendered,
                expected,
                "{encoding:?}: rows {start}..{} must render exactly as Utf8 does",
                start + size
            );
        }
    }
    // The Utf8 rendering is the baseline the other encodings must match.
    let whole = prompt.render_batch(&renderer, &utf8, 0, 2).unwrap();
    assert!(
        whole.contains("[R.1] zeta: first row text, alpha: 1"),
        "{whole}"
    );
    assert!(
        whole.contains("[R.2] zeta: second row text, alpha: 2"),
        "{whole}"
    );
}

#[test]
fn selection_prompt_bytes_do_not_depend_on_the_string_encoding() {
    for mode in ReasoningMode::all() {
        let (prompt, renderer) = selection_prompt_and_renderer_for(mode);
        let expected = prompt.render_batch(&renderer, &test_batch(), 0, 2).unwrap();
        for encoding in [DataType::LargeUtf8, DataType::Utf8View] {
            let batch = test_batch_with_text_as(encoding.clone());
            let rendered = prompt.render_batch(&renderer, &batch, 0, 2).unwrap();
            assert_eq!(
                rendered, expected,
                "{mode:?} / {encoding:?}: must render exactly as Utf8 does"
            );
        }
    }
}

/// One text column plus Date / Timestamp / Float32 / small-integer columns,
/// with a NULL in the second row of each.
fn mixed_type_batch() -> RecordBatch {
    use datafusion::arrow::array::{
        Date32Array, Decimal128Array, Float32Array, Int8Array, StringViewArray,
        TimestampMillisecondArray,
    };
    use datafusion::arrow::datatypes::TimeUnit;

    let schema = Arc::new(Schema::new(vec![
        Field::new("note", DataType::Utf8View, true),
        Field::new("released", DataType::Date32, true),
        Field::new(
            "seen",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new("score", DataType::Float32, true),
        Field::new("rank", DataType::Int8, true),
        Field::new("price", DataType::Decimal128(10, 2), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(vec![
                Some("a note that is longer than twelve bytes"),
                None,
            ])) as ArrayRef,
            Arc::new(Date32Array::from(vec![Some(19_737), None])),
            Arc::new(TimestampMillisecondArray::from(vec![
                Some(1_705_314_600_123),
                None,
            ])),
            Arc::new(Float32Array::from(vec![Some(4.5), None])),
            Arc::new(Int8Array::from(vec![Some(7), None])),
            Arc::new(
                Decimal128Array::from(vec![Some(1234), None])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap()
}

const MIXED_COLUMNS: [&str; 6] = ["note", "released", "seen", "score", "rank", "price"];

/// What each of the two rows of `mixed_type_batch()` renders to, without the
/// template's per-row marker (`[R.n]` for extend, `n.` for selection).
const MIXED_BODY_1: &str = "note: a note that is longer than twelve bytes, released: 2024-01-15, \
     seen: 2024-01-15T10:30:00.123, score: 4.5, rank: 7, price: 12.34";
const MIXED_BODY_2: &str = "note: NULL, released: NULL, seen: NULL, score: NULL, rank: NULL, \
     price: NULL";

#[test]
fn extend_renders_dates_timestamps_and_other_scalar_columns() {
    let renderer = ExtendBatchRenderer::new(
        "Summarise the row".to_string(),
        MIXED_COLUMNS.map(String::from).to_vec(),
        vec!["summary".to_string()],
    );
    let mode = ReasoningMode::Off;
    let prompt = Prompt::new(
        "Summarise the row".to_string(),
        Some(mode.system_template(OperatorKind::Extend)),
        mode.user_template(OperatorKind::Extend),
        Arc::new(Schema::new(vec![Field::new(
            "summary",
            DataType::Utf8,
            false,
        )])),
    )
    .with_reasoning(mode);

    let rendered = prompt
        .render_batch(&renderer, &mixed_type_batch(), 0, 2)
        .unwrap();
    assert!(
        rendered.contains(&format!("[R.1] {MIXED_BODY_1}")),
        "{rendered}"
    );
    assert!(
        rendered.contains(&format!("[R.2] {MIXED_BODY_2}")),
        "{rendered}"
    );
}

#[test]
fn selection_renders_dates_timestamps_and_other_scalar_columns() {
    let renderer = SemanticSelectionRenderer::new(
        "the film is recent".to_string(),
        MIXED_COLUMNS.map(String::from).to_vec(),
    );
    for mode in ReasoningMode::all() {
        let (prompt, _) = selection_prompt_and_renderer_for(mode);
        let rendered = prompt
            .render_batch(&renderer, &mixed_type_batch(), 0, 2)
            .unwrap();
        assert!(
            rendered.contains(&format!("1. {MIXED_BODY_1}")),
            "{mode:?}: {rendered}"
        );
        assert!(
            rendered.contains(&format!("2. {MIXED_BODY_2}")),
            "{mode:?}: {rendered}"
        );
    }
}
