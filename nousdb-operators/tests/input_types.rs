//! Streams over inputs whose text is not `Utf8` and whose other columns are
//! Date / Timestamp / Float32 / small-integer types.
//!
//! Plans built elsewhere (parquet, `CAST .. AS VARCHAR`, other DataFrame
//! libraries) can hand the semantic operators `Utf8View` / `LargeUtf8` text and
//! such columns. The render-level pins live in `prompt_prefix.rs`; these run the
//! real extend and selection streams, so the whole path (coalescing, slicing,
//! rendering) must carry such a column, and the prompt the model receives must
//! be the same bytes whichever encoding the text arrived in.

use async_trait::async_trait;
use datafusion::arrow::array::{
    Array, ArrayRef, Date32Array, Float32Array, Int8Array, StringArray,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use std::sync::{Arc, Mutex};

use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::prompt::reasoning::ReasoningMode;
use nousdb_core::prompt::renderer::BatchRenderer;
use nousdb_core::providers::metadata::ModelCard;
use nousdb_core::providers::metrics::{InferenceMetrics, PredictResult};
use nousdb_core::providers::provider::ModelProvider;
use nousdb_core::providers::schemas::ModelOutput;
use nousdb_operators::extend::stream::ExtendStream;
use nousdb_operators::metrics::{SemanticOperatorMetrics, WallClockTimer};
use nousdb_operators::selection::stream::SemanticSelectionStream;

/// Renders every request exactly as a real provider would (through the
/// operator's own `Prompt` and renderer), records the user prompt, and answers
/// with a fixed value so the stream can finish.
#[derive(Debug, Default)]
struct RecordingProvider {
    prompts: Mutex<Vec<String>>,
    boolean_mode: bool,
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
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
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let rendered = prompt.render_batch(renderer, batch, start_row, batch_size)?;
        self.prompts.lock().unwrap().push(rendered);

        let outputs = (0..batch_size)
            .map(|_| {
                if self.boolean_mode {
                    ModelOutput::Boolean(true)
                } else {
                    ModelOutput::Text("ok".to_string())
                }
            })
            .collect();
        Ok(PredictResult::new(
            outputs,
            InferenceMetrics::new(10, 5, 1_000),
            false,
        ))
    }
}

fn input_schema(text: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("text", text, true),
        Field::new("released", DataType::Date32, true),
        Field::new("score", DataType::Float32, true),
        Field::new("rank", DataType::Int8, true),
    ]))
}

/// Five rows; the text column is `Utf8` first, then re-encoded to `text`.
fn input_batch(text: DataType) -> RecordBatch {
    let utf8: ArrayRef = Arc::new(StringArray::from(vec![
        Some("the first review, long enough to leave Utf8View's inline form"),
        Some("short"),
        None,
        Some("a fourth review with unicode: caf\u{e9}"),
        Some(""),
    ]));
    RecordBatch::try_new(
        input_schema(text.clone()),
        vec![
            cast(&utf8, &text).unwrap(),
            // 2024-01-15 + 0..4 days, and one NULL
            Arc::new(Date32Array::from(vec![
                Some(19_737),
                Some(19_738),
                None,
                Some(19_740),
                Some(19_741),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(1.5),
                None,
                Some(3.25),
                Some(4.0),
                Some(5.5),
            ])),
            Arc::new(Int8Array::from(vec![
                Some(1),
                Some(2),
                Some(3),
                None,
                Some(5),
            ])),
        ],
    )
    .unwrap()
}

fn input_stream(batch: RecordBatch) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        batch.schema(),
        futures::stream::iter(vec![Ok(batch)]),
    ))
}

const INPUT_COLUMNS: [&str; 4] = ["text", "released", "score", "rank"];

fn input_columns() -> Vec<String> {
    INPUT_COLUMNS.map(String::from).to_vec()
}

/// Run the extend stream over `text`-encoded input; returns the prompts the
/// provider saw and the output batches.
async fn run_extend(text: DataType) -> (Vec<String>, Vec<RecordBatch>) {
    let provider = Arc::new(RecordingProvider::default());
    let output_schema = Arc::new(Schema::new(vec![Field::new("out", DataType::Utf8, true)]));
    let mut full = input_schema(text.clone()).fields().to_vec();
    full.push(Arc::new(Field::new("out", DataType::Utf8, true)));

    let stream = ExtendStream::new(
        input_stream(input_batch(text)),
        "summarise the review".to_string(),
        input_columns(),
        output_schema,
        "mock-model".to_string(),
        provider.clone(),
        ModelCard::new("mock", 8192),
        Arc::new(Schema::new(full)),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Off,
        None,
    );
    let out = stream.map(|r| r.unwrap()).collect().await;
    let prompts = provider.prompts.lock().unwrap().clone();
    (prompts, out)
}

async fn run_selection(text: DataType) -> (Vec<String>, Vec<RecordBatch>) {
    let provider = Arc::new(RecordingProvider {
        prompts: Mutex::default(),
        boolean_mode: true,
    });
    let stream = SemanticSelectionStream::new(
        input_stream(input_batch(text.clone())),
        "the review is recent".to_string(),
        input_columns(),
        provider.clone(),
        "mock-model".to_string(),
        ModelCard::new("mock", 8192),
        input_schema(text),
        SemanticOperatorMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::new(WallClockTimer::new()),
        Some(2),
        1,
        8192,
        ReasoningMode::Off,
        None,
    );
    let out = stream.map(|r| r.unwrap()).collect().await;
    let prompts = provider.prompts.lock().unwrap().clone();
    (prompts, out)
}

fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

const ENCODINGS: [DataType; 2] = [DataType::LargeUtf8, DataType::Utf8View];

#[tokio::test]
async fn extend_prompts_are_identical_whatever_the_text_encoding() {
    let (baseline, out) = run_extend(DataType::Utf8).await;
    assert_eq!(baseline.len(), 3, "5 rows at k=2 are requests of 2, 2, 1");
    assert_eq!(row_count(&out), 5);

    // The non-string columns and the NULLs reached the model as text.
    let all = baseline.join("\n");
    assert!(all.contains("released: 2024-01-15"), "{all}");
    assert!(all.contains("released: NULL"), "{all}");
    assert!(all.contains("score: 1.5"), "{all}");
    assert!(all.contains("rank: 5"), "{all}");
    assert!(all.contains("text: NULL"), "{all}");

    for encoding in ENCODINGS {
        let (prompts, out) = run_extend(encoding.clone()).await;
        assert_eq!(
            prompts, baseline,
            "{encoding:?}: every request must carry the bytes the Utf8 input produced"
        );
        assert_eq!(row_count(&out), 5, "{encoding:?}");
        // The input column keeps its encoding in the output: the operator
        // appends the derived column and rewrites nothing else.
        assert_eq!(
            out[0].column_by_name("text").unwrap().data_type(),
            &encoding
        );
        let derived = out[0].column_by_name("out").unwrap();
        assert_eq!(derived.null_count(), 0, "{encoding:?}: every row answered");
    }
}

#[tokio::test]
async fn selection_prompts_are_identical_whatever_the_text_encoding() {
    let (baseline, out) = run_selection(DataType::Utf8).await;
    assert_eq!(baseline.len(), 3, "5 rows at k=2 are requests of 2, 2, 1");
    assert_eq!(row_count(&out), 5, "the mock keeps every row");

    let all = baseline.join("\n");
    assert!(all.contains("released: 2024-01-15"), "{all}");
    assert!(all.contains("text: NULL"), "{all}");

    for encoding in ENCODINGS {
        let (prompts, out) = run_selection(encoding.clone()).await;
        assert_eq!(
            prompts, baseline,
            "{encoding:?}: every request must carry the bytes the Utf8 input produced"
        );
        assert_eq!(row_count(&out), 5, "{encoding:?}");
        assert_eq!(
            out[0].column_by_name("text").unwrap().data_type(),
            &encoding
        );
    }
}
