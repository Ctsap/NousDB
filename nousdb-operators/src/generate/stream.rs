//! Execution stream for Generate: chunked requests with retry and row-dropping recovery.

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::error::Result;
use datafusion::physical_plan::RecordBatchStream;

use crate::extend::stream::outputs_to_json;
use crate::generate::renderer::GenerateBatchRenderer;
use crate::metrics::{SemanticOperatorMetrics, WallClockTimer};
use nousdb_core::prompt::prompt::Prompt;
use nousdb_core::providers::metrics::{InferenceMetrics, Rejection};
use nousdb_core::providers::provider::{AdmissionControl, ModelProvider};
use nousdb_core::providers::schemas::ModelOutput;

use futures::Future;
use futures::stream::{FuturesOrdered, Stream, StreamExt};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// System template for Generate. Fixed rather than taken from
/// `ReasoningMode::system_template`: the reasoning modes scaffold a judgement
/// per input item, and a generate request has no input item.
pub const GENERATE_SYSTEM_TEMPLATE: &str = "generate/system.md";
/// User template for Generate.
pub const GENERATE_USER_TEMPLATE: &str = "generate/user.md";

/// The rows `[offset, offset + len)` of the table being generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    offset: usize,
    len: usize,
}

enum ChunkOutcome {
    Rows(RecordBatch),
    /// The response could not be used. `transport` failures never reached the
    /// model and are retried as they are; anything else is retried smaller.
    Rejected {
        reason: String,
        transport: bool,
    },
}

struct RequestResult {
    chunk: Chunk,
    outcome: ChunkOutcome,
    metrics: InferenceMetrics,
    truncated: bool,
}

type RequestFuture = Pin<Box<dyn Future<Output = Result<RequestResult>> + Send>>;

/// Generates `num_rows` rows in requests of at most `chunk_size` rows, up to
/// `request_window` of them in flight, emitting one batch per completed
/// request in submission order.
///
/// With a fetch ([`Self::with_fetch`]) only the first `fetch` rows are asked
/// for. The prompt still describes the whole table ("rows a–b of N"), so a
/// fetch reads a prefix of the same table instead of defining a smaller one.
///
/// A request whose response is unusable is retried: as it is when it never
/// reached the model, split in half otherwise. Halving relieves an output
/// budget the request overran and changes the prompt, which matters because
/// decoding is deterministic and the same prompt would fail the same way. A
/// failing single-row request is dropped, and so is every failure once the
/// retry budget (twice the initial request count) is spent. Dropped rows are
/// counted in `failed_rows`; the stream finishes short rather than failing.
pub struct GenerateStream {
    schema: SchemaRef,
    /// Zero columns, `num_rows` rows: every request is a slice of it.
    slots: RecordBatch,
    renderer: GenerateBatchRenderer,
    prompt: Prompt,
    model_id: String,
    model_provider: Arc<dyn ModelProvider>,
    /// Endpoint-global permit pool shared across partitions and operators.
    admission: Option<AdmissionControl>,
    request_window: usize,
    chunk_size: usize,
    pending: VecDeque<Chunk>,
    in_flight: FuturesOrdered<RequestFuture>,
    retries_left: usize,
    metrics: SemanticOperatorMetrics,
    inference_timer: Arc<WallClockTimer>,
    done: bool,
}

impl GenerateStream {
    /// Creates a stream that generates `num_rows` rows of `schema` in requests
    /// of at most `chunk_size` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        schema: SchemaRef,
        instruction: String,
        examples: Vec<String>,
        num_rows: usize,
        chunk_size: usize,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        request_window: usize,
        metrics: SemanticOperatorMetrics,
        inference_timer: Arc<WallClockTimer>,
    ) -> Result<Self> {
        let renderer =
            GenerateBatchRenderer::new(instruction.clone(), &schema, examples, num_rows)?;
        let prompt = Prompt::new(
            instruction,
            Some(GENERATE_SYSTEM_TEMPLATE),
            GENERATE_USER_TEMPLATE,
            schema.clone(),
        );
        let slots = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?;

        let chunk_size = chunk_size.max(1);
        let pending = plan_chunks(num_rows, chunk_size);
        let retries_left = 2 * pending.len();
        let admission = model_provider.admission_control();

        Ok(Self {
            schema,
            slots,
            renderer,
            prompt,
            model_id,
            model_provider,
            admission,
            request_window: request_window.max(1),
            chunk_size,
            pending,
            in_flight: FuturesOrdered::new(),
            retries_left,
            metrics,
            inference_timer,
            done: false,
        })
    }

    /// Ask only for the first `fetch` rows of the table (`None`: all of it).
    /// Call before the first poll: it re-plans the requests.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        let num_rows = self.slots.num_rows();
        let rows = fetch.map_or(num_rows, |f| f.min(num_rows));
        self.pending = plan_chunks(rows, self.chunk_size);
        self.retries_left = 2 * self.pending.len();
        self
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_chunk(
        chunk: Chunk,
        slots: RecordBatch,
        renderer: GenerateBatchRenderer,
        prompt: Prompt,
        schema: SchemaRef,
        model_id: String,
        model_provider: Arc<dyn ModelProvider>,
        admission: Option<AdmissionControl>,
    ) -> Result<RequestResult> {
        // One permit per in-flight request; held until the response is
        // decoded so the endpoint-global cap bounds actual load.
        let _permit = match admission {
            Some(ac) => Some(ac.acquire().await),
            None => None,
        };

        let predict_result = model_provider
            .predict(
                &model_id,
                &prompt,
                &slots,
                chunk.offset,
                chunk.len,
                &renderer,
            )
            .await?;

        let rejection = predict_result.rejection(chunk.len);
        let truncated = predict_result.truncated;
        let metrics = predict_result.metrics;

        let outcome = match rejection {
            Some(rejection) => ChunkOutcome::Rejected {
                reason: rejection.to_string(),
                transport: matches!(rejection, Rejection::Transport { .. }),
            },
            // Extend treats a decode error as fatal; here it is one more
            // unusable response, and the smaller retry gets a different prompt.
            None => match decode(&predict_result.outputs, &prompt, &schema, chunk.len) {
                Ok(batch) => ChunkOutcome::Rows(batch),
                Err(e) => ChunkOutcome::Rejected {
                    reason: format!("response did not decode against the schema: {e}"),
                    transport: false,
                },
            },
        };

        Ok(RequestResult {
            chunk,
            outcome,
            metrics,
            truncated,
        })
    }

    /// Dispatch one request into the in-flight window.
    fn spawn_request(&mut self, chunk: Chunk) {
        self.inference_timer.start();
        let fut = Self::run_chunk(
            chunk,
            self.slots.clone(),
            self.renderer.clone(),
            self.prompt.clone(),
            self.schema.clone(),
            self.model_id.clone(),
            self.model_provider.clone(),
            self.admission.clone(),
        );
        self.in_flight.push_back(Box::pin(fut));
    }

    /// Requeue a rejected chunk, or drop it once it cannot usefully be retried.
    fn retry_or_drop(&mut self, chunk: Chunk, reason: String, transport: bool) {
        let first = chunk.offset + 1;
        let last = chunk.offset + chunk.len;

        if self.retries_left == 0 || (!transport && chunk.len == 1) {
            log::warn!(
                "GenerateRecovery: {reason} — dropping rows {first}–{last} ({} rows); \
                 the result will be short",
                chunk.len
            );
            self.metrics.failed_rows.add(chunk.len);
            return;
        }
        self.retries_left -= 1;

        if transport {
            log::warn!("GenerateRecovery: {reason} — retrying rows {first}–{last}");
            self.pending.push_front(chunk);
        } else {
            let head = Chunk {
                offset: chunk.offset,
                len: chunk.len - chunk.len / 2,
            };
            let tail = Chunk {
                offset: head.offset + head.len,
                len: chunk.len / 2,
            };
            log::warn!(
                "GenerateRecovery: {reason} — retrying rows {first}–{last} as two requests \
                 of {} and {} rows",
                head.len,
                tail.len
            );
            self.pending.push_front(tail);
            self.pending.push_front(head);
        }
    }
}

/// The requests that cover rows `[0, rows)`: `chunk_size` rows each and one
/// short tail.
fn plan_chunks(rows: usize, chunk_size: usize) -> VecDeque<Chunk> {
    (0..rows)
        .step_by(chunk_size)
        .map(|offset| Chunk {
            offset,
            len: chunk_size.min(rows - offset),
        })
        .collect()
}

/// Decode a usable response into a batch of `expected` generated rows.
fn decode(
    outputs: &[ModelOutput],
    prompt: &Prompt,
    schema: &SchemaRef,
    expected: usize,
) -> Result<RecordBatch> {
    let json = outputs_to_json(outputs, schema)?;
    let arrays = prompt.parse_response(&json, expected)?;
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
}

impl Stream for GenerateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if this.done {
                return Poll::Ready(None);
            }

            // Keep the request window full. Requests are independent, so they
            // all dispatch up front, bounded only by the window.
            while this.in_flight.len() < this.request_window {
                match this.pending.pop_front() {
                    Some(chunk) => this.spawn_request(chunk),
                    None => break,
                }
            }

            match this.in_flight.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(result))) => {
                    this.metrics.record_inference(&result.metrics);
                    this.inference_timer.stop();
                    if result.truncated {
                        this.metrics.record_truncation();
                    }
                    match result.outcome {
                        ChunkOutcome::Rows(batch) => {
                            this.metrics.record_rows_produced(batch.num_rows());
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        ChunkOutcome::Rejected { reason, transport } => {
                            this.metrics.failed_batches.add(1);
                            this.retry_or_drop(result.chunk, reason, transport);
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                // Nothing in flight, and the window loop above found nothing
                // pending: every row is generated or dropped.
                Poll::Ready(None) => {
                    this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl RecordBatchStream for GenerateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
