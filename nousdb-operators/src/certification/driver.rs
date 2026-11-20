//! Shared calibration-prologue state machine for semantic operator streams.
//!
//! Every certified stream (selection, extend) runs the same prologue: buffer
//! rows while donating a copy of them to the shared [`PrefixPool`], race to
//! install the once-per-instance calibration, await its outcome, then hand its
//! own buffered rows back to the owning stream's normal dispatch machinery at
//! the certified batch size `k̂`. The owning streams differ only in the
//! [`CalibrationParts`] they pre-clone at construction.
//!
//! Each partition stops buffering at its quota (`ceil(target/P)` rows) rather
//! than at the whole design. Every partition recycles the rows it buffered
//! itself: the pooled sample holds copies from every partition, so
//! re-injecting it would duplicate rows and emit them from the wrong output
//! partition.
//!
//! The state is a field on the owning stream; each `poll_next` calls
//! [`poll_calibration`] with a [`CalibrationHost`] of disjoint `&mut` borrows
//! of the stream's dispatch fields (rebuilt per poll, never outliving the
//! call). The parts are pre-cloned instead of borrowed from the stream so the
//! install closure only captures locals.

use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::SendableRecordBatchStream;

use crate::certification::calibrate::{ProbeCalibrationContext, run_probe_calibration};
use crate::certification::probe::{ItemSpace, ProbeInputBuilder, ProbeScorer, build_probe_input};
use crate::certification::types::{CalibrationOutcome, StreamCertification};
use crate::join::stream::OnceFut;
use crate::metrics::WallClockTimer;

use futures::stream::StreamExt;
use std::collections::VecDeque;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Per-operator inputs the install step needs to build a
/// [`ProbeCalibrationContext`]. Pre-cloned at stream construction (cheap
/// `Arc`s), so the state machine never borrows the owning stream.
///
/// The probe-plan factory, task context and concurrency budget arrive on
/// [`StreamCertification`], because only the exec can build a probe copy of
/// itself.
pub(crate) struct CalibrationParts {
    /// Operator-specific agreement on output rows.
    pub scorer: ProbeScorer,
    /// Timer for wall-clock time spent in inference.
    pub inference_timer: Arc<WallClockTimer>,
}

/// Quality-certification prologue state, checked before the normal dispatch
/// machine on every poll. `Disabled` short-circuits, so streams without a
/// `CertificationSpec` pay nothing.
pub(crate) enum CalibrationState {
    /// No certification requested for this operator instance.
    Disabled,
    /// Contributing this partition's quota to the shared calibration sample.
    Buffering {
        cert: StreamCertification,
        /// Boxed to keep the enum (a field on every stream) small.
        parts: Box<CalibrationParts>,
        buffered: Vec<RecordBatch>,
        rows: usize,
    },
    /// Waiting on the shared once-per-instance calibration. `buffered` holds
    /// every row this partition read during the prologue; all of it re-enters
    /// normal processing at `k̂` afterwards, in order.
    Awaiting {
        fut: OnceFut<CalibrationOutcome>,
        buffered: Vec<RecordBatch>,
    },
    /// Calibration resolved; the stream's `optimal_batch_size` now holds `k̂`.
    Done,
}

impl CalibrationState {
    /// Entry state for a stream: `Buffering` when certification is requested
    /// for this instance, `Disabled` otherwise. `parts` is only evaluated
    /// when certification is on.
    pub(crate) fn new(
        certification: Option<StreamCertification>,
        parts: impl FnOnce() -> CalibrationParts,
    ) -> Self {
        match certification {
            Some(cert) => CalibrationState::Buffering {
                cert,
                parts: Box::new(parts()),
                buffered: Vec::new(),
                rows: 0,
            },
            None => CalibrationState::Disabled,
        }
    }
}

/// Disjoint `&mut` borrows of the owning stream's dispatch fields: the
/// state machine's window into the stream. Built fresh on every poll.
pub(crate) struct CalibrationHost<'a> {
    pub input: &'a mut SendableRecordBatchStream,
    pub optimal_batch_size: &'a mut Option<usize>,
    pub pending_slices: &'a mut VecDeque<RecordBatch>,
    pub coalesce_buffer: &'a mut Vec<RecordBatch>,
    pub coalesce_rows: &'a mut usize,
    pub input_exhausted: &'a mut bool,
}

impl CalibrationHost<'_> {
    /// Concatenate the coalesce buffer into a single RecordBatch and reset.
    /// Mirror of the streams' own `take_coalesced_batch`; keep in sync.
    fn take_coalesced_batch(&mut self) -> Result<RecordBatch> {
        let batches = std::mem::take(self.coalesce_buffer);
        *self.coalesce_rows = 0;
        let input_schema = self.input.schema();
        concat_batches(&input_schema, &batches).map_err(|e| {
            DataFusionError::Execution(format!("Failed to concat coalesced batches: {}", e))
        })
    }

    /// Split a coalesced batch into request-sized slices (zero-copy).
    /// Mirror of the streams' own `enqueue_request_slices`; keep in sync.
    ///
    /// `keep_remainder` returns a short trailing slice to the coalesce buffer
    /// rather than dispatching it, so production requests land on exactly the
    /// size the certificate was issued for instead of a mix of `k̂` and
    /// under-filled fragments.
    fn enqueue_request_slices(&mut self, combined: RecordBatch, keep_remainder: bool) {
        match *self.optimal_batch_size {
            Some(max_size) if combined.num_rows() > max_size => {
                let total = combined.num_rows();
                let mut offset = 0;
                while offset + max_size <= total {
                    self.pending_slices
                        .push_back(combined.slice(offset, max_size));
                    offset += max_size;
                }
                if offset < total {
                    let tail = combined.slice(offset, total - offset);
                    if keep_remainder {
                        *self.coalesce_rows += tail.num_rows();
                        self.coalesce_buffer.push(tail);
                    } else {
                        self.pending_slices.push_back(tail);
                    }
                }
            }
            _ => self.pending_slices.push_back(combined),
        }
    }
}

/// Drive the calibration prologue. `Ready(Ok(()))` means the normal
/// dispatch machine may proceed (calibration disabled or resolved).
pub(crate) fn poll_calibration(
    state: &mut CalibrationState,
    cx: &mut Context<'_>,
    mut host: CalibrationHost<'_>,
) -> Poll<Result<()>> {
    loop {
        match std::mem::replace(state, CalibrationState::Disabled) {
            CalibrationState::Disabled => return Poll::Ready(Ok(())),
            CalibrationState::Done => {
                *state = CalibrationState::Done;
                return Poll::Ready(Ok(()));
            }
            CalibrationState::Buffering {
                cert,
                parts,
                mut buffered,
                rows,
            } => match host.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    if batch.num_rows() == 0 {
                        *state = CalibrationState::Buffering {
                            cert,
                            parts,
                            buffered,
                            rows,
                        };
                        continue;
                    }
                    // Donate a zero-copy slice to the shared sample and keep the
                    // rows for this partition's own processing. The pool takes
                    // at most the remaining quota.
                    cert.pool.donate(cert.partition, &batch);
                    let rows = rows + batch.num_rows();
                    buffered.push(batch);

                    if cert.pool.quota_met(cert.partition) {
                        install(state, host.input.schema(), cert, *parts, buffered)?;
                    } else {
                        *state = CalibrationState::Buffering {
                            cert,
                            parts,
                            buffered,
                            rows,
                        };
                    }
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                Poll::Ready(None) => {
                    *host.input_exhausted = true;
                    // Tell the pool to expect nothing more from this partition;
                    // otherwise the assembling partition waits out the full
                    // deadline.
                    cert.pool.seal_partition(cert.partition);
                    if rows == 0 {
                        // Empty partition: never touches the shared OnceAsync
                        // (so it cannot poison it with k=1 or deadlock) and
                        // finishes through the normal machine.
                        *state = CalibrationState::Done;
                        return Poll::Ready(Ok(()));
                    }
                    // Fewer rows than the quota: install anyway. The runner
                    // commits the probe budget to whatever the pool gathered.
                    install(state, host.input.schema(), cert, *parts, buffered)?;
                }
                Poll::Pending => {
                    *state = CalibrationState::Buffering {
                        cert,
                        parts,
                        buffered,
                        rows,
                    };
                    return Poll::Pending;
                }
            },
            CalibrationState::Awaiting { mut fut, buffered } => match fut.get(cx) {
                Poll::Pending => {
                    *state = CalibrationState::Awaiting { fut, buffered };
                    return Poll::Pending;
                }
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(outcome)) => {
                    let k_hat = outcome.k_hat;
                    *host.optimal_batch_size = Some(k_hat);

                    // Recycling: this partition re-processes exactly the rows it
                    // read, in order, at k̂. Calibration only read copies of
                    // them from the pool, so nothing has been emitted yet.
                    // Zero-copy: Arrow columns are shared.
                    for batch in buffered {
                        if batch.num_rows() > 0 {
                            *host.coalesce_rows += batch.num_rows();
                            host.coalesce_buffer.push(batch);
                        }
                    }
                    // Flush what already fills request batches;
                    // `next_request_batch` only flushes on new input or
                    // exhaustion, and neither may come for a while.
                    if *host.coalesce_rows >= k_hat {
                        let combined = host.take_coalesced_batch()?;
                        // `keep_remainder`: a short tail waits for the rows that
                        // follow it, so production requests land on exactly k̂
                        // rather than a mix of k̂ and under-filled fragments.
                        host.enqueue_request_slices(combined, true);
                    }
                    *state = CalibrationState::Done;
                    return Poll::Ready(Ok(()));
                }
            },
        }
    }
}

/// Race to install the shared calibration future.
///
/// Exactly one partition's closure runs (`OnceAsync` invokes it synchronously
/// under its lock); every other partition gets a handle to the same shared
/// future. Which one wins does not affect the sample, which comes from the
/// pool.
///
/// The closure only builds the future. Assembling the pool waits on the other
/// partitions, and the closure runs inside `poll_next` while holding a
/// `std::sync::Mutex`; waiting there would park a tokio worker thread on the
/// partitions that need one to make progress. The wait lives inside the async
/// block instead.
fn install(
    state: &mut CalibrationState,
    input_schema: SchemaRef,
    cert: StreamCertification,
    parts: CalibrationParts,
    buffered: Vec<RecordBatch>,
) -> Result<()> {
    let factory = cert.factory.clone();
    let fut = cert.once.try_once(|| {
        let ctx = ProbeCalibrationContext {
            factory,
            task_ctx: cert.task_ctx.clone(),
            window_budget: cert.window_budget,
            scorer: parts.scorer.clone(),
            metrics: cert.metrics.clone(),
            inference_timer: parts.inference_timer.clone(),
        };
        let spec = cert.spec.clone();
        let cache = cert.cache.clone();
        let pool = cert.pool.clone();
        Ok(async move {
            let prefix_batches = pool.assemble().await;
            let prefix = concat_batches(&input_schema, &prefix_batches).map_err(|e| {
                DataFusionError::Execution(format!(
                    "calibration: failed to concat prefix batches: {e}"
                ))
            })?;
            // Selection and extend designate over rows: the item id is the
            // prefix row index.
            let space = ItemSpace::Rows {
                rows: prefix.num_rows(),
            };
            let build: ProbeInputBuilder =
                Arc::new(move |indices: &[usize]| build_probe_input(&prefix, indices));
            run_probe_calibration(spec, space, build, ctx, cache).await
        })
    })?;

    *state = CalibrationState::Awaiting { fut, buffered };
    Ok(())
}
