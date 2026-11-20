//! Output stream of a repartitioned partition.

use std::pin::Pin;
use std::task::{Context, Poll};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::physical_plan::RecordBatchStream;
use futures::Stream;
use tokio::sync::mpsc;

use crate::metrics::TokenRepartitionMetrics;

/// Stream that receives repartitioned batches from the wave coordinator.
///
/// Each output partition has its own stream backed by an `mpsc::Receiver`.
/// The coordinator task estimates tokens, assigns rows to partitions, and
/// sends filtered batches through the channels, so downstream operators
/// can begin processing as soon as the first batch arrives.
pub struct TokenRepartitionStream {
    /// Channel receiver for this partition's batches
    receiver: mpsc::Receiver<Result<RecordBatch>>,

    /// Output schema
    schema: SchemaRef,

    /// Metrics for this partition
    metrics: TokenRepartitionMetrics,
}

impl TokenRepartitionStream {
    /// Creates a stream over the batches received on `receiver`.
    pub fn new(
        receiver: mpsc::Receiver<Result<RecordBatch>>,
        schema: SchemaRef,
        metrics: TokenRepartitionMetrics,
    ) -> Self {
        Self {
            receiver,
            schema,
            metrics,
        }
    }
}

impl Stream for TokenRepartitionStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.metrics.record_rows_repartitioned(batch.num_rows());
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for TokenRepartitionStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
