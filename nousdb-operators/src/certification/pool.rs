//! The calibration sample, pooled across every partition of an operator.
//!
//! The prologue estimates `phi(k)`, the probability that a batched answer
//! agrees with the solo one, over the input the operator is about to process.
//! The sample must therefore represent the whole input, not the slice one
//! partition holds: a single partition may hold too few rows to fill the
//! design, and under round-robin repartitioning its rows are a fixed stride
//! through the input.
//!
//! A [`PrefixPool`] fixes both without re-executing the input plan: each
//! partition copies its share of rows into a shared pool while keeping the
//! originals for its own processing.
//!
//! # Quota
//!
//! `designate` draws each probe's fillers from the whole pooled sample (see
//! [`crate::certification::probe`]), so every partition contributes
//! `ceil(target / P)` rows and a probe batch may mix rows from all of them.
//!
//! # Ordering
//!
//! [`PrefixPool::assemble`] concatenates contributions in partition order, so a
//! deterministic input yields a deterministic prefix regardless of which
//! partition finished first, and re-running a query re-draws the same probes.
//!
//! # Liveness
//!
//! The pool is a barrier: calibration does not start until it is sealed.
//! [`PrefixPool::assemble`] waits at most `deadline` for stragglers before
//! sealing with whatever arrived. The shortfall is absorbed by the realized
//! probe budget rather than reported as an error, because
//! `OnceAsync::try_once` memoizes errors permanently and a transient timeout
//! must never surface as one.
//!
//! The wait is async and happens inside the calibration future, not inside the
//! `try_once` closure. That closure runs synchronously under a
//! `std::sync::Mutex` from within `poll_next`; blocking there would park a
//! tokio worker thread while the partitions it waits for need one to make
//! progress, and deadlocks on a current-thread runtime.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use datafusion::arrow::record_batch::RecordBatch;
use tokio::sync::Notify;

/// How long a partition that has met its quota waits for the others.
pub const DEFAULT_POOL_DEADLINE: Duration = Duration::from_secs(30);

/// One partition's contribution.
#[derive(Default)]
struct Contribution {
    batches: Vec<RecordBatch>,
    rows: usize,
    /// This partition will send nothing further (quota met or input exhausted).
    done: bool,
}

struct PoolState {
    contributions: Vec<Contribution>,
    /// Set once the pool is closed for donations; assembly is stable afterwards.
    sealed: bool,
}

/// Shared calibration sample for one operator instance.
///
/// Created by `CertifyBatchSizeRule` alongside the operator's `OnceAsync`, and
/// cloned into every partition's `StreamCertification`.
pub struct PrefixPool {
    state: Mutex<PoolState>,
    /// Woken whenever a partition meets its quota or seals. `notify_one` (not
    /// `notify_waiters`) because there is exactly one waiter, the partition that
    /// won the install race, and its permit semantics close the race where every
    /// donation lands before the waiter first polls.
    ready: Notify,
    /// Rows each partition contributes: `ceil(target / partitions)`.
    /// Atomic because [`Self::retarget`] revises it at execution time for
    /// operators whose units are not known at plan time.
    quota_rows: AtomicUsize,
    /// Rows the design asks for in total.
    target_rows: AtomicUsize,
    partitions: usize,
    deadline: Duration,
}

impl std::fmt::Debug for PrefixPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixPool")
            .field("quota_rows", &self.quota_rows)
            .field("target_rows", &self.target_rows)
            .field("partitions", &self.partitions)
            .finish_non_exhaustive()
    }
}

impl PrefixPool {
    /// Build a pool that gathers `target` rows across `partitions` partitions.
    pub fn new(target: usize, partitions: usize) -> Self {
        Self::with_deadline(target, partitions, DEFAULT_POOL_DEADLINE)
    }

    /// Like [`Self::new`], with a custom bound on how long
    /// [`Self::assemble`] waits for stragglers.
    pub fn with_deadline(target: usize, partitions: usize, deadline: Duration) -> Self {
        let partitions = partitions.max(1);
        let target = target.max(1);
        Self {
            state: Mutex::new(PoolState {
                contributions: (0..partitions).map(|_| Contribution::default()).collect(),
                sealed: false,
            }),
            ready: Notify::new(),
            // Rounded up so the pool reaches the target even when the
            // partitions do not divide it evenly.
            quota_rows: AtomicUsize::new(target.div_ceil(partitions)),
            target_rows: AtomicUsize::new(target),
            partitions,
            deadline,
        }
    }

    /// Rows one partition should contribute before it stops buffering.
    pub fn quota_rows(&self) -> usize {
        self.quota_rows.load(Ordering::Acquire)
    }

    /// Rows the design asks for across all partitions.
    pub fn target_rows(&self) -> usize {
        self.target_rows.load(Ordering::Acquire)
    }

    /// Restate the target in the operator's own units, once they are known.
    ///
    /// The pool is sized at plan time in rows, which fits selection and extend
    /// (their items are rows). A join's item is a (left, right) pair, so it
    /// pools the number of left rows whose cross product with the right relation
    /// covers the design's item budget; the right relation's size is only known
    /// at runtime.
    ///
    /// Idempotent: every partition computes the same value from the same shared
    /// right relation.
    pub fn retarget(&self, total_rows: usize) {
        let partitions = self.partitions.max(1);
        self.quota_rows
            .store(total_rows.div_ceil(partitions).max(1), Ordering::Release);
        self.target_rows.store(total_rows.max(1), Ordering::Release);
    }

    /// Copy up to this partition's remaining quota out of `batch`.
    ///
    /// Returns the number of rows accepted, which is zero once the quota is met.
    /// The batch is sliced zero-copy; Arrow columns are shared with the
    /// partition's own copy.
    pub fn donate(&self, partition: usize, batch: &RecordBatch) -> usize {
        let mut state = self.state.lock().expect("prefix pool poisoned");
        if state.sealed {
            return 0;
        }
        let Some(slot) = state.contributions.get_mut(partition) else {
            return 0;
        };
        if slot.done {
            return 0;
        }
        let room = self.quota_rows().saturating_sub(slot.rows);
        let take = room.min(batch.num_rows());
        if take > 0 {
            slot.batches.push(batch.slice(0, take));
            slot.rows += take;
        }
        if slot.rows >= self.quota_rows() {
            slot.done = true;
            drop(state);
            self.ready.notify_one();
        }
        take
    }

    /// Mark this partition as finished donating (input exhausted).
    pub fn seal_partition(&self, partition: usize) {
        let mut state = self.state.lock().expect("prefix pool poisoned");
        if let Some(slot) = state.contributions.get_mut(partition)
            && !slot.done
        {
            slot.done = true;
            drop(state);
            self.ready.notify_one();
        }
    }

    /// True once no partition will contribute more.
    fn all_done(state: &PoolState) -> bool {
        state.contributions.iter().all(|c| c.done)
    }

    /// Whether this partition has contributed everything it was asked for.
    pub fn quota_met(&self, partition: usize) -> bool {
        let state = self.state.lock().expect("prefix pool poisoned");
        state.contributions.get(partition).is_some_and(|c| c.done)
    }

    /// Wait until every partition is done, or the deadline expires, then take
    /// the pooled rows in partition order.
    ///
    /// Awaited once, inside the calibration future, by the partition that won the
    /// install race. Every other partition either fills its `1/P` share of the
    /// design or hits end of input, so the wait is short in practice.
    pub async fn assemble(&self) -> Vec<RecordBatch> {
        let start = Instant::now();
        loop {
            // Register interest before re-checking, so a donation landing
            // between the check and the await is not missed.
            let notified = self.ready.notified();

            {
                let state = self.state.lock().expect("prefix pool poisoned");
                if Self::all_done(&state) {
                    break;
                }
            }

            let Some(remaining) = self
                .deadline
                .checked_sub(start.elapsed())
                .filter(|d| !d.is_zero())
            else {
                let state = self.state.lock().expect("prefix pool poisoned");
                let outstanding = state.contributions.iter().filter(|c| !c.done).count();
                log::warn!(
                    "[certify] prefix pool sealed on deadline with {outstanding}/{} \
                     partitions still buffering; calibrating on what arrived",
                    self.partitions
                );
                break;
            };

            if tokio::time::timeout(remaining, notified).await.is_err() {
                continue; // deadline check at the top of the next iteration
            }
        }

        let mut state = self.state.lock().expect("prefix pool poisoned");
        state.sealed = true;
        let mut out = Vec::new();
        for slot in state.contributions.iter_mut() {
            out.append(&mut slot.batches);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(values: std::ops::Range<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let array = Int64Array::from(values.collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// The target is split evenly, rounded up so the pool still reaches it
    /// when the partitions do not divide it: 480 rows over 15 partitions.
    #[test]
    fn quota_splits_the_target_across_partitions() {
        let pool = PrefixPool::new(480, 15);
        assert_eq!(pool.quota_rows(), 32);
        assert_eq!(pool.target_rows(), 480);
        assert!(pool.quota_rows() * 15 >= pool.target_rows());
    }

    /// A single partition takes the whole design.
    #[tokio::test]
    async fn single_partition_takes_the_whole_design() {
        let pool = PrefixPool::new(20, 1);
        assert_eq!(pool.quota_rows(), 20);
        assert_eq!(pool.donate(0, &batch(0..50)), 20);
        assert_eq!(values(&pool.assemble().await), (0..20).collect::<Vec<_>>());
    }

    /// Contributions are concatenated in partition order, not arrival order.
    #[tokio::test]
    async fn assembly_is_deterministic_in_partition_order() {
        let pool = PrefixPool::new(6, 3); // quota 2 rows each
        // Donate out of order.
        pool.donate(2, &batch(200..202));
        pool.donate(0, &batch(0..2));
        pool.donate(1, &batch(100..102));
        assert_eq!(
            values(&pool.assemble().await),
            vec![0, 1, 100, 101, 200, 201]
        );
    }

    /// A partition never contributes more than its quota, however much it holds.
    #[test]
    fn donation_stops_at_the_quota() {
        let pool = PrefixPool::new(8, 2); // quota 4 rows each
        assert_eq!(pool.donate(0, &batch(0..3)), 3);
        assert_eq!(pool.donate(0, &batch(3..10)), 1);
        assert_eq!(pool.donate(0, &batch(10..20)), 0);
        assert!(pool.quota_met(0));
    }

    /// A partition that runs dry seals itself, and the pool completes with
    /// whatever the others provided. The realized budget absorbs the shortfall;
    /// it is not an error.
    #[tokio::test]
    async fn exhausted_partitions_seal_and_do_not_block() {
        let pool = PrefixPool::new(8, 2); // quota 4 rows each
        pool.donate(0, &batch(0..4));
        pool.donate(1, &batch(100..101));
        pool.seal_partition(1);
        assert_eq!(values(&pool.assemble().await), vec![0, 1, 2, 3, 100]);
    }

    /// Donations after sealing are refused rather than silently reordering an
    /// already-assembled prefix.
    #[tokio::test]
    async fn donations_after_sealing_are_refused() {
        let pool = PrefixPool::new(2, 1);
        pool.donate(0, &batch(0..2));
        let _ = pool.assemble().await;
        assert_eq!(pool.donate(0, &batch(50..52)), 0);
    }

    /// A join restates the target in left rows once the right relation is
    /// materialized, and the split follows.
    #[tokio::test]
    async fn retarget_restates_the_target_in_left_rows() {
        let pool = PrefixPool::new(640, 4); // plan-time: pairs
        pool.retarget(23);
        assert_eq!(pool.quota_rows(), 6); // ceil(23/4)
        assert_eq!(pool.target_rows(), 23);
        assert_eq!(pool.donate(0, &batch(0..20)), 6);
    }

    /// A partition that never donates and never seals must not hang the query:
    /// the deadline seals the pool with what arrived.
    #[tokio::test]
    async fn deadline_seals_a_stalled_pool() {
        let pool = PrefixPool::with_deadline(8, 2, Duration::from_millis(50));
        pool.donate(0, &batch(0..4));
        // Partition 1 never donates and never seals.
        let assembled = pool.assemble().await;
        assert_eq!(values(&assembled), vec![0, 1, 2, 3]);
    }
}
