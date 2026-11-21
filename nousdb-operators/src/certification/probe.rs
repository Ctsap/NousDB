//! Probe plans for the calibration prologue.
//!
//! The prologue does not dispatch LLM requests itself: it builds a designed
//! batch of probe items, wraps it in an in-memory source, and asks the operator
//! to build a probe exec over it at a fixed batch size. Executing that plan
//! reuses the operator's own coalescer, request window, admission gating,
//! recovery and metrics.
//!
//! # The sampling design
//!
//! The probe set is fixed first (`n` items, one per stratum of the sample) and
//! everything else is drawn around it:
//!
//! - **Fillers come from the whole sample, minus the probes.** Probe `j`'s
//!   batch is its own item plus `k-1` others drawn uniformly without
//!   replacement from the shared pool, redrawn independently for every probe
//!   and every rung. Two batches may contain the same filler row. That is not
//!   a dependence: `Z_j` is a function of `(F_j, slot_j)` alone and those are
//!   drawn independently per probe, so the `n` indicators stay mutually
//!   independent and the agreement count stays `Binomial(n, φ(k))`.
//! - **The sample requirement is `n + k_max - 1` items.**
//! - **Every draw happens before the first model call.** `designate` fully
//!   materializes each rung's index vector, so nothing about the design can
//!   depend on an observation.
//! - **Packing is the coalescer's.** A rung-`k` probe input holds exactly
//!   `n · k` items and the exec runs with `optimal_batch_size = k`, so
//!   `enqueue_request_slices` cuts it into `n` requests of `k` on exact
//!   boundaries. Probe `j` lands at index `j·k + slot_j` at every rung.
//!
//! # What this measures
//!
//! `φ(k)` is the probability that a uniformly drawn row's answer, placed at a
//! uniformly drawn slot among `k-1` rows drawn uniformly from the relation,
//! equals its solo answer. It describes batching, not the scan order a
//! particular table happens to have: production packs contiguous runs, and the
//! design deliberately does not.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, UInt32Array};
use datafusion::arrow::compute::take_record_batch;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::ExecutionPlan;

use nousdb_core::certification::cache::PackingPolicy;
use nousdb_core::certification::{CertificationSpec, SplitMix64};
use nousdb_core::providers::metrics::Rejection;

/// Row-identity column appended to every probe input. It never reaches the
/// prompt: selection and extend project by their explicit `input_columns`
/// list, and the join renderer reads the pair batch positionally and stops at
/// the declared left+right split, so probe requests stay byte-identical to
/// production ones.
pub const PROBE_ID_COLUMN: &str = "__nousdb_probe_id";

/// How one probe request ended, as the calibration prologue must see it.
///
/// Both failure variants score as disagreement: the streams' recovery paths
/// (selection passing rows through, extend NULL-filling, join emitting no
/// matches) must never read as agreement, so a dead endpoint cannot mint a
/// certificate. They are told apart because an unusable answer is evidence
/// about the model at this batch size, while a missing answer is evidence about
/// the endpoint and must not decide what the session caches
/// (`nousdb_core::certification::Certificate::is_cacheable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestOutcome {
    /// The model answered and the operator used the answer.
    Answered,
    /// The model answered, but unusably: truncated, undecodable, the wrong
    /// number of items, or a value the operator cannot coerce. A property of
    /// the model under this packing, which calibration is meant to see.
    Unusable,
    /// No answer at all: a transport failure ([`Rejection::Transport`]), a
    /// provider error (`predict` returned `Err`), or a probe arm that faulted
    /// before reporting.
    NoAnswer,
}

impl RequestOutcome {
    /// Classify a completed request by the rejection its response carries.
    pub fn from_rejection(rejection: Option<&Rejection>) -> Self {
        match rejection {
            None => Self::Answered,
            Some(Rejection::Transport { .. }) => Self::NoAnswer,
            Some(_) => Self::Unusable,
        }
    }

    /// Any outcome other than a usable answer; scored as a disagreement.
    pub fn failed(self) -> bool {
        self != Self::Answered
    }
}

/// Reported once per completed request, in submission order:
/// `(request ordinal, how the request ended)`.
///
/// Production leaves this unset. The prologue installs one because it must
/// score transport errors, `row_count_mismatch` and wrong-length outputs as
/// disagreement: the streams' recovery paths (selection passing rows through
/// unfiltered, extend NULL-filling) would otherwise masquerade as agreement.
///
/// Installing a sink also makes the stream request-error tolerant: a failed
/// request is reported and replaced by an empty batch instead of ending the
/// stream, so one bad probe costs one disagreement rather than the whole rung.
#[derive(Clone)]
pub struct RequestOutcomeSink(Arc<dyn Fn(usize, RequestOutcome) + Send + Sync>);

impl RequestOutcomeSink {
    /// Wraps `f` as a sink.
    pub fn new(f: impl Fn(usize, RequestOutcome) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Record the outcome of request `ordinal` (0-based, submission order).
    pub fn report(&self, ordinal: usize, outcome: RequestOutcome) {
        (self.0)(ordinal, outcome)
    }
}

impl std::fmt::Debug for RequestOutcomeSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RequestOutcomeSink")
    }
}

/// Builds a single-purpose copy of a semantic operator over `input`, fixed at
/// `k` items per request with a request window of `window`, certification off,
/// and `sink` installed. Constructed by each exec in `execute()`.
pub type ProbeExecFactory = Arc<
    dyn Fn(
            Arc<dyn ExecutionPlan>,
            usize,
            usize,
            RequestOutcomeSink,
        ) -> Result<Arc<dyn ExecutionPlan>>
        + Send
        + Sync,
>;

/// Materializes a designed list of item ids into the batch a probe exec reads.
///
/// The item id space is the operator's: a row index into the pooled prefix for
/// selection and extend, a `(left, right)` pair id for the join. The closure
/// lets one runner serve both without the join materializing its whole cross
/// product.
pub type ProbeInputBuilder = Arc<dyn Fn(&[usize]) -> Result<RecordBatch> + Send + Sync>;

/// The space the design draws items from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemSpace {
    /// A flat sequence of rows (selection and extend, whose item is a row).
    Rows {
        /// Number of rows.
        rows: usize,
    },
    /// A `left × right` grid; item id `l * right + r` is the pair `(l, r)`.
    /// The join's item is a pair, and a request is a cross-product tile.
    Grid {
        /// Number of left rows.
        left: usize,
        /// Number of right rows.
        right: usize,
    },
}

impl ItemSpace {
    /// Items available to the design.
    pub fn len(&self) -> usize {
        match *self {
            ItemSpace::Rows { rows } => rows,
            ItemSpace::Grid { left, right } => left.saturating_mul(right),
        }
    }

    /// Whether the space holds no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn grid(&self) -> (usize, usize) {
        match *self {
            ItemSpace::Grid { left, right } => (left, right),
            ItemSpace::Rows { rows } => (rows, 1),
        }
    }
}

/// What one calibration arm observed for one probe.
///
/// This is an operator output, not a raw `ModelOutput`: agreement is decided on
/// the rows the operator produces, so per-operator coercion comes from the
/// operator itself.
#[derive(Debug, Clone)]
pub struct ProbeObservation {
    /// Output batch of the request that carried this probe.
    pub batch: RecordBatch,
    /// Row index of the probe inside `batch`. `None` means the operator did
    /// not emit it, which for selection is the predicate's answer (false) and
    /// for extend means rows were dropped.
    pub row: Option<usize>,
    /// How the request that carried this probe ended. Anything but
    /// [`RequestOutcome::Answered`] is scored as disagreement, never as
    /// agreement.
    pub outcome: RequestOutcome,
}

impl ProbeObservation {
    /// The request failed (no answer, or an unusable one).
    pub fn failed(&self) -> bool {
        self.outcome.failed()
    }
}

/// Operator-specific agreement between a batched observation and its solo
/// reference. The runner has already excluded failed requests, so a scorer
/// only decides whether the two outputs match.
pub type ProbeScorer = Arc<dyn Fn(&ProbeObservation, &ProbeObservation) -> bool + Send + Sync>;

/// The geometry a request of `k` items has for this operator.
///
/// A wrong shape would calibrate a batch distribution the operator never
/// dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchShape {
    /// An unordered set of `k` items (selection and extend).
    Flat,
    /// A cross-product tile: `k = height · width` where `width` is the tile
    /// width the join plans, so a rung-`k` batch is `k/width` left rows crossed
    /// with `width` right rows (or one left row and `k` right values when `k`
    /// is narrower than a tile). These are the geometries
    /// `decompose_certified_pairs` hands production.
    Tile { right_block: usize },
}

impl BatchShape {
    /// The `(height, width)` a nominal `k` realizes in `space`; the rung
    /// measures `height · width` items. A `k` that is not a whole number of tile
    /// rows rounds down to one, and no side may ask for more distinct items
    /// than it has: `flat_pool` bounds a flat batch (its probe plus every
    /// non-probe item), the grid bounds a tiled one.
    fn geometry(&self, k: usize, space: &ItemSpace, flat_pool: usize) -> (usize, usize) {
        match *self {
            BatchShape::Flat => (1, k.clamp(1, flat_pool + 1)),
            BatchShape::Tile { right_block } => {
                let (left, right) = space.grid();
                let width = right_block.clamp(1, right.max(1));
                if k >= width {
                    ((k / width).clamp(1, left.max(1)), width)
                } else {
                    (1, k.clamp(1, right.max(1)))
                }
            }
        }
    }
}

/// One rung, fully drawn. Both vectors are materialized during [`designate`],
/// before the first model call.
#[derive(Debug, Clone)]
struct RungDesign {
    k: usize,
    indices: Vec<usize>,
    probe_ids: Vec<u32>,
}

/// A rung's probe input, ready to hand to a [`ProbeInputBuilder`].
#[derive(Debug, Clone)]
pub struct Rung {
    /// Items per request, the batch size the probe exec runs at, and the size
    /// the certificate records. Under `Tile` this can be below the nominal
    /// ladder entry; it is always the size that will actually execute.
    pub k: usize,
    /// Item ids, `n · k` of them, probe `j`'s batch occupying `[j·k, (j+1)·k)`.
    pub indices: Vec<usize>,
    /// Item id probe `j` carries in the built input.
    pub probe_ids: Vec<u32>,
}

/// Probe items, batch slots and every filler, all drawn before the first model
/// call.
#[derive(Debug, Clone)]
pub struct Design {
    /// Item id of each probe. Distinct, one per stratum of the space.
    pub probe_item: Vec<usize>,
    rungs: Vec<RungDesign>,
}

/// Persistent urn for drawing distinct items with a fixed RNG budget.
///
/// Partial Fisher–Yates over a permutation that is *kept* between draws: `m`
/// draws cost exactly `m` `uniform` calls (no rejection loop, so the RNG trace
/// stays a function of the ladder alone), and leaving the permutation shuffled
/// keeps every subsequent draw uniform.
struct Urn {
    items: Vec<usize>,
    /// Position of each value, maintained only when hold-outs are needed.
    at: Vec<usize>,
}

impl Urn {
    /// Urn over `0..size`, supporting a per-draw hold-out.
    fn range(size: usize) -> Self {
        Self {
            items: (0..size).collect(),
            at: (0..size).collect(),
        }
    }

    /// Urn over an explicit value list. Does not support hold-outs.
    fn values(items: Vec<usize>) -> Self {
        Self {
            items,
            at: Vec::new(),
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.items.swap(i, j);
        if !self.at.is_empty() {
            self.at[self.items[i]] = i;
            self.at[self.items[j]] = j;
        }
    }

    /// Replace `out` with `m` distinct values, none equal to `hold_out`.
    fn draw(
        &mut self,
        m: usize,
        hold_out: Option<usize>,
        rng: &mut SplitMix64,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let mut avail = self.items.len();
        if let Some(v) = hold_out {
            debug_assert!(!self.at.is_empty(), "hold-out needs a positional urn");
            avail -= 1;
            let i = self.at[v];
            self.swap(i, avail);
        }
        for i in 0..m.min(avail) {
            let p = i + rng.uniform(avail - i);
            self.swap(i, p);
            out.push(self.items[i]);
        }
    }
}

/// The item at position `i` of a run that carries `own` at position `at` and
/// the `drawn` fillers everywhere else.
fn splice(i: usize, at: usize, own: usize, drawn: &[usize]) -> usize {
    if i == at {
        own
    } else {
        drawn[if i < at { i } else { i - 1 }]
    }
}

/// The urns a rung draws its fillers from, plus their scratch buffers.
///
/// A flat batch draws from a single shared urn of every non-probe item. A tile
/// must stay a cross product, so it draws per side and crosses the two,
/// holding out the probe's own left row and right row so the tile cannot
/// contain it twice.
enum Fillers {
    Flat {
        pool: Urn,
        drawn: Vec<usize>,
    },
    Tile {
        left: Urn,
        right: Urn,
        lefts: Vec<usize>,
        rights: Vec<usize>,
        /// Row stride of the grid: item id `p` is the pair `(p / stride, p % stride)`.
        stride: usize,
    },
}

impl Fillers {
    fn new(shape: BatchShape, space: &ItemSpace, probes: &[usize]) -> Self {
        match shape {
            BatchShape::Flat => {
                let total = space.len();
                let mut is_probe = vec![false; total];
                for &p in probes {
                    is_probe[p] = true;
                }
                Fillers::Flat {
                    pool: Urn::values((0..total).filter(|i| !is_probe[*i]).collect()),
                    drawn: Vec::new(),
                }
            }
            BatchShape::Tile { .. } => {
                let (left, right) = space.grid();
                Fillers::Tile {
                    left: Urn::range(left),
                    right: Urn::range(right),
                    lefts: Vec::new(),
                    rights: Vec::new(),
                    stride: right,
                }
            }
        }
    }

    /// Append one probe's `height · width` item ids to `out`, with the probe on
    /// a uniformly drawn slot. Returns that slot.
    fn draw_batch(
        &mut self,
        probe: usize,
        (height, width): (usize, usize),
        rng: &mut SplitMix64,
        out: &mut Vec<usize>,
    ) -> usize {
        match self {
            Fillers::Flat { pool, drawn } => {
                let slot = rng.uniform(width);
                pool.draw(width - 1, None, rng, drawn);
                out.extend((0..width).map(|s| splice(s, slot, probe, drawn)));
                slot
            }
            Fillers::Tile {
                left,
                right,
                lefts,
                rights,
                stride,
            } => {
                let (l, c) = (probe / *stride, probe % *stride);
                let row = rng.uniform(height);
                let col = rng.uniform(width);
                left.draw(height - 1, Some(l), rng, lefts);
                right.draw(width - 1, Some(c), rng, rights);
                // Emitted left-major, exactly as production enumerates a left
                // block crossed with a right tile.
                for i in 0..height {
                    let li = splice(i, row, l, lefts);
                    out.extend((0..width).map(|x| li * *stride + splice(x, col, c, rights)));
                }
                row * width + col
            }
        }
    }
}

/// Shape for `spec` in `space`, falling back to `Flat` when the declared tile
/// width cannot describe the space.
fn batch_shape(spec: &CertificationSpec, space: &ItemSpace) -> BatchShape {
    match spec.cache_key.packing_policy {
        PackingPolicy::Flat => BatchShape::Flat,
        PackingPolicy::JoinTile { right_block } => {
            if matches!(space, ItemSpace::Grid { .. }) && right_block >= 1 {
                BatchShape::Tile { right_block }
            } else {
                log::warn!(
                    "[certify] tile width {right_block} cannot describe {space:?}; \
                     falling back to flat rung batches"
                );
                BatchShape::Flat
            }
        }
    }
}

/// Draw the design for `spec` over `space`. Deterministic in `spec.seed`.
///
/// Draw order (probe items, then per rung one slot and its fillers per probe,
/// in rung-major order) is a function of `(seed, n, ladder, space)` alone.
/// Rungs whose realized geometry collapses onto the previous one still consume
/// their draws before being dropped, so the trace never depends on which rungs
/// survive.
pub fn designate(spec: &CertificationSpec, space: &ItemSpace) -> Design {
    let total = space.len();
    let n = spec.n.min(total);
    let shape = batch_shape(spec, space);
    let mut rng = SplitMix64::new(spec.seed);

    // Probe items: one per stratum, so they are distinct and spread over the
    // whole sample; the last stratum absorbs the remainder.
    let stride = total.checked_div(n).unwrap_or(1).max(1);
    let probe_item: Vec<usize> = (0..n)
        .map(|j| {
            let start = j * stride;
            let width = if j + 1 == n { total - start } else { stride };
            start + rng.uniform(width.max(1))
        })
        .collect();

    // Fillers are everything the probes are not: `total - n` items, since the
    // probes are distinct.
    let mut fillers = Fillers::new(shape, space, &probe_item);
    let flat_pool = total - n;

    let mut rungs: Vec<RungDesign> = Vec::with_capacity(spec.ladder.len());
    for &k in &spec.ladder {
        let (height, width) = shape.geometry(k, space, flat_pool);
        let realized = height * width;
        let collapses = rungs.last().is_some_and(|prev| prev.k == realized);

        let mut indices = Vec::with_capacity(n * realized);
        let mut probe_ids = Vec::with_capacity(n);
        for (j, &probe) in probe_item.iter().enumerate() {
            let slot = fillers.draw_batch(probe, (height, width), &mut rng, &mut indices);
            probe_ids.push((j * realized + slot) as u32);
        }

        // Drawn first, dropped second: a collapsed rung still spends its draws,
        // so the RNG trace never depends on which rungs survive.
        if collapses {
            log::debug!("[certify] ladder rung k={k} collapses onto k={realized}; skipping");
            continue;
        }
        rungs.push(RungDesign {
            k: realized,
            indices,
            probe_ids,
        });
    }

    Design { probe_item, rungs }
}

impl Design {
    /// Item ids for the reference arm: the `n` probes, in order. Run at
    /// `k = 1`, so probe `j` is the whole of request `j`.
    pub fn reference_indices(&self) -> Vec<usize> {
        self.probe_item.clone()
    }

    /// Rungs to try, descending. May be shorter than `spec.ladder` when two
    /// entries share a realized geometry.
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Items per request at rung `rung_idx`.
    pub fn rung_size(&self, rung_idx: usize) -> usize {
        self.rungs[rung_idx].k
    }

    /// Everything one calibration arm needs for rung `rung_idx`.
    pub fn rung(&self, rung_idx: usize) -> Rung {
        let rung = &self.rungs[rung_idx];
        Rung {
            k: rung.k,
            indices: rung.indices.clone(),
            probe_ids: rung.probe_ids.clone(),
        }
    }
}

/// Input schema plus the row-identity column.
fn probe_input_schema(input: &SchemaRef) -> SchemaRef {
    let mut fields = input.fields().to_vec();
    fields.push(Arc::new(Field::new(
        PROBE_ID_COLUMN,
        DataType::UInt32,
        false,
    )));
    Arc::new(Schema::new_with_metadata(fields, input.metadata().clone()))
}

/// Append [`PROBE_ID_COLUMN`], numbered by position. `batch` must already hold
/// the designed items in design order.
pub fn attach_probe_ids(batch: RecordBatch) -> Result<RecordBatch> {
    let ids: ArrayRef = Arc::new(UInt32Array::from_iter_values(0..batch.num_rows() as u32));
    let schema = probe_input_schema(&batch.schema());
    let mut columns = batch.columns().to_vec();
    columns.push(ids);
    RecordBatch::try_new(schema, columns).map_err(|e| {
        DataFusionError::Execution(format!("calibration: failed to build probe input: {e}"))
    })
}

/// Gather `indices` out of `prefix` and append [`PROBE_ID_COLUMN`], for the
/// row-index item space (selection and extend).
pub fn build_probe_input(prefix: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    let positions = UInt32Array::from_iter_values(indices.iter().map(|&i| i as u32));
    let taken = take_record_batch(prefix, &positions).map_err(|e| {
        DataFusionError::Execution(format!("calibration: failed to gather probe rows: {e}"))
    })?;
    attach_probe_ids(taken)
}

/// Locate a probe row by its id in an operator's output batch.
///
/// Returns `None` when the row is absent: for selection that is the predicate's
/// answer (the row was filtered out), and for extend it means the operator
/// dropped or reordered rows.
pub fn find_probe_row(batch: &RecordBatch, probe_id: u32) -> Option<usize> {
    let ids = batch
        .column_by_name(PROBE_ID_COLUMN)?
        .as_any()
        .downcast_ref::<UInt32Array>()?;
    (0..ids.len()).find(|&i| !ids.is_null(i) && ids.value(i) == probe_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use nousdb_core::certification::cache::{CertKey, OperatorKind, PackingPolicy};

    fn spec_with(ladder: Vec<usize>, n: usize, packing_policy: PackingPolicy) -> CertificationSpec {
        CertificationSpec {
            ladder,
            eta: 0.10,
            delta_prime: 1.0 / 30.0,
            n,
            s_star: n,
            seed: 42,
            cache_key: CertKey {
                provider: "mock".into(),
                model_id: "m".into(),
                operator_kind: OperatorKind::Selection,
                template_hash: 1,
                packing_policy,
            },
            delta_share: 0.10,
            tolerated: None,
            max_probes: nousdb_core::certification::DEFAULT_MAX_PROBES,
        }
    }

    fn spec(ladder: Vec<usize>, n: usize) -> CertificationSpec {
        spec_with(ladder, n, PackingPolicy::Flat)
    }

    fn join_spec(ladder: Vec<usize>, n: usize, right_block: usize) -> CertificationSpec {
        spec_with(ladder, n, PackingPolicy::JoinTile { right_block })
    }

    fn rows(n: usize) -> ItemSpace {
        ItemSpace::Rows { rows: n }
    }

    fn prefix(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));
        let vals = Int64Array::from_iter_values(0..rows as i64);
        RecordBatch::try_new(schema, vec![Arc::new(vals)]).unwrap()
    }

    /// Same spec, same space => same design, down to the filler draws.
    #[test]
    fn designation_is_seed_deterministic() {
        let s = spec(vec![16, 8, 4], 40);
        let space = rows(400);
        let a = designate(&s, &space);
        let b = designate(&s, &space);
        assert_eq!(a.probe_item, b.probe_item);
        assert_eq!(a.rung_count(), 3);
        for (rung_idx, &k) in s.ladder.iter().enumerate() {
            let (x, y) = (a.rung(rung_idx), b.rung(rung_idx));
            assert_eq!(x.k, y.k);
            assert_eq!(x.indices, y.indices);
            assert_eq!(x.probe_ids, y.probe_ids);
            assert_eq!(x.k, k, "flat rungs are the ladder");
        }
    }

    /// Probes are one per stratum: distinct, and spread over the whole sample
    /// rather than clustered.
    #[test]
    fn probes_are_stratified_and_distinct() {
        let s = spec(vec![16, 8, 4], 40);
        let design = designate(&s, &rows(400));
        let stride = 400 / 40;
        for (j, &p) in design.probe_item.iter().enumerate() {
            assert!(
                p >= j * stride && p < (j + 1) * stride,
                "probe {j} at {p} left stratum {j}"
            );
        }
        let mut sorted = design.probe_item.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 40, "probes must be distinct");
    }

    /// The realized slot equals the designated slot at EVERY rung, under both
    /// shapes. This is what makes `probe_ids[j] = j*k + slot_j` the right
    /// lookup key for the batch the coalescer cuts.
    #[test]
    fn probe_lands_on_its_designated_slot_at_every_rung() {
        for (s, space) in [
            (spec(vec![16, 8, 4, 2], 40), rows(400)),
            (
                join_spec(vec![16, 8, 4, 2], 40, 4),
                ItemSpace::Grid {
                    left: 40,
                    right: 20,
                },
            ),
        ] {
            let design = designate(&s, &space);
            for rung_idx in 0..design.rung_count() {
                let rung = design.rung(rung_idx);
                let k = rung.k;
                assert_eq!(rung.indices.len(), 40 * k);
                for j in 0..40 {
                    let slot = rung.probe_ids[j] as usize - j * k;
                    assert!(slot < k);
                    assert_eq!(
                        rung.indices[j * k + slot],
                        design.probe_item[j],
                        "rung k={k}, probe {j} must sit on slot {slot}"
                    );
                }
            }
        }
    }

    /// A batch never repeats an item and never contains a probe other than its
    /// own. Fillers may be shared between batches, which is not a dependence.
    #[test]
    fn fillers_are_distinct_and_exclude_every_probe() {
        let s = spec(vec![16, 8, 4, 2], 12);
        let design = designate(&s, &rows(120));
        let probes: std::collections::HashSet<usize> = design.probe_item.iter().copied().collect();
        for rung_idx in 0..design.rung_count() {
            let rung = design.rung(rung_idx);
            let k = rung.k;
            for j in 0..12 {
                let batch = &rung.indices[j * k..(j + 1) * k];
                let mut sorted = batch.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), k, "rung k={k}, batch {j} repeated an item");
                let slot = rung.probe_ids[j] as usize - j * k;
                for (s_idx, item) in batch.iter().enumerate() {
                    if s_idx == slot {
                        continue;
                    }
                    assert!(
                        !probes.contains(item),
                        "rung k={k}, batch {j}: filler {item} is a probe"
                    );
                }
            }
        }
    }

    /// Fillers really are drawn from the whole pool, not a neighbourhood: over
    /// 12 probes at k=16 the union of the fillers must cover far more of the
    /// sample than any per-probe window could.
    #[test]
    fn fillers_span_the_whole_pool() {
        let s = spec(vec![16], 12);
        let design = designate(&s, &rows(400));
        let touched: std::collections::HashSet<usize> =
            design.rung(0).indices.iter().copied().collect();
        assert!(
            touched.len() > 100,
            "only {} distinct items touched; fillers look localized",
            touched.len()
        );
    }

    /// `n + k - 1` items suffice to seat the top rung.
    #[test]
    fn a_minimal_sample_still_seats_the_top_rung() {
        let (n, k) = (30usize, 16usize);
        let s = spec(vec![16, 8], n);
        let design = designate(&s, &rows(n + k - 1));
        let rung = design.rung(0);
        assert_eq!(rung.k, k);
        assert_eq!(rung.indices.len(), n * k);
        for j in 0..n {
            let batch = &rung.indices[j * k..(j + 1) * k];
            let mut sorted = batch.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), k);
        }
    }

    /// Join rungs must be shapes the join stream actually emits: `k/R` left
    /// rows crossed with the full tile, left-major.
    #[test]
    fn join_rungs_are_whole_tile_rows() {
        const R: usize = 4;
        let space = ItemSpace::Grid {
            left: 40,
            right: 20,
        };
        let s = join_spec(vec![16, 8, 4], 12, R);
        let design = designate(&s, &space);
        for rung_idx in 0..design.rung_count() {
            let rung = design.rung(rung_idx);
            let k = rung.k;
            assert_eq!(k % R, 0, "rung {k} is not a whole number of tile rows");
            for j in 0..12 {
                let batch = &rung.indices[j * k..(j + 1) * k];
                // Read back as (left, right): every tile row must repeat the
                // same right selection, and left rows must be distinct.
                let want_rights: Vec<usize> = batch[..R].iter().map(|p| p % 20).collect();
                let mut lefts = Vec::new();
                for row in batch.chunks(R) {
                    let l: Vec<usize> = row.iter().map(|p| p / 20).collect();
                    assert!(l.windows(2).all(|w| w[0] == w[1]), "tile row spans lefts");
                    let rights: Vec<usize> = row.iter().map(|p| p % 20).collect();
                    assert_eq!(rights, want_rights, "tile rows disagree on the right set");
                    lefts.push(l[0]);
                }
                let mut sorted = lefts.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), lefts.len(), "left rows repeated in a tile");
            }
        }
    }

    /// Below the tile width the geometry is (1, k): one left row, k distinct
    /// right values.
    #[test]
    fn join_rungs_below_the_tile_stay_in_one_row() {
        const R: usize = 4;
        let space = ItemSpace::Grid {
            left: 40,
            right: 20,
        };
        let s = join_spec(vec![16, 8, 4, 2, 1], 12, R);
        let design = designate(&s, &space);
        let rung = (0..design.rung_count())
            .map(|i| design.rung(i))
            .find(|r| r.k == 2)
            .expect("k=2 rung");
        for j in 0..12 {
            let batch = &rung.indices[j * 2..(j + 1) * 2];
            let lefts: Vec<usize> = batch.iter().map(|p| p / 20).collect();
            assert_eq!(lefts[0], lefts[1], "batch {j} straddles two left rows");
            assert_ne!(batch[0] % 20, batch[1] % 20, "batch {j} repeated a right");
        }
    }

    /// A nominal rung that is not a whole number of tile rows is measured at
    /// the geometry production would run it at.
    #[test]
    fn join_rungs_round_down_to_whole_tile_rows() {
        // Geometry (4, 5) => k*_alpha = 20, ladder {20, 16, 8, 4, 2, 1}.
        // 16 pairs is three whole tile rows plus one, so it is measured at 15.
        let space = ItemSpace::Grid {
            left: 40,
            right: 20,
        };
        let design = designate(&join_spec(vec![20, 16, 8, 4, 2, 1], 8, 5), &space);
        let sizes: Vec<usize> = (0..design.rung_count()).map(|i| design.rung(i).k).collect();
        assert_eq!(sizes, vec![20, 15, 5, 4, 2, 1]);
        for rung_idx in 0..design.rung_count() {
            let rung = design.rung(rung_idx);
            assert_eq!(rung.indices.len(), 8 * rung.k);
        }
    }

    /// A tile policy over a flat space cannot describe a tile; fall back rather
    /// than produce nonsense geometry.
    #[test]
    fn a_tile_policy_over_a_flat_space_falls_back() {
        let ladder = vec![16, 8, 4];
        let space = rows(400);
        let fallback = designate(&join_spec(ladder.clone(), 6, 5), &space);
        let flat = designate(&spec(ladder, 6), &space);
        for rung_idx in 0..flat.rung_count() {
            assert_eq!(fallback.rung(rung_idx).indices, flat.rung(rung_idx).indices);
        }
    }

    #[test]
    fn probe_input_carries_row_ids_and_gathered_values() {
        let s = spec(vec![16, 8], 5);
        let design = designate(&s, &rows(80));
        let batch = prefix(80);

        let reference = build_probe_input(&batch, &design.reference_indices()).unwrap();
        assert_eq!(reference.num_rows(), 5);
        assert_eq!(design.probe_item.len(), 5);
        for (j, &item) in design.probe_item.iter().enumerate() {
            let row = find_probe_row(&reference, j as u32).unwrap();
            assert_eq!(row, j);
            let vals = reference
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(vals.value(row), item as i64);
        }

        let r1 = design.rung(1);
        let rung = build_probe_input(&batch, &r1.indices).unwrap();
        assert_eq!(rung.num_rows(), 40);
        for j in 0..5 {
            let row = find_probe_row(&rung, r1.probe_ids[j]).unwrap();
            let vals = rung
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(vals.value(row), design.probe_item[j] as i64);
        }
    }
}
