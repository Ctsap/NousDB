//! Batch sizing: how many rows or pairs fit in one request.

use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;

use crate::certification::cache::OperatorKind;
use crate::prompt::reasoning::ReasoningMode;
use crate::providers::metadata::ModelCard;
use crate::token_estimator::estimator::TokenEstimator;
use crate::token_estimator::histogram::TokenHistogram;

/// Configuration for context-window-safe batch sizing.
#[derive(Debug, Clone)]
pub struct SafeBatchSizingConfig {
    /// Context-window overflow probability budget for a batch. Unrelated to the
    /// certification risk budget `delta`.
    pub alpha: f64,
    /// Number of buckets in the token histograms.
    pub bucket_count: usize,
    /// Maximum number of fixed-point iterations of the size search.
    pub max_iterations: usize,
    /// Smallest batch size ever returned.
    pub min_batch_size: usize,
}

impl Default for SafeBatchSizingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            bucket_count: 32,
            max_iterations: 12,
            min_batch_size: 1,
        }
    }
}

/// Which constraint decided a batch size. Diagnostic only, but it separates
/// "the rows are huge", "the card says no" and "the model gets confused past
/// this size", which have different remedies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingBound {
    /// The context-window token budget decided the size.
    TokenBudget,
    /// [`ModelCard::max_batch_size`], the capacity/protocol ceiling.
    ModelCapacity,
    /// [`ModelCard::quality_max_batch_size`], the accuracy-derived ceiling.
    QualityCap,
    /// The output-token budget bound the count (join: pairs times output per
    /// pair).
    OutputBudget,
    /// Nothing fits, so the configured minimum batch size was used.
    MinFloor,
}

impl std::fmt::Display for SizingBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SizingBound::TokenBudget => "token_budget",
            SizingBound::ModelCapacity => "model_capacity",
            SizingBound::QualityCap => "quality_cap",
            SizingBound::OutputBudget => "output_budget",
            SizingBound::MinFloor => "min_floor",
        };
        f.write_str(s)
    }
}

/// Token budget for flat (extend/selection) batches.
#[derive(Debug, Clone)]
pub struct BatchBudgetModel {
    /// Effective input token budget ([`ModelCard::effective_input_budget`]).
    pub context_window_tokens: usize,
    /// Static prompt tokens, constant across requests.
    pub fixed_overhead_tokens: usize,
    /// Per-row delimiter and framing tokens in the prompt.
    pub per_row_delimiter_tokens: usize,
    /// Expected output tokens per row.
    pub expected_output_tokens_per_row: usize,
    /// The model card's own ceiling on rows per request
    /// ([`ModelCard::max_batch_size`]).
    ///
    /// `None` when no card could be resolved, in which case sizing is bounded
    /// only by the token budget.
    pub model_max_batch_size: Option<usize>,
    /// The card's accuracy-derived ceiling
    /// ([`ModelCard::quality_max_batch_size`]): the size beyond which per-row
    /// accuracy degrades even if the tokens would fit. Applied as `min` with
    /// `model_max_batch_size`.
    pub quality_max_batch_size: Option<usize>,
    /// [`ModelCard::max_output_tokens`]: the generation ceiling the response
    /// has to fit inside.
    ///
    /// Under the JSON grammar a truncated response is an unterminated array,
    /// so the whole request is lost rather than partially recovered. `None`
    /// means no card was resolved and no output bound is applied.
    pub max_output_tokens: Option<usize>,
    /// Extra generated tokens per row beyond `expected_output_tokens_per_row`
    /// (a per-row rationale, say). Zero when the reasoning mode adds none.
    ///
    /// Charged only against the output cap, never against the input-side
    /// denominator: `context_window_tokens` is already
    /// [`ModelCard::effective_input_budget`], which has subtracted
    /// `max_output_tokens` once, so adding a reasoning term to the input
    /// denominator would bill the same tokens again.
    pub reasoning_output_tokens_per_row: usize,
}

/// JSON framing the response carries regardless of item count (the array
/// brackets and separators). Subtracted from the output budget before it is
/// divided into per-item costs, on both the flat and the join path.
const OUTPUT_FRAMING_TOKENS: usize = 20;

/// Result of [`estimate_safe_batch_size`].
#[derive(Debug, Clone)]
pub struct SafeBatchEstimate {
    /// Batch size at the start of the search.
    pub initial_batch_size: usize,
    /// The chosen batch size.
    pub final_batch_size: usize,
    /// Iterations the search ran.
    pub iterations: usize,
    /// Token quantile used for the final size.
    pub final_quantile: f64,
    /// Probability level of that quantile (`1 - alpha / k`).
    pub final_quantile_probability: f64,
    /// True when a card-derived cap (capacity, quality or the output budget,
    /// not the input token budget) decided `final_batch_size`.
    pub capped_by_model: bool,
    /// Which constraint decided `final_batch_size` (finer-grained than
    /// `capped_by_model`).
    pub bound: SizingBound,
}

/// Finds the largest batch size whose rows fit the token budget with
/// probability at least `1 - alpha`, subject to the model card's caps.
///
/// The size `k` is a fixed point of `k = available / (Q(1 - alpha/k) + per-row
/// costs)`, where `Q` is the histogram's token quantile function; it is found
/// by iteration.
pub fn estimate_safe_batch_size(
    histogram: &TokenHistogram,
    budget: &BatchBudgetModel,
    config: &SafeBatchSizingConfig,
) -> SafeBatchEstimate {
    let min_batch = config.min_batch_size.max(1);
    // Rows whose answers still fit the generation ceiling. The reasoning term
    // is charged here and nowhere else (see `BatchBudgetModel`).
    let output_row_cap = budget.max_output_tokens.and_then(|max_output| {
        let per_row =
            budget.expected_output_tokens_per_row + budget.reasoning_output_tokens_per_row;
        (per_row > 0).then(|| max_output.saturating_sub(OUTPUT_FRAMING_TOKENS) / per_row)
    });
    // The tightest of the card's three ceilings. Later entries win ties, so a
    // tie reports the stronger claim: quality ("the model degrades here") over
    // capacity ("the card says no") over the output budget. The join sizing
    // uses the same order.
    let mut card_cap: Option<usize> = None;
    let mut cap_bound = SizingBound::TokenBudget;
    for (cap, bound) in [
        (output_row_cap, SizingBound::OutputBudget),
        (budget.model_max_batch_size, SizingBound::ModelCapacity),
        (budget.quality_max_batch_size, SizingBound::QualityCap),
    ] {
        if let Some(c) = cap
            && card_cap.is_none_or(|current| c <= current)
        {
            card_cap = Some(c);
            cap_bound = bound;
        }
    }
    // `.max(min_batch)` is required: `usize::clamp` panics when min > max, and
    // a card whose `max_batch_size` is below the configured floor is possible.
    let max_batch = card_cap.unwrap_or(usize::MAX).max(min_batch);

    if histogram.total_rows() == 0 {
        return SafeBatchEstimate {
            initial_batch_size: min_batch,
            final_batch_size: min_batch,
            iterations: 0,
            final_quantile: 0.0,
            final_quantile_probability: 1.0,
            capped_by_model: false,
            bound: SizingBound::MinFloor,
        };
    }

    let available = budget
        .context_window_tokens
        .saturating_sub(budget.fixed_overhead_tokens);

    if available == 0 {
        return SafeBatchEstimate {
            initial_batch_size: min_batch,
            final_batch_size: min_batch,
            iterations: 0,
            final_quantile: histogram.quantile(1.0),
            final_quantile_probability: 1.0,
            capped_by_model: false,
            bound: SizingBound::MinFloor,
        };
    }

    let stats = histogram.stats();
    let mean_denominator = stats.mean
        + budget.per_row_delimiter_tokens as f64
        + budget.expected_output_tokens_per_row as f64;

    let seed = if mean_denominator > 0.0 {
        (available as f64 / mean_denominator).floor() as usize
    } else {
        min_batch
    };
    let mut k = seed.clamp(min_batch, max_batch);

    let initial_batch_size = k;
    let mut final_quantile_probability = 1.0;
    let mut final_quantile = histogram.quantile(1.0);

    // The last unclamped candidate. `capped_by_model` describes the returned
    // value, so it is read from the candidate that produced it, not from an
    // intermediate iteration.
    let mut raw_k = seed;
    let capped = |raw: usize| card_cap.is_some() && raw > max_batch;
    let bound_for = |raw: usize| {
        if capped(raw) {
            cap_bound
        } else {
            SizingBound::TokenBudget
        }
    };

    for iteration in 1..=config.max_iterations.max(1) {
        let safe_k = k.max(min_batch);
        let p = (1.0 - config.alpha / safe_k as f64).clamp(0.0, 1.0);
        let q = histogram.quantile(p);
        let denominator = q
            + budget.per_row_delimiter_tokens as f64
            + budget.expected_output_tokens_per_row as f64;

        final_quantile_probability = p;
        final_quantile = q;

        if denominator <= 0.0 {
            return SafeBatchEstimate {
                initial_batch_size,
                final_batch_size: min_batch,
                iterations: iteration,
                final_quantile,
                final_quantile_probability,
                capped_by_model: false,
                bound: SizingBound::MinFloor,
            };
        }

        raw_k = (available as f64 / denominator).floor() as usize;
        let next_k = raw_k.clamp(min_batch, max_batch);

        // Once the cap binds, `next_k` stays at `max_batch` and the search
        // converges on the following pass: the quantile is recomputed at
        // `p = 1 - alpha/max_batch`, but the clamp absorbs the difference.
        if next_k == k {
            return SafeBatchEstimate {
                initial_batch_size,
                final_batch_size: next_k,
                iterations: iteration,
                final_quantile,
                final_quantile_probability,
                capped_by_model: capped(raw_k),
                bound: bound_for(raw_k),
            };
        }

        k = next_k;
    }

    SafeBatchEstimate {
        initial_batch_size,
        final_batch_size: k.clamp(min_batch, max_batch),
        iterations: config.max_iterations.max(1),
        final_quantile,
        final_quantile_probability,
        capped_by_model: capped(raw_k),
        bound: bound_for(raw_k),
    }
}

/// Estimates the output tokens per row from the output schema.
///
/// Uses heuristic token counts per data type: booleans are cheap (about 2
/// tokens), strings are expensive (about 24), with numeric types in between,
/// plus a fixed JSON framing overhead (about 6 tokens per row).
pub fn estimate_output_tokens_per_row(
    output_schema: &[datafusion::arrow::datatypes::FieldRef],
) -> usize {
    let mut estimate = 6usize; // object/JSON framing overhead

    for field in output_schema {
        estimate += (field.name().len() / 4).max(1);
        estimate += match field.data_type() {
            DataType::Boolean => 2,
            DataType::Int32 | DataType::Int64 => 4,
            DataType::Float32 | DataType::Float64 => 6,
            DataType::Utf8 | DataType::LargeUtf8 => 24,
            _ => 12,
        };
    }

    estimate.max(20)
}

/// Output tokens a generated Utf8 field costs per row.
///
/// Twice the 24 used by [`estimate_output_tokens_per_row`]: that figure suits
/// values an extend extracts (a label, a short phrase), while a generated field
/// is free text the model composes. It errs high because an output overrun
/// truncates the JSON array and loses the whole request.
pub const GENERATE_UTF8_TOKENS_PER_FIELD: usize = 48;

/// Estimates the output tokens one generated row costs: the same structure as
/// [`estimate_output_tokens_per_row`] with Utf8 charged at
/// [`GENERATE_UTF8_TOKENS_PER_FIELD`].
pub fn estimate_generation_tokens_per_row(
    output_schema: &[datafusion::arrow::datatypes::FieldRef],
) -> usize {
    let mut estimate = 6usize; // object/JSON framing overhead

    for field in output_schema {
        estimate += (field.name().len() / 4).max(1);
        estimate += match field.data_type() {
            DataType::Boolean => 2,
            DataType::Int32 | DataType::Int64 => 4,
            DataType::Float32 | DataType::Float64 => 6,
            DataType::Utf8 | DataType::LargeUtf8 => GENERATE_UTF8_TOKENS_PER_FIELD,
            _ => 12,
        };
    }

    estimate.max(20)
}

/// Rows the generate operator asks for in one request.
///
/// A generate request has no input rows, so the output budget bounds it:
/// `(budget - framing) / tokens_per_row`, clamped by the card's capacity and
/// quality caps and by the rows still wanted. `budget` is the card's
/// `max_output_tokens`, or, when the card has none, the same 20% output reserve
/// that [`ModelCard::effective_input_budget`] assumes.
pub fn generate_chunk_size(
    card: &ModelCard,
    output_schema: &[datafusion::arrow::datatypes::FieldRef],
    num_rows: usize,
) -> usize {
    let budget = card
        .max_output_tokens
        .unwrap_or_else(|| (card.context_window * 2) / 10);
    let per_row = estimate_generation_tokens_per_row(output_schema);
    let by_budget = budget.saturating_sub(OUTPUT_FRAMING_TOKENS) / per_row;

    let cap = card
        .quality_max_batch_size
        .map_or(card.max_batch_size, |q| q.min(card.max_batch_size))
        .min(num_rows);

    by_budget.min(cap).max(1)
}

/// Budget model for the reference-table join prompt format.
///
/// In this format, left and right rows are listed once as reference tables,
/// and pairs are compact ID references. The total input cost is:
///
/// ```text
/// total = fixed_overhead
///       + L × (left_row_content + left_row_format)
///       + R × (right_row_content + right_row_format)
///       + L×R × pair_reference_tokens
/// ```
///
/// Row *content* tokens come from the token histograms passed to
/// [`estimate_join_block_sizes_v2`]; this struct carries only what the
/// histograms cannot know. Output tokens are not part of the input cost:
/// `context_window_tokens` is the effective input budget
/// ([`ModelCard::effective_input_budget`]), which already reserves the output
/// window. `output_tokens_per_pair` only bounds the pair count by the output
/// budget.
#[derive(Debug, Clone)]
pub struct JoinBatchBudgetModel {
    /// Effective input token budget (context window minus output reserve).
    pub context_window_tokens: usize,
    /// Static prompt tokens: system template + user template preamble
    /// (task rules, output-format section, headers) + join condition.
    /// Measure with `prompt::utils::join_static_prompt_tokens`.
    pub fixed_overhead_tokens: usize,
    /// Per-row markdown/label overhead on the left side: `**L{i}:**` plus
    /// `  - {col}: ` per column. Compute with [`row_format_tokens`].
    pub left_row_format_tokens: usize,
    /// Same for the right side.
    pub right_row_format_tokens: usize,
    /// Tokens per pair reference line (for example `- Pair 1: L1 <-> R2`, about
    /// 15 tokens). This is the only per-pair input cost: the output-format
    /// section is constant-size and counted in `fixed_overhead_tokens`.
    pub pair_reference_tokens: usize,
    /// Expected output tokens per pair (for example `{"join_match": true}`,
    /// about 8 tokens). Used only for the output-budget cap on pairs.
    pub output_tokens_per_pair: usize,
}

/// Per-row markdown/label overhead of the reference-table row rendering:
/// `**L{i}:**\n` (about 4 tokens) plus, per column, `  - {name}: ` and the
/// trailing newline (about 3 tokens plus the column name itself).
pub fn row_format_tokens(col_names: &[String], estimator: &TokenEstimator) -> usize {
    4 + col_names
        .iter()
        .map(|c| estimator.estimate_text_tokens(c) + 3)
        .sum::<usize>()
}

/// Result of the reference-table join block sizing.
#[derive(Debug, Clone)]
pub struct JoinBlockEstimate {
    /// Number of left rows per request.
    pub left_block: usize,
    /// Number of right rows per request.
    pub right_block: usize,
    /// `left_block * right_block` (may sit below the searched pair target due
    /// to integer decomposition).
    pub pairs: usize,
    /// Which constraint decided the pair budget.
    pub bound: SizingBound,
}

/// Estimates optimal (left_block, right_block) sizes for the reference-table
/// join format, the join's counterpart of [`estimate_safe_batch_size`].
///
/// Maximizes `L×R` subject to the α-safe input cost fitting the budget:
///
/// ```text
/// cost(P) = L·(Q_left(1 − α/2L) + fmt_L)
///         + R·(Q_right(1 − α/2R) + fmt_R)
///         + L·R·pair_ref     with  L/R = sqrt(mean_R / mean_L)  (Lagrange)
/// ```
///
/// The α accounting mirrors the flat path's union bound, split evenly across
/// the two operands: each of the `L` left rows exceeds its quantile with
/// probability `α/2L`, so the left block exceeds `L·Q_left` with probability
/// ≤ α/2; same on the right; the whole request overflows its envelope with
/// probability ≤ α. The join thus responds to `config.alpha` like the flat
/// path.
///
/// `cost(P)` is monotone non-decreasing in `P` (both blocks grow with `P`, and
/// each side's quantile is non-decreasing as its argument approaches 1), so a
/// binary search over `P` converges in `⌈log₂ hi⌉` evaluations. Integer
/// rounding in the `L/R` decomposition can make the realized cost dip slightly
/// below the searched envelope; it never rises above it.
///
/// The caps (capacity, quality, output budget) bound the search space, so the
/// L/R ratio is chosen within the cap instead of being scaled down afterwards,
/// which would crush the smaller dimension.
pub fn estimate_join_block_sizes_v2(
    left_histogram: Option<&TokenHistogram>,
    right_histogram: &TokenHistogram,
    budget: &JoinBatchBudgetModel,
    config: &SafeBatchSizingConfig,
    capacity_cap: Option<usize>,
    quality_cap: Option<usize>,
    output_budget: Option<usize>,
) -> JoinBlockEstimate {
    let min_b = config.min_batch_size.max(1);
    let floor = |bound: SizingBound| JoinBlockEstimate {
        left_block: min_b,
        right_block: min_b,
        pairs: min_b * min_b,
        bound,
    };

    let available = budget
        .context_window_tokens
        .saturating_sub(budget.fixed_overhead_tokens);

    if available == 0 || right_histogram.total_rows() == 0 {
        return floor(SizingBound::MinFloor);
    }

    // The left side is often the token-dominant one in an asymmetric join, so
    // proxying it from the right histogram can undercount. Use the proxy only
    // as a last resort, with a warning.
    let (left_hist, left_proxied) = match left_histogram {
        Some(h) if h.total_rows() > 0 => (h, false),
        _ => (right_histogram, true),
    };
    if left_proxied {
        log::warn!(
            "[BatchOptimizer] no left histogram — using the right side as a proxy; \
             block sizes may overflow the context when left rows are longer"
        );
    }

    let l_fmt = budget.left_row_format_tokens as f64;
    let r_fmt = budget.right_row_format_tokens as f64;
    let pair_ref = budget.pair_reference_tokens as f64;
    let alpha = config.alpha.clamp(1e-9, 1.0);

    let left_mean = left_hist.stats().mean + l_fmt;
    let right_mean = right_histogram.stats().mean + r_fmt;
    if left_mean <= 0.0 || right_mean <= 0.0 {
        return floor(SizingBound::MinFloor);
    }

    // Upper bound on the pair search space: the trivial input bound, then the
    // tightest of the explicit caps. Later entries win ties so the reported
    // bound prefers the stronger claim (quality > capacity > output budget).
    let mut hi: usize = if pair_ref > 0.0 {
        ((available as f64) / pair_ref).floor() as usize
    } else {
        available
    };
    let mut hi_bound = SizingBound::TokenBudget;
    let output_pair_cap = output_budget.and_then(|ob| {
        (budget.output_tokens_per_pair > 0)
            .then(|| ob.saturating_sub(OUTPUT_FRAMING_TOKENS) / budget.output_tokens_per_pair)
    });
    for (cap, bound) in [
        (output_pair_cap, SizingBound::OutputBudget),
        (capacity_cap, SizingBound::ModelCapacity),
        (quality_cap, SizingBound::QualityCap),
    ] {
        if let Some(c) = cap
            && c <= hi
        {
            hi = c;
            hi_bound = bound;
        }
    }
    hi = hi.max(1);

    // Mean-based Lagrange ratio for the L/R split; α-quantiles for the cost.
    let shape = |p: usize| -> (usize, usize) {
        let l = ((p as f64 * right_mean / left_mean).sqrt().round() as usize).clamp(1, p);
        let r = (p / l).max(1);
        (l, r)
    };
    let side_cost = |hist: &TokenHistogram, n: usize, fmt: f64| -> f64 {
        let p = (1.0 - alpha / (2.0 * n as f64)).clamp(0.0, 1.0);
        n as f64 * (hist.quantile(p) + fmt)
    };
    let cost_of = |l: usize, r: usize| -> f64 {
        side_cost(left_hist, l, l_fmt)
            + side_cost(right_histogram, r, r_fmt)
            + (l * r) as f64 * pair_ref
    };
    let cost = |p: usize| -> f64 {
        let (l, r) = shape(p);
        cost_of(l, r)
    };

    // Binary search the largest pair target whose α-safe cost fits.
    let (mut lo, mut hi_s) = (1usize, hi);
    let mut best_p = 0usize;
    while lo <= hi_s {
        let mid = lo + (hi_s - lo) / 2;
        if cost(mid) <= available as f64 {
            best_p = mid;
            lo = mid + 1;
        } else {
            if mid == 1 {
                break;
            }
            hi_s = mid - 1;
        }
    }

    if best_p == 0 {
        // Not even one pair fits the α-safe envelope; the request must still
        // carry one, so the floor is used and reported through `bound`.
        return floor(SizingBound::MinFloor);
    }

    let bound = if best_p == hi {
        hi_bound
    } else {
        SizingBound::TokenBudget
    };

    let (mut left_block, mut right_block) = shape(best_p);

    // Use a left block of at least 2 when affordable. With L=1 every left row
    // needs a separate right-side scan, which defeats the reference-table
    // format (each row listed once). Raising L to 2 and reducing R halves the
    // number of right scans at the cost of slightly fewer pairs.
    if left_block == 1 && best_p >= 4 {
        let r_at_min_left = best_p / 2;
        if r_at_min_left >= 1 && cost_of(2, r_at_min_left) <= available as f64 {
            left_block = 2;
            right_block = r_at_min_left;
        }
    }

    let pairs = (left_block * right_block).max(1);
    log::info!(
        "[BatchOptimizer] join sizing: pairs={} (target {}), left_block={}, right_block={}, \
         bound={}, cost={:.1}/{} tokens, alpha={}",
        pairs,
        best_p,
        left_block,
        right_block,
        bound,
        cost_of(left_block, right_block),
        available,
        alpha,
    );
    JoinBlockEstimate {
        left_block,
        right_block,
        pairs,
        bound,
    }
}

/// Result of computing optimal pair batch sizes for a semantic join.
#[derive(Debug, Clone)]
pub struct JoinBatchSizingResult {
    /// Maximum number of pairs to evaluate in one LLM call.
    pub max_pairs_per_call: usize,
    /// Number of left rows per block.
    pub left_block_size: usize,
    /// Number of right rows per block.
    pub right_block_size: usize,
    /// Number of right rows used to build the right-side histogram.
    pub histogram_rows: usize,
    /// Number of left rows sampled for the left-side histogram. Zero means no
    /// left sample was available and the right side was used as a proxy.
    pub left_histogram_rows: usize,
    /// Which constraint decided the pair budget.
    pub bound: SizingBound,
}

/// Builds a per-row token histogram over the join columns of `batches`.
/// Returns `None` when no rows are available.
fn join_column_histogram(
    batches: &[RecordBatch],
    join_indices: &[usize],
    estimator: &TokenEstimator,
    bucket_count: usize,
) -> Result<Option<(TokenHistogram, usize)>> {
    let Some(first_batch) = batches.iter().find(|b| b.num_rows() > 0) else {
        return Ok(None);
    };
    let column_names: Vec<String> = join_indices
        .iter()
        .map(|&idx| first_batch.schema().field(idx).name().clone())
        .collect();

    let mut all_tokens: Vec<usize> = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        all_tokens.extend(estimator.estimate_batch_tokens(batch, &column_names)?);
    }
    if all_tokens.is_empty() {
        return Ok(None);
    }
    let rows = all_tokens.len();
    Ok(Some((
        TokenHistogram::from_token_counts(&all_tokens, bucket_count),
        rows,
    )))
}

/// Column names at `join_indices` of the first non-empty batch.
fn join_column_names(batches: &[RecordBatch], join_indices: &[usize]) -> Vec<String> {
    batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .map(|b| {
            join_indices
                .iter()
                .map(|&idx| b.schema().field(idx).name().clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Computes token-aware block sizes for the semantic join operator at run
/// time, for plans that carry no block sizes.
///
/// Uses the reference-table cost model via [`estimate_join_block_sizes_v2`]:
/// rows are tokenized per side over the join columns rendered into the prompt,
/// the static template overhead is measured by rendering the real templates,
/// and the model card contributes three caps: capacity (`max_batch_size`),
/// quality (`quality_max_batch_size`) and the output budget.
///
/// `left_batches` is a sample of the streamed left relation. It may be empty,
/// in which case sizing falls back to the right-side proxy with a warning.
pub fn compute_optimal_pair_batch_size(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    join_condition: &str,
    left_join_indices: &[usize],
    right_join_indices: &[usize],
    model_card: &ModelCard,
    reasoning: ReasoningMode,
) -> Result<JoinBatchSizingResult> {
    let empty_result = || JoinBatchSizingResult {
        max_pairs_per_call: 1,
        left_block_size: 1,
        right_block_size: 1,
        histogram_rows: 0,
        left_histogram_rows: 0,
        bound: SizingBound::MinFloor,
    };

    let estimator = TokenEstimator::default_estimator()?;
    let config = SafeBatchSizingConfig::default();

    // Per-row histograms over the join columns of each side, which are the
    // columns projected into the prompt.
    let Some((right_histogram, right_rows)) = join_column_histogram(
        right_batches,
        right_join_indices,
        &estimator,
        config.bucket_count,
    )?
    else {
        return Ok(empty_result());
    };
    let left_sample = join_column_histogram(
        left_batches,
        left_join_indices,
        &estimator,
        config.bucket_count,
    )?;
    let (left_histogram, left_rows) = match left_sample {
        Some((h, n)) => (Some(h), n),
        None => (None, 0),
    };

    // Static prompt overhead, measured from the real templates (system + user
    // preamble + join condition).
    let fixed_overhead = crate::prompt::utils::join_static_prompt_tokens(
        &reasoning.system_template(OperatorKind::Join),
        reasoning.user_template(OperatorKind::Join),
        join_condition,
        &estimator,
    )
    .unwrap_or_else(|e| {
        log::warn!(
            "[SemanticJoin::BatchSizing] failed to render join templates for overhead \
             measurement ({e}); falling back to a conservative constant"
        );
        estimator.estimate_text_tokens(join_condition) + 450
    });

    let budget = JoinBatchBudgetModel {
        context_window_tokens: model_card.effective_input_budget(),
        fixed_overhead_tokens: fixed_overhead,
        left_row_format_tokens: row_format_tokens(
            &join_column_names(left_batches, left_join_indices),
            &estimator,
        ),
        right_row_format_tokens: row_format_tokens(
            &join_column_names(right_batches, right_join_indices),
            &estimator,
        ),
        pair_reference_tokens: 15,
        // The reasoning mode's per-item output is charged here because
        // `output_tokens_per_pair` feeds only the output-budget cap, never the
        // input denominator. The plan-time budget uses the same figure.
        output_tokens_per_pair: 8 + reasoning.reasoning_output_tokens_per_row(),
    };

    let output_budget = model_card
        .max_output_tokens
        .unwrap_or_else(|| (model_card.context_window * 3) / 10);

    let estimate = estimate_join_block_sizes_v2(
        left_histogram.as_ref(),
        &right_histogram,
        &budget,
        &config,
        Some(model_card.max_batch_size),
        model_card.quality_max_batch_size,
        Some(output_budget),
    );

    log::debug!(
        "[SemanticJoin::BatchSizing] left_block={}, right_block={}, pairs={}, bound={}, \
         context_budget={}, fixed_overhead={}, left_sample_rows={}, right_rows={}, \
         model_max_batch={}, quality_max_batch={:?}",
        estimate.left_block,
        estimate.right_block,
        estimate.pairs,
        estimate.bound,
        budget.context_window_tokens,
        fixed_overhead,
        left_rows,
        right_rows,
        model_card.max_batch_size,
        model_card.quality_max_batch_size,
    );

    Ok(JoinBatchSizingResult {
        max_pairs_per_call: estimate.pairs,
        left_block_size: estimate.left_block,
        right_block_size: estimate.right_block,
        histogram_rows: right_rows,
        left_histogram_rows: left_rows,
        bound: estimate.bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_estimator::histogram::TokenHistogram;

    fn startup_fields() -> Vec<datafusion::arrow::datatypes::FieldRef> {
        use datafusion::arrow::datatypes::Field;
        use std::sync::Arc;
        vec![
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::new(Field::new("industry", DataType::Utf8, true)),
            Arc::new(Field::new("pitch", DataType::Utf8, true)),
            Arc::new(Field::new("founded", DataType::Int64, true)),
        ]
    }

    #[test]
    fn generation_charges_utf8_at_the_generation_rate() {
        // 6 framing + name lengths (1+2+1+1) + 3×48 Utf8 + 4 Int64.
        assert_eq!(estimate_generation_tokens_per_row(&startup_fields()), 159);
        assert!(
            estimate_generation_tokens_per_row(&startup_fields())
                > estimate_output_tokens_per_row(&startup_fields())
        );
    }

    #[test]
    fn generate_chunk_is_bound_by_the_output_budget() {
        // No max_output_tokens, so the budget is 20% of 8192 = 1638;
        // (1638 - 20) / 159 = 10.
        let card = ModelCard::new("mock", 8192).with_max_batch_size(4096);
        assert_eq!(generate_chunk_size(&card, &startup_fields(), 100), 10);
    }

    #[test]
    fn generate_chunk_respects_card_caps_and_rows_wanted() {
        let card = ModelCard::new("mock", 40960)
            .with_max_output_tokens(9000)
            .with_max_batch_size(4096)
            .with_quality_max_batch_size(8);
        assert_eq!(generate_chunk_size(&card, &startup_fields(), 100), 8);
        assert_eq!(generate_chunk_size(&card, &startup_fields(), 3), 3);

        // A budget too small for one row still asks for one.
        let tiny = card.clone().with_max_output_tokens(50);
        assert_eq!(generate_chunk_size(&tiny, &startup_fields(), 100), 1);
    }

    #[test]
    fn test_optimizer_converges() {
        let counts = vec![100, 120, 140, 160, 180, 200, 220, 240, 260, 280];
        let hist = TokenHistogram::from_token_counts(&counts, 8);
        let budget = BatchBudgetModel {
            context_window_tokens: 4096,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 5,
            expected_output_tokens_per_row: 20,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };

        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());
        assert!(est.final_batch_size >= 1);
        assert!(est.iterations >= 1);
        assert!(!est.capped_by_model);
    }

    #[test]
    fn test_optimizer_handles_tiny_budget() {
        let counts = vec![1000, 1100, 1200];
        let hist = TokenHistogram::from_token_counts(&counts, 4);
        let budget = BatchBudgetModel {
            context_window_tokens: 100,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 5,
            expected_output_tokens_per_row: 20,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };

        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());
        assert_eq!(est.final_batch_size, 1);
    }

    #[test]
    fn test_optimizer_reduces_k_with_heavy_tail() {
        let mut counts = vec![487; 90];
        counts.extend(vec![12847; 10]);

        let hist = TokenHistogram::from_token_counts(&counts, 10);
        let budget = BatchBudgetModel {
            context_window_tokens: 128000,
            fixed_overhead_tokens: 230,
            per_row_delimiter_tokens: 0,
            expected_output_tokens_per_row: 25,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };

        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());

        // The mean-based batch is much larger than the tail-aware safe batch
        // for heavy-tailed distributions.
        assert!(est.initial_batch_size > est.final_batch_size);
        assert!(est.final_batch_size >= 1);
    }

    /// With a generous budget the token math alone would pick a batch far
    /// above the card's ceiling; the card wins and is reported as the bound.
    #[test]
    fn model_max_batch_size_caps_the_estimate() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let mut budget = BatchBudgetModel {
            context_window_tokens: 128000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let config = SafeBatchSizingConfig::default();

        let uncapped = estimate_safe_batch_size(&hist, &budget, &config);
        assert!(
            uncapped.final_batch_size > 25,
            "fixture must exceed the cap to be meaningful, got {}",
            uncapped.final_batch_size
        );

        budget.model_max_batch_size = Some(25);
        let capped = estimate_safe_batch_size(&hist, &budget, &config);
        assert_eq!(capped.final_batch_size, 25);
        assert!(capped.capped_by_model);
    }

    /// A cap above what the tokens allow changes neither the size nor the
    /// diagnostic flag.
    #[test]
    fn model_max_batch_size_above_the_token_budget_is_inert() {
        let counts = vec![100, 120, 140, 160, 180, 200, 220, 240, 260, 280];
        let hist = TokenHistogram::from_token_counts(&counts, 8);
        let mut budget = BatchBudgetModel {
            context_window_tokens: 4096,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 5,
            expected_output_tokens_per_row: 20,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let config = SafeBatchSizingConfig::default();

        let uncapped = estimate_safe_batch_size(&hist, &budget, &config);
        budget.model_max_batch_size = Some(uncapped.final_batch_size + 1000);
        let capped = estimate_safe_batch_size(&hist, &budget, &config);

        assert_eq!(capped.final_batch_size, uncapped.final_batch_size);
        assert_eq!(capped.initial_batch_size, uncapped.initial_batch_size);
        assert!(!capped.capped_by_model);
    }

    /// `usize::clamp` panics when min > max, so a floor above the card's cap
    /// must be handled.
    #[test]
    fn model_cap_below_the_minimum_batch_size_yields_the_minimum() {
        let counts: Vec<usize> = (0..50).map(|_| 30).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 16);
        let budget = BatchBudgetModel {
            context_window_tokens: 32000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: Some(2),
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let config = SafeBatchSizingConfig {
            min_batch_size: 4,
            ..SafeBatchSizingConfig::default()
        };

        let est = estimate_safe_batch_size(&hist, &budget, &config);
        assert_eq!(est.final_batch_size, 4);
    }

    fn join_budget(ctx: usize) -> JoinBatchBudgetModel {
        JoinBatchBudgetModel {
            context_window_tokens: ctx,
            fixed_overhead_tokens: 450,
            left_row_format_tokens: 8,
            right_row_format_tokens: 8,
            pair_reference_tokens: 15,
            output_tokens_per_pair: 8,
        }
    }

    /// Independent reimplementation of the estimator's shape and cost, used to
    /// verify the binary search by brute force. It must stay in step with the
    /// production code; a divergence fails `join_binary_search_matches_brute_force`.
    fn reference_cost(
        left: &TokenHistogram,
        right: &TokenHistogram,
        budget: &JoinBatchBudgetModel,
        alpha: f64,
        p: usize,
    ) -> (usize, usize, f64) {
        let l_fmt = budget.left_row_format_tokens as f64;
        let r_fmt = budget.right_row_format_tokens as f64;
        let left_mean = left.stats().mean + l_fmt;
        let right_mean = right.stats().mean + r_fmt;
        let l = ((p as f64 * right_mean / left_mean).sqrt().round() as usize).clamp(1, p);
        let r = (p / l).max(1);
        let side = |h: &TokenHistogram, n: usize, fmt: f64| {
            let q = (1.0 - alpha / (2.0 * n as f64)).clamp(0.0, 1.0);
            n as f64 * (h.quantile(q) + fmt)
        };
        let cost = side(left, l, l_fmt)
            + side(right, r, r_fmt)
            + (l * r) as f64 * budget.pair_reference_tokens as f64;
        (l, r, cost)
    }

    #[test]
    fn join_symmetric_histograms_give_square_blocks() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 20)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(8192),
            &SafeBatchSizingConfig::default(),
            None,
            None,
            None,
        );
        assert!(est.left_block >= 1 && est.right_block >= 1);
        let ratio = est.left_block as f64 / est.right_block as f64;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "Expected roughly square blocks, got L={}, R={}",
            est.left_block,
            est.right_block
        );
        assert_eq!(est.pairs, est.left_block * est.right_block);
    }

    /// Long rows on the left (~300 tokens) and short rows on the right (~5
    /// tokens): the left block must be small, and the α-safe cost of the chosen
    /// blocks must fit the budget.
    #[test]
    fn join_left_dominant_histogram_shrinks_left_block() {
        let left_counts: Vec<usize> = (0..100).map(|i| 280 + (i % 40)).collect();
        let right_counts: Vec<usize> = (0..100).map(|i| 4 + (i % 3)).collect();
        let left = TokenHistogram::from_token_counts(&left_counts, 32);
        let right = TokenHistogram::from_token_counts(&right_counts, 32);
        let budget = join_budget(8192);
        let config = SafeBatchSizingConfig::default();

        let est =
            estimate_join_block_sizes_v2(Some(&left), &right, &budget, &config, None, None, None);
        assert!(
            est.right_block > est.left_block,
            "Expected R > L when left rows dominate, got L={}, R={}",
            est.left_block,
            est.right_block
        );

        // The realized blocks respect the α-safe envelope.
        let available = (budget.context_window_tokens - budget.fixed_overhead_tokens) as f64;
        let l_fmt = budget.left_row_format_tokens as f64;
        let r_fmt = budget.right_row_format_tokens as f64;
        let side = |h: &TokenHistogram, n: usize, fmt: f64| {
            let q = (1.0 - config.alpha / (2.0 * n as f64)).clamp(0.0, 1.0);
            n as f64 * (h.quantile(q) + fmt)
        };
        let cost = side(&left, est.left_block, l_fmt)
            + side(&right, est.right_block, r_fmt)
            + est.pairs as f64 * budget.pair_reference_tokens as f64;
        assert!(
            cost <= available,
            "α-safe cost {cost:.0} exceeds available {available:.0}"
        );
    }

    #[test]
    fn join_proxy_fallback_matches_explicit_right_histogram() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 20)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let budget = join_budget(8192);
        let config = SafeBatchSizingConfig::default();

        let proxied = estimate_join_block_sizes_v2(None, &hist, &budget, &config, None, None, None);
        let explicit =
            estimate_join_block_sizes_v2(Some(&hist), &hist, &budget, &config, None, None, None);
        assert_eq!(proxied.left_block, explicit.left_block);
        assert_eq!(proxied.right_block, explicit.right_block);
    }

    #[test]
    fn join_tiny_budget_forces_min_floor() {
        let counts: Vec<usize> = (0..100).map(|_| 500).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(100), // fixed_overhead 450 > 100 → available == 0
            &SafeBatchSizingConfig::default(),
            None,
            None,
            None,
        );
        assert_eq!(est.left_block, 1);
        assert_eq!(est.right_block, 1);
        assert_eq!(est.bound, SizingBound::MinFloor);
    }

    #[test]
    fn join_quality_cap_binds_and_reports() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(128_000),
            &SafeBatchSizingConfig::default(),
            Some(1000), // capacity roomy
            Some(16),   // quality tight
            None,
        );
        assert_eq!(est.pairs, 16, "L={}, R={}", est.left_block, est.right_block);
        assert_eq!(est.bound, SizingBound::QualityCap);
    }

    #[test]
    fn join_capacity_cap_binds_without_quality() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(128_000),
            &SafeBatchSizingConfig::default(),
            Some(25),
            None,
            None,
        );
        assert_eq!(est.pairs, 25);
        assert_eq!(est.left_block, 5);
        assert_eq!(est.right_block, 5);
        assert_eq!(est.bound, SizingBound::ModelCapacity);
    }

    #[test]
    fn join_quality_outranks_capacity_on_tie() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(128_000),
            &SafeBatchSizingConfig::default(),
            Some(16),
            Some(16),
            None,
        );
        assert_eq!(est.bound, SizingBound::QualityCap);
    }

    #[test]
    fn join_output_budget_binds_when_tightest() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        // (100 - 20) / 8 = 10 pairs from the output budget.
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(128_000),
            &SafeBatchSizingConfig::default(),
            None,
            None,
            Some(100),
        );
        assert!(est.pairs <= 10);
        assert_eq!(est.bound, SizingBound::OutputBudget);
    }

    #[test]
    fn join_token_budget_binds_when_caps_are_loose() {
        let counts: Vec<usize> = (0..100).map(|i| 100 + (i % 50)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let est = estimate_join_block_sizes_v2(
            Some(&hist),
            &hist,
            &join_budget(4096),
            &SafeBatchSizingConfig::default(),
            Some(1_000_000),
            Some(1_000_000),
            Some(1_000_000),
        );
        assert_eq!(est.bound, SizingBound::TokenBudget);
        // And the chosen blocks fit the α-safe envelope.
        let (_, _, cost) = reference_cost(
            &hist,
            &hist,
            &join_budget(4096),
            SafeBatchSizingConfig::default().alpha,
            est.pairs,
        );
        assert!(cost <= (4096 - 450) as f64);
    }

    /// A stricter overflow budget can only shrink the batch.
    #[test]
    fn join_alpha_is_monotone() {
        // Spread-out distribution so the quantile actually moves with p.
        let counts: Vec<usize> = (0..200).map(|i| 20 + i * 3).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let budget = join_budget(16_000);

        let pairs_at = |alpha: f64| {
            let config = SafeBatchSizingConfig {
                alpha,
                ..SafeBatchSizingConfig::default()
            };
            estimate_join_block_sizes_v2(Some(&hist), &hist, &budget, &config, None, None, None)
                .pairs
        };
        assert!(
            pairs_at(0.001) <= pairs_at(0.5),
            "stricter alpha must not grow the batch: {} vs {}",
            pairs_at(0.001),
            pairs_at(0.5)
        );
    }

    /// The binary search finds the largest feasible pair target, checked
    /// against a brute-force scan with an independent copy of the cost model.
    #[test]
    fn join_binary_search_matches_brute_force() {
        let left_counts: Vec<usize> = (0..150).map(|i| 60 + (i % 90)).collect();
        let right_counts: Vec<usize> = (0..150).map(|i| 10 + (i % 25)).collect();
        let left = TokenHistogram::from_token_counts(&left_counts, 32);
        let right = TokenHistogram::from_token_counts(&right_counts, 32);
        let budget = join_budget(8192);
        let config = SafeBatchSizingConfig::default();
        let available = (budget.context_window_tokens - budget.fixed_overhead_tokens) as f64;

        let hi = (available / budget.pair_reference_tokens as f64).floor() as usize;
        let mut brute_best = 0usize;
        for p in 1..=hi {
            let (_, _, cost) = reference_cost(&left, &right, &budget, config.alpha, p);
            if cost <= available {
                brute_best = p;
            }
        }
        let (bl, br, _) = reference_cost(&left, &right, &budget, config.alpha, brute_best);

        let est =
            estimate_join_block_sizes_v2(Some(&left), &right, &budget, &config, None, None, None);
        assert_eq!((est.left_block, est.right_block), (bl, br));
        assert_eq!(est.bound, SizingBound::TokenBudget);
    }

    #[test]
    fn quality_cap_caps_the_flat_estimate() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let mut budget = BatchBudgetModel {
            context_window_tokens: 128000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let config = SafeBatchSizingConfig::default();

        let uncapped = estimate_safe_batch_size(&hist, &budget, &config);
        assert!(uncapped.final_batch_size > 16);

        budget.quality_max_batch_size = Some(16);
        let capped = estimate_safe_batch_size(&hist, &budget, &config);
        assert_eq!(capped.final_batch_size, 16);
        assert!(capped.capped_by_model);
        assert_eq!(capped.bound, SizingBound::QualityCap);
    }

    #[test]
    fn quality_cap_tie_with_capacity_reports_quality() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let budget = BatchBudgetModel {
            context_window_tokens: 128000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: Some(16),
            quality_max_batch_size: Some(16),
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());
        assert_eq!(est.final_batch_size, 16);
        assert_eq!(est.bound, SizingBound::QualityCap);
    }

    #[test]
    fn capacity_below_quality_reports_capacity() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let budget = BatchBudgetModel {
            context_window_tokens: 128000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: Some(12),
            quality_max_batch_size: Some(16),
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());
        assert_eq!(est.final_batch_size, 12);
        assert_eq!(est.bound, SizingBound::ModelCapacity);
    }

    /// The flat counterpart of `join_output_budget_binds_when_tightest`.
    /// Nothing else keeps `k × output_tokens_per_row` under the generation
    /// ceiling, and under the JSON grammar an overrun truncates the array and
    /// loses the whole request.
    #[test]
    fn flat_output_budget_binds_when_tightest() {
        let counts: Vec<usize> = (0..100).map(|i| 20 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        // (100 - 20) / 8 = 10 rows from the output budget; the token budget
        // and both card ceilings are deliberately far looser.
        let budget = BatchBudgetModel {
            context_window_tokens: 128_000,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 8,
            model_max_batch_size: Some(1_000),
            quality_max_batch_size: Some(1_000),
            max_output_tokens: Some(100),
            reasoning_output_tokens_per_row: 0,
        };
        let est = estimate_safe_batch_size(&hist, &budget, &SafeBatchSizingConfig::default());
        assert_eq!(est.final_batch_size, 10);
        assert_eq!(est.bound, SizingBound::OutputBudget);
        assert!(est.capped_by_model);
    }

    /// A generous output ceiling changes nothing, exactly like a loose row cap.
    #[test]
    fn flat_output_budget_above_the_token_budget_is_inert() {
        let counts = vec![100, 120, 140, 160, 180, 200, 220, 240, 260, 280];
        let hist = TokenHistogram::from_token_counts(&counts, 8);
        let mut budget = BatchBudgetModel {
            context_window_tokens: 4096,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 5,
            expected_output_tokens_per_row: 20,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: None,
            reasoning_output_tokens_per_row: 0,
        };
        let config = SafeBatchSizingConfig::default();

        let unbounded = estimate_safe_batch_size(&hist, &budget, &config);
        budget.max_output_tokens = Some(1_000_000);
        let bounded = estimate_safe_batch_size(&hist, &budget, &config);

        assert_eq!(bounded.final_batch_size, unbounded.final_batch_size);
        assert_eq!(bounded.bound, unbounded.bound);
        assert!(!bounded.capped_by_model);
    }

    /// `reasoning_output_tokens_per_row` is charged to the output cap only.
    /// `context_window_tokens` is an input budget that has already reserved the
    /// output window, so a reasoning term in the input denominator would bill
    /// the same tokens again and shrink k needlessly.
    #[test]
    fn reasoning_tokens_are_charged_to_the_output_cap_only() {
        let counts: Vec<usize> = (0..100).map(|i| 100 + (i % 10)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let config = SafeBatchSizingConfig::default();
        let budget = |reasoning: usize, max_output: Option<usize>| BatchBudgetModel {
            context_window_tokens: 8192,
            fixed_overhead_tokens: 200,
            per_row_delimiter_tokens: 2,
            expected_output_tokens_per_row: 4,
            model_max_batch_size: None,
            quality_max_batch_size: None,
            max_output_tokens: max_output,
            reasoning_output_tokens_per_row: reasoning,
        };

        // Output ceiling far above what the input budget allows: a 500-token
        // per-row rationale must leave the token-bound k untouched.
        let plain = estimate_safe_batch_size(&hist, &budget(0, Some(1_000_000)), &config);
        let reasoning = estimate_safe_batch_size(&hist, &budget(500, Some(1_000_000)), &config);
        assert_eq!(reasoning.final_batch_size, plain.final_batch_size);
        assert_eq!(reasoning.bound, SizingBound::TokenBudget);

        // And it does reach the output cap: (5000 - 20) / (4 + 500) = 9.
        let bounded = estimate_safe_batch_size(&hist, &budget(500, Some(5_000)), &config);
        assert_eq!(bounded.final_batch_size, 9);
        assert_eq!(bounded.bound, SizingBound::OutputBudget);
    }

    /// For a card with a quality cap of 25, measuring the real templates
    /// instead of guessing from the instruction length, and bounding k by the
    /// generation ceiling, leaves the batch size at that cap for both extend and
    /// selection.
    #[test]
    fn qwen3_8b_batch_size_is_unchanged_by_the_measured_overhead_and_output_bound() {
        use crate::prompt::utils::flat_static_prompt_tokens;
        use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
        use std::sync::Arc;

        // Card: context 32768, max_output 9000, max_batch_size 25,
        // quality_max_batch_size 25.
        const EFFECTIVE_INPUT_BUDGET: usize = 32768 - 9000;
        const MAX_OUTPUT_TOKENS: usize = 9000;
        let instruction = "the review expresses a positive opinion about the movie";

        let estimator = TokenEstimator::default_estimator().unwrap();
        let derived: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "sentiment",
            DataType::Utf8,
            true,
        )]));
        let guessed = instruction.len() / 3 + 100;

        // Review-like row lengths.
        let counts: Vec<usize> = (0..1000).map(|i| 60 + (i % 240)).collect();
        let hist = TokenHistogram::from_token_counts(&counts, 32);
        let config = SafeBatchSizingConfig::default();

        for (operator, measured, output_per_row) in [
            (
                "selection",
                flat_static_prompt_tokens(
                    "selection/system.md",
                    "selection/user.md",
                    instruction,
                    None,
                    &estimator,
                )
                .unwrap(),
                4usize,
            ),
            (
                "extend",
                flat_static_prompt_tokens(
                    "extend/system.md",
                    "extend/user.md",
                    instruction,
                    Some(&derived),
                    &estimator,
                )
                .unwrap(),
                estimate_output_tokens_per_row(derived.fields()),
            ),
        ] {
            assert!(
                measured > guessed,
                "{operator}: the measured overhead {measured} must exceed the length-based guess {guessed}"
            );

            let build = |fixed: usize, max_output: Option<usize>| BatchBudgetModel {
                context_window_tokens: EFFECTIVE_INPUT_BUDGET,
                fixed_overhead_tokens: fixed,
                per_row_delimiter_tokens: 2,
                expected_output_tokens_per_row: output_per_row,
                model_max_batch_size: Some(25),
                quality_max_batch_size: Some(25),
                max_output_tokens: max_output,
                reasoning_output_tokens_per_row: 0,
            };

            let before = estimate_safe_batch_size(&hist, &build(guessed, None), &config);
            let after =
                estimate_safe_batch_size(&hist, &build(measured, Some(MAX_OUTPUT_TOKENS)), &config);

            assert_eq!(
                after.final_batch_size, before.final_batch_size,
                "{operator}: k moved from {} to {}",
                before.final_batch_size, after.final_batch_size
            );
            assert_eq!(after.bound, before.bound, "{operator}: bound changed");
            assert_eq!(
                after.final_batch_size, 25,
                "{operator}: the card's quality cap should still bind"
            );

            // A reasoning system template is much longer than the plain
            // preamble, so the static prefix grows. The card must still be
            // quality-capped afterwards.
            const REASONING_PREFIX_TOKENS: usize = 600;
            let reasoning = estimate_safe_batch_size(
                &hist,
                &build(measured + REASONING_PREFIX_TOKENS, Some(MAX_OUTPUT_TOKENS)),
                &config,
            );
            assert_eq!(
                reasoning.final_batch_size, 25,
                "{operator}: a reasoning-sized static prefix must not shrink k"
            );
            assert_eq!(reasoning.bound, SizingBound::QualityCap);
        }
    }
}
