//! Inference metrics and the result type returned by model providers.

use std::time::Duration;

use crate::providers::schemas::ModelOutput;

/// Metrics from a single model inference call.
///
/// Token counts and timing returned by the model provider, used for cost
/// estimation and performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Number of input (prompt) tokens processed.
    pub input_tokens: u64,
    /// Number of output (completion) tokens generated.
    pub output_tokens: u64,
    /// Total tokens (input + output).
    pub total_tokens: u64,
    /// Tokens the model spent on a `<think>` block, when the endpoint reports
    /// them; zero otherwise.
    ///
    /// A subset of [`Self::output_tokens`], not an addition to it: the OpenAI
    /// usage schema bills reasoning out of the completion budget, so it must
    /// stay out of `total_tokens` or the same tokens are counted twice. It is
    /// kept separately so that a costly batch can be attributed to a long
    /// answer or to long reasoning.
    pub reasoning_tokens: u64,
    /// Time spent in model inference, in nanoseconds.
    pub inference_duration_ns: u64,
}

impl InferenceMetrics {
    /// Creates metrics from token counts and duration; `total_tokens` is the
    /// sum of input and output tokens and `reasoning_tokens` is zero.
    pub fn new(input_tokens: u64, output_tokens: u64, inference_duration_ns: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            reasoning_tokens: 0,
            inference_duration_ns,
        }
    }

    /// Sets the reasoning-token count reported by the endpoint.
    ///
    /// This is a builder rather than a `new()` parameter because only some
    /// providers can report reasoning tokens.
    pub fn with_reasoning_tokens(mut self, reasoning_tokens: u64) -> Self {
        self.reasoning_tokens = reasoning_tokens;
        self
    }

    /// Creates metrics with all values zero.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The inference duration as a [`Duration`].
    pub fn inference_duration(&self) -> Duration {
        Duration::from_nanos(self.inference_duration_ns)
    }

    /// Adds another set of metrics into this one, for aggregation.
    pub fn merge(&mut self, other: &InferenceMetrics) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.inference_duration_ns += other.inference_duration_ns;
    }
}

/// Result of a `predict()` call: the outputs and the inference metrics.
#[derive(Debug)]
pub struct PredictResult {
    /// The model outputs for each row in the batch.
    pub outputs: Vec<ModelOutput>,
    /// Metrics from this inference call.
    pub metrics: InferenceMetrics,
    /// Whether the model returned a different number of outputs than expected.
    /// When `true`, `outputs` contains null-equivalent values and operator
    /// streams should activate their recovery paths.
    pub row_count_mismatch: bool,
    /// How long the request ran before the transport failed, when it never
    /// produced a response at all. `None` means a response was received, so any
    /// remaining problem is a decoding problem, not a connectivity one. The two
    /// are reported differently; see [`Rejection::Transport`].
    pub transport_failure_ms: Option<u64>,
    /// The model stopped because it exhausted its output budget
    /// (`finish_reason == "length"`) rather than finishing its answer.
    ///
    /// Structured output is always requested, and a grammar-constrained array
    /// that runs out of budget is unterminated, so the whole batch is lost and
    /// the loss looks like a parse failure. This flag distinguishes "the output
    /// budget was too small for this batch size" from "the model answered
    /// badly".
    pub truncated: bool,
}

impl PredictResult {
    /// Creates a result with no transport failure and no truncation.
    pub fn new(
        outputs: Vec<ModelOutput>,
        metrics: InferenceMetrics,
        row_count_mismatch: bool,
    ) -> Self {
        Self {
            outputs,
            metrics,
            row_count_mismatch,
            transport_failure_ms: None,
            truncated: false,
        }
    }

    /// Records whether the model ran out of output budget mid-response.
    ///
    /// This is a builder rather than a `new()` parameter because truncation is
    /// only observable where the endpoint reports a finish reason.
    pub fn with_truncation(mut self, truncated: bool) -> Self {
        self.truncated = truncated;
        self
    }

    /// Creates a result with outputs and zero metrics, for providers that do
    /// not report metrics.
    pub fn with_outputs(outputs: Vec<ModelOutput>) -> Self {
        Self {
            outputs,
            metrics: InferenceMetrics::zero(),
            row_count_mismatch: false,
            transport_failure_ms: None,
            truncated: false,
        }
    }

    /// A result for a request that never came back: no response, no tokens, no
    /// outputs.
    ///
    /// `row_count_mismatch` stays `true`, so readers that only check it still
    /// see a failure: a request that never completed is not agreement, and a
    /// dead endpoint must not be able to certify a batch size. The
    /// certification prologue reads [`Self::rejection`]: a
    /// [`Rejection::Transport`] is scored as a disagreement and kept out of
    /// what the session caches, since a missing answer is not evidence about
    /// the model.
    pub fn transport_failure(elapsed_ms: u64) -> Self {
        Self {
            outputs: Vec::new(),
            metrics: InferenceMetrics::zero(),
            row_count_mismatch: true,
            transport_failure_ms: Some(elapsed_ms),
            truncated: false,
        }
    }

    /// Why this response is unusable for a request of `expected` items, if it is.
    ///
    /// Operators should not decide this by comparing `outputs.len()` alone: on
    /// a parse failure the provider null-fills to `expected`, so the length
    /// matches and only [`Self::row_count_mismatch`] reveals the problem.
    ///
    /// Checks run in priority order. Transport comes first, because a failed
    /// request also sets `row_count_mismatch` and would otherwise be reported
    /// as a parse failure. [`Rejection::Truncated`] outranks the two decoding
    /// verdicts because truncation causes the parse failure, so reporting the
    /// effect would hide the cause. Truncation only relabels a response that is
    /// already rejected: the `truncated` flag never turns a usable answer into
    /// a rejection.
    pub fn rejection(&self, expected: usize) -> Option<Rejection> {
        let undecodable = self.row_count_mismatch || self.outputs.len() != expected;

        if let Some(elapsed_ms) = self.transport_failure_ms {
            Some(Rejection::Transport { elapsed_ms })
        } else if self.truncated && undecodable {
            Some(Rejection::Truncated {
                output_tokens: self.metrics.output_tokens,
            })
        } else if self.row_count_mismatch {
            Some(Rejection::Unusable)
        } else if self.outputs.len() != expected {
            Some(Rejection::WrongLength {
                got: self.outputs.len(),
                expected,
            })
        } else {
            None
        }
    }
}

/// Why an operator cannot use a model response, worded for logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The request to the endpoint never produced a response, so nothing was
    /// decoded.
    ///
    /// The elapsed time helps tell the causes apart: sub-second means the
    /// connection was refused or reset, while a very long wait means a client
    /// timeout fired because the endpoint's queue never reached the request.
    Transport {
        /// Milliseconds the request ran before failing.
        elapsed_ms: u64,
    },
    /// The model exhausted its output budget mid-response, so what arrived was
    /// a fragment.
    ///
    /// Distinct from [`Self::Unusable`] even though it surfaces as the same
    /// parse failure: the batch is lost because the budget was too small for
    /// this batch size, not because the model answered badly. The remedies
    /// differ (lower the batch size or raise `ModelCard.max_output_tokens`,
    /// versus fix the prompt), and reasoning modes make truncation more likely
    /// because the `<think>` block is billed from the same budget.
    Truncated {
        /// Output tokens generated before the cutoff.
        output_tokens: u64,
    },
    /// The provider could not parse the response (or the model returned the
    /// wrong number of items) and null-filled to the requested length.
    Unusable,
    /// The provider returned a short or long vector directly, with no null
    /// fill.
    WrongLength {
        /// Number of outputs returned.
        got: usize,
        /// Number of outputs requested.
        expected: usize,
    },
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport { elapsed_ms } => {
                write!(
                    f,
                    "the request to the model endpoint failed after {:.1}s without returning a \
                     response — a transport failure, NOT a decoding problem (~600s means the \
                     litellm-rs client timeout fired on a saturated endpoint: lower \
                     NOUSDB_MAX_CONCURRENT_REQUESTS, --parallelism or --request-window)",
                    *elapsed_ms as f64 / 1000.0,
                )
            }
            Self::Truncated { output_tokens } => {
                write!(
                    f,
                    "the model hit its output token budget after {output_tokens} tokens and the \
                     response was cut off — a TRUNCATION, not a bad answer (structured output is \
                     always on, so a cut-off array is unterminated and the whole batch is lost): \
                     lower the batch size or raise ModelCard.max_output_tokens, and note that a \
                     <think> block is billed out of this same budget",
                )
            }
            Self::Unusable => f.write_str(
                "the model response could not be decoded into per-row outputs \
                 (parse failure or wrong item count); the provider null-filled it",
            ),
            Self::WrongLength { got, expected } => {
                write!(
                    f,
                    "the provider returned {got} outputs for {expected} items"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed request also carries `row_count_mismatch: true` (the
    /// certification prologue needs it), but it must not be reported as a parse
    /// failure.
    #[test]
    fn a_transport_failure_never_reports_as_a_parse_failure() {
        let result = PredictResult::transport_failure(600_123);

        let rejection = result
            .rejection(25)
            .expect("a request that never returned is always a rejection");
        assert_eq!(
            rejection,
            Rejection::Transport {
                elapsed_ms: 600_123
            }
        );

        let rendered = rejection.to_string();
        assert!(
            !rendered.contains("parse"),
            "a transport failure must not name parsing: {rendered}"
        );
        assert!(
            rendered.contains("600.1s"),
            "the elapsed time is the only clue to the cause: {rendered}"
        );

        assert!(
            result.row_count_mismatch,
            "certification's RequestOutcomeSink reads this flag to score the \
             request as disagreement; a dead endpoint must never certify a \
             batch size"
        );
    }

    /// The transport check must come first. An empty output vector on its own
    /// would otherwise be reported as a length problem, which says nothing
    /// about why the request failed.
    #[test]
    fn transport_outranks_the_other_rejections() {
        let result = PredictResult::transport_failure(37);
        assert!(matches!(
            result.rejection(8),
            Some(Rejection::Transport { .. })
        ));
    }

    /// The counterpart: a decoded-but-unusable response keeps saying so.
    #[test]
    fn a_parse_failure_still_reports_as_a_parse_failure() {
        let result = PredictResult::new(Vec::new(), InferenceMetrics::zero(), true);
        assert_eq!(result.rejection(4), Some(Rejection::Unusable));
        assert!(result.rejection(4).unwrap().to_string().contains("parse"));
    }

    /// A usable response is not a rejection at all.
    #[test]
    fn a_complete_response_is_not_rejected() {
        let outputs = vec![ModelOutput::Boolean(true), ModelOutput::Boolean(false)];
        let result = PredictResult::new(outputs, InferenceMetrics::zero(), false);
        assert_eq!(result.rejection(2), None);
    }

    /// A batch lost to an exhausted output budget must not read as a batch the
    /// model answered badly, although both arrive as the same null-filled parse
    /// failure.
    #[test]
    fn a_truncated_response_is_not_reported_as_a_parse_failure() {
        let result = PredictResult::new(
            Vec::new(),
            InferenceMetrics::new(120, 9000, 1_000),
            true, // grammar-constrained output cut mid-array never parses
        )
        .with_truncation(true);

        let rejection = result
            .rejection(25)
            .expect("an unterminated response is still a rejection");
        assert_eq!(
            rejection,
            Rejection::Truncated {
                output_tokens: 9000
            }
        );

        let rendered = rejection.to_string();
        assert!(
            rendered.contains("9000") && rendered.contains("max_output_tokens"),
            "the budget and the knob that fixes it are the whole diagnosis: {rendered}"
        );
        assert!(
            !rendered.contains("parse failure"),
            "truncation causes the parse failure; naming the effect hides the \
             cause: {rendered}"
        );
    }

    /// Truncation only relabels a rejection. A response that decoded cleanly
    /// stays usable even if the model stopped on `length`; rejecting it would
    /// send an already-good batch down an operator's recovery path.
    #[test]
    fn truncation_never_rejects_an_otherwise_usable_response() {
        let outputs = vec![ModelOutput::Boolean(true), ModelOutput::Boolean(false)];
        let result =
            PredictResult::new(outputs, InferenceMetrics::zero(), false).with_truncation(true);

        assert_eq!(result.rejection(2), None);
        assert!(
            result.truncated,
            "the fact is still recorded for whoever counts truncations"
        );
    }

    /// Transport still outranks truncation: a request that never returned has
    /// no finish reason to be truncated by.
    #[test]
    fn transport_outranks_truncation() {
        let result = PredictResult::transport_failure(600_000).with_truncation(true);
        assert!(matches!(
            result.rejection(8),
            Some(Rejection::Transport { .. })
        ));
    }

    /// Reasoning tokens are billed out of the completion budget, so they are a
    /// subset of `output_tokens` and adding them to `total_tokens` would count
    /// the same tokens twice.
    #[test]
    fn reasoning_tokens_are_not_added_to_the_total() {
        let metrics = InferenceMetrics::new(100, 500, 1_000).with_reasoning_tokens(300);

        assert_eq!(metrics.reasoning_tokens, 300);
        assert_eq!(metrics.output_tokens, 500);
        assert_eq!(metrics.total_tokens, 600, "input + output, nothing else");

        // And they aggregate.
        let mut acc = metrics.clone();
        acc.merge(&metrics);
        assert_eq!(acc.reasoning_tokens, 600);
        assert_eq!(acc.total_tokens, 1200);
    }

    /// The default is zero, so a provider that cannot report reasoning tokens
    /// does not need to say so.
    #[test]
    fn reasoning_tokens_default_to_zero() {
        assert_eq!(InferenceMetrics::new(1, 2, 3).reasoning_tokens, 0);
        assert_eq!(InferenceMetrics::zero().reasoning_tokens, 0);
    }
}
