//! Per-row reasoning arms.
//!
//! A [`ReasoningMode`] selects which system template an operator renders and
//! what the model is asked to emit alongside its answer. [`ReasoningMode::Off`]
//! is the baseline and must stay byte-identical on the wire.
//!
//! The mode is not carried on logical plan nodes. It is threaded as a
//! constructor or builder parameter into the physical operators, because a
//! logical node's hand-written `with_exprs_and_inputs` clone path silently
//! drops any field it does not copy during DataFusion's tree reconstruction.

use datafusion::arrow::datatypes::{DataType, Field};

use crate::certification::OperatorKind;

/// Generated tokens a `rationale` costs per item, charged against the model
/// card's output ceiling (`BatchBudgetModel::reasoning_output_tokens_per_row`).
///
/// `_reasoning/brevity.md` bounds the rationale at one short sentence per item
/// (about 25 words, roughly 33 tokens). The JSON framing (the `"rationale":`
/// key, quotes, separator) adds a few more and models overshoot soft word
/// budgets, so the figure carries 1.4x headroom: `33 * 1.4 = 46`, rounded to 48.
///
/// It is deliberately an over-estimate. Under the JSON grammar an output
/// overrun truncates the array mid-object and loses the whole request, so a
/// slightly smaller batch size is much cheaper than an underestimate.
pub const RATIONALE_OUTPUT_TOKENS_PER_ROW: usize = 48;

/// Output tokens `Think` spends per item, reasoning channel included.
///
/// Per-row cost falls with batch size (the model pays a fixed orientation cost
/// per request and a small marginal cost per item), so this is a high per-row
/// figure with headroom rather than an extrapolation from `k = 1`.
///
/// Like [`RATIONALE_OUTPUT_TOKENS_PER_ROW`] it deliberately errs high, because
/// an output overrun truncates the array and loses the whole request. Reasoning
/// length depends on the deployed model and its sampling configuration, so
/// re-measure it before relying on it for a different model.
pub const THINK_OUTPUT_TOKENS_PER_ROW: usize = 160;

/// How much reasoning the model is asked to do per item, and where it goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReasoningMode {
    /// Answer only, with no extra reasoning. The baseline.
    #[default]
    Off,
    /// A reasoning procedure stated in the system prompt; the answer shape is
    /// unchanged.
    Scaffold,
    /// Reason-then-decide: every response object carries a `rationale` string
    /// *before* the decision fields. The rationale is a wire field only and
    /// never becomes an Arrow column (see [`Self::prompt_only_fields`]).
    Rationale,
    /// Scaffold plus the server-side thinking channel
    /// ([`Self::enable_thinking`]).
    Think,
}

impl ReasoningMode {
    /// Stable identifier used in cache keys and environment variables.
    /// Round-trips through [`Self::from_label`].
    pub fn label(&self) -> &'static str {
        match self {
            ReasoningMode::Off => "off",
            ReasoningMode::Scaffold => "scaffold",
            ReasoningMode::Rationale => "rationale",
            ReasoningMode::Think => "think",
        }
    }

    /// Parses a [`Self::label`], ignoring case and surrounding whitespace.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "off" => Some(ReasoningMode::Off),
            "scaffold" => Some(ReasoningMode::Scaffold),
            "rationale" => Some(ReasoningMode::Rationale),
            "think" => Some(ReasoningMode::Think),
            _ => None,
        }
    }

    /// Every mode, for exhaustive iteration.
    pub fn all() -> [ReasoningMode; 4] {
        [
            ReasoningMode::Off,
            ReasoningMode::Scaffold,
            ReasoningMode::Rationale,
            ReasoningMode::Think,
        ]
    }

    /// The system template `Prompt::render_system` loads for `kind`.
    ///
    /// This is the single place where operator and mode combine into a template
    /// name. Every consumer (operator streams, the certificate cache key,
    /// plan-time batch sizing) must use it, or it describes a prompt that is
    /// never sent.
    pub fn system_template(&self, kind: OperatorKind) -> String {
        let dir = operator_dir(kind);
        match self {
            ReasoningMode::Off => format!("{dir}/system.md"),
            _ => format!("{dir}/system.{}.md", self.label()),
        }
    }

    /// The mode a [`Self::system_template`] name came from.
    ///
    /// The static-overhead measurements
    /// (`prompt::utils::{flat,join}_static_prompt_tokens`) receive template
    /// names rather than a mode, and must supply the `reasoning_mode` template
    /// variable that the user templates branch on.
    ///
    /// Any name this module did not produce (`system.md`, or a template
    /// registered under another name) reads back as [`ReasoningMode::Off`],
    /// whose response contract has no extra wire field.
    pub fn from_system_template(template: &str) -> Self {
        template
            .rsplit('/')
            .next()
            .unwrap_or(template)
            .strip_prefix("system.")
            .and_then(|rest| rest.strip_suffix(".md"))
            .and_then(Self::from_label)
            .unwrap_or(ReasoningMode::Off)
    }

    /// The user template `Prompt::render_batch` loads for `kind`.
    ///
    /// The row data and its framing do not vary with the mode; only the system
    /// prompt and the response contract do.
    pub fn user_template(&self, kind: OperatorKind) -> &'static str {
        match kind {
            OperatorKind::Selection => "selection/user.md",
            OperatorKind::Extend => "extend/user.md",
            OperatorKind::Join => "join/user.md",
        }
    }

    /// Whether the server-side thinking channel should be enabled.
    pub fn enable_thinking(&self) -> bool {
        matches!(self, ReasoningMode::Think)
    }

    /// Generated tokens this mode adds per item, beyond the answer itself.
    ///
    /// Feeds `BatchBudgetModel::reasoning_output_tokens_per_row`, which is
    /// charged against the card's output ceiling only (see that field's
    /// documentation for why the input side must not carry it as well).
    pub fn reasoning_output_tokens_per_row(&self) -> usize {
        match self {
            // The answer shape is unchanged: `Scaffold` moves reasoning into
            // the system prompt (input side), not into the response.
            ReasoningMode::Off | ReasoningMode::Scaffold => 0,
            ReasoningMode::Rationale => RATIONALE_OUTPUT_TOKENS_PER_ROW,
            // `Think` spends on the server-side thinking channel, whose length
            // depends on the deployed model and is not bounded by the
            // templates. The constant is a fixed per-row estimate; too low
            // truncates whole requests, too high needlessly shrinks batches.
            ReasoningMode::Think => THINK_OUTPUT_TOKENS_PER_ROW,
        }
    }

    /// Fields the model must emit that are not operator output columns.
    ///
    /// These lead the wire schema (`Prompt::wire_fields`), so the model writes
    /// them before it commits to an answer. They are kept out of the output
    /// schema: a `rationale` registered as a derived column would be scored by
    /// the certification probe comparator, and two independent free-text
    /// generations never agree, which would drive every certified size to 1.
    pub fn prompt_only_fields(&self) -> Vec<Field> {
        match self {
            ReasoningMode::Rationale => vec![Field::new("rationale", DataType::Utf8, true)],
            _ => Vec::new(),
        }
    }
}

fn operator_dir(kind: OperatorKind) -> &'static str {
    match kind {
        OperatorKind::Selection => "selection",
        OperatorKind::Extend => "extend",
        OperatorKind::Join => "join",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip() {
        for mode in ReasoningMode::all() {
            assert_eq!(ReasoningMode::from_label(mode.label()), Some(mode));
        }
        assert_eq!(
            ReasoningMode::from_label(" SCAFFOLD "),
            Some(ReasoningMode::Scaffold)
        );
        assert_eq!(ReasoningMode::from_label("chain-of-thought"), None);
    }

    #[test]
    fn default_is_off() {
        assert_eq!(ReasoningMode::default(), ReasoningMode::Off);
    }

    /// The baseline mode resolves to the plain `system.md` templates.
    #[test]
    fn off_resolves_to_todays_templates() {
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
    }

    /// Each operator resolves to its own system template. The selection and
    /// extend files can be byte-identical under `Off`, so a mix-up would not
    /// show in rendered output.
    #[test]
    fn each_operator_gets_its_own_system_template() {
        for mode in ReasoningMode::all() {
            let names: Vec<String> = [
                OperatorKind::Selection,
                OperatorKind::Extend,
                OperatorKind::Join,
            ]
            .into_iter()
            .map(|kind| mode.system_template(kind))
            .collect();
            assert!(
                names[0].starts_with("selection/")
                    && names[1].starts_with("extend/")
                    && names[2].starts_with("join/"),
                "{mode:?} crossed operators: {names:?}"
            );
        }
    }

    #[test]
    fn only_rationale_declares_a_wire_only_field() {
        for mode in ReasoningMode::all() {
            let fields = mode.prompt_only_fields();
            if mode == ReasoningMode::Rationale {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name(), "rationale");
                assert_eq!(fields[0].data_type(), &DataType::Utf8);
            } else {
                assert!(fields.is_empty(), "{mode:?} must declare no wire fields");
            }
        }
    }

    #[test]
    fn only_think_enables_thinking() {
        for mode in ReasoningMode::all() {
            assert_eq!(mode.enable_thinking(), mode == ReasoningMode::Think);
        }
    }

    /// `from_system_template` recovers the mode from a template name; a
    /// round-trip failure would size reasoning modes against the baseline's
    /// example.
    #[test]
    fn system_template_names_round_trip_to_their_mode() {
        for mode in ReasoningMode::all() {
            for kind in [
                OperatorKind::Selection,
                OperatorKind::Extend,
                OperatorKind::Join,
            ] {
                assert_eq!(
                    ReasoningMode::from_system_template(&mode.system_template(kind)),
                    mode,
                    "{mode:?}/{kind:?}"
                );
            }
        }
    }

    /// A name this module did not produce reads back as the mode with no extra
    /// wire field.
    #[test]
    fn unknown_template_names_read_back_as_off() {
        for name in [
            "__test_only__/system_with_worked_example.md",
            "selection/system.chain_of_thought.md",
            "system.md",
            "",
            "selection/",
        ] {
            assert_eq!(
                ReasoningMode::from_system_template(name),
                ReasoningMode::Off,
                "{name:?}"
            );
        }
    }

    /// Every mode that spends extra output tokens must budget them, whether
    /// they land in a wire field or on the thinking channel. Spending without
    /// budgeting truncates whole requests; budgeting without spending shrinks
    /// batches needlessly.
    #[test]
    fn every_arm_that_spends_output_tokens_budgets_them() {
        for mode in ReasoningMode::all() {
            let budgeted = mode.reasoning_output_tokens_per_row() > 0;
            let declares_fields = !mode.prompt_only_fields().is_empty();
            match mode {
                ReasoningMode::Rationale => {
                    assert_eq!(
                        mode.reasoning_output_tokens_per_row(),
                        RATIONALE_OUTPUT_TOKENS_PER_ROW
                    );
                    assert!(budgeted && declares_fields);
                }
                // `Think` spends on the server-side thinking channel, so it
                // budgets without declaring a wire field.
                ReasoningMode::Think => {
                    assert_eq!(
                        mode.reasoning_output_tokens_per_row(),
                        THINK_OUTPUT_TOKENS_PER_ROW
                    );
                    assert!(budgeted && !declares_fields);
                }
                ReasoningMode::Off | ReasoningMode::Scaffold => {
                    assert!(!budgeted && !declares_fields, "{mode:?}");
                }
            }
        }
    }

    /// With a 9 000-token output ceiling, the budgets leave a quality cap of
    /// 25 as the binding limit: `Think` at 160 tokens/row caps k at 54. If a
    /// budget grows past that, batch size changes with the mode.
    #[test]
    fn measured_budgets_leave_the_quality_cap_binding_on_qwen3_8b() {
        const OUTPUT_CEILING: usize = 9000;
        const FRAMING: usize = 20;
        const SELECTION_EXPECTED_PER_ROW: usize = 4;
        const QWEN3_8B_QUALITY_CAP: usize = 25;

        for mode in [ReasoningMode::Rationale, ReasoningMode::Think] {
            let per_row = SELECTION_EXPECTED_PER_ROW + mode.reasoning_output_tokens_per_row();
            let cap = (OUTPUT_CEILING - FRAMING) / per_row;
            assert!(
                cap > QWEN3_8B_QUALITY_CAP,
                "{mode:?} caps k at {cap}, at or below the quality cap of \
                 {QWEN3_8B_QUALITY_CAP}: the arm now moves batch size"
            );
        }
    }
}
