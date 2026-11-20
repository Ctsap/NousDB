//! The prompt templates, compiled into the binary.
//!
//! `include_str!` embeds every `templates/**/*.md` file in the crate, so a
//! built artifact renders prompts without the source tree. Keys are
//! `/`-separated names relative to `templates/` (`selection/system.scaffold.md`,
//! `_reasoning/brevity.md`).
//!
//! Adding a template file means adding a row here too;
//! `tests/embedded_templates.rs` fails if the table and the directory diverge.

/// Every template, sorted by name.
static TEMPLATES: &[(&str, &str)] = &[
    (
        "_reasoning/brevity.md",
        include_str!("templates/_reasoning/brevity.md"),
    ),
    (
        "_reasoning/calibration.md",
        include_str!("templates/_reasoning/calibration.md"),
    ),
    (
        "_reasoning/independence.md",
        include_str!("templates/_reasoning/independence.md"),
    ),
    (
        "extend/system.md",
        include_str!("templates/extend/system.md"),
    ),
    (
        "extend/system.rationale.md",
        include_str!("templates/extend/system.rationale.md"),
    ),
    (
        "extend/system.scaffold.md",
        include_str!("templates/extend/system.scaffold.md"),
    ),
    (
        "extend/system.think.md",
        include_str!("templates/extend/system.think.md"),
    ),
    ("extend/user.md", include_str!("templates/extend/user.md")),
    (
        "generate/system.md",
        include_str!("templates/generate/system.md"),
    ),
    (
        "generate/user.md",
        include_str!("templates/generate/user.md"),
    ),
    ("join/system.md", include_str!("templates/join/system.md")),
    (
        "join/system.rationale.md",
        include_str!("templates/join/system.rationale.md"),
    ),
    (
        "join/system.scaffold.md",
        include_str!("templates/join/system.scaffold.md"),
    ),
    (
        "join/system.think.md",
        include_str!("templates/join/system.think.md"),
    ),
    ("join/user.md", include_str!("templates/join/user.md")),
    (
        "selection/system.md",
        include_str!("templates/selection/system.md"),
    ),
    (
        "selection/system.rationale.md",
        include_str!("templates/selection/system.rationale.md"),
    ),
    (
        "selection/system.scaffold.md",
        include_str!("templates/selection/system.scaffold.md"),
    ),
    (
        "selection/system.think.md",
        include_str!("templates/selection/system.think.md"),
    ),
    (
        "selection/user.md",
        include_str!("templates/selection/user.md"),
    ),
];

/// Source of the embedded template `name`, or `None` when no such template exists.
pub fn embedded_template(name: &str) -> Option<&'static str> {
    TEMPLATES
        .iter()
        .find(|(template, _)| *template == name)
        .map(|(_, source)| *source)
}

/// Names of all embedded templates, in table order.
pub fn embedded_template_names() -> impl Iterator<Item = &'static str> {
    TEMPLATES.iter().map(|(name, _)| *name)
}
