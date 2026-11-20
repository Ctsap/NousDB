//! The compiled-in template table must mirror `src/prompt/templates/` exactly.

use nousdb_core::prompt::environment::ENV;
use nousdb_core::prompt::templates::{embedded_template, embedded_template_names};
use std::collections::BTreeSet;
use std::path::Path;

const TEMPLATES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/prompt/templates");

/// Collects every `*.md` under `dir` as a `/`-separated path relative to `root`.
fn collect_md(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(dir).expect("read templates dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_md(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "md") {
            let rel = path.strip_prefix(root).expect("path under root");
            let name = rel
                .components()
                .map(|c| c.as_os_str().to_str().expect("utf-8 file name"))
                .collect::<Vec<_>>()
                .join("/");
            out.insert(name);
        }
    }
}

fn on_disk() -> BTreeSet<String> {
    let root = Path::new(TEMPLATES_DIR);
    let mut names = BTreeSet::new();
    collect_md(root, root, &mut names);
    names
}

/// Catches a template file added on disk but not to the table in `templates.rs` (and the reverse).
#[test]
fn embedded_table_lists_exactly_the_template_files() {
    let embedded: BTreeSet<String> = embedded_template_names().map(str::to_owned).collect();
    let disk = on_disk();
    assert!(!disk.is_empty(), "no templates found under {TEMPLATES_DIR}");
    assert_eq!(
        embedded.len(),
        embedded_template_names().count(),
        "duplicate names in the embedded table"
    );
    let missing_from_table: Vec<_> = disk.difference(&embedded).collect();
    let missing_on_disk: Vec<_> = embedded.difference(&disk).collect();
    assert!(
        missing_from_table.is_empty() && missing_on_disk.is_empty(),
        "templates.rs is out of sync with {TEMPLATES_DIR}: \
         on disk but not embedded = {missing_from_table:?}, embedded but not on disk = {missing_on_disk:?}"
    );
}

#[test]
fn embedded_sources_are_byte_identical_to_the_files() {
    for name in on_disk() {
        let from_disk = std::fs::read_to_string(Path::new(TEMPLATES_DIR).join(&name))
            .unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert_eq!(
            embedded_template(&name),
            Some(from_disk.as_str()),
            "embedded copy of {name} differs from the file on disk (stale build?)"
        );
    }
}

#[test]
fn every_embedded_template_loads_through_the_environment() {
    let env = ENV.read().unwrap();
    for name in embedded_template_names() {
        env.get_template(name)
            .unwrap_or_else(|e| panic!("get_template({name}) failed: {e}"));
    }
}

#[test]
fn unknown_template_is_a_not_found_error() {
    assert!(embedded_template("nope/missing.md").is_none());
    let env = ENV.read().unwrap();
    let err = env
        .get_template("nope/missing.md")
        .expect_err("unknown template must not load");
    assert_eq!(err.kind(), minijinja::ErrorKind::TemplateNotFound);
}
