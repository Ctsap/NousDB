//! The shared MiniJinja environment used to render prompt templates.

use super::templates::embedded_template;
use minijinja::{AutoEscape, Environment, UndefinedBehavior};
use std::sync::LazyLock as Lazy;
use std::sync::RwLock;

/// Global MiniJinja environment serving the embedded prompt templates.
pub static ENV: Lazy<RwLock<Environment<'static>>> = Lazy::new(|| {
    let mut env = Environment::new();
    // Serve templates from the table compiled into the binary (`templates.rs`),
    // so nothing is read from disk at run time. `set_loader` is lazy and
    // caches; an unknown name yields `None`, which minijinja reports as
    // "template not found". Requires the `loader` feature.
    env.set_loader(|name| Ok(embedded_template(name).map(str::to_owned)));
    // Prompts are plain text, not HTML: turn off auto-escape.
    env.set_auto_escape_callback(|_| AutoEscape::None);
    // A referenced but unprovided variable is a bug in a renderer or template:
    // fail the render instead of silently emitting "".
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    // Resolve `{% include "../_reasoning/.." %}` relative to the including template.
    env.set_path_join_callback(|name, parent| {
        let mut parts: Vec<&str> = parent.split('/').collect();
        parts.pop();
        for seg in name.split('/') {
            match seg {
                "." => {}
                ".." => {
                    parts.pop();
                }
                _ => parts.push(seg),
            }
        }
        parts.join("/").into()
    });

    RwLock::new(env)
});
