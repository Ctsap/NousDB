//! SQL functions for NousDB's semantic operators.
//!
//! ```sql
//! SELECT *, EXTEND('classify sentiment', 'model', review) AS sentiment FROM reviews;
//! SELECT * FROM EXTEND(reviews, 'extract both', 'model', review, 'sentiment: Utf8, genre: Utf8');
//! SELECT * FROM papers WHERE SEM_FILTER('is about databases', 'model', abstract);
//! SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'model', a.title, b.name);
//! SELECT * FROM GENERATE('20 fictional startups', 'model', 'name: Utf8, pitch: Utf8', 20);
//! ```
//!
//! The scalar functions are markers: [`markers`] registers them so stock SQL
//! parses and plans, and [`rewrite::SemanticFunctionRewriteRule`] (an analyzer
//! rule) replaces them with NousDB's logical nodes before optimization. The
//! relation forms, `GENERATE(...)` and `EXTEND(...)` in `FROM`, are planned
//! directly by [`relation::SemanticRelationPlanner`]. Argument positions live
//! in [`args`] and nowhere else.
//!
//! [`install`] adds all three to a session under construction; [`register`]
//! adds them to an existing one.

#![warn(missing_docs)]

pub mod args;
mod columns;
pub mod markers;
pub mod relation;
pub mod rewrite;
pub mod schema_dsl;

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableFunction;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::execution::session_state::SessionStateBuilder;

pub use relation::SemanticRelationPlanner;
pub use rewrite::SemanticFunctionRewriteRule;

/// Names of the scalar marker functions, as registered (SQL lowercases an
/// unquoted function name, so `EXTEND(...)` resolves to `extend`).
pub const SCALAR_FUNCTION_NAMES: [&str; 3] = ["extend", "sem_filter", "sem_join"];

/// Names the relation planner handles in `FROM`, matched in any case.
pub const RELATION_FUNCTION_NAMES: [&str; 2] = ["extend", "generate"];

/// Names of the table functions this crate registers. The only one is the
/// hidden resolver behind `EXTEND(table_name, ...)`; see
/// [`relation::TABLE_RESOLVER_NAME`].
pub const TABLE_FUNCTION_NAMES: [&str; 1] = [relation::TABLE_RESOLVER_NAME];

/// Adds the semantic SQL functions to a session under construction: the marker
/// functions, the relation planner and the analyzer rule.
///
/// Call it after `with_default_features()`; the rule is appended after
/// DataFusion's own analyzer rules.
pub fn install(mut builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .scalar_functions()
        .get_or_insert_with(Vec::new)
        .extend(markers::scalar_functions());
    // First in the list, so it sees `EXTEND(...)` / `GENERATE(...)` before any
    // other planner or DataFusion's case-sensitive table-function lookup.
    builder
        .relation_planners()
        .get_or_insert_with(Vec::new)
        .insert(0, Arc::new(SemanticRelationPlanner::new()));
    // How the relation planner reaches a table named as `EXTEND`'s input.
    builder
        .table_functions()
        .get_or_insert_with(HashMap::new)
        .insert(
            relation::TABLE_RESOLVER_NAME.to_string(),
            Arc::new(TableFunction::new(
                relation::TABLE_RESOLVER_NAME.to_string(),
                relation::table_resolver(),
            )),
        );
    builder.with_analyzer_rule(Arc::new(SemanticFunctionRewriteRule::new()))
}

/// Adds the semantic SQL functions to an existing session.
///
/// The session must plan NousDB's logical nodes (a `NousDBContext` does), or
/// physical planning of a rewritten query fails.
pub fn register(ctx: &SessionContext) -> Result<()> {
    for udf in markers::scalar_functions() {
        ctx.register_udf(udf.as_ref().clone());
    }
    ctx.register_relation_planner(Arc::new(SemanticRelationPlanner::new()))?;
    ctx.register_udtf(relation::TABLE_RESOLVER_NAME, relation::table_resolver());
    ctx.add_analyzer_rule(Arc::new(SemanticFunctionRewriteRule::new()));
    Ok(())
}
