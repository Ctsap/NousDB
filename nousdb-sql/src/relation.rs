//! The relation planner for `GENERATE(...)` and relation-form `EXTEND(...)`.
//!
//! ```sql
//! SELECT * FROM GENERATE('20 fictional startups', 'm', 'name: Utf8, pitch: Utf8', 20);
//! SELECT * FROM EXTEND(reviews, 'extract both', 'm', review, 'sentiment: Utf8, genre: Utf8');
//! ```
//!
//! DataFusion looks table functions up case-sensitively and without
//! normalizing the name, so these two are not UDTFs: [`SemanticRelationPlanner`]
//! sees every raw `FROM` item before DataFusion's own planning and claims the
//! calls whose (normalized) name is `generate` or `extend`. Anything else,
//! including a table named `extend` used without parentheses, is passed on
//! untouched.

use std::io::Cursor;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll, Waker};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::json::ReaderBuilder;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{TableFunctionArgs as UdtfArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{DFSchema, ScalarValue, TableReference, plan_err};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use datafusion::logical_expr::simplify::SimplifyContext;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, lit};
use datafusion::optimizer::simplify_expressions::ExprSimplifier;
use datafusion::sql::sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, ObjectName, ObjectNamePart, TableAlias,
    TableFactor, TableFunctionArgs,
};

use nousdb_operators::extend::create_extend;
use nousdb_operators::generate::create_generate;

use crate::args::{GenerateCall, RelationExtendCall};
use crate::columns::bare_input_name;

/// Plans `FROM GENERATE(...)` into a `GenerateNode` and `FROM EXTEND(...)`
/// into an `ExtendNode`, matching the function name in any case.
///
/// The node is the same one the DataFrame API builds
/// (`create_generate` / `create_extend`); an alias written after the call
/// (`AS g`) is applied by DataFusion as a `SubqueryAlias`.
#[derive(Debug, Default)]
pub struct SemanticRelationPlanner;

impl SemanticRelationPlanner {
    /// Creates the planner.
    pub fn new() -> Self {
        Self
    }
}

/// Which relation function a `FROM` item calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelationFunction {
    Generate,
    Extend,
}

impl RelationFunction {
    fn name(self) -> &'static str {
        match self {
            Self::Generate => "GENERATE",
            Self::Extend => "EXTEND",
        }
    }

    fn of(normalized: &str) -> Option<Self> {
        match normalized {
            "generate" => Some(Self::Generate),
            "extend" => Some(Self::Extend),
            _ => None,
        }
    }
}

/// The function `name` calls, when it is a single identifier whose normalized
/// form is one of [`crate::RELATION_FUNCTION_NAMES`].
fn function_of(
    name: &ObjectName,
    context: &dyn RelationPlannerContext,
) -> Option<RelationFunction> {
    let [ObjectNamePart::Identifier(ident)] = name.0.as_slice() else {
        return None;
    };
    let normalized = context.normalize_ident(ident.clone());
    if !crate::RELATION_FUNCTION_NAMES.contains(&normalized.as_str()) {
        return None;
    }
    RelationFunction::of(&normalized)
}

impl RelationPlanner for SemanticRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let function = match &relation {
            TableFactor::Table {
                name,
                args: Some(_),
                ..
            } => function_of(name, context),
            _ => None,
        };
        let Some(function) = function else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let TableFactor::Table {
            name: _,
            alias,
            args: Some(args),
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            sample,
            index_hints,
        } = relation
        else {
            unreachable!("matched as a table function call above");
        };

        let f = function.name();
        let unsupported = |what: &str| -> Result<()> {
            plan_err!("{f}: {what} is not supported on {f}(...) in FROM")
        };
        if !with_hints.is_empty() {
            unsupported("WITH (...) table hints")?;
        }
        if version.is_some() {
            unsupported("a version (time-travel) clause")?;
        }
        if with_ordinality {
            unsupported("WITH ORDINALITY")?;
        }
        if !partitions.is_empty() {
            unsupported("PARTITION (...)")?;
        }
        if json_path.is_some() {
            unsupported("a JSON path")?;
        }
        if sample.is_some() {
            unsupported("TABLESAMPLE (sample a subquery instead)")?;
        }
        if !index_hints.is_empty() {
            unsupported("an index hint")?;
        }
        let args = positional_args(f, args)?;

        let plan = match function {
            RelationFunction::Generate => plan_generate(args, context)?,
            RelationFunction::Extend => plan_extend(args, context)?,
        };
        Ok(planned(plan, alias))
    }
}

fn planned(plan: LogicalPlan, alias: Option<TableAlias>) -> RelationPlanning {
    RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, alias)))
}

/// The call's arguments as plain positional SQL expressions; named arguments,
/// wildcards and a `SETTINGS` clause are errors.
fn positional_args(f: &str, args: TableFunctionArgs) -> Result<Vec<SqlExpr>> {
    let TableFunctionArgs { args, settings } = args;
    if settings.is_some() {
        return plan_err!("{f}: SETTINGS inside the argument list is not supported");
    }
    args.into_iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Ok(expr),
            FunctionArg::Unnamed(other) => plan_err!(
                "{f}: argument {} is {other}; wildcards are not accepted as arguments",
                i + 1
            ),
            FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => plan_err!(
                "{f}: argument {} is a named argument ({arg}); {f} takes positional arguments only",
                i + 1
            ),
        })
        .collect()
}

/// Folds constant expressions (`10 + 10`, `-5`); an expression the simplifier
/// cannot handle is returned unchanged and left for the argument parser to
/// reject.
fn fold_constant(expr: Expr) -> Expr {
    let simplifier = ExprSimplifier::new(SimplifyContext::default());
    simplifier.simplify(expr.clone()).unwrap_or(expr)
}

/// Adds the function and argument position to an error from expression
/// planning.
fn in_argument(f: &str, at: usize, e: DataFusionError) -> DataFusionError {
    e.context(format!("{f}: planning argument {}", at + 1))
}

/// `GENERATE(instruction, model, 'schema', num_rows [, examples])`.
fn plan_generate(
    args: Vec<SqlExpr>,
    context: &mut dyn RelationPlannerContext,
) -> Result<LogicalPlan> {
    let empty = DFSchema::empty();
    let exprs = args
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            context
                .sql_to_expr(arg, &empty)
                .map(fold_constant)
                .map_err(|e| in_argument("GENERATE", i, e))
        })
        .collect::<Result<Vec<_>>>()?;
    let call = GenerateCall::parse(&exprs)?;
    let examples = match &call.examples {
        Some(text) => decode_examples(text, &call.schema)?,
        None => Vec::new(),
    };
    create_generate(
        call.schema,
        call.model,
        call.instruction,
        call.num_rows,
        &examples,
    )
    .map_err(|e| prefix("GENERATE", e))
}

/// Decodes the examples literal: one JSON object per line (blank lines are
/// skipped), each into a one-row batch of `schema`. Unknown keys are errors;
/// missing keys are NULL.
fn decode_examples(text: &str, schema: &SchemaRef) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let lineno = i + 1;
        let bad = |why: String| -> Result<Vec<RecordBatch>> {
            plan_err!(
                "GENERATE: examples (argument 5), line {lineno}: {why}. Expected one JSON object \
                 per line whose keys are columns of the schema, e.g. {{\"name\": \"Acme\"}}"
            )
        };
        if !line.starts_with('{') {
            return bad(format!("'{line}' is not a JSON object"));
        }
        let reader = ReaderBuilder::new(schema.clone())
            .with_strict_mode(true)
            .build(Cursor::new(line.as_bytes()));
        let mut reader = match reader {
            Ok(r) => r,
            Err(e) => return bad(e.to_string()),
        };
        let mut rows = 0;
        loop {
            match reader.next() {
                None => break,
                Some(Ok(batch)) => {
                    rows += batch.num_rows();
                    batches.push(batch);
                }
                Some(Err(e)) => return bad(e.to_string()),
            }
        }
        if rows != 1 {
            return bad(format!("expected exactly one JSON object, found {rows}"));
        }
    }
    if batches.is_empty() {
        return plan_err!(
            "GENERATE: examples (argument 5) is empty; give one JSON object per line, \
             or leave the argument out"
        );
    }
    Ok(batches)
}

/// Prefixes a plan error from an operator constructor with the SQL function.
fn prefix(f: &str, e: DataFusionError) -> DataFusionError {
    match e {
        DataFusionError::Plan(msg) => DataFusionError::Plan(format!("{f}: {msg}")),
        other => other.context(f.to_string()),
    }
}

/// The input relation of `EXTEND`, from its raw first argument: a table, view
/// or CTE name, or a parenthesized subquery.
fn input_relation(arg: SqlExpr) -> Result<TableFactor> {
    let mut arg = arg;
    while let SqlExpr::Nested(inner) = arg {
        arg = *inner;
    }
    let name = match arg {
        SqlExpr::Subquery(subquery) => {
            return Ok(TableFactor::Derived {
                lateral: false,
                subquery,
                alias: None,
                sample: None,
            });
        }
        SqlExpr::Identifier(ident) => ObjectName::from(ident),
        SqlExpr::CompoundIdentifier(idents) => ObjectName::from(idents),
        other => {
            return plan_err!(
                "EXTEND: the first argument of EXTEND in FROM is a table name or a parenthesized \
                 subquery, found {other}"
            );
        }
    };
    Ok(TableFactor::Table {
        name,
        alias: None,
        args: None,
        with_hints: vec![],
        version: None,
        with_ordinality: false,
        partitions: vec![],
        json_path: None,
        sample: None,
        index_hints: vec![],
    })
}

/// Name under which [`table_resolver`] must be registered as a table function
/// (`ctx.register_udtf(TABLE_RESOLVER_NAME, table_resolver())`).
///
/// DataFusion loads the tables a statement uses before planning it, from the
/// names it finds in `FROM`; a table named only as `EXTEND`'s first argument
/// is an expression there, so it is not loaded. The relation planner looks
/// such a table up through this hidden function instead.
pub const TABLE_RESOLVER_NAME: &str = "__nousdb_resolve_table";

/// The hidden table function that returns a catalog table's provider by name:
/// its arguments are one to three string literals, `[catalog,] [schema,] table`.
///
/// The catalog is read synchronously (SQL planning is synchronous), so a
/// schema whose lookup has to wait on I/O (a remote catalog) is reported as
/// not resolvable; reading from the table in a subquery,
/// `EXTEND((SELECT * FROM t), ...)`, works for every catalog.
#[derive(Debug, Default)]
pub struct TableResolver;

/// The [`TableResolver`] to register under [`TABLE_RESOLVER_NAME`].
pub fn table_resolver() -> Arc<dyn TableFunctionImpl> {
    Arc::new(TableResolver)
}

impl TableFunctionImpl for TableResolver {
    fn call_with_args(&self, args: UdtfArgs) -> Result<Arc<dyn TableProvider>> {
        let parts = args
            .exprs()
            .iter()
            .map(|e| match e {
                Expr::Literal(ScalarValue::Utf8(Some(s)), _)
                | Expr::Literal(ScalarValue::Utf8View(Some(s)), _)
                | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Ok(s.as_str()),
                other => {
                    plan_err!("{TABLE_RESOLVER_NAME}: expected string literals, found {other}")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let table_ref = match parts.as_slice() {
            [t] => TableReference::bare(*t),
            [s, t] => TableReference::partial(*s, *t),
            [c, s, t] => TableReference::full(*c, *s, *t),
            _ => return plan_err!("{TABLE_RESOLVER_NAME}: expected 1 to 3 name parts"),
        };
        let Some(state) = args.session().as_any().downcast_ref::<SessionState>() else {
            return plan_err!("{TABLE_RESOLVER_NAME}: the session is not a SessionState");
        };
        let schema = state.schema_for_ref(table_ref.clone())?;
        match now_or_never(schema.table(table_ref.table())) {
            Some(Ok(Some(provider))) => Ok(provider),
            Some(Ok(None)) => plan_err!("table '{table_ref}' not found"),
            Some(Err(e)) => Err(e),
            None => plan_err!(
                "table '{table_ref}' cannot be looked up while planning (its catalog is \
                 asynchronous); write EXTEND((SELECT * FROM {table_ref}), ...) instead"
            ),
        }
    }
}

/// Polls `fut` once: its output when it is ready at once, else `None`.
fn now_or_never<F: Future>(fut: F) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = TaskContext::from_waker(Waker::noop());
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(out) => Some(out),
        Poll::Pending => None,
    }
}

/// Plans the input relation of `EXTEND`. A bare table name DataFusion did not
/// load for this statement is looked up through [`TABLE_RESOLVER_NAME`].
fn plan_input(
    factor: TableFactor,
    context: &mut dyn RelationPlannerContext,
) -> Result<LogicalPlan> {
    let name = match &factor {
        TableFactor::Table { name, .. } => Some(name.clone()),
        _ => None,
    };
    let error = match context.plan(factor) {
        Ok(plan) => return Ok(plan),
        Err(e) => e,
    };
    let Some(name) = name else {
        return Err(error);
    };
    let table_ref = context.object_name_to_table_reference(name)?;
    let parts: Vec<Expr> = match &table_ref {
        TableReference::Bare { table } => vec![lit(table.as_ref())],
        TableReference::Partial { schema, table } => {
            vec![lit(schema.as_ref()), lit(table.as_ref())]
        }
        TableReference::Full {
            catalog,
            schema,
            table,
        } => vec![
            lit(catalog.as_ref()),
            lit(schema.as_ref()),
            lit(table.as_ref()),
        ],
    };
    let source = match context
        .context_provider()
        .get_table_function_source(TABLE_RESOLVER_NAME, parts)
    {
        Ok(source) => source,
        Err(e) if resolver_is_missing(&e) => {
            return Err(error.context(format!(
                "EXTEND: table '{table_ref}' was not loaded for this statement and the session \
                 has no {TABLE_RESOLVER_NAME} table function to look it up (register the SQL \
                 functions with nousdb_sql::register or install); \
                 EXTEND((SELECT * FROM {table_ref}), ...) works without it"
            )));
        }
        Err(e) => return Err(e.context("EXTEND: resolving the input table")),
    };
    LogicalPlanBuilder::scan(table_ref, source, None)?.build()
}

/// Whether `e`, returned by `get_table_function_source(TABLE_RESOLVER_NAME, ..)`,
/// says the resolver is not registered, as opposed to an error from the
/// resolver itself. DataFusion 55.2's `ContextProvider` cannot be asked which
/// table functions exist, so this matches the variant and the exact text of
/// the two "not there" errors: the session's `Plan("table function '<name>'
/// not found")` and the trait default's `NotImplemented`.
fn resolver_is_missing(e: &DataFusionError) -> bool {
    match e.find_root() {
        DataFusionError::Plan(msg) => {
            msg.starts_with(&format!("table function '{TABLE_RESOLVER_NAME}' not found"))
        }
        DataFusionError::NotImplemented(msg) => {
            msg.starts_with("Table Functions are not supported")
        }
        _ => false,
    }
}

/// `EXTEND(<table or subquery>, instruction, model, column [, column ...], 'schema')`.
fn plan_extend(
    args: Vec<SqlExpr>,
    context: &mut dyn RelationPlannerContext,
) -> Result<LogicalPlan> {
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        // Let the argument parser report the usage.
        RelationExtendCall::parse(&[])?;
        return plan_err!("EXTEND: missing the input relation");
    };
    let input = plan_input(input_relation(first)?, context)?;
    let exprs = args
        .enumerate()
        .map(|(i, arg)| {
            context
                .sql_to_expr(arg, input.schema())
                .map_err(|e| in_argument("EXTEND", i + 1, e))
        })
        .collect::<Result<Vec<_>>>()?;
    let call = RelationExtendCall::parse(&exprs)?;
    let input_names = call
        .inputs
        .iter()
        .map(|c| bare_input_name(c, input.schema(), "EXTEND"))
        .collect::<Result<Vec<_>>>()?;
    for field in call.schema.fields() {
        if !input
            .schema()
            .fields_with_unqualified_name(field.name())
            .is_empty()
        {
            return plan_err!(
                "EXTEND: derived column '{}' already exists in the input relation; \
                 choose a different name in the schema literal",
                field.name()
            );
        }
    }
    create_extend(
        input,
        call.instruction,
        call.model,
        input_names,
        call.schema,
    )
    .map_err(|e| prefix("EXTEND", e))
}
