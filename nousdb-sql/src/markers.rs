//! The marker scalar functions `extend`, `sem_filter` and `sem_join`.
//!
//! A marker only has to parse and plan; the analyzer rule replaces it before
//! optimization, so invoking one is an error.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::expr::schema_name_from_exprs;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};

/// Which semantic operator a marker stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerKind {
    /// `EXTEND(instruction, model, column ... [, spec])`
    Extend,
    /// `SEM_FILTER(instruction, model, column ...)`
    SemFilter,
    /// `SEM_JOIN(instruction, model, column, column ...)`
    SemJoin,
}

impl MarkerKind {
    /// Every kind, in [`crate::SCALAR_FUNCTION_NAMES`] order.
    pub const ALL: [MarkerKind; 3] = [Self::Extend, Self::SemFilter, Self::SemJoin];

    /// The registered (lowercase) function name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Extend => "extend",
            Self::SemFilter => "sem_filter",
            Self::SemJoin => "sem_join",
        }
    }

    /// The function name as written in SQL documentation (uppercase).
    pub fn sql_name(self) -> &'static str {
        match self {
            Self::Extend => "EXTEND",
            Self::SemFilter => "SEM_FILTER",
            Self::SemJoin => "SEM_JOIN",
        }
    }
}

/// The `ScalarUDFImpl` behind every marker function.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SemanticMarker {
    kind: MarkerKind,
    signature: Signature,
}

impl SemanticMarker {
    /// Creates the marker for `kind`.
    pub fn new(kind: MarkerKind) -> Self {
        Self {
            kind,
            // Volatile, so the optimizer never constant-folds the call or
            // pushes it into a scan if it ever survives the rewrite.
            signature: Signature::variadic_any(Volatility::Volatile),
        }
    }

    /// Which operator this marker stands for.
    pub fn kind(&self) -> MarkerKind {
        self.kind
    }
}

impl ScalarUDFImpl for SemanticMarker {
    fn name(&self) -> &str {
        self.kind.name()
    }

    /// The output column's name. A named output spec (`'stars: Int64'`) names
    /// the column, as an alias would; any other call gets the usual
    /// `name(args)`.
    fn schema_name(&self, args: &[Expr]) -> Result<String> {
        if self.kind == MarkerKind::Extend
            && let Ok(call) = crate::args::ExtendCall::parse(args)
            && let Some(name) = call.output.name
        {
            return Ok(name);
        }
        Ok(format!(
            "{}({})",
            self.name(),
            schema_name_from_exprs(args)?
        ))
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(DataFusionError::Internal(format!(
            "{} derives its type from return_field_from_args",
            self.kind.sql_name()
        )))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        crate::args::return_field(self.kind, &args)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let name = self.kind.sql_name();
        Err(DataFusionError::Execution(format!(
            "{name}(...) is a NousDB semantic function that must be rewritten into a semantic \
             operator before execution; it was left in an unsupported position, or this session \
             lacks NousDB's analyzer rule (build it with NousDBContextBuilder)"
        )))
    }
}

/// The marker functions to register, in [`crate::SCALAR_FUNCTION_NAMES`] order.
pub fn scalar_functions() -> Vec<Arc<ScalarUDF>> {
    MarkerKind::ALL
        .into_iter()
        .map(|kind| Arc::new(ScalarUDF::new_from_impl(SemanticMarker::new(kind))))
        .collect()
}

/// The marker kind of `expr`, if it is a call to one of the marker functions.
///
/// Detects by the UDF's implementation type, not by name, so a user function
/// that happens to be called `extend` is not mistaken for a marker.
pub fn marker_kind(expr: &Expr) -> Option<MarkerKind> {
    let Expr::ScalarFunction(call) = expr else {
        return None;
    };
    let inner: &dyn Any = &**call.func.inner();
    inner
        .downcast_ref::<SemanticMarker>()
        .map(SemanticMarker::kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::lit;

    #[test]
    fn functions_follow_the_published_names() {
        let names: Vec<String> = scalar_functions()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        assert_eq!(names, crate::SCALAR_FUNCTION_NAMES);
        for kind in MarkerKind::ALL {
            assert_eq!(kind.name().to_uppercase(), kind.sql_name());
        }
    }

    #[test]
    fn marker_kind_detects_by_type() {
        for (udf, kind) in scalar_functions().into_iter().zip(MarkerKind::ALL) {
            let expr = udf.call(vec![lit("i"), lit("m")]);
            assert_eq!(marker_kind(&expr), Some(kind));
        }
        assert_eq!(marker_kind(&lit(1_i64)), None);
        let other = datafusion::functions::string::expr_fn::upper(lit("x"));
        assert_eq!(marker_kind(&other), None);
    }
}
