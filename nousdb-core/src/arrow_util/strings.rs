//! Encoding-independent access to Arrow string columns.
//!
//! Arrow spells "a column of strings" three ways: `Utf8` (i32 offsets),
//! `LargeUtf8` (i64 offsets) and `Utf8View` (per-value views over shared
//! buffers). Code that downcasts straight to [`StringArray`] therefore only
//! handles one of them, and the failure is silent wherever the downcast sits
//! behind an `if let Some(..)`: the column reads as "not a string" and the
//! fallback path takes over.
//!
//! This matters because large text columns are often registered as `Utf8View`:
//! a single `Utf8` array cannot exceed `i32::MAX` bytes. Under that schema a
//! `StringArray`-only downcast in the token estimator would score every text
//! value as one token and mis-size every batch without erroring.
//!
//! [`StringColumn`] borrows any of the three encodings. Use
//! [`StringColumn::try_new`] where a non-string column is legitimate and has its
//! own handling, and [`StringColumn::new`] where it is a bug.

use datafusion::arrow::array::{Array, LargeStringArray, StringArray, StringViewArray};
use datafusion::error::{DataFusionError, Result};

/// A borrowed string column, whatever its Arrow encoding.
#[derive(Debug, Clone, Copy)]
pub enum StringColumn<'a> {
    /// `DataType::Utf8`: i32 offsets, capped at `i32::MAX` bytes per array.
    Utf8(&'a StringArray),
    /// `DataType::LargeUtf8`: i64 offsets.
    Large(&'a LargeStringArray),
    /// `DataType::Utf8View`: views over shared buffers, no per-array byte cap.
    View(&'a StringViewArray),
}

impl<'a> StringColumn<'a> {
    /// Borrows `array` if it holds strings in any encoding, else `None`.
    ///
    /// For call sites where a non-string column is expected and handled
    /// separately (the token estimator charges those a flat per-row estimate).
    pub fn try_new(array: &'a dyn Array) -> Option<Self> {
        let any = array.as_any();
        if let Some(a) = any.downcast_ref::<StringArray>() {
            Some(Self::Utf8(a))
        } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
            Some(Self::Large(a))
        } else {
            any.downcast_ref::<StringViewArray>().map(Self::View)
        }
    }

    /// Borrows `array`, returning an error if it does not hold strings.
    ///
    /// The error names the offending type, so a schema that drifts to an
    /// unexpected encoding surfaces here instead of silently degrading.
    pub fn new(array: &'a dyn Array) -> Result<Self> {
        Self::try_new(array).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "expected a string column (Utf8, LargeUtf8 or Utf8View), found {}",
                array.data_type()
            ))
        })
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Self::Utf8(a) => a.len(),
            Self::Large(a) => a.len(),
            Self::View(a) => a.len(),
        }
    }

    /// Whether the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether row `idx` is null.
    pub fn is_null(&self, idx: usize) -> bool {
        match self {
            Self::Utf8(a) => a.is_null(idx),
            Self::Large(a) => a.is_null(idx),
            Self::View(a) => a.is_null(idx),
        }
    }

    /// Value at `idx`. Panics if `idx` is out of bounds, as the Arrow
    /// accessors it delegates to do.
    pub fn value(&self, idx: usize) -> &'a str {
        match self {
            Self::Utf8(a) => a.value(idx),
            Self::Large(a) => a.value(idx),
            Self::View(a) => a.value(idx),
        }
    }

    /// Value at `idx`, or `None` when null.
    pub fn get(&self, idx: usize) -> Option<&'a str> {
        (!self.is_null(idx)).then(|| self.value(idx))
    }

    /// Iterate over every row.
    pub fn iter(&self) -> impl Iterator<Item = Option<&'a str>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;

    #[test]
    fn borrows_every_string_encoding() {
        let utf8 = StringArray::from(vec![Some("a"), None]);
        let large = LargeStringArray::from(vec![Some("a"), None]);
        let view = StringViewArray::from(vec![Some("a"), None]);

        for col in [
            StringColumn::new(&utf8).unwrap(),
            StringColumn::new(&large).unwrap(),
            StringColumn::new(&view).unwrap(),
        ] {
            assert_eq!(col.len(), 2);
            assert_eq!(col.get(0), Some("a"));
            assert!(col.is_null(1));
            assert_eq!(col.get(1), None);
            assert_eq!(col.iter().collect::<Vec<_>>(), vec![Some("a"), None]);
        }
    }

    #[test]
    fn try_new_declines_a_non_string_column() {
        let ints = Int64Array::from(vec![1, 2]);
        assert!(StringColumn::try_new(&ints).is_none());
    }

    #[test]
    fn new_names_the_offending_type() {
        let ints = Int64Array::from(vec![1, 2]);
        let err = StringColumn::new(&ints).unwrap_err().to_string();
        assert!(err.contains("Int64"), "unhelpful error: {err}");
    }
}
