//! Offline preparation of the token histograms that [`HistogramCatalog`]
//! reads: one equi-depth [`TokenHistogram`] per string column of every CSV
//! file in a folder, written as `{output_dir}/{table}/{column}.json`.
//!
//! The pieces are exposed separately so that a progress-printing CLI and a
//! silent library caller ([`prepare_csv_folder`]) run the same reading,
//! column-selection, counting and serialization code and write identical
//! files.
//!
//! [`HistogramCatalog`]: super::catalog::HistogramCatalog

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::csv::ReaderBuilder;
use datafusion::arrow::csv::reader::Format;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};

use super::estimator::TokenEstimator;
use super::histogram::TokenHistogram;

/// Bucket count of every histogram written by this module.
/// `HistogramCatalog::get_combined` works at the same resolution.
pub const HISTOGRAM_BUCKETS: usize = 32;

/// Rows used to infer a CSV file's schema.
const SCHEMA_INFERENCE_ROWS: usize = 100;

/// One histogram written by [`prepare_csv_folder`].
#[derive(Debug, Clone)]
pub struct PreparedHistogram {
    /// Table name: the CSV file's stem, which is also the output sub-directory.
    pub table: String,
    /// Column name, which is also the output file's stem.
    pub column: String,
    /// Rows counted (`TokenHistogram::total_rows`), NULLs included.
    pub rows: usize,
    /// Mean tokens per row.
    pub mean: f64,
    /// 99th-percentile tokens per row.
    pub p99: f64,
    /// The `{output_dir}/{table}/{column}.json` file that was written.
    pub path: PathBuf,
}

/// What [`prepare_csv_folder`] did.
#[derive(Debug, Clone, Default)]
pub struct PreparationReport {
    /// Every histogram written, in CSV-path order and, within a file, in
    /// schema column order.
    pub histograms: Vec<PreparedHistogram>,
    /// Files that produced nothing, with the reason: unreadable files (the
    /// message says which step failed), files with no data rows, and files
    /// with no string column. A skipped file never fails the run.
    pub skipped: Vec<(PathBuf, String)>,
}

/// The `*.csv` files directly inside `folder`, sorted by path so a run is
/// deterministic (`read_dir` order is platform-dependent). The extension match
/// is exact and case-sensitive.
pub fn list_csv_files(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(folder).map_err(|e| DataFusionError::External(Box::new(e)))? {
        let path = entry
            .map_err(|e| DataFusionError::External(Box::new(e)))?
            .path();
        if path.extension().and_then(|e| e.to_str()) == Some("csv") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Reads a headed CSV file into batches.
///
/// The schema is inferred from the first 100 rows, then every field is made
/// nullable and rows shorter than the header are read with the missing cells as
/// NULL (`with_truncated_rows`), so ragged files do not abort the read. An
/// empty vector means the file has a header but no data rows.
///
/// Errors carry a short "failed to …" message naming the step that failed.
pub fn read_csv_for_histograms(path: &Path) -> Result<Vec<RecordBatch>> {
    let step = |what: &str, e: &dyn std::fmt::Display| {
        DataFusionError::Execution(format!("failed to {what}: {e}"))
    };

    let file = std::fs::File::open(path).map_err(|e| step("open", &e))?;
    let (inferred, _) = Format::default()
        .with_header(true)
        .infer_schema(std::io::BufReader::new(&file), Some(SCHEMA_INFERENCE_ROWS))
        .map_err(|e| step("infer the schema", &e))?;
    let schema = Arc::new(Schema::new(
        inferred
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), f.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));

    let file = std::fs::File::open(path).map_err(|e| step("reopen", &e))?;
    let reader = ReaderBuilder::new(schema)
        .with_header(true)
        .with_truncated_rows(true)
        .build(file)
        .map_err(|e| step("build the CSV reader", &e))?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| step("read batches", &e))
}

/// Names of the string columns of `schema`, in schema order: `Utf8`,
/// `LargeUtf8` and `Utf8View`. (The CSV reader only ever yields `Utf8`; the
/// other two are accepted because `TokenEstimator` counts every string
/// encoding.)
pub fn histogram_string_columns(schema: &Schema) -> Vec<String> {
    schema
        .fields()
        .iter()
        .filter(|f| {
            matches!(
                f.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
        })
        .map(|f| f.name().clone())
        .collect()
}

/// Token histogram of one column across all `batches`: per-row token counts
/// (NULL counts as 0 tokens) bucketed into [`HISTOGRAM_BUCKETS`] equi-depth
/// buckets.
pub fn column_histogram(
    estimator: &TokenEstimator,
    batches: &[RecordBatch],
    column: &str,
) -> Result<TokenHistogram> {
    let columns = [column.to_string()];
    let mut counts: Vec<usize> = Vec::new();
    for batch in batches {
        counts.extend(estimator.estimate_batch_tokens(batch, &columns)?);
    }
    Ok(TokenHistogram::from_token_counts(
        &counts,
        HISTOGRAM_BUCKETS,
    ))
}

/// Writes `histogram` as pretty-printed JSON to
/// `{output_dir}/{table}/{column}.json` (creating the table directory) and
/// returns that path. This is the layout `HistogramCatalog::load` reads.
pub fn write_histogram(
    output_dir: &Path,
    table: &str,
    column: &str,
    histogram: &TokenHistogram,
) -> Result<PathBuf> {
    let table_dir = output_dir.join(table);
    std::fs::create_dir_all(&table_dir).map_err(|e| DataFusionError::External(Box::new(e)))?;

    let json = serde_json::to_string_pretty(histogram)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let path = table_dir.join(format!("{column}.json"));
    std::fs::write(&path, json).map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(path)
}

/// Builds and writes a histogram for every string column of every `*.csv` file
/// in `csv_folder` (non-recursive), without printing.
///
/// Files are processed in path order. A file that cannot be read, has no data
/// rows or has no string column is recorded in [`PreparationReport::skipped`]
/// and the run continues; a failure to count tokens or to write an output file
/// is returned as an error. Histograms already written stay on disk in that
/// case. A folder with no CSV files yields an empty report.
pub fn prepare_csv_folder(csv_folder: &Path, output_dir: &Path) -> Result<PreparationReport> {
    let estimator = TokenEstimator::default_estimator()?;
    let mut report = PreparationReport::default();

    for csv_path in list_csv_files(csv_folder)? {
        let table = csv_path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let batches = match read_csv_for_histograms(&csv_path) {
            Ok(batches) => batches,
            Err(e) => {
                report.skipped.push((csv_path, e.message().into_owned()));
                continue;
            }
        };
        if batches.is_empty() {
            report.skipped.push((csv_path, "empty".to_string()));
            continue;
        }

        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let string_columns = histogram_string_columns(batches[0].schema().as_ref());
        if string_columns.is_empty() {
            report
                .skipped
                .push((csv_path, "no string columns".to_string()));
            continue;
        }

        for column in string_columns {
            let histogram = column_histogram(&estimator, &batches, &column)?;
            let path = write_histogram(output_dir, &table, &column, &histogram)?;
            report.histograms.push(PreparedHistogram {
                table: table.clone(),
                rows,
                mean: histogram.stats().mean,
                p99: histogram.quantile(0.99),
                column,
                path,
            });
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_estimator::catalog::HistogramCatalog;
    use datafusion::arrow::array::{Array, Int64Array, StringArray};
    use tempfile::TempDir;

    fn write_file(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).unwrap();
    }

    /// 2 string columns + 1 int column, 130 rows: a quoted field with commas and
    /// escaped quotes, then rows shorter than the header at rows 110 and 120.
    /// They sit past the 100-row schema-inference window; see
    /// `short_rows_inside_the_inference_window_make_the_file_unreadable`.
    fn people_csv() -> String {
        let mut csv = String::from("id,name,bio\n");
        for i in 0..130 {
            match i {
                0 => csv.push_str("0,alice,\"likes cats, dogs, and \"\"birds\"\"\"\n"),
                110 => csv.push_str("110,carol\n"),
                120 => csv.push_str("120\n"),
                _ => csv.push_str(&format!(
                    "{i},person {i},some words about person number {i}\n"
                )),
            }
        }
        csv
    }

    #[test]
    fn string_columns_are_detected_in_schema_order() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new("b", DataType::LargeUtf8, true),
            Field::new("c", DataType::Utf8View, true),
            Field::new("f", DataType::Float64, true),
        ]);
        assert_eq!(histogram_string_columns(&schema), vec!["a", "b", "c"]);
    }

    #[test]
    fn csv_reading_infers_types_and_tolerates_truncated_rows() {
        let dir = TempDir::new().unwrap();
        write_file(dir.path(), "people.csv", &people_csv());

        let batches = read_csv_for_histograms(&dir.path().join("people.csv")).unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 130);
        let batch = &batches[0];
        let schema = batch.schema();
        assert!(schema.fields().iter().all(|f| f.is_nullable()));
        assert_eq!(
            histogram_string_columns(schema.as_ref()),
            vec!["name", "bio"]
        );
        assert_eq!(
            schema.field_with_name("id").unwrap().data_type(),
            &DataType::Int64
        );

        let strings = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone()
        };
        let (name, bio) = (strings("name"), strings("bio"));
        // The quoted cell keeps its commas and unescaped quotes.
        assert_eq!(bio.value(0), "likes cats, dogs, and \"birds\"");
        // "110,carol" has no bio; "120" has neither string cell.
        assert_eq!(name.value(110), "carol");
        assert!(bio.is_null(110));
        assert!(name.is_null(120) && bio.is_null(120));
        assert!(!bio.is_null(121));
        let id = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.value(120), 120);
    }

    /// Schema inference does not enable `with_truncated_rows`, so a short row
    /// among the first 100 makes the whole file unreadable and
    /// `prepare_csv_folder` skips it.
    #[test]
    fn short_rows_inside_the_inference_window_make_the_file_unreadable() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in");
        std::fs::create_dir_all(&input).unwrap();
        write_file(&input, "early.csv", "t,u\nhello,1\nworld\nagain,3\n");

        let err = read_csv_for_histograms(&input.join("early.csv")).unwrap_err();
        assert!(
            err.message().starts_with("failed to infer the schema: "),
            "{err}"
        );

        let report = prepare_csv_folder(&input, &dir.path().join("out")).unwrap();
        assert!(report.histograms.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(
            report.skipped[0]
                .1
                .starts_with("failed to infer the schema: ")
        );
    }

    #[test]
    fn header_only_file_reads_as_no_batches_and_missing_file_errors() {
        let dir = TempDir::new().unwrap();
        write_file(dir.path(), "empty.csv", "a,b\n");
        assert!(
            read_csv_for_histograms(&dir.path().join("empty.csv"))
                .unwrap()
                .is_empty()
        );

        let err = read_csv_for_histograms(&dir.path().join("missing.csv")).unwrap_err();
        assert!(err.message().starts_with("failed to open: "), "{err}");
    }

    #[test]
    fn column_histogram_counts_every_row_including_null_cells() {
        let dir = TempDir::new().unwrap();
        write_file(dir.path(), "people.csv", &people_csv());
        let batches = read_csv_for_histograms(&dir.path().join("people.csv")).unwrap();
        let estimator = TokenEstimator::default_estimator().unwrap();

        let bio = column_histogram(&estimator, &batches, "bio").unwrap();
        assert_eq!(bio.total_rows(), 130);
        assert_eq!(bio.requested_bucket_count(), HISTOGRAM_BUCKETS);
        assert!(bio.stats().mean > 0.0);

        assert!(column_histogram(&estimator, &batches, "no_such_column").is_err());
    }

    #[test]
    fn written_histogram_round_trips_through_the_catalog() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in");
        let output = dir.path().join("out");
        std::fs::create_dir_all(&input).unwrap();
        write_file(&input, "people.csv", &people_csv());
        write_file(&input, "ignored.txt", "name\nnot a csv\n");

        let report = prepare_csv_folder(&input, &output).unwrap();
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        let columns: Vec<_> = report
            .histograms
            .iter()
            .map(|h| h.column.as_str())
            .collect();
        assert_eq!(columns, vec!["name", "bio"]);

        let name = &report.histograms[0];
        assert_eq!(name.table, "people");
        assert_eq!(name.rows, 130);
        assert_eq!(name.path, output.join("people").join("name.json"));
        assert!(name.path.is_file());

        let catalog = HistogramCatalog::load(&output).unwrap();
        for prepared in &report.histograms {
            let loaded = catalog
                .get("people", &prepared.column)
                .unwrap_or_else(|| panic!("people.{} missing from the catalog", prepared.column));
            assert_eq!(loaded.total_rows(), prepared.rows);
            assert_eq!(loaded.stats().mean, prepared.mean);
            assert_eq!(loaded.quantile(0.99), prepared.p99);
        }

        // Re-running overwrites with identical bytes.
        let before = std::fs::read(&name.path).unwrap();
        prepare_csv_folder(&input, &output).unwrap();
        assert_eq!(std::fs::read(&name.path).unwrap(), before);
    }

    #[test]
    fn files_without_string_columns_produce_no_histograms() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in");
        let output = dir.path().join("out");
        std::fs::create_dir_all(&input).unwrap();
        write_file(&input, "numbers.csv", "a,b,c\n1,2,0.5\n3,4,1.5\n");
        write_file(&input, "empty.csv", "x,y\n");

        let report = prepare_csv_folder(&input, &output).unwrap();
        assert!(report.histograms.is_empty());
        let skipped: Vec<_> = report
            .skipped
            .iter()
            .map(|(p, why)| (p.file_name().unwrap().to_str().unwrap(), why.as_str()))
            .collect();
        assert_eq!(
            skipped,
            vec![("empty.csv", "empty"), ("numbers.csv", "no string columns")]
        );
        assert!(
            !output.exists(),
            "nothing should be written for a skipped file"
        );
        assert!(
            HistogramCatalog::load(&output)
                .unwrap()
                .get("numbers", "a")
                .is_none()
        );
    }

    #[test]
    fn csv_files_are_listed_in_path_order_and_an_unreadable_file_is_skipped() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in");
        std::fs::create_dir_all(&input).unwrap();
        write_file(&input, "b.csv", "t\nhello\n");
        write_file(&input, "a.csv", "t\nworld\n");
        write_file(&input, "c.CSV", "t\nignored\n");
        // A directory named like a CSV opens but cannot be read.
        std::fs::create_dir_all(input.join("d.csv")).unwrap();

        let names: Vec<_> = list_csv_files(&input)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["a.csv", "b.csv", "d.csv"]);

        let report = prepare_csv_folder(&input, &dir.path().join("out")).unwrap();
        let tables: Vec<_> = report.histograms.iter().map(|h| h.table.as_str()).collect();
        assert_eq!(tables, vec!["a", "b"]);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].0.ends_with("d.csv"));

        assert!(list_csv_files(&input.join("missing")).is_err());
    }
}
