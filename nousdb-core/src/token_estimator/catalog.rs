//! Catalog of pre-computed token histograms.

use std::collections::HashMap;
use std::path::Path;

use datafusion::error::{DataFusionError, Result};

use super::histogram::TokenHistogram;

/// On-disk catalog of pre-computed per-table, per-column token histograms.
///
/// On-disk format:
/// ```text
/// histograms/
///   {table_name}/
///     {column_name}.json   # serialized TokenHistogram
/// ```
///
/// In memory it is a map keyed by table name, then column name.
///
/// A table may also be looked up through an alias (see [`Self::add_alias`]),
/// for example a registered table `orders` resolving to the histogram
/// directory `orders_1000`.
#[derive(Debug, Clone)]
pub struct HistogramCatalog {
    histograms: HashMap<String, HashMap<String, TokenHistogram>>,
    /// Registered table name to histogram directory name.
    aliases: HashMap<String, String>,
}

impl HistogramCatalog {
    /// Loads a catalog from a directory.
    ///
    /// Scans `dir/{table}/` subdirectories and loads `{column}.json` files.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut histograms: HashMap<String, HashMap<String, TokenHistogram>> = HashMap::new();

        if !dir.exists() {
            return Ok(Self {
                histograms,
                aliases: HashMap::new(),
            });
        }

        let entries = std::fs::read_dir(dir).map_err(|e| DataFusionError::External(Box::new(e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| DataFusionError::External(Box::new(e)))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let table_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    DataFusionError::Internal("Invalid directory name in histogram catalog".into())
                })?
                .to_string();

            let table_entries =
                std::fs::read_dir(&path).map_err(|e| DataFusionError::External(Box::new(e)))?;

            let mut columns = HashMap::new();

            for col_entry in table_entries {
                let col_entry = col_entry.map_err(|e| DataFusionError::External(Box::new(e)))?;
                let col_path = col_entry.path();

                if col_path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }

                let column_name = col_path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| {
                        DataFusionError::Internal("Invalid file name in histogram catalog".into())
                    })?
                    .to_string();

                let data = std::fs::read_to_string(&col_path)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                let histogram: TokenHistogram = serde_json::from_str(&data)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                log::debug!(
                    "[HistogramCatalog] Loaded histogram for {}.{} ({} rows, {} buckets)",
                    table_name,
                    column_name,
                    histogram.total_rows(),
                    histogram.buckets().len(),
                );

                columns.insert(column_name, histogram);
            }

            if !columns.is_empty() {
                log::debug!(
                    "[HistogramCatalog] Table '{}': {} column histograms loaded",
                    table_name,
                    columns.len(),
                );
                histograms.insert(table_name, columns);
            }
        }

        Ok(Self {
            histograms,
            aliases: HashMap::new(),
        })
    }

    /// Registers an alias so lookups for `alias` resolve to the histogram
    /// directory `target`. Exact table names always win over aliases.
    ///
    /// Histogram directories are keyed by their on-disk names, which may carry
    /// a scale suffix (`orders_1000`) while the engine registers the table
    /// under its logical name (`orders`).
    pub fn add_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        let (alias, target) = (alias.into(), target.into());
        if !self.histograms.contains_key(&target) {
            log::warn!(
                "[HistogramCatalog] alias '{alias}' points at unknown histogram table '{target}'"
            );
        }
        self.aliases.insert(alias, target);
    }

    /// Resolves a lookup name through the alias map (exact names win).
    fn resolve<'a>(&'a self, table: &'a str) -> &'a str {
        if self.histograms.contains_key(table) {
            return table;
        }
        self.aliases.get(table).map(String::as_str).unwrap_or(table)
    }

    /// Looks up the histogram for a table and column.
    pub fn get(&self, table: &str, column: &str) -> Option<&TokenHistogram> {
        self.histograms.get(self.resolve(table))?.get(column)
    }

    /// Combines the histograms of several columns of a table.
    ///
    /// Reconstructs approximate per-row token counts from each column's
    /// histogram (each bucket contributes `tuple_count` values at its
    /// `average_token_count`), sums them, and rebuilds a combined histogram.
    pub fn get_combined(&self, table: &str, columns: &[String]) -> Option<TokenHistogram> {
        let table_histograms = self.histograms.get(self.resolve(table))?;

        // Collect the histograms available for the requested columns.
        let mut found: Vec<&TokenHistogram> = Vec::new();
        for col in columns {
            if let Some(h) = table_histograms.get(col) {
                found.push(h);
            }
        }

        if found.is_empty() {
            return None;
        }

        // A single column needs no combining.
        if found.len() == 1 {
            return Some(found[0].clone());
        }

        // Exact per-row data is not available, so reconstruct approximate
        // counts (for each bucket, `tuple_count` values at its
        // `average_token_count`) and sum corresponding positions across
        // columns.
        let reconstruct = |h: &TokenHistogram| -> Vec<usize> {
            let mut counts = Vec::with_capacity(h.total_rows());
            for bucket in h.buckets() {
                for _ in 0..bucket.tuple_count {
                    counts.push(bucket.average_token_count.round() as usize);
                }
            }
            counts
        };

        let first_counts = reconstruct(found[0]);
        let total_rows = first_counts.len();
        let mut combined = first_counts;

        for h in &found[1..] {
            let counts = reconstruct(h);
            let len = counts.len().min(total_rows);
            for i in 0..len {
                combined[i] += counts[i];
            }
        }

        Some(TokenHistogram::from_token_counts(&combined, 32))
    }

    /// Lists the table names in the catalog.
    pub fn tables(&self) -> impl Iterator<Item = &str> {
        self.histograms.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_histogram(counts: &[usize]) -> TokenHistogram {
        TokenHistogram::from_token_counts(counts, 32)
    }

    #[test]
    fn test_load_empty_dir() {
        let dir = TempDir::new().unwrap();
        let catalog = HistogramCatalog::load(dir.path()).unwrap();
        assert_eq!(catalog.tables().count(), 0);
    }

    #[test]
    fn test_load_nonexistent_dir() {
        let catalog = HistogramCatalog::load(Path::new("/nonexistent/path")).unwrap();
        assert_eq!(catalog.tables().count(), 0);
    }

    #[test]
    fn test_roundtrip_serde() {
        let dir = TempDir::new().unwrap();
        let table_dir = dir.path().join("reviews");
        fs::create_dir(&table_dir).unwrap();

        let counts = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let hist = make_histogram(&counts);
        let json = serde_json::to_string_pretty(&hist).unwrap();
        fs::write(table_dir.join("text.json"), &json).unwrap();

        let catalog = HistogramCatalog::load(dir.path()).unwrap();
        let loaded = catalog.get("reviews", "text").unwrap();
        assert_eq!(loaded.total_rows(), hist.total_rows());
        assert_eq!(loaded.buckets().len(), hist.buckets().len());
    }

    #[test]
    fn test_get_combined_single_column() {
        let dir = TempDir::new().unwrap();
        let table_dir = dir.path().join("reviews");
        fs::create_dir(&table_dir).unwrap();

        let hist = make_histogram(&[10, 20, 30]);
        fs::write(
            table_dir.join("text.json"),
            serde_json::to_string(&hist).unwrap(),
        )
        .unwrap();

        let catalog = HistogramCatalog::load(dir.path()).unwrap();
        let combined = catalog
            .get_combined("reviews", &["text".to_string()])
            .unwrap();
        assert_eq!(combined.total_rows(), 3);
    }

    /// A registered table name may differ from the scale-suffixed histogram
    /// directory (`join_reviews` vs `join_reviews_1000`); an alias bridges it,
    /// and exact directory names win over aliases.
    #[test]
    fn test_alias_resolves_scaled_directory() {
        let dir = TempDir::new().unwrap();
        let table_dir = dir.path().join("join_reviews_1000");
        fs::create_dir(&table_dir).unwrap();
        let hist = make_histogram(&[100, 200, 300]);
        fs::write(
            table_dir.join("review.json"),
            serde_json::to_string(&hist).unwrap(),
        )
        .unwrap();

        let mut catalog = HistogramCatalog::load(dir.path()).unwrap();
        assert!(catalog.get("join_reviews", "review").is_none());

        catalog.add_alias("join_reviews", "join_reviews_1000");
        assert!(catalog.get("join_reviews", "review").is_some());
        assert!(
            catalog
                .get_combined("join_reviews", &["review".to_string()])
                .is_some()
        );
        // The real directory name still resolves.
        assert!(catalog.get("join_reviews_1000", "review").is_some());
    }

    #[test]
    fn test_get_combined_multiple_columns() {
        let dir = TempDir::new().unwrap();
        let table_dir = dir.path().join("reviews");
        fs::create_dir(&table_dir).unwrap();

        let hist_a = make_histogram(&[10, 20, 30, 40]);
        let hist_b = make_histogram(&[5, 15, 25, 35]);
        fs::write(
            table_dir.join("title.json"),
            serde_json::to_string(&hist_a).unwrap(),
        )
        .unwrap();
        fs::write(
            table_dir.join("body.json"),
            serde_json::to_string(&hist_b).unwrap(),
        )
        .unwrap();

        let catalog = HistogramCatalog::load(dir.path()).unwrap();
        let combined = catalog
            .get_combined("reviews", &["title".to_string(), "body".to_string()])
            .unwrap();
        assert_eq!(combined.total_rows(), 4);
        // Combined stats should have higher mean than either individual
        let stats = combined.stats();
        assert!(stats.mean > 0.0);
    }
}
