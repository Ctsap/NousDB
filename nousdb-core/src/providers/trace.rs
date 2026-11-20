//! Per-request trace of packed LLM calls, for offline analysis.
//!
//! Off unless `NOUSDB_TRACE_DIR` is set. When it is, every packed request sent
//! through a [`ModelProvider`](super::ModelProvider) appends one JSON line to
//! `<dir>/trace_<tag>_<pid>_<unix_ms>.jsonl`: the rows in the batch, the
//! rendered prompts, the raw response and the parsed outputs. `NOUSDB_TRACE_TAG`
//! (default `run`) names the file and is copied into every line.
//!
//! The phase tells production requests apart from certification probe
//! requests, whose answers are scored and then discarded. It is process-wide
//! state set by [`PhaseGuard`], which is sufficient for single-query traces
//! because calibration runs before the production requests of the operator it
//! calibrates.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use serde_json::{Map, Value, json};

use super::schemas::ModelOutput;

struct TraceSink {
    tag: String,
    file: Mutex<File>,
}

static SINK: OnceLock<Option<TraceSink>> = OnceLock::new();
static PHASE: Mutex<&'static str> = Mutex::new("production");

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn sink() -> Option<&'static TraceSink> {
    SINK.get_or_init(|| {
        let dir = std::env::var_os("NOUSDB_TRACE_DIR")?;
        let tag = std::env::var("NOUSDB_TRACE_TAG").unwrap_or_else(|_| "run".to_string());
        let dir = PathBuf::from(dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!("trace: cannot create {}: {e}", dir.display());
            return None;
        }
        let path = dir.join(format!(
            "trace_{tag}_{}_{}.jsonl",
            std::process::id(),
            now_ms()
        ));
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => {
                log::info!("trace: writing per-request trace to {}", path.display());
                Some(TraceSink {
                    tag,
                    file: Mutex::new(file),
                })
            }
            Err(e) => {
                log::warn!("trace: cannot open {}: {e}", path.display());
                None
            }
        }
    })
    .as_ref()
}

/// Whether tracing is on. Cheap after the first call.
pub fn enabled() -> bool {
    sink().is_some()
}

/// The current request phase.
pub fn phase() -> &'static str {
    *PHASE.lock().unwrap_or_else(|p| p.into_inner())
}

/// Sets the phase for its lifetime and restores the previous one on drop.
pub struct PhaseGuard {
    previous: &'static str,
}

impl PhaseGuard {
    /// Sets the phase to `phase` until the returned guard is dropped.
    pub fn enter(phase: &'static str) -> Self {
        let mut slot = PHASE.lock().unwrap_or_else(|p| p.into_inner());
        let previous = *slot;
        *slot = phase;
        Self { previous }
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        *PHASE.lock().unwrap_or_else(|p| p.into_inner()) = self.previous;
    }
}

/// Rows `start..start+len` of `batch` as JSON objects, every value rendered as
/// its display string (NULL as JSON null).
pub fn rows_as_json(batch: &RecordBatch, start: usize, len: usize) -> Vec<Value> {
    let end = (start + len).min(batch.num_rows());
    let options = FormatOptions::default();
    let schema = batch.schema();
    let formatters: Vec<_> = batch
        .columns()
        .iter()
        .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).ok())
        .collect();
    (start..end)
        .map(|r| {
            let mut obj = Map::new();
            for (i, field) in schema.fields().iter().enumerate() {
                let value = if batch.column(i).is_null(r) {
                    Value::Null
                } else {
                    match &formatters[i] {
                        Some(f) => Value::String(f.value(r).to_string()),
                        None => Value::Null,
                    }
                };
                obj.insert(field.name().clone(), value);
            }
            Value::Object(obj)
        })
        .collect()
}

/// A parsed model output as plain JSON.
pub fn output_as_json(output: &ModelOutput) -> Value {
    match output {
        ModelOutput::Text(s) => json!(s),
        ModelOutput::Float(f) => json!(f),
        ModelOutput::Classification { label, confidence } => {
            json!({ "label": label, "confidence": confidence })
        }
        ModelOutput::Boolean(b) => json!(b),
        ModelOutput::Embedding(v) => json!({ "embedding_len": v.len() }),
        ModelOutput::Structured(v) => v.clone(),
        ModelOutput::Null => Value::Null,
        ModelOutput::Multiple(v) => Value::Array(v.iter().map(output_as_json).collect()),
    }
}

/// Appends one record. `fields` is merged under the common header
/// (`ts_ms`, `tag`, `phase`). A no-op when tracing is off.
pub fn record(fields: Map<String, Value>) {
    let Some(sink) = sink() else { return };
    let mut line = Map::new();
    line.insert("ts_ms".into(), json!(now_ms() as u64));
    line.insert("tag".into(), json!(sink.tag));
    line.insert("phase".into(), json!(phase()));
    line.extend(fields);
    let mut text = Value::Object(line).to_string();
    text.push('\n');
    let mut file = sink.file.lock().unwrap_or_else(|p| p.into_inner());
    if let Err(e) = file.write_all(text.as_bytes()) {
        log::warn!("trace: write failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn rows_and_outputs_serialize() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();
        let rows = rows_as_json(&batch, 1, 5);
        assert_eq!(
            rows,
            vec![
                json!({"id": "2", "text": null}),
                json!({"id": "3", "text": "c"})
            ]
        );

        let out = ModelOutput::Multiple(vec![ModelOutput::Boolean(true), ModelOutput::Null]);
        assert_eq!(output_as_json(&out), json!([true, null]));
    }

    #[test]
    fn phase_guard_restores() {
        assert_eq!(phase(), "production");
        {
            let _g = PhaseGuard::enter("calibration");
            assert_eq!(phase(), "calibration");
        }
        assert_eq!(phase(), "production");
    }
}
