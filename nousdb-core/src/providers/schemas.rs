//! The [`ModelOutput`] type returned by model providers.

use std::fmt;

/// A model's output for one request item.
#[derive(Debug, Clone)]
pub enum ModelOutput {
    /// Text generation output.
    Text(String),

    /// Floating-point number.
    Float(f64),

    /// Classification result.
    Classification {
        /// Predicted label.
        label: String,
        /// Confidence in the label.
        confidence: f64,
    },

    /// Boolean decision (for predicates).
    Boolean(bool),

    /// Embedding vector.
    Embedding(Vec<f32>),

    /// Structured output (JSON).
    Structured(serde_json::Value),

    /// The model answered `null`: an explicit abstention, distinct from the
    /// empty string. Operators that build a nullable Arrow column must turn
    /// this into a NULL, since decoding it as `""` would manufacture a value
    /// the model declined to give.
    Null,

    /// Multiple outputs.
    Multiple(Vec<ModelOutput>),
}

impl fmt::Display for ModelOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelOutput::Text(s) => write!(f, "{s}"),
            ModelOutput::Classification { label, confidence } => {
                write!(
                    f,
                    "Classification(label: {label}, confidence: {confidence:.2})"
                )
            }
            ModelOutput::Boolean(b) => write!(f, "{b}"),
            ModelOutput::Embedding(vec) => {
                write!(f, "Embedding(len={}, first={:?})", vec.len(), vec.first())
            }
            ModelOutput::Structured(json) => write!(f, "{}", json),
            ModelOutput::Null => write!(f, "null"),
            ModelOutput::Multiple(outputs) => {
                let joined = outputs
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "[{joined}]")
            }
            ModelOutput::Float(val) => write!(f, "{val}"),
        }
    }
}
