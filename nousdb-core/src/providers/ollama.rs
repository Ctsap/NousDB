//! [`OllamaProvider`]: a local Ollama server through its native API.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::{FormatType, JsonStructure};
use ollama_rs::models::{LocalModel, ModelOptions};

use crate::prompt::prompt::Prompt;
use crate::prompt::renderer::BatchRenderer;
use crate::providers::error::ProviderError;
use crate::providers::metadata::{ModelCard, ModelCatalog};
use crate::providers::metrics::PredictResult;
use crate::providers::pipeline::{Completion, Request};
use crate::providers::provider::{AdmissionControl, ModelProvider};

/// Where Ollama listens by default.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Context window NousDB asks Ollama for when no card says otherwise.
///
/// Ollama allocates the context per loaded model, so asking for a model's
/// full trained length (often 128k tokens) can exhaust memory. Each request
/// sends the card's `context_window` as `num_ctx`, so the card and the
/// server agree.
pub const DEFAULT_OLLAMA_CONTEXT_WINDOW: usize = 8192;

/// Ollama sets no batch limit; the context window bounds the batch instead.
const OLLAMA_MAX_BATCH_SIZE: usize = 4096;

/// Seed for every request, so repeated calls give the same answer.
const SEED: i32 = 1337;

/// A provider for a local Ollama server.
///
/// On [`build`](OllamaBuilder::build) it lists the installed models and
/// gives each one a card, using the model's trained context length capped at
/// [`DEFAULT_OLLAMA_CONTEXT_WINDOW`]. Cards you add with
/// [`OllamaBuilder::model`] replace those.
///
/// ```no_run
/// # async fn example() -> Result<(), nousdb_core::providers::ProviderError> {
/// use nousdb_core::OllamaProvider;
///
/// let ollama = OllamaProvider::builder().build().await?;
/// # Ok(()) }
/// ```
pub struct OllamaProvider {
    name: String,
    url: String,
    client: Ollama,
    catalog: ModelCatalog,
    admission: AdmissionControl,
}

impl fmt::Debug for OllamaProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OllamaProvider")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("models", &self.catalog.len())
            .field("max_concurrent", &self.admission.max_concurrent())
            .finish_non_exhaustive()
    }
}

impl OllamaProvider {
    /// Starts configuring a provider for the server at [`DEFAULT_OLLAMA_URL`].
    pub fn builder() -> OllamaBuilder {
        OllamaBuilder {
            name: "ollama".into(),
            url: DEFAULT_OLLAMA_URL.into(),
            max_concurrent: 4,
            discover: true,
            catalog: ModelCatalog::new(),
        }
    }

    /// The models this provider serves.
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    async fn installed_models(client: &Ollama, url: &str) -> Result<Vec<ModelCard>, ProviderError> {
        let models = client
            .list_local_models()
            .await
            .map_err(|e| ProviderError::Connect {
                endpoint: url.to_string(),
                message: e.to_string(),
            })?;
        let mut cards = Vec::with_capacity(models.len());
        for model in models {
            let trained = Self::trained_context_length(client, &model).await;
            let context_window = trained.map_or(DEFAULT_OLLAMA_CONTEXT_WINDOW, |n| {
                n.min(DEFAULT_OLLAMA_CONTEXT_WINDOW)
            });
            cards.push(
                ModelCard::new(model.name, context_window)
                    .with_max_batch_size(OLLAMA_MAX_BATCH_SIZE),
            );
        }
        Ok(cards)
    }

    /// The `<arch>.context_length` that `/api/show` reports, if any.
    async fn trained_context_length(client: &Ollama, model: &LocalModel) -> Option<usize> {
        let info = client.show_model_info(model.name.clone()).await.ok()?;
        info.model_info
            .iter()
            .find(|(key, _)| key.ends_with(".context_length"))
            .and_then(|(_, value)| value.as_u64())
            .map(|n| n as usize)
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelCard> {
        self.catalog.cards().to_vec()
    }

    fn model(&self, model_id: &str) -> Option<ModelCard> {
        self.catalog.get(model_id).cloned()
    }

    async fn predict(
        &self,
        model_id: &str,
        prompt: &Prompt,
        batch: &RecordBatch,
        start_row: usize,
        batch_size: usize,
        renderer: &(dyn BatchRenderer + Send + Sync),
    ) -> Result<PredictResult> {
        let schema = self.construct_output_schema(prompt, batch_size);
        let request = Request::render(
            model_id, prompt, batch, start_row, batch_size, renderer, &schema,
        )?;

        let mut options = ModelOptions::default().temperature(0.0).seed(SEED);
        if let Some(card) = self.catalog.get(model_id) {
            options = options.num_ctx(card.context_window as u64);
        }
        let schema = schemars::Schema::try_from(request.schema.clone())
            .map_err(|e| DataFusionError::Execution(format!("invalid output schema: {e}")))?;
        let mut generation = GenerationRequest::new(model_id.to_string(), request.user.clone())
            .format(FormatType::StructuredJson(Box::new(
                JsonStructure::new_for_schema(schema),
            )))
            .options(options);
        if let Some(system) = &request.system {
            generation = generation.system(system.clone());
        }

        let start = Instant::now();
        let response = match self.client.generate(generation).await {
            Ok(response) => response,
            Err(e) => return Ok(request.transport_failure(start.elapsed(), e)),
        };
        let elapsed = start.elapsed();
        let completion = Completion {
            text: response.response,
            input_tokens: response.prompt_eval_count.unwrap_or(0),
            output_tokens: response.eval_count.unwrap_or(0),
            reasoning_tokens: 0,
            truncated: false,
        };
        Ok(request.complete(completion, elapsed))
    }

    fn admission_control(&self) -> Option<AdmissionControl> {
        Some(self.admission.clone())
    }
}

/// Configures an [`OllamaProvider`].
#[derive(Debug)]
#[must_use]
pub struct OllamaBuilder {
    name: String,
    url: String,
    max_concurrent: usize,
    discover: bool,
    catalog: ModelCatalog,
}

impl OllamaBuilder {
    /// Renames the provider. Defaults to `"ollama"`.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The server URL. Defaults to [`DEFAULT_OLLAMA_URL`].
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Cap on requests in flight to the server. Defaults to 4; raise it
    /// together with Ollama's `OLLAMA_NUM_PARALLEL`.
    pub fn max_concurrent(mut self, requests: usize) -> Self {
        self.max_concurrent = requests;
        self
    }

    /// Whether to list the installed models when building. Defaults to
    /// `true`; with `false` the provider serves only the models you add.
    pub fn discover_models(mut self, discover: bool) -> Self {
        self.discover = discover;
        self
    }

    /// Adds a model, replacing the card derived from the server. The id must
    /// match the Ollama model name, tag included (`llama3.1:latest`).
    pub fn model(mut self, card: ModelCard) -> Self {
        self.catalog = self.catalog.with_card(card);
        self
    }

    /// Adds several models, as [`model`](Self::model) does.
    pub fn models(mut self, cards: impl IntoIterator<Item = ModelCard>) -> Self {
        self.catalog = self.catalog.with_cards(cards);
        self
    }

    /// Connects to the server and builds the catalog.
    pub async fn build(self) -> Result<Arc<OllamaProvider>, ProviderError> {
        let client = Ollama::try_new(self.url.as_str())
            .map_err(|e| ProviderError::Config(format!("invalid Ollama URL {}: {e}", self.url)))?;
        let mut catalog = ModelCatalog::new();
        if self.discover {
            let installed = OllamaProvider::installed_models(&client, &self.url).await?;
            for card in self.catalog.cards() {
                if !installed.iter().any(|m| m.id == card.id) {
                    log::warn!(
                        "ollama: model {} is not installed at {}; requests to it will fail",
                        card.id,
                        self.url
                    );
                }
            }
            catalog = catalog.with_cards(installed);
        }
        Ok(Arc::new(OllamaProvider {
            name: self.name,
            url: self.url,
            client,
            catalog: catalog.merge(self.catalog),
            admission: AdmissionControl::new(self.max_concurrent),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn server_with(installed: &[(&str, Option<u64>)]) -> MockServer {
        let server = MockServer::start().await;
        let models: Vec<serde_json::Value> = installed
            .iter()
            .map(|(name, _)| {
                serde_json::json!({"name": name, "modified_at": "2024-01-01T00:00:00Z", "size": 1})
            })
            .collect();
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "models": models })),
            )
            .mount(&server)
            .await;
        for (name, context_length) in installed {
            let info = match context_length {
                Some(n) => serde_json::json!({ "llama.context_length": n }),
                None => serde_json::json!({}),
            };
            Mock::given(method("POST"))
                .and(path("/api/show"))
                .and(body_partial_json(serde_json::json!({ "name": name })))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({ "model_info": info })),
                )
                .mount(&server)
                .await;
        }
        server
    }

    async fn provider(server: &MockServer, cards: Vec<ModelCard>) -> Arc<OllamaProvider> {
        OllamaProvider::builder()
            .url(server.uri())
            .models(cards)
            .build()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn the_context_window_comes_from_the_server_capped_at_the_default() {
        let server =
            server_with(&[("big:latest", Some(131_072)), ("small:latest", Some(2048))]).await;
        let p = provider(&server, vec![]).await;
        assert_eq!(
            p.model("big:latest").unwrap().context_window,
            DEFAULT_OLLAMA_CONTEXT_WINDOW
        );
        assert_eq!(p.model("small:latest").unwrap().context_window, 2048);
    }

    #[tokio::test]
    async fn an_unreported_context_length_falls_back_to_the_default() {
        let server = server_with(&[("m:latest", None)]).await;
        let p = provider(&server, vec![]).await;
        assert_eq!(
            p.model("m:latest").unwrap().context_window,
            DEFAULT_OLLAMA_CONTEXT_WINDOW
        );
    }

    #[tokio::test]
    async fn a_supplied_card_replaces_the_derived_one() {
        let server =
            server_with(&[("llama3.1:latest", Some(131_072)), ("mistral:latest", None)]).await;
        let card = ModelCard::new("llama3.1:latest", 32_768)
            .with_max_output_tokens(2048)
            .with_max_concurrent_requests(4);
        let p = provider(&server, vec![card]).await;

        let ids: Vec<String> = p.models().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, ["llama3.1:latest", "mistral:latest"]);
        let llama = p.model("llama3.1:latest").unwrap();
        assert_eq!(llama.context_window, 32_768);
        assert_eq!(llama.max_output_tokens, Some(2048));
        assert_eq!(llama.max_concurrent_requests, Some(4));
    }

    #[tokio::test]
    async fn a_card_for_a_missing_model_is_still_added() {
        let server = server_with(&[("llama3.1:latest", None)]).await;
        let p = provider(&server, vec![ModelCard::new("not-pulled:7b", 4096)]).await;
        assert_eq!(p.model("not-pulled:7b").unwrap().context_window, 4096);
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_connect_error() {
        let err = OllamaProvider::builder()
            .url("http://127.0.0.1:9")
            .build()
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Connect { .. }));
    }
}
