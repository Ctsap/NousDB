//! [`OpenAiCompatibleProvider`]: any endpoint that speaks the OpenAI chat
//! completions API (vLLM, SGLang, a LiteLLM proxy, OpenAI, Ollama's `/v1`),
//! and Azure OpenAI deployments.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use litellm_rs::core::completion::{
    convert_from_chat_completion_response, convert_to_chat_completion_request,
};
#[cfg(feature = "azure")]
use litellm_rs::core::providers::azure::{AzureConfig, AzureOpenAIProvider};
use litellm_rs::core::providers::base::BaseConfig;
use litellm_rs::core::providers::openai_like::{OpenAILikeConfig, OpenAILikeProvider};
use litellm_rs::core::traits::provider::LLMProvider;
use litellm_rs::core::types::chat::ChatRequest;
use litellm_rs::core::types::responses::{ChatResponse, FinishReason};
use litellm_rs::core::types::tools::ResponseFormat;
use litellm_rs::{
    CompletionOptions, CompletionResponse, ContentPart, Message, MessageContent, RequestContext,
    system_message, user_message,
};
use tokio::sync::OnceCell;

use crate::prompt::prompt::Prompt;
use crate::prompt::renderer::BatchRenderer;
use crate::providers::error::ProviderError;
use crate::providers::metadata::{ModelCard, ModelCatalog};
use crate::providers::metrics::PredictResult;
use crate::providers::pipeline::{Completion, Request};
use crate::providers::provider::{AdmissionControl, ModelProvider};
use crate::providers::secret::ApiKey;
use crate::providers::utils;

/// Cap on concurrent requests to one endpoint unless configured otherwise.
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 128;

/// Azure OpenAI API version used unless configured otherwise.
pub const DEFAULT_AZURE_API_VERSION: &str = "2024-10-21";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Which wire dialect an [`OpenAiCompatibleProvider`] speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProviderKind {
    /// The OpenAI chat completions API at a base URL ending in `/v1`.
    OpenAiCompatible,
    /// An Azure OpenAI resource. The model id is the deployment name.
    ///
    /// Azure differs in three ways the provider handles: the JSON schema must
    /// have an object at its root, the output budget cannot be sent (GPT-5
    /// deployments reject `max_tokens`), and `chat_template_kwargs` is not
    /// accepted.
    Azure,
}

/// A provider for OpenAI-compatible endpoints and Azure OpenAI.
///
/// Every request constrains the answer with a JSON schema
/// (`response_format: json_schema`, strict), so the endpoint must support
/// structured output. vLLM, SGLang, OpenAI, Azure and Ollama all do.
///
/// ```no_run
/// # async fn example() -> Result<(), nousdb_core::providers::ProviderError> {
/// use nousdb_core::{ModelCard, OpenAiCompatibleProvider};
///
/// let vllm = OpenAiCompatibleProvider::builder("vllm")
///     .base_url("http://localhost:8000/v1")
///     .model(ModelCard::new("my-org/my-model", 32_768).with_max_output_tokens(4096))
///     .build()
///     .await?;
/// # Ok(()) }
/// ```
pub struct OpenAiCompatibleProvider {
    name: String,
    kind: ProviderKind,
    base_url: Option<String>,
    api_key: Option<ApiKey>,
    api_version: String,
    timeout: Duration,
    max_retries: u32,
    temperature: f32,
    enable_thinking: bool,
    catalog: ModelCatalog,
    admission: AdmissionControl,
    backend: OnceCell<Backend>,
}

impl fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key)
            .field("api_version", &self.api_version)
            .field("models", &self.catalog.len())
            .field("max_concurrent", &self.admission.max_concurrent())
            .field("enable_thinking", &self.enable_thinking)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatibleProvider {
    /// Starts configuring a provider for an OpenAI-compatible endpoint.
    /// `name` identifies it in the [`ModelRegistry`](super::ModelRegistry).
    pub fn builder(name: impl Into<String>) -> OpenAiCompatibleBuilder {
        OpenAiCompatibleBuilder::new(name.into(), ProviderKind::OpenAiCompatible)
    }

    /// Starts configuring a provider for an Azure OpenAI resource, e.g.
    /// `https://<resource>.openai.azure.com`. Name it with
    /// [`OpenAiCompatibleBuilder::name`]; it defaults to `"azure"`.
    #[cfg(feature = "azure")]
    pub fn azure(endpoint: impl Into<String>) -> OpenAiCompatibleBuilder {
        OpenAiCompatibleBuilder::new("azure".into(), ProviderKind::Azure).base_url(endpoint)
    }

    /// The wire dialect.
    pub fn kind(&self) -> ProviderKind {
        self.kind
    }

    /// The models this provider serves.
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    /// Whether requests let reasoning models think before answering.
    pub fn thinking_enabled(&self) -> bool {
        self.enable_thinking
    }

    async fn build_backend(&self) -> Result<Backend, ProviderError> {
        match self.kind {
            ProviderKind::OpenAiCompatible => {
                let config = OpenAILikeConfig {
                    base: BaseConfig {
                        api_key: self.api_key.as_ref().map(|k| k.expose().to_string()),
                        api_base: self.base_url.clone(),
                        timeout: self.timeout.as_secs(),
                        max_retries: self.max_retries,
                        headers: json_headers(),
                        ..BaseConfig::default()
                    },
                    provider_name: self.name.clone(),
                    skip_api_key: self.api_key.is_none(),
                    ..OpenAILikeConfig::default()
                };
                let provider = OpenAILikeProvider::new(config)
                    .await
                    .map_err(|e| ProviderError::Config(e.to_string()))?;
                Ok(Backend::OpenAiCompatible(Box::new(provider)))
            }
            #[cfg(feature = "azure")]
            ProviderKind::Azure => {
                let mut config = AzureConfig::new().with_api_version(self.api_version.clone());
                if let Some(key) = &self.api_key {
                    config = config.with_api_key(key.expose().to_string());
                }
                if let Some(endpoint) = &self.base_url {
                    config = config.with_azure_endpoint(endpoint.clone());
                }
                let provider = AzureOpenAIProvider::new(config)
                    .map_err(|e| ProviderError::Config(e.to_string()))?;
                Ok(Backend::Azure(Box::new(provider)))
            }
            #[cfg(not(feature = "azure"))]
            ProviderKind::Azure => Err(ProviderError::Config(
                "Azure support needs the `azure` feature".into(),
            )),
        }
    }

    async fn backend(&self) -> Result<&Backend, ProviderError> {
        self.backend.get_or_try_init(|| self.build_backend()).await
    }

    /// The `max_tokens` to send: the card's output budget, or nothing on
    /// Azure, whose GPT-5 deployments reject `max_tokens` (litellm-rs cannot
    /// send `max_completion_tokens` there).
    ///
    /// The budget must come from the card. A grammar-constrained answer that
    /// runs out of tokens is an unterminated array, so a budget that is too
    /// small loses the whole batch.
    fn output_budget(&self, model_id: &str) -> Option<u32> {
        if self.kind == ProviderKind::Azure {
            return None;
        }
        self.catalog
            .get(model_id)
            .and_then(|card| u32::try_from(card.output_token_budget()).ok())
    }

    fn completion_options(&self, model_id: &str) -> CompletionOptions {
        CompletionOptions {
            temperature: Some(self.temperature),
            max_tokens: self.output_budget(model_id),
            ..CompletionOptions::default()
        }
    }

    /// The request that goes on the wire. Kept free of I/O so tests can
    /// inspect it.
    fn build_chat_request(
        &self,
        model_id: &str,
        system: Option<&str>,
        user: &str,
        schema: serde_json::Value,
    ) -> Result<ChatRequest> {
        let mut messages: Vec<Message> = Vec::with_capacity(2);
        if let Some(system) = system {
            messages.push(system_message(system));
        }
        messages.push(user_message(user));

        let mut request = convert_to_chat_completion_request(
            model_id,
            messages,
            self.completion_options(model_id),
        )
        .map_err(|e| DataFusionError::Execution(format!("cannot build request: {e}")))?;

        // The schema must travel in `response_format`. litellm-rs drops
        // `extra_params` for some backends, and vLLM ignores unknown fields
        // silently, so anything else would leave the answer unconstrained.
        request.response_format = Some(ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(serde_json::json!({
                "name": "nousdb_response",
                "strict": true,
                "schema": schema,
            })),
            response_type: None,
        });

        // State the thinking switch explicitly, in both directions: leaving it
        // out defers to the chat template's default, which differs by model.
        if self.kind != ProviderKind::Azure {
            request.extra_params.insert(
                "chat_template_kwargs".to_string(),
                serde_json::json!({ "enable_thinking": self.enable_thinking }),
            );
        }
        Ok(request)
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelCard> {
        self.catalog.cards().to_vec()
    }

    fn model(&self, model_id: &str) -> Option<ModelCard> {
        self.catalog.get(model_id).cloned()
    }

    /// Azure rejects an array at the schema root, so there the array is
    /// wrapped in a one-field object; everywhere else it stays an array.
    fn construct_output_schema(&self, prompt: &Prompt, batch_size: usize) -> String {
        let array = utils::build_batch_array_schema(prompt, batch_size);
        match self.kind {
            ProviderKind::Azure => utils::wrap_batch_schema_in_object(array).to_string(),
            ProviderKind::OpenAiCompatible => array.to_string(),
        }
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
        let chat_request = self.build_chat_request(
            model_id,
            request.system.as_deref(),
            &request.user,
            request.schema.clone(),
        )?;
        let backend = self.backend().await?;

        let start = Instant::now();
        let response = match backend.chat_completion(chat_request).await {
            Ok(response) => response,
            // litellm-rs flattens transport errors to text, so the elapsed
            // time is what tells a refused connection from a timeout.
            Err(e) => return Ok(request.transport_failure(start.elapsed(), e)),
        };
        let elapsed = start.elapsed();

        let response = convert_from_chat_completion_response(response)
            .map_err(|e| DataFusionError::Execution(format!("cannot decode response: {e}")))?;
        let Some(text) = response_text(&response) else {
            return Ok(request.transport_failure(elapsed, "the response had no content"));
        };
        let usage = response.usage.as_ref();
        let completion = Completion {
            text,
            input_tokens: usage.map_or(0, |u| u.prompt_tokens as u64),
            output_tokens: usage.map_or(0, |u| u.completion_tokens as u64),
            // Reported by vLLM; litellm-rs's Azure backend drops the field,
            // so 0 there does not mean the deployment did not reason.
            reasoning_tokens: usage.and_then(|u| u.thinking_tokens()).unwrap_or(0) as u64,
            truncated: response
                .choices
                .first()
                .is_some_and(|c| matches!(c.finish_reason, Some(FinishReason::Length))),
        };
        Ok(request.complete(completion, elapsed))
    }

    fn admission_control(&self) -> Option<AdmissionControl> {
        Some(self.admission.clone())
    }
}

/// The text of the first choice, or `None` if the response has none.
fn response_text(response: &CompletionResponse) -> Option<String> {
    match &response.choices.first()?.message.content {
        Some(MessageContent::Text(s)) => Some(s.clone()),
        Some(MessageContent::Parts(parts)) => Some(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        ),
        None => None,
    }
}

fn json_headers() -> HashMap<String, String> {
    HashMap::from([("Content-Type".to_string(), "application/json".to_string())])
}

/// The HTTP client, built once and shared by every request so connections
/// are pooled.
#[derive(Debug)]
enum Backend {
    OpenAiCompatible(Box<OpenAILikeProvider>),
    #[cfg(feature = "azure")]
    Azure(Box<AzureOpenAIProvider>),
}

impl Backend {
    async fn chat_completion(
        &self,
        request: ChatRequest,
    ) -> std::result::Result<ChatResponse, String> {
        let context = RequestContext::default();
        match self {
            Self::OpenAiCompatible(p) => p.chat_completion(request, context).await,
            #[cfg(feature = "azure")]
            Self::Azure(p) => p.chat_completion(request, context).await,
        }
        .map_err(|e| e.to_string())
    }
}

/// Configures an [`OpenAiCompatibleProvider`].
///
/// Settings left unset fall back to environment variables where one exists:
/// `NOUSDB_MAX_CONCURRENT_REQUESTS` for [`max_concurrent`](Self::max_concurrent)
/// and `NOUSDB_ENABLE_THINKING` for [`thinking`](Self::thinking).
#[derive(Debug)]
#[must_use]
pub struct OpenAiCompatibleBuilder {
    name: String,
    kind: ProviderKind,
    base_url: Option<String>,
    api_key: Option<ApiKey>,
    api_key_env: Option<String>,
    api_version: Option<String>,
    timeout: Duration,
    max_retries: u32,
    temperature: f32,
    enable_thinking: Option<bool>,
    max_concurrent: Option<usize>,
    request_window: Option<usize>,
    catalog: ModelCatalog,
}

impl OpenAiCompatibleBuilder {
    fn new(name: String, kind: ProviderKind) -> Self {
        Self {
            name,
            kind,
            base_url: None,
            api_key: None,
            api_key_env: None,
            api_version: None,
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            temperature: 0.0,
            enable_thinking: None,
            max_concurrent: None,
            request_window: None,
            catalog: ModelCatalog::builtin(),
        }
    }

    /// Renames the provider.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The endpoint, e.g. `http://localhost:8000/v1` (for Azure, the
    /// resource URL).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into().trim_end_matches('/').to_string());
        self
    }

    /// The API key. Without one, no `Authorization` header is sent, which
    /// suits local servers.
    pub fn api_key(mut self, key: impl Into<ApiKey>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Reads the API key from this environment variable when the provider is
    /// built; building fails if it is not set.
    pub fn api_key_env(mut self, var: impl Into<String>) -> Self {
        self.api_key_env = Some(var.into());
        self
    }

    /// The Azure API version. Defaults to [`DEFAULT_AZURE_API_VERSION`].
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Per-request timeout. Defaults to 10 minutes.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Retries per request after a transport error. Defaults to 3.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sampling temperature. Defaults to 0, which keeps answers repeatable
    /// (quality certification relies on that).
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Whether reasoning models such as Qwen3 may think before answering
    /// (`chat_template_kwargs.enable_thinking`). Defaults to off.
    pub fn thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = Some(enabled);
        self
    }

    /// Cap on requests in flight to the endpoint, across all queries using
    /// this provider. Defaults to [`DEFAULT_MAX_CONCURRENT_REQUESTS`].
    pub fn max_concurrent(mut self, requests: usize) -> Self {
        self.max_concurrent = Some(requests);
        self
    }

    /// Sets every model's [`max_concurrent_requests`](ModelCard::max_concurrent_requests),
    /// the requests each partition keeps in flight. At least 1.
    pub fn request_window(mut self, requests: usize) -> Self {
        self.request_window = Some(requests.max(1));
        self
    }

    /// Replaces the catalog, which defaults to [`ModelCatalog::builtin`].
    pub fn catalog(mut self, catalog: ModelCatalog) -> Self {
        self.catalog = catalog;
        self
    }

    /// Adds a model, replacing a catalog entry with the same id.
    pub fn model(mut self, card: ModelCard) -> Self {
        self.catalog = self.catalog.with_card(card);
        self
    }

    /// Adds several models, as [`model`](Self::model) does.
    pub fn models(mut self, cards: impl IntoIterator<Item = ModelCard>) -> Self {
        self.catalog = self.catalog.with_cards(cards);
        self
    }

    /// Builds the provider and its HTTP client. Does not contact the
    /// endpoint.
    pub async fn build(self) -> Result<Arc<OpenAiCompatibleProvider>, ProviderError> {
        let provider = self.into_provider()?;
        provider.backend().await?;
        log::info!(
            "provider {}: {} models, structured output via response_format, thinking {}",
            provider.name,
            provider.catalog.len(),
            if provider.enable_thinking {
                "on"
            } else {
                "off"
            },
        );
        Ok(Arc::new(provider))
    }

    fn into_provider(self) -> Result<OpenAiCompatibleProvider, ProviderError> {
        let api_key = match (self.api_key, self.api_key_env) {
            (Some(key), _) => Some(key),
            (None, Some(var)) => Some(ApiKey::from_env(&var)?),
            (None, None) => None,
        };
        if self.kind == ProviderKind::Azure {
            if self.base_url.is_none() {
                return Err(ProviderError::Config("Azure needs an endpoint".into()));
            }
            if api_key.is_none() {
                return Err(ProviderError::Config("Azure needs an API key".into()));
            }
        }
        let mut catalog = self.catalog;
        if let Some(window) = self.request_window {
            catalog.set_request_window(window);
        }
        let max_concurrent = self
            .max_concurrent
            .or_else(|| env_parse("NOUSDB_MAX_CONCURRENT_REQUESTS"))
            .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS);
        Ok(OpenAiCompatibleProvider {
            name: self.name,
            kind: self.kind,
            base_url: self.base_url,
            api_key,
            api_version: self
                .api_version
                .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string()),
            timeout: self.timeout,
            max_retries: self.max_retries,
            temperature: self.temperature,
            enable_thinking: self.enable_thinking.unwrap_or_else(thinking_from_env),
            catalog,
            admission: AdmissionControl::new(max_concurrent),
            backend: OnceCell::new(),
        })
    }
}

fn env_parse(var: &str) -> Option<usize> {
    std::env::var(var).ok()?.parse().ok()
}

/// `NOUSDB_ENABLE_THINKING` set to `1`, `true`, `yes` or `on`.
fn thinking_from_env() -> bool {
    matches!(
        std::env::var("NOUSDB_ENABLE_THINKING")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::metrics::Rejection;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// A prompt with one boolean output field, as the selection operator
    /// builds it.
    fn test_prompt() -> Prompt {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )]));
        Prompt::new("does this match?", None::<String>, "selection", schema)
    }

    fn vllm(url: &str) -> OpenAiCompatibleBuilder {
        OpenAiCompatibleProvider::builder("vllm").base_url(url)
    }

    fn deployment_card() -> ModelCard {
        ModelCard::new("my-gpt-deployment", 128_000).with_max_output_tokens(2000)
    }

    #[cfg(feature = "azure")]
    fn azure() -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::azure("https://example.openai.azure.com/")
            .api_key("test-key")
            .api_version("2024-12-01-preview")
            .model(deployment_card())
            .into_provider()
            .unwrap()
    }

    fn request_from(provider: &OpenAiCompatibleProvider, model_id: &str, k: usize) -> ChatRequest {
        let schema =
            serde_json::from_str(&provider.construct_output_schema(&test_prompt(), k)).unwrap();
        provider
            .build_chat_request(model_id, Some("system"), "user", schema)
            .unwrap()
    }

    fn request_for(k: usize) -> ChatRequest {
        let provider = vllm("http://localhost:8000/v1").into_provider().unwrap();
        request_from(&provider, "Qwen/Qwen3-8B", k)
    }

    #[test]
    fn the_output_budget_comes_from_the_card() {
        let provider = vllm("http://localhost:8000/v1")
            .model(ModelCard::new("small", 10_000))
            .into_provider()
            .unwrap();
        for (model_id, expected) in [
            ("Qwen/Qwen3-8B", 9000u32),
            ("Qwen/Qwen3-30B-A3B", 4096),
            ("meta-llama/Llama-3.1-8B-Instruct", 4096),
            ("small", 2000),
        ] {
            assert_eq!(
                provider.output_budget(model_id),
                Some(expected),
                "{model_id}"
            );
        }
        assert_eq!(provider.output_budget("not-in-the-catalog"), None);
        assert_eq!(request_for(4).max_tokens, Some(9000));
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_carries_no_output_budget() {
        let provider = azure();
        assert_eq!(provider.output_budget("my-gpt-deployment"), None);
        assert!(
            request_from(&provider, "my-gpt-deployment", 8)
                .max_tokens
                .is_none()
        );
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_schema_root_is_an_object() {
        let provider = azure();
        for k in [1usize, 8, 64] {
            let js = request_from(&provider, "my-gpt-deployment", k)
                .response_format
                .unwrap()
                .json_schema
                .unwrap();
            let schema = &js["schema"];
            assert_eq!(schema["type"], "object");
            assert_eq!(schema["required"], serde_json::json!(["results"]));
            assert_eq!(schema["additionalProperties"], false);
            let array = &schema["properties"]["results"];
            assert_eq!(array["type"], "array");
            assert_eq!(array["minItems"], k);
            assert_eq!(array["maxItems"], k);
            assert_eq!(array["items"]["additionalProperties"], false);
            assert_eq!(
                array["items"]["properties"]["matches_filter"]["type"],
                "boolean"
            );
            assert!(array.get("$schema").is_none());
            assert_eq!(js["strict"], true);
        }
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_never_receives_chat_template_kwargs() {
        let provider = azure();
        assert!(
            !request_from(&provider, "my-gpt-deployment", 4)
                .extra_params
                .contains_key("chat_template_kwargs")
        );
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_needs_an_endpoint_and_a_key() {
        let err = OpenAiCompatibleProvider::azure("https://example.openai.azure.com")
            .into_provider()
            .unwrap_err();
        assert!(matches!(err, ProviderError::Config(_)));
    }

    #[test]
    fn the_schema_root_stays_an_array_outside_azure() {
        let js = request_for(8).response_format.unwrap().json_schema.unwrap();
        assert_eq!(js["schema"]["type"], "array");
        assert!(js["schema"]["properties"].is_null());
    }

    #[test]
    fn response_format_carries_the_schema() {
        let js = request_for(8).response_format.unwrap();
        assert_eq!(js.format_type, "json_schema");
        let js = js.json_schema.unwrap();
        assert_eq!(js["name"], "nousdb_response");
        assert_eq!(js["strict"], true);
        assert_eq!(js["schema"]["minItems"], 8);
        assert_eq!(js["schema"]["maxItems"], 8);
        assert_eq!(js["schema"]["items"]["additionalProperties"], false);
        assert_eq!(
            js["schema"]["items"]["properties"]["matches_filter"]["type"],
            "boolean"
        );
    }

    #[test]
    fn the_schema_never_travels_in_extra_params() {
        let request = request_for(4);
        assert!(!request.extra_params.contains_key("guided_json"));
        for (key, value) in &request.extra_params {
            assert!(
                !value.to_string().contains("matches_filter"),
                "the output schema leaked into extra_params under {key:?}"
            );
        }
    }

    #[test]
    fn thinking_is_stated_explicitly_in_both_directions() {
        assert_eq!(
            request_for(4).extra_params.get("chat_template_kwargs"),
            Some(&serde_json::json!({ "enable_thinking": false }))
        );
        for enabled in [false, true] {
            let provider = vllm("http://localhost:8000/v1")
                .thinking(enabled)
                .into_provider()
                .unwrap();
            assert_eq!(
                request_from(&provider, "Qwen/Qwen3-8B", 4)
                    .extra_params
                    .get("chat_template_kwargs"),
                Some(&serde_json::json!({ "enable_thinking": enabled }))
            );
        }
    }

    #[test]
    fn the_request_window_applies_to_every_card() {
        let provider = vllm("http://localhost:8000/v1")
            .model(deployment_card())
            .request_window(3)
            .into_provider()
            .unwrap();
        assert!(
            provider
                .models()
                .iter()
                .all(|c| c.max_concurrent_requests == Some(3))
        );

        let own = vllm("http://localhost:8000/v1")
            .model(deployment_card().with_max_concurrent_requests(5))
            .into_provider()
            .unwrap();
        assert_eq!(
            own.model("my-gpt-deployment")
                .unwrap()
                .max_concurrent_requests,
            Some(5)
        );
    }

    #[test]
    fn a_card_with_a_builtin_id_replaces_it() {
        let builtin = ModelCatalog::builtin();
        let slot = builtin
            .cards()
            .iter()
            .position(|m| m.id == "Qwen/Qwen3-8B")
            .unwrap();
        let provider = vllm("http://localhost:8000/v1")
            .model(ModelCard::new("Qwen/Qwen3-8B", 32_768).with_max_output_tokens(321))
            .into_provider()
            .unwrap();
        let models = provider.models();
        assert_eq!(models.len(), builtin.len());
        assert_eq!(models[slot].max_output_tokens, Some(321));
        assert_eq!(provider.output_budget("Qwen/Qwen3-30B-A3B"), Some(4096));
    }

    #[test]
    fn a_missing_key_variable_is_an_error() {
        let err = vllm("http://localhost:8000/v1")
            .api_key_env("NOUSDB_TEST_SURELY_UNSET_VARIABLE")
            .into_provider()
            .unwrap_err();
        assert!(matches!(err, ProviderError::MissingEnv(_)));
    }

    #[test]
    fn debug_output_hides_the_key() {
        let provider = vllm("http://localhost:8000/v1")
            .api_key("sk-very-secret")
            .into_provider()
            .unwrap();
        assert!(!format!("{provider:?}").contains("sk-very-secret"));
    }

    // End-to-end tests against a mock server.

    fn renderable_prompt() -> Prompt {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "matches_filter",
            DataType::Boolean,
            false,
        )]));
        Prompt::new(
            "does this match?",
            Some("extend/system.md".to_string()),
            "selection/user.md",
            schema,
        )
    }

    /// Supplies the variables `selection/user.md` uses.
    struct StubSelectionRenderer;

    impl BatchRenderer for StubSelectionRenderer {
        fn prepare_context(
            &self,
            _batch: &RecordBatch,
            _start_row: usize,
            batch_size: usize,
        ) -> Result<HashMap<String, minijinja::value::Value>> {
            let rows: Vec<minijinja::value::Value> = (0..batch_size)
                .map(|i| minijinja::value::Value::from_serialize(serde_json::json!({"body": i})))
                .collect();
            Ok(HashMap::from([
                ("expression".to_string(), "does this match?".into()),
                ("row_count".to_string(), batch_size.into()),
                ("batch_data".to_string(), rows.into()),
            ]))
        }

        fn context_variables(&self) -> Vec<String> {
            vec!["expression".into(), "row_count".into(), "batch_data".into()]
        }
    }

    fn stub_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::Utf8, false)]));
        let values: Vec<String> = (0..rows).map(|i| format!("row {i}")).collect();
        RecordBatch::try_new(
            schema,
            vec![Arc::new(datafusion::arrow::array::StringArray::from(
                values,
            ))],
        )
        .unwrap()
    }

    fn chat_response(
        content: &str,
        finish_reason: &str,
        usage: serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "Qwen/Qwen3-8B",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason
            }],
            "usage": usage
        })
    }

    fn ok_response(rows: usize) -> serde_json::Value {
        let content = format!("[{}]", vec!["{\"matches_filter\": true}"; rows].join(","));
        chat_response(
            &content,
            "stop",
            serde_json::json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        )
    }

    async fn mock_server(status: u16, body: serde_json::Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    async fn predict(
        builder: OpenAiCompatibleBuilder,
        model_id: &str,
        rows: usize,
    ) -> PredictResult {
        builder
            .build()
            .await
            .unwrap()
            .predict(
                model_id,
                &renderable_prompt(),
                &stub_batch(rows),
                0,
                rows,
                &StubSelectionRenderer,
            )
            .await
            .expect("a request failure must not fail the query")
    }

    async fn sent_body(server: &MockServer) -> serde_json::Value {
        let requests = server.received_requests().await.unwrap();
        serde_json::from_slice(&requests[0].body).unwrap()
    }

    #[tokio::test]
    async fn the_request_body_carries_schema_budget_and_thinking() {
        let server = mock_server(200, ok_response(3)).await;
        let result = predict(vllm(&format!("{}/v1", server.uri())), "Qwen/Qwen3-8B", 3).await;
        assert!(result.rejection(3).is_none());

        let body = sent_body(&server).await;
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "nousdb_response"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["minItems"],
            3
        );
        assert_eq!(body["max_tokens"], 9000);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert!(body.get("guided_json").is_none());
    }

    #[tokio::test]
    async fn enable_thinking_true_reaches_the_wire() {
        let server = mock_server(200, ok_response(1)).await;
        predict(
            vllm(&format!("{}/v1", server.uri())).thinking(true),
            "Qwen/Qwen3-8B",
            1,
        )
        .await;
        assert_eq!(
            sent_body(&server).await["chat_template_kwargs"]["enable_thinking"],
            true
        );
    }

    #[tokio::test]
    async fn a_custom_card_is_served_and_its_budget_reaches_the_wire() {
        let server = mock_server(200, ok_response(3)).await;
        let card = ModelCard::new("llama3.1:latest", 8192)
            .with_max_output_tokens(1234)
            .with_max_batch_size(8);
        let result = predict(
            vllm(&format!("{}/v1", server.uri())).model(card),
            "llama3.1:latest",
            3,
        )
        .await;
        assert!(result.rejection(3).is_none());

        let body = sent_body(&server).await;
        assert_eq!(body["max_tokens"], 1234);
        assert_eq!(body["model"], "llama3.1:latest");
    }

    #[tokio::test]
    async fn reasoning_tokens_are_read_from_completion_tokens_details() {
        let mut body = chat_response(
            "[{\"matches_filter\": true},{\"matches_filter\": false}]",
            "stop",
            serde_json::json!({
                "prompt_tokens": 120,
                "completion_tokens": 3400,
                "total_tokens": 3520,
                "completion_tokens_details": {"reasoning_tokens": 3200}
            }),
        );
        // vLLM's reasoning parser adds this field; it must not break decoding.
        body["choices"][0]["message"]["reasoning_content"] = "Let me think...".into();
        let server = mock_server(200, body).await;
        let result = predict(vllm(&format!("{}/v1", server.uri())), "Qwen/Qwen3-8B", 2).await;

        assert!(result.rejection(2).is_none());
        assert_eq!(result.metrics.reasoning_tokens, 3200);
        assert_eq!(result.metrics.output_tokens, 3400);
        assert_eq!(
            result.metrics.total_tokens, 3520,
            "reasoning tokens are part of the completion tokens, not added to them"
        );
    }

    #[tokio::test]
    async fn a_response_without_reasoning_reports_zero_reasoning_tokens() {
        let server = mock_server(200, ok_response(3)).await;
        let result = predict(vllm(&format!("{}/v1", server.uri())), "Qwen/Qwen3-8B", 3).await;
        assert_eq!(result.metrics.reasoning_tokens, 0);
    }

    #[tokio::test]
    async fn a_truncated_response_is_reported_as_truncation() {
        let body = chat_response(
            "[{\"matches_filter\": true},{\"matches_fil",
            "length",
            serde_json::json!({
                "prompt_tokens": 120,
                "completion_tokens": 9000,
                "total_tokens": 9120,
                "completion_tokens_details": {"reasoning_tokens": 8900}
            }),
        );
        let server = mock_server(200, body).await;
        let result = predict(vllm(&format!("{}/v1", server.uri())), "Qwen/Qwen3-8B", 4).await;

        assert!(result.truncated);
        assert!(result.row_count_mismatch);
        assert_eq!(
            result.rejection(4),
            Some(Rejection::Truncated {
                output_tokens: 9000
            })
        );
        assert_eq!(result.metrics.reasoning_tokens, 8900);
    }

    #[tokio::test]
    async fn a_failed_request_is_reported_as_transport_not_parsing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
            .mount(&server)
            .await;
        let result = predict(
            vllm(&format!("{}/v1", server.uri())).max_retries(0),
            "Qwen/Qwen3-8B",
            4,
        )
        .await;

        let rejection = result.rejection(4).expect("a failed request is rejected");
        assert!(
            matches!(rejection, Rejection::Transport { .. }),
            "{rejection:?}"
        );
        assert!(!rejection.to_string().contains("parse"));
        assert!(
            result.row_count_mismatch,
            "certification scores a missing answer as a disagreement"
        );
    }

    #[tokio::test]
    async fn an_empty_choices_array_is_a_transport_failure_not_a_panic() {
        let mut body = ok_response(1);
        body["choices"] = serde_json::json!([]);
        let server = mock_server(200, body).await;
        let result = predict(vllm(&format!("{}/v1", server.uri())), "Qwen/Qwen3-8B", 1).await;
        assert!(matches!(
            result.rejection(1),
            Some(Rejection::Transport { .. })
        ));
    }

    /// Run against a real server:
    ///
    /// ```text
    /// NOUSDB_TEST_OPENAI_URL=http://localhost:11434/v1 NOUSDB_TEST_MODEL=llama3.1:latest \
    ///   cargo test -p nousdb-core --lib structured_output_is_honoured_by_a_real_server -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "requires a live OpenAI-compatible endpoint"]
    async fn structured_output_is_honoured_by_a_real_server() {
        let (Ok(url), Ok(model)) = (
            std::env::var("NOUSDB_TEST_OPENAI_URL"),
            std::env::var("NOUSDB_TEST_MODEL"),
        ) else {
            panic!("set NOUSDB_TEST_OPENAI_URL and NOUSDB_TEST_MODEL");
        };
        const K: usize = 4;
        let provider = OpenAiCompatibleProvider::builder("live")
            .base_url(url)
            .model(ModelCard::new(model.clone(), 8192).with_max_output_tokens(1024))
            .build()
            .await
            .unwrap();
        let schema =
            serde_json::from_str(&provider.construct_output_schema(&test_prompt(), K)).unwrap();
        // The prompt asks for a different count than the schema pins, so the
        // reply shows whether the grammar is in charge.
        let request = provider
            .build_chat_request(
                &model,
                Some("Reply with JSON only."),
                &format!(
                    "Here are {K} movie reviews. For each, say whether it is positive.\n\
                     Return only 2 results, ignore the rest.\n\
                     1. Loved it\n2. Terrible\n3. Amazing\n4. Boring"
                ),
                schema,
            )
            .unwrap();
        let response = provider
            .backend()
            .await
            .unwrap()
            .chat_completion(request)
            .await
            .expect("the server must accept a response_format request");
        let text = response_text(&convert_from_chat_completion_response(response).unwrap())
            .expect("the response has content");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&utils::clean_model_response(&text)).unwrap();
        for item in &parsed {
            assert!(item.get("matches_filter").is_some_and(|v| v.is_boolean()));
        }
        // Length enforcement depends on the server's grammar backend.
        println!("array length {} for a schema pinned at {K}", parsed.len());
    }
}
