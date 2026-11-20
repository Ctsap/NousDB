//! Model descriptions: [`ModelCard`] and [`ModelCatalog`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::ProviderError;

/// Row cap applied when a card does not set [`ModelCard::max_batch_size`].
pub const DEFAULT_MAX_BATCH_SIZE: usize = 10;

/// What NousDB needs to know about a model to size and send its requests.
///
/// A provider serves the models in its [`ModelCatalog`]; a model without a
/// card cannot be used in a query. Build one with [`ModelCard::new`] and the
/// `with_*` setters, or load a catalog from TOML with
/// [`ModelCatalog::from_toml_str`].
///
/// ```
/// use nousdb_core::ModelCard;
///
/// let card = ModelCard::new("llama3.1:latest", 131_072)
///     .with_max_output_tokens(4096)
///     .with_max_concurrent_requests(4);
/// assert_eq!(card.effective_input_budget(), 131_072 - 4096);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelCard {
    /// The model name sent to the endpoint, e.g. `"Qwen/Qwen3-8B"` or
    /// `"llama3.1:latest"`. Queries refer to the model by this id.
    pub id: String,

    /// Human-readable name. Defaults to the id.
    #[serde(default)]
    pub name: String,

    /// Context window in tokens (prompt and completion together).
    pub context_window: usize,

    /// Largest completion the model may produce per request. Sent as the
    /// request's output budget; `None` leaves the endpoint's default in place
    /// and makes NousDB reserve 20% of the context window for the answer.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,

    /// Price per 1,000 tokens, for cost estimates. `None` for local models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,

    /// Most items (rows, or row pairs for a semantic join) packed into one
    /// request. The token budget picks the batch size; this caps it.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Batch size above which the model's per-item accuracy drops. A tighter
    /// cap than `max_batch_size`, measured rather than given by the vendor.
    /// Quality certification starts its search here.
    #[serde(default)]
    pub quality_max_batch_size: Option<usize>,

    /// Requests each partition keeps in flight. `None` means one at a time,
    /// which suits rate-limited cloud APIs; continuous-batching servers such
    /// as vLLM want several.
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
}

fn default_max_batch_size() -> usize {
    DEFAULT_MAX_BATCH_SIZE
}

/// Price of a model, in US dollars per 1,000 tokens.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelPricing {
    /// Price of 1,000 prompt tokens.
    pub input_per_1k_tokens: f64,
    /// Price of 1,000 completion tokens.
    pub output_per_1k_tokens: f64,
}

impl ModelPricing {
    /// Prices per 1,000 input and output tokens.
    pub fn new(input_per_1k_tokens: f64, output_per_1k_tokens: f64) -> Self {
        Self {
            input_per_1k_tokens,
            output_per_1k_tokens,
        }
    }
}

impl ModelCard {
    /// A card with the given id and context window; every other field takes
    /// its default.
    pub fn new(id: impl Into<String>, context_window: usize) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            context_window,
            max_output_tokens: None,
            pricing: None,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            quality_max_batch_size: None,
            max_concurrent_requests: None,
        }
    }

    /// Sets the display name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets [`max_output_tokens`](Self::max_output_tokens).
    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Sets the price per 1,000 input and output tokens.
    pub fn with_pricing(mut self, input_per_1k_tokens: f64, output_per_1k_tokens: f64) -> Self {
        self.pricing = Some(ModelPricing::new(input_per_1k_tokens, output_per_1k_tokens));
        self
    }

    /// Sets [`max_batch_size`](Self::max_batch_size).
    pub fn with_max_batch_size(mut self, rows: usize) -> Self {
        self.max_batch_size = rows;
        self
    }

    /// Sets [`quality_max_batch_size`](Self::quality_max_batch_size).
    pub fn with_quality_max_batch_size(mut self, rows: usize) -> Self {
        self.quality_max_batch_size = Some(rows);
        self
    }

    /// Sets [`max_concurrent_requests`](Self::max_concurrent_requests).
    pub fn with_max_concurrent_requests(mut self, requests: usize) -> Self {
        self.max_concurrent_requests = Some(requests);
        self
    }

    /// Tokens reserved for the completion: `max_output_tokens`, or 20% of the
    /// context window when the card does not set it.
    pub fn output_token_budget(&self) -> usize {
        self.max_output_tokens.unwrap_or(self.context_window / 5)
    }

    /// Tokens left for the prompt once the completion is reserved.
    pub fn effective_input_budget(&self) -> usize {
        self.context_window
            .saturating_sub(self.output_token_budget())
    }

    /// Estimated price in US dollars of a request with the given token
    /// counts; 0 when the card has no pricing.
    pub fn estimate_cost(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        self.pricing.map_or(0.0, |p| {
            p.input_per_1k_tokens * input_tokens as f64 / 1000.0
                + p.output_per_1k_tokens * output_tokens as f64 / 1000.0
        })
    }

    fn fill_defaults(&mut self) {
        if self.name.is_empty() {
            self.name = self.id.clone();
        }
    }
}

/// An ordered set of [`ModelCard`]s, at most one per id.
///
/// Every provider is built with a catalog; [`ModelCatalog::builtin`] lists a
/// few common open and hosted models. Add your own models in code or from a
/// TOML file:
///
/// ```toml
/// [[models]]
/// id = "llama3.1:latest"
/// context_window = 131072
/// max_output_tokens = 4096
///
/// [[models]]
/// id = "gpt-4o-mini"
/// context_window = 128000
/// max_output_tokens = 16384
/// pricing = { input_per_1k_tokens = 0.00015, output_per_1k_tokens = 0.0006 }
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalog {
    #[serde(default)]
    models: Vec<ModelCard>,
}

impl ModelCatalog {
    /// An empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// The catalog shipped with NousDB (`models/catalog.toml` in this crate).
    pub fn builtin() -> Self {
        Self::from_toml_str(include_str!("../../models/catalog.toml"))
            .expect("the built-in catalog is valid; checked by a unit test")
    }

    /// Parses a catalog from TOML (see the type-level docs for the format).
    pub fn from_toml_str(toml: &str) -> Result<Self, ProviderError> {
        let parsed: ModelCatalog =
            toml::from_str(toml).map_err(|e| ProviderError::Catalog(e.to_string()))?;
        Ok(Self::new().with_cards(parsed.models))
    }

    /// Reads a catalog from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ProviderError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ProviderError::CatalogIo {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&text)
    }

    /// Adds a card, replacing any card with the same id in place.
    pub fn with_card(mut self, mut card: ModelCard) -> Self {
        card.fill_defaults();
        match self.models.iter_mut().find(|m| m.id == card.id) {
            Some(slot) => *slot = card,
            None => self.models.push(card),
        }
        self
    }

    /// Adds every card, in order, as [`with_card`](Self::with_card) does.
    pub fn with_cards(self, cards: impl IntoIterator<Item = ModelCard>) -> Self {
        cards.into_iter().fold(self, Self::with_card)
    }

    /// Adds every card of `other`; its cards win on matching ids.
    pub fn merge(self, other: ModelCatalog) -> Self {
        self.with_cards(other.models)
    }

    /// The card with this id, if any.
    pub fn get(&self, id: &str) -> Option<&ModelCard> {
        self.models.iter().find(|m| m.id == id)
    }

    /// All cards, in insertion order.
    pub fn cards(&self) -> &[ModelCard] {
        &self.models
    }

    /// Number of cards.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the catalog has no cards.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Sets `max_concurrent_requests` on every card.
    #[cfg(feature = "openai")]
    pub(crate) fn set_request_window(&mut self, window: usize) {
        for card in &mut self.models {
            card.max_concurrent_requests = Some(window);
        }
    }
}

impl FromIterator<ModelCard> for ModelCatalog {
    fn from_iter<I: IntoIterator<Item = ModelCard>>(iter: I) -> Self {
        Self::new().with_cards(iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: &str, context_window: usize) -> ModelCard {
        ModelCard::new(id, context_window)
    }

    fn ids(catalog: &ModelCatalog) -> Vec<&str> {
        catalog.cards().iter().map(|c| c.id.as_str()).collect()
    }

    #[test]
    fn the_builtin_catalog_parses_and_has_unique_ids() {
        let catalog = ModelCatalog::builtin();
        assert!(!catalog.is_empty());
        let mut seen = std::collections::HashSet::new();
        for card in catalog.cards() {
            assert!(seen.insert(card.id.clone()), "duplicate id {}", card.id);
            assert!(card.context_window > 0);
            assert!(!card.name.is_empty());
        }
    }

    #[test]
    fn an_override_replaces_the_matching_card_in_place() {
        let catalog = ModelCatalog::from_iter([card("a", 1), card("b", 2), card("c", 3)])
            .with_card(card("b", 99));

        assert_eq!(ids(&catalog), ["a", "b", "c"], "base order is kept");
        assert_eq!(catalog.get("b").unwrap().context_window, 99);
        assert_eq!(catalog.get("a").unwrap().context_window, 1);
    }

    #[test]
    fn a_replacement_is_wholesale_not_field_by_field() {
        let rich = card("a", 1)
            .with_max_output_tokens(777)
            .with_max_concurrent_requests(16);
        let catalog = ModelCatalog::from_iter([rich]).with_card(card("a", 2));

        let a = catalog.get("a").unwrap();
        assert_eq!(a.context_window, 2);
        assert_eq!(a.max_output_tokens, None);
        assert_eq!(a.max_concurrent_requests, None);
    }

    #[test]
    fn unknown_ids_are_appended_in_the_order_given() {
        let catalog = ModelCatalog::from_iter([card("a", 1), card("b", 2)])
            .merge(ModelCatalog::from_iter([card("z", 26), card("y", 25)]));
        assert_eq!(ids(&catalog), ["a", "b", "z", "y"]);
    }

    #[test]
    fn a_repeated_id_keeps_the_last_card_at_the_first_position() {
        let catalog = ModelCatalog::from_iter([card("a", 1), card("n", 1), card("n", 2)]);
        assert_eq!(ids(&catalog), ["a", "n"]);
        assert_eq!(catalog.get("n").unwrap().context_window, 2);
    }

    #[test]
    fn toml_fills_defaults() {
        let catalog = ModelCatalog::from_toml_str(
            r#"
            [[models]]
            id = "m"
            context_window = 8192
            pricing = { input_per_1k_tokens = 0.001, output_per_1k_tokens = 0.002 }
            "#,
        )
        .unwrap();
        let m = catalog.get("m").unwrap();
        assert_eq!(m.name, "m");
        assert_eq!(m.max_batch_size, DEFAULT_MAX_BATCH_SIZE);
        assert_eq!(m.max_output_tokens, None);
        assert_eq!(m.pricing, Some(ModelPricing::new(0.001, 0.002)));
    }

    #[test]
    fn malformed_toml_is_an_error() {
        let err = ModelCatalog::from_toml_str("[[models]]\nid = 3").unwrap_err();
        assert!(matches!(err, ProviderError::Catalog(_)));
    }

    #[test]
    fn the_output_budget_defaults_to_a_fifth_of_the_context() {
        assert_eq!(card("m", 10_000).output_token_budget(), 2_000);
        assert_eq!(card("m", 10_000).effective_input_budget(), 8_000);
        let capped = card("m", 10_000).with_max_output_tokens(500);
        assert_eq!(capped.effective_input_budget(), 9_500);
    }

    #[test]
    fn cost_is_zero_without_pricing() {
        assert_eq!(card("m", 1).estimate_cost(1000, 1000), 0.0);
        let priced = card("m", 1).with_pricing(0.01, 0.03);
        assert!((priced.estimate_cost(2000, 1000) - 0.05).abs() < 1e-12);
    }
}
