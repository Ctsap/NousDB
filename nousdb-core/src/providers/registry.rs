//! The [`ModelRegistry`].

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::providers::error::ProviderError;
use crate::providers::metadata::ModelCard;
use crate::providers::provider::ModelProvider;

/// The providers a session can send requests to.
///
/// Queries name a model by id. [`resolve`](Self::resolve) finds the first
/// registered provider whose catalog has that id; to pick a provider
/// explicitly, write `"<provider>/<model>"`, e.g. `"local/llama3.1:latest"`.
/// An exact model id always wins over that form, so ids that contain a slash
/// (`"Qwen/Qwen3-8B"`) keep working.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    providers: RwLock<Vec<(String, Arc<dyn ModelProvider>)>>,
}

impl ModelRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<(String, Arc<dyn ModelProvider>)>> {
        self.providers.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, Vec<(String, Arc<dyn ModelProvider>)>> {
        self.providers.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers `provider` under its own [`name`](ModelProvider::name).
    pub fn register(&self, provider: Arc<dyn ModelProvider>) -> Result<(), ProviderError> {
        let name = provider.name().to_string();
        self.register_as(name, provider)
    }

    /// Registers `provider` under `name`. Providers registered earlier take
    /// precedence for models that several of them serve.
    pub fn register_as(
        &self,
        name: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
    ) -> Result<(), ProviderError> {
        let name = name.into();
        let mut providers = self.write();
        if providers.iter().any(|(n, _)| *n == name) {
            return Err(ProviderError::DuplicateProvider(name));
        }
        providers.push((name, provider));
        Ok(())
    }

    /// Removes the provider registered under `name`.
    pub fn unregister(&self, name: &str) -> Result<Arc<dyn ModelProvider>, ProviderError> {
        let mut providers = self.write();
        let index = providers
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| ProviderError::UnknownProvider(name.to_string()))?;
        Ok(providers.remove(index).1)
    }

    /// The provider registered under `name`.
    pub fn provider(&self, name: &str) -> Option<Arc<dyn ModelProvider>> {
        self.read()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| Arc::clone(p))
    }

    /// The names of the registered providers, in registration order.
    pub fn provider_names(&self) -> Vec<String> {
        self.read().iter().map(|(n, _)| n.clone()).collect()
    }

    /// The provider that serves `model` and the model's card.
    pub fn resolve(
        &self,
        model: &str,
    ) -> Result<(Arc<dyn ModelProvider>, ModelCard), ProviderError> {
        let providers = self.read();
        for (_, provider) in providers.iter() {
            if let Some(card) = provider.model(model) {
                return Ok((Arc::clone(provider), card));
            }
        }
        if let Some((provider_name, model_id)) = model.split_once('/')
            && let Some((_, provider)) = providers.iter().find(|(n, _)| n == provider_name)
            && let Some(card) = provider.model(model_id)
        {
            return Ok((Arc::clone(provider), card));
        }
        Err(ProviderError::UnknownModel(model.to_string()))
    }

    /// The card of `model`, as [`resolve`](Self::resolve) finds it.
    pub fn model_card(&self, model: &str) -> Option<ModelCard> {
        self.resolve(model).ok().map(|(_, card)| card)
    }

    /// Every model of every provider, as `(provider name, card)` pairs.
    pub fn models(&self) -> Vec<(String, ModelCard)> {
        self.read()
            .iter()
            .flat_map(|(name, p)| p.models().into_iter().map(move |c| (name.clone(), c)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::prompt::Prompt;
    use crate::prompt::renderer::BatchRenderer;
    use crate::providers::metrics::PredictResult;
    use async_trait::async_trait;
    use datafusion::arrow::record_batch::RecordBatch;

    #[derive(Debug)]
    struct Fixed(&'static str, Vec<ModelCard>);

    #[async_trait]
    impl ModelProvider for Fixed {
        fn name(&self) -> &str {
            self.0
        }
        fn models(&self) -> Vec<ModelCard> {
            self.1.clone()
        }
        async fn predict(
            &self,
            _: &str,
            _: &Prompt,
            _: &RecordBatch,
            _: usize,
            _: usize,
            _: &(dyn BatchRenderer + Send + Sync),
        ) -> datafusion::error::Result<PredictResult> {
            unreachable!("not called in these tests")
        }
    }

    fn registry() -> ModelRegistry {
        let r = ModelRegistry::new();
        r.register(Arc::new(Fixed(
            "a",
            vec![ModelCard::new("m", 1), ModelCard::new("Qwen/Qwen3-8B", 2)],
        )))
        .unwrap();
        r.register(Arc::new(Fixed("b", vec![ModelCard::new("m", 3)])))
            .unwrap();
        r
    }

    #[test]
    fn the_first_registered_provider_wins() {
        let (p, card) = registry().resolve("m").unwrap();
        assert_eq!(p.name(), "a");
        assert_eq!(card.context_window, 1);
    }

    #[test]
    fn a_provider_prefix_selects_the_provider() {
        let (p, card) = registry().resolve("b/m").unwrap();
        assert_eq!(p.name(), "b");
        assert_eq!(card.context_window, 3);
    }

    #[test]
    fn an_exact_id_with_a_slash_is_not_split() {
        let (p, _) = registry().resolve("Qwen/Qwen3-8B").unwrap();
        assert_eq!(p.name(), "a");
    }

    #[test]
    fn unknown_models_and_duplicate_names_are_errors() {
        let r = registry();
        assert!(matches!(
            r.resolve("x"),
            Err(ProviderError::UnknownModel(_))
        ));
        assert!(matches!(
            r.register(Arc::new(Fixed("a", vec![]))),
            Err(ProviderError::DuplicateProvider(_))
        ));
        r.unregister("a").unwrap();
        assert_eq!(r.resolve("m").unwrap().0.name(), "b");
    }
}
