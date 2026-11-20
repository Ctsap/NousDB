# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-20

First release, built on DataFusion 55.2 and Arrow 59.

### Added

- Semantic operators on DataFusion DataFrames: `extend`, `semantic_select`,
  `semantic_join` and `generate`.
- SQL functions for the semantic operators: `EXTEND`, `SEM_FILTER`,
  `SEM_JOIN` and `GENERATE`, in the crate `nousdb-sql`. They are on by
  default; `NousDBContextBuilder::with_sql_functions(false)` opts out.
- The extend operator allows DataFusion's limit pushdown (logical node and
  physical operator): a `LIMIT` above an extend is applied below it, so the
  model is called for the limited rows only.
- Semantic selection and generate take a fetch, as DataFusion's filter and
  scan do (`ExecutionPlan::with_fetch`). A `LIMIT` above a semantic selection
  ends it after that many accepted rows, and it sends a request only while
  the ones in flight cannot fill the limit. A `LIMIT` above `GENERATE` asks
  the model for the first rows of the table only.
- `OpenAiCompatibleProvider` for OpenAI-compatible endpoints (vLLM, SGLang,
  LiteLLM, OpenAI, Ollama `/v1`) and Azure OpenAI, with JSON-schema
  constrained answers.
- `OllamaProvider` for Ollama's native API.
- `ModelCard` and `ModelCatalog` to describe models in code or TOML, and a
  built-in catalog of common models.
- Optimizer rules: semantic rewrites, token-histogram injection, token
  repartitioning and batch sizing.
- Quality-certified batch sizing.
- Reasoning modes for operator prompts.

[Unreleased]: https://github.com/Ctsap/nousdb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ctsap/nousdb/releases/tag/v0.1.0
