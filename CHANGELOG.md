# Changelog

## [Unreleased]

### Added
- **Multi-format provider endpoints** — a single provider can now expose multiple API formats (`anthropic`, `openai_chat`, `openai_responses`, `google`) under one `api_key`. Each format has its own `base_url` and `models` list. Replaces the old flat `type`/`api_format` field on `ProviderConfig`.
- **Google Gemini native format support** (`google` endpoint) — full bidirectional Anthropic ↔ Gemini `generateContent` conversion:
  - Text messages, system prompt → `systemInstruction`
  - Tools → `functionDeclarations`; `tool_result` resolves function name from message history
  - Tool choice → `functionCallingConfig` (AUTO / NONE / ANY / specific function)
  - Image blocks → `inlineData` (base64) or `fileData` (URL)
  - `finishReason` mapping (STOP → `end_turn`, MAX_TOKENS → `max_tokens`, SAFETY → `end_turn`)
  - Full SSE streaming — text deltas, tool use blocks, usage in `message_delta`
- **Usage logging** — every request logged to `router_usage.db` (SQLite)
- **`simple-api-router usage`** subcommand — per-provider/model table with token counts and cost; supports `--last N`, `--period`, `--daily`, `--model`, `--provider`, `--format json`
- **`simple-api-router models`** subcommand — lists configured providers, endpoints, models with multimodal/text-only label and per-currency pricing
- **Multi-currency pricing** — `PricingEntry.currency` field (`"CNY"` default, `"USD"` supported); usage table shows `¥ Cost` and `$ Cost` columns separately so mixed-currency configs are reported accurately
- **Tiered pricing** (`PricingEntry.tiers`) for models like Gemini 2.5 Pro; entire request billed at the matching tier
- **Inline pricing** on `ModelEntry` — pricing attached directly to the model entry; falls back to top-level `pricing:` section
- **Multimodal fallback routing** — `text_only` models automatically re-route image/video requests to a configurable fallback model (`server.multimodal_fallback` or per-model `multimodal_fallback`)
- **Service management CLI** (`install`, `uninstall`, `start`, `stop`, `restart`, `status`, `log`) — replaces `scripts/service.sh`; supports both macOS launchd and Linux systemd
- `EndpointConfig` model with `base_url`, `models`, `model_map`, `deepseek_reasoning`, default URL resolution per format
- `ProviderConfig.find_model()` — exact match first, wildcard (empty `models`) as fallback; duplicate model detection across endpoints

### Fixed
- **Streaming errors return correct HTTP status codes** — upstream 401/403/404 now propagate as the actual HTTP status instead of always 200
- **Retry exhaustion returns last upstream status code** — persistent 429/503 now propagated correctly instead of always 502
- **Unexpanded `${VAR}` placeholders raise at startup** — `load_config` detects unset env vars and raises `ValueError` listing all affected paths
- **Dual auth headers for Anthropic-type providers** — both `x-api-key` and `Authorization: Bearer` sent (fixes ollama.com compatibility)
- **Non-JSON upstream error bodies handled** — plain-text error responses no longer cause 500
- **Cache token fallback** — if `cache_read`/`cache_write` pricing is absent, those tokens are billed at the `input` rate

## [0.1.0] — Initial Release

### Added
- Unified Anthropic Messages API (`POST /v1/messages`) routing to multiple backends
- `provider/model` routing via model name prefix
- `type: anthropic` backend — pure HTTP proxy, zero conversion
- `type: openai` backend — full bidirectional Anthropic ↔ OpenAI Chat Completions conversion:
  - Text, multi-turn, system prompts
  - Tool use / function calling (including streaming `input_json_delta`)
  - Vision (base64 + URL images)
  - Extended thinking → OpenAI reasoning effort
  - Cache control preservation
  - Streaming SSE with all Anthropic event types
- `model_map` per provider for external→backend name remapping
- `GET /v1/models`, `GET /health`, `GET /stats`, `GET /stats/data` endpoints
- YAML config with `${ENV_VAR}` expansion
- 64 tests including ported cc-switch-cli test suite
