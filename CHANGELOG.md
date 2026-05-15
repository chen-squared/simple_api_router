# Changelog

## [Unreleased]

### Added
- OpenAI **Responses API** support (`api_format: openai_responses`) — full request, response, and streaming conversion
- **DeepSeek `reasoning_content`** passthrough — thinking blocks preserved across request/response/streaming; auto-enabled for any `deepseek-*` model (`deepseek_reasoning` config option)
- `[1m]` context suffix handling — Claude Code's `model[1m]` suffix stripped before forwarding; full 1M context window honoured
- `api_format` and `deepseek_reasoning` fields on `ProviderConfig`
- 33 new tests for Responses API, DeepSeek reasoning, and config validation (97 total)

### Fixed
- **Streaming errors return correct HTTP status codes** — upstream 401/403/404 now propagate as the actual HTTP status instead of always 200; achieved by inspecting upstream response headers before committing to `StreamingResponse`
- **Retry exhaustion returns last upstream status code** — persistent 429/503 now propagated correctly instead of always 502; network-level exhaustion still returns 502
- **Unexpanded `${VAR}` placeholders raise at startup** — `load_config` detects unset env vars and raises `ValueError` listing all affected paths, preventing silent auth failures
- **Dual auth headers for Anthropic-type providers** — both `x-api-key` and `Authorization: Bearer` are sent; providers use whichever they recognise (fixes ollama.com compatibility without extra config)
- **Fake key values ignored** — `api_key: "none"` / `"null"` / `"false"` / `"no"` / `"0"` treated as absent; no spurious auth header sent to upstream
- **Non-JSON upstream error bodies handled** — providers returning plain-text error bodies (e.g. ollama.com's `"unauthorized\n"`) no longer cause 500; body falls back to raw text
- **`service.sh status` false negative on macOS** — replaced `launchctl list | grep` (SIGPIPE + pipefail caused false "not loaded") with `launchctl list <label>` (direct exit code)
- 21 new tests for the above fixes (118 total)

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
- `GET /v1/models`, `GET /health`, `GET /stats` endpoints
- YAML config with `${ENV_VAR}` expansion
- 64 tests including ported cc-switch-cli test suite
