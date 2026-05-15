# Changelog

## [Unreleased]

### Added
- OpenAI **Responses API** support (`api_format: openai_responses`) — full request, response, and streaming conversion
- **DeepSeek `reasoning_content`** passthrough — thinking blocks preserved across request/response/streaming; auto-enabled for any `deepseek-*` model (`deepseek_reasoning` config option)
- `[1m]` context suffix handling — Claude Code's `model[1m]` suffix stripped before forwarding; full 1M context window honoured
- `api_format` and `deepseek_reasoning` fields on `ProviderConfig`
- 33 new tests for Responses API, DeepSeek reasoning, and config validation (97 total)

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
