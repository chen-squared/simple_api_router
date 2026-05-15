# Simple API Router

A lightweight multi-provider LLM API router that exposes a **unified Anthropic Messages API** (`/v1/messages`) and routes requests to multiple backend providers — Anthropic, OpenAI, DeepSeek, or any OpenAI-compatible endpoint.

Designed for use with tools like [Claude Code](https://claude.ai/code) that speak the Anthropic API but need flexible model routing.

---

## Features

- **Unified Anthropic API** — any Anthropic-compatible client (Claude Code, etc.) works out of the box
- **`provider/model` routing** — `model: "openai/gpt-4o"`, `model: "deepseek/deepseek-chat"`, `model: "ant2/claude-opus-4-5"`, etc.
- **Anthropic backend** — pure HTTP proxy, zero conversion; all Anthropic features (extended thinking, prompt caching, `anthropic-beta` headers, streaming) pass through verbatim
- **OpenAI backend** — full bidirectional Anthropic ↔ OpenAI conversion:
  - Text, multi-turn, system prompts
  - Tool use / function calling (including streaming `input_json_delta`)
  - Vision (base64 + URL images)
  - Extended thinking → OpenAI reasoning effort (adaptive maps to `high`)
  - Streaming SSE with all event types
  - **OpenAI Responses API** (`api_format: openai_responses`) for providers that use `/v1/responses`
- **DeepSeek reasoning passthrough** — `reasoning_content` preserved across request/response/streaming; auto-enabled for any `deepseek-*` model
- **1M context suffix** — Claude Code's `[1m]` model suffix is stripped before forwarding but the full context window is honoured
- **`model_map`** per provider — remap external model names to backend names
- **Model catalog** — `GET /v1/models` lists all configured models

---

## Quick Start

### 1. Install

```bash
pip install -e .
# with dev/test dependencies:
pip install -e ".[dev]"
```

### 2. Configure

Create or edit `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  max_retries: 3        # retry upstream errors (429/5xx/529/network), default 3

providers:
  anthropic:
    type: anthropic
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - claude-opus-4-5
      - claude-sonnet-4-5

  openai:
    type: openai
    api_key: "${OPENAI_API_KEY}"
    models:
      - gpt-4o
      - gpt-4o-mini

  deepseek:
    type: openai
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    deepseek_reasoning: true   # enable reasoning_content passthrough
    models:
      - deepseek-chat
      - deepseek-reasoner
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

### 3. Run

```bash
# Foreground (for testing)
simple-api-router --config config.yaml

# Or via python -m
python -m simple_api_router --config config.yaml
```

### 4. Run as a macOS background service (auto-start on login)

```bash
# From the project directory:
./scripts/service.sh install
```

This will:
- Auto-detect the `simple-api-router` executable
- Create `~/.config/simple-api-router/env` for your API keys (edit this file)
- Install a launchd plist to `~/Library/LaunchAgents/`
- Start the service immediately and on every login

**Control commands:**
```bash
./scripts/service.sh start    # start
./scripts/service.sh stop     # stop
./scripts/service.sh restart  # restart
./scripts/service.sh status   # show launchd state + recent logs
./scripts/service.sh log      # tail live logs
./scripts/service.sh uninstall
```

**Hot reload:** just save `config.yaml` — provider/model/key/retry changes apply automatically within a second, with no restart and no dropped connections. Changes to `host` or `port` require a restart.

### 5. Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 claude
```

Then pick any configured model:

```
/model deepseek/deepseek-reasoner
/model openai/gpt-4o
/model anthropic/claude-opus-4-5
```

---

## Configuration Reference

### `server`

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `"0.0.0.0"` | Bind host |
| `port` | `8080` | Bind port |
| `log_level` | `"INFO"` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `log_file` | `"router.log"` | Log file path; set to `null` to log to stdout only |

### `providers`

Each key is the provider name used as the prefix in `provider/model` routing:

```yaml
providers:
  <provider-name>:
    type: anthropic | openai   # required
    api_key: "..."             # required; supports ${ENV_VAR} expansion
    base_url: "..."            # optional; defaults to official endpoint
    models:                    # optional; if omitted, any model name is accepted
      - model-name
    model_map:                 # optional; remap external name → backend name
      external-name: backend-name
    api_format: openai_chat | openai_responses   # (openai type only) default: openai_chat
    deepseek_reasoning: true | false             # (openai type only) default: auto-detect
```

**`type: anthropic`** — pure HTTP proxy. All Anthropic headers and features (extended thinking, prompt caching, beta headers) pass through unchanged.

**`type: openai`** — full bidirectional conversion layer. Supports any OpenAI Chat Completions-compatible endpoint, plus the Responses API (`api_format: openai_responses`).

#### `api_format`

| Value | Endpoint | Use case |
|---|---|---|
| `openai_chat` (default) | `POST /chat/completions` | Standard OpenAI, DeepSeek, Ollama, etc. |
| `openai_responses` | `POST /responses` | OpenAI Responses API |

#### `deepseek_reasoning`

Controls whether `reasoning_content` is passed through in requests/responses:

- `true` — always enabled
- `false` — always disabled
- omitted — auto-detected from model name (enabled for any `deepseek-*` model)

---

## API Endpoints

### `POST /v1/messages`

Standard [Anthropic Messages API](https://docs.anthropic.com/en/api/messages). Use `model: "provider/model"` to route:

```bash
# Anthropic backend (pure proxy)
curl http://localhost:8080/v1/messages \
  -H "x-api-key: any" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"Hello!"}]}'

# OpenAI backend (converted)
curl http://localhost:8080/v1/messages \
  -H "x-api-key: any" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"Hello!"}]}'
```

If the provider prefix is omitted (e.g. `"model": "claude-opus-4-5"`), the first configured provider that lists that model is used.

### `GET /v1/models`

Returns all configured models in Anthropic-compatible format:

```json
{
  "object": "list",
  "data": [
    {"id": "anthropic/claude-opus-4-5", "object": "model", "owned_by": "anthropic"},
    {"id": "openai/gpt-4o", "object": "model", "owned_by": "openai"},
    {"id": "deepseek/deepseek-chat", "object": "model", "owned_by": "deepseek"}
  ]
}
```

### `GET /health`

```json
{"status": "ok", "uptime_seconds": 42.1, "providers": ["anthropic", "openai", "deepseek"]}
```

### `GET /stats`

```json
{
  "providers": {
    "anthropic": {"type": "anthropic", "models": ["claude-opus-4-5"], "base_url": "..."},
    "openai":    {"type": "openai",    "models": ["gpt-4o"], "base_url": "..."}
  }
}
```

---

## Advanced Examples

### Multiple Anthropic Accounts

Spread load across multiple API keys:

```yaml
providers:
  ant1:
    type: anthropic
    api_key: "${ANTHROPIC_KEY_1}"
    models: [claude-opus-4-5, claude-sonnet-4-5]
  ant2:
    type: anthropic
    api_key: "${ANTHROPIC_KEY_2}"
    models: [claude-opus-4-5, claude-sonnet-4-5]
```

Use `model: "ant1/claude-opus-4-5"` or `model: "ant2/claude-opus-4-5"`.

### Local / Self-Hosted (Ollama, vLLM, LM Studio)

```yaml
providers:
  local:
    type: openai
    api_key: "none"
    base_url: "http://localhost:11434/v1"
    models: [llama3.2, qwen2.5-coder]
```

### OpenAI Responses API

```yaml
providers:
  openai-responses:
    type: openai
    api_key: "${OPENAI_API_KEY}"
    api_format: openai_responses
    models: [gpt-4o, o3]
```

### Model Name Remapping

```yaml
providers:
  myapi:
    type: openai
    api_key: "${MY_KEY}"
    base_url: "https://api.my-provider.com/v1"
    models: [fast, smart]
    model_map:
      fast: gpt-4o-mini
      smart: gpt-4o
```

Client sends `model: "myapi/fast"`; backend receives `gpt-4o-mini`.

### 1M Context Models

Claude Code appends `[1m]` to model names to signal 1M-token context support. The router strips the suffix before forwarding and passes through all content — no truncation is applied.

```
/model deepseek/deepseek-chat[1m]
```

---

## How Conversion Works

For `type: openai` providers the router performs full bidirectional conversion between the Anthropic Messages format and OpenAI Chat Completions (or Responses API):

| Anthropic concept | OpenAI equivalent |
|---|---|
| `system` (string or array) | `messages[0].role = "system"` |
| `content[].type = "image_url"` | `content[].type = "image_url"` |
| `content[].type = "tool_use"` | `tool_calls[].function` |
| `content[].type = "tool_result"` | `role = "tool"` message |
| `content[].type = "thinking"` | `reasoning_content` (DeepSeek) / `reasoning.effort` (Responses API) |
| `cache_control` | forwarded as-is (provider-dependent) |
| `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` |
| Streaming `content_block_delta` | Streaming `delta.content` / `delta.tool_calls` |

---

## Development

```bash
# Run all tests (97 tests)
python -m pytest tests/ -v

# Run with auto-reload during development
uvicorn simple_api_router.app:app --reload --port 8080
```

### Module Structure

```
simple_api_router/
  config.py     — Pydantic config models + YAML loader with ${ENV_VAR} expansion
  app.py        — FastAPI application factory and endpoint wiring
  proxy.py      — Request routing, provider resolution, dispatch
  converter.py  — Stateless Anthropic ↔ OpenAI format conversion (request/response/streaming)
  logger.py     — Logging setup
  cli.py        — CLI entry point
```

---

## License

MIT
