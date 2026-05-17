# Simple API Router

A lightweight multi-provider LLM API router that exposes a **unified Anthropic Messages API** (`/v1/messages`) and routes requests to multiple backend providers — Anthropic, OpenAI, Google Gemini, DeepSeek, or any OpenAI-compatible endpoint.

Designed for use with tools like [Claude Code](https://claude.ai/code) that speak the Anthropic API but need flexible model routing.

---

## Features

- **Unified Anthropic API** — any Anthropic-compatible client (Claude Code, etc.) works out of the box
- **`provider/model` routing** — `model: "openai/gpt-4o"`, `model: "deepseek/deepseek-chat"`, `model: "ant2/claude-opus-4-5"`, etc.
- **Anthropic backend** — pure HTTP proxy; all features (extended thinking, prompt caching, `anthropic-beta` headers, streaming) pass through verbatim
- **OpenAI backend** — full bidirectional Anthropic ↔ OpenAI Chat Completions conversion:
  - Text, multi-turn, system prompts, tool use / function calling
  - Vision (base64 + URL images), streaming SSE with all event types
  - Extended thinking → reasoning effort; `input_json_delta` streaming
  - **Responses API** (`openai_responses` endpoint) for `/v1/responses`
- **Google Gemini backend** — full bidirectional Anthropic ↔ Gemini `generateContent` conversion:
  - System prompt, tools, tool choice, images, streaming
- **DeepSeek reasoning passthrough** — `reasoning_content` preserved; auto-enabled for `deepseek-*` models
- **Multimodal fallback routing** — text-only models automatically re-route image/video requests to a configurable multimodal model
- **Usage logging** — every request logged to `router.usage.jsonl`; view with `simple-api-router usage`
- **Hot reload** — provider/model/key changes apply within a second, no restart needed
- **`model_map`** — remap external model names to backend names per endpoint
- **1M context suffix** — Claude Code's `[1m]` suffix stripped before forwarding

---

## Quick Start

### 1. Install

```bash
pip install -e .
# with dev/test dependencies:
pip install -e ".[dev]"
```

### 2. Configure

The default config location is `~/.config/simple-api-router/config.yaml`.

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  log_level: "INFO"
  log_file: "router.log"   # relative to config directory; null = stdout only
  max_retries: 3

providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    endpoints:
      anthropic:
        models:
          - claude-opus-4-5
          - claude-sonnet-4-5

  openai:
    api_key: "${OPENAI_API_KEY}"
    endpoints:
      openai_chat:
        models: [gpt-4o, gpt-4o-mini]
      openai_responses:
        models: [o3, o4-mini]

  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    endpoints:
      openai_chat:
        base_url: "https://api.deepseek.com"
        models: [deepseek-chat, deepseek-reasoner]
```

Store API keys in `~/.config/simple-api-router/env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

### 3. Run

```bash
# Foreground (for testing)
simple-api-router

# Explicit config path
simple-api-router --config /path/to/config.yaml
```

### 4. Run as a background service

```bash
# Install and start (macOS launchd / Linux systemd)
simple-api-router install

# Control commands
simple-api-router start
simple-api-router stop
simple-api-router restart
simple-api-router status
simple-api-router log        # tail live logs
simple-api-router uninstall
```

**macOS:** installs to `~/Library/LaunchAgents/` (auto-starts on login).  
**Linux:** installs to `~/.config/systemd/user/` (systemd user unit).

### 5. Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 claude
```

Then pick any configured model:

```
/model anthropic/claude-opus-4-5
/model openai/gpt-4o
/model deepseek/deepseek-reasoner
/model google/gemini-2.5-pro
```

---

## Configuration Reference

### `server`

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `"0.0.0.0"` | Bind host |
| `port` | `8080` | Bind port |
| `log_level` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `log_file` | `"router.log"` | Log file path (relative = next to config); `null` = stdout only |
| `max_retries` | `3` | Retry attempts on upstream 429/5xx/network errors |
| `multimodal_fallback` | `null` | Global fallback model for text-only → multimodal re-routing (`"provider/model"`) |

### `providers`

Each key is the **provider name** used as the prefix in `provider/model` routing.

```yaml
providers:
  <provider-name>:
    api_key: "${ENV_VAR}"     # supports ${ENV_VAR} expansion; omit for keyless endpoints
    base_url: "..."           # shared base URL; individual endpoints inherit unless overridden
    endpoints:
      <format>:               # one of: anthropic, openai_chat, openai_responses, google
        base_url: "..."       # endpoint-level override (optional)
        models:               # list of model names (plain strings or dicts)
          - plain-model-name
          - name: model-name
            text_only: true
            multimodal_fallback: "provider/model"
            pricing:
              input: 3.0
              output: 15.0
        model_map:            # remap client name → backend name (optional)
          client-name: backend-name
        deepseek_reasoning: true | false   # None = auto-detect from model name
```

#### Endpoint formats

| Format | Default base URL | Use case |
|---|---|---|
| `anthropic` | `https://api.anthropic.com` | Pure proxy — all Anthropic features pass through |
| `openai_chat` | `https://api.openai.com` | OpenAI, DeepSeek, Ollama, vLLM, LM Studio, etc. |
| `openai_responses` | `https://api.openai.com` | OpenAI Responses API (`/v1/responses`) |
| `google` | `https://generativelanguage.googleapis.com` | Google Gemini native API |

A single provider can have multiple endpoint formats, each with its own model list.

#### `text_only` and multimodal fallback

Mark text-only models so the router re-routes image/video requests instead of forwarding them and getting an error:

```yaml
server:
  multimodal_fallback: "google/gemini-2.5-flash"   # global default

providers:
  local:
    endpoints:
      openai_chat:
        base_url: "http://localhost:11434"
        models:
          - llava                              # plain string = multimodal capable
          - name: deepseek-r1
            text_only: true                   # uses server.multimodal_fallback
          - name: qwen2.5-coder:32b
            text_only: true
            multimodal_fallback: "local/llava" # per-model override
```

Priority: **per-model `multimodal_fallback`** > **`server.multimodal_fallback`** > forward as-is (with a warning).

### `pricing`

Pricing can be set **inline** on each model entry (recommended) or in a top-level `pricing` section (fallback). Prices are per million tokens.

```yaml
providers:
  anthropic:
    endpoints:
      anthropic:
        models:
          - name: claude-opus-4-5
            pricing:
              currency: USD          # "CNY" (default) or "USD"
              input: 15.0
              output: 75.0
              cache_read: 1.50
              cache_write: 18.75

  deepseek:
    endpoints:
      openai_chat:
        models:
          - name: deepseek-chat
            pricing:
              currency: CNY
              input: 1.0
              output: 2.0
              cache_read: 0.1        # omit = billed at input rate
              cache_write: 1.0       # omit = billed at input rate

  google:
    endpoints:
      google:
        models:
          - name: gemini-2.5-pro
            pricing:
              currency: USD
              tiers:                 # tiered: whole request billed at matching tier
                - threshold: 0       # input_tokens < 200 K
                  input: 1.25
                  output: 10.0
                  cache_read: 0.31
                - threshold: 200000  # input_tokens ≥ 200 K
                  input: 2.50
                  output: 15.0
                  cache_read: 0.625
```

Top-level fallback (when inline pricing is absent):

```yaml
pricing:
  "anthropic/claude-opus-4-5":
    currency: USD
    input: 15.0
    output: 75.0
```

---

## CLI Reference

### `simple-api-router [run]`

Start the API server.

```
simple-api-router [--config PATH] [--env-file PATH]
simple-api-router run [--config PATH] [--env-file PATH]
```

### `simple-api-router models`

List all configured providers and models with capability and pricing info.

```
simple-api-router models [--config PATH]
```

Output example:
```
anthropic  [anthropic]
  claude-opus-4-5                                    multimodal  ¥109.00in ¥545.00out ¥10.89cr ¥136.73cw  /MTok
  claude-sonnet-4-5                                  multimodal  ¥21.82in ¥109.10out ¥2.18cr ¥27.27cw  /MTok

openai  [openai_chat]
  gpt-4o                                             multimodal
```

### `simple-api-router usage`

Show API usage statistics from the usage log.

```
simple-api-router usage [--last N] [--period day|week|month]
                        [--daily] [--model PATTERN] [--provider NAME]
                        [--format table|json] [--config PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--last N` | `7` | Number of days to include |
| `--period` | — | Preset: `day` (1), `week` (7), `month` (30) |
| `--daily` | off | Show per-day breakdown instead of summary |
| `--model PATTERN` | — | Filter by model name (substring) |
| `--provider NAME` | — | Filter by provider name |
| `--format` | `table` | `table` or `json` |

The `¥ Cost` and `$ Cost` columns show costs in CNY and USD respectively; a `-` means no pricing is configured for that model.

### Service management commands

```
simple-api-router install [--config PATH] [--exe PATH]
simple-api-router uninstall
simple-api-router start | stop | restart | status | log
```

---

## HTTP API

### `POST /v1/messages`

Standard [Anthropic Messages API](https://docs.anthropic.com/en/api/messages). Use `model: "provider/model"` to route:

```bash
curl http://localhost:8080/v1/messages \
  -H "x-api-key: any" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"Hello!"}]}'
```

If the provider prefix is omitted (e.g. `"model": "claude-opus-4-5"`), the first configured provider that lists that model is used.

### `GET /v1/models`

Returns all configured models in Anthropic-compatible format.

### `GET /health`

```json
{"status": "ok", "uptime_seconds": 42.1, "providers": ["anthropic", "openai"]}
```

### `GET /stats`

Returns per-provider model lists and base URLs.

---

## Advanced Examples

### Multiple Anthropic Accounts

```yaml
providers:
  ant1:
    api_key: "${ANTHROPIC_KEY_1}"
    endpoints:
      anthropic:
        models: [claude-opus-4-5, claude-sonnet-4-5]
  ant2:
    api_key: "${ANTHROPIC_KEY_2}"
    endpoints:
      anthropic:
        models: [claude-opus-4-5, claude-sonnet-4-5]
```

Use `model: "ant1/claude-opus-4-5"` or `model: "ant2/claude-opus-4-5"`.

### Local / Self-Hosted (Ollama, vLLM, LM Studio)

```yaml
providers:
  local:
    endpoints:
      openai_chat:
        base_url: "http://localhost:11434"
        models: [llama3.2, qwen2.5-coder]
```

The trailing `/v1` on a `base_url` is stripped automatically.

### Multi-format provider (one key, multiple API styles)

```yaml
providers:
  myupstream:
    api_key: "${MY_KEY}"
    base_url: "https://api.myupstream.com"
    endpoints:
      anthropic:
        models: [claude-opus-4-5]
      openai_chat:
        models: [gpt-4o]
      google:
        models: [gemini-2.5-pro]
```

### Model Name Remapping

```yaml
providers:
  myapi:
    api_key: "${MY_KEY}"
    endpoints:
      openai_chat:
        base_url: "https://api.my-provider.com/v1"
        models: [fast, smart]
        model_map:
          fast: gpt-4o-mini
          smart: gpt-4o
```

Client sends `model: "myapi/fast"`; backend receives `gpt-4o-mini`.

---

## How Conversion Works

| Anthropic concept | OpenAI equivalent |
|---|---|
| `system` (string or array) | `messages[0].role = "system"` |
| `content[].type = "image"` | `content[].type = "image_url"` |
| `content[].type = "tool_use"` | `tool_calls[].function` |
| `content[].type = "tool_result"` | `role = "tool"` message |
| `content[].type = "thinking"` | `reasoning_content` (DeepSeek) / `reasoning.effort` (Responses API) |
| `cache_control` | forwarded as-is (provider-dependent) |
| `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` |
| Streaming `content_block_delta` | Streaming `delta.content` / `delta.tool_calls` |

| Anthropic concept | Google Gemini equivalent |
|---|---|
| `system` | `systemInstruction` |
| `content[].type = "image"` | `inlineData` (base64) or `fileData` (URL) |
| `content[].type = "tool_use"` | `functionCall` |
| `content[].type = "tool_result"` | `functionResponse` |
| `tools` | `functionDeclarations` |
| `tool_choice` | `functionCallingConfig` (AUTO / NONE / ANY / specific) |

---

## Development

```bash
# Run all tests
python -m pytest tests/ -v

# Run with auto-reload during development
uvicorn simple_api_router.app:app --reload --port 8080
```

### Module Structure

```
simple_api_router/
  config.py           — Pydantic config models + YAML loader with ${ENV_VAR} expansion
  app.py              — FastAPI application factory and endpoint wiring
  proxy.py            — Request routing, provider resolution, dispatch
  converter.py        — Anthropic ↔ OpenAI conversion (request/response/streaming)
  converter_google.py — Anthropic ↔ Google Gemini conversion (request/response/streaming)
  usage_logger.py     — Per-request JSONL usage logging (router.usage.jsonl)
  usage_cli.py        — `usage` subcommand: load/aggregate/display usage stats
  service.py          — Service management (launchd / systemd install/start/stop/…)
  logger.py           — Logging setup
  cli.py              — CLI entry point
```

---

## License

MIT
