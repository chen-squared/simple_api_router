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
- **Per-type media fallback** — when a model receives image/audio/video/PDF content it doesn't support, the content is auto-described by a configurable fallback model and replaced with text; the original model then handles the request normally. Configurable per media type and per model.
- **Media MCP server** — optional `image_understanding`, `audio_understanding`, `video_understanding`, `pdf_understanding` MCP tools, mounted on the same port; lets models describe screenshots, audio, video, and PDFs on demand
- **`auto-config` command** — auto-generate config entries from [models.dev](https://models.dev) metadata: infers endpoint format, modalities, and pricing; smart-merges into existing config preserving user-set fields; `--dry-run` shows a git-style diff before writing
- **Usage logging** — every request logged to `router_usage.db` (SQLite); view with `simple-api-router usage` or `/stats`
- **Config GUI model tests** — `/config` page tests now send requests through the router’s own `/v1/messages` path, so they follow normal routing and appear in usage/stats
- **Hot reload** — provider/model/key changes apply within a second, no restart needed
- **`model_map`** — remap external model names to backend names per endpoint; global server-level aliases (`server.model_map`) let clients use short names without a `provider/` prefix, billing tracks the resolved model
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
          - name: claude-opus-4-5
            multimodality: [image]
          - name: claude-sonnet-4-5
            multimodality: [image]

  openai:
    api_key: "${OPENAI_API_KEY}"
    endpoints:
      openai_chat:
        models:
          - name: gpt-4o
            multimodality: [image]
          - name: gpt-4o-mini
            multimodality: [image]
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
| `image_fallback` | `null` | Global fallback model for image description (`"provider/model"`) |
| `audio_fallback` | `null` | Global fallback model for audio description (`"provider/model"`) |
| `video_fallback` | `null` | Global fallback model for video description (`"provider/model"`) |
| `pdf_fallback` | `null` | Global fallback model for PDF description (`"provider/model"`) |
| `multimodal_fallback_max_concurrency` | `3` | Max concurrent media description calls during fallback |
| `image_model` | `null` | Enable `image_understanding` MCP tool at `/mcp` (`"provider/model"`) |
| `audio_model` | `null` | Enable `audio_understanding` MCP tool at `/mcp` (`"provider/model"`) |
| `video_model` | `null` | Enable `video_understanding` MCP tool at `/mcp` (`"provider/model"`) |
| `pdf_model` | `null` | Enable `pdf_understanding` MCP tool at `/mcp` (`"provider/model"`) |
| `debug_log` | `null` | Path to debug log file; all 4 request/response stages logged per request |
| `model_map` | `{}` | Global model aliases (`alias: "provider/model"`); clients send the alias, billing uses the resolved model |

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
        models:               # list of model names (plain strings or ModelEntry dicts)
          - plain-model-name
          - name: model-name
            multimodality: [image, video]      # media types natively supported
            image_fallback: "provider/model"   # per-model override for image fallback
            audio_fallback: "provider/model"   # per-model override for audio fallback
            video_fallback: "provider/model"   # per-model override for video fallback
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

#### `multimodality` and per-type fallback

Each model declares which media types it natively supports via the `multimodality` list. When a model receives content of an unsupported type, the router automatically describes it using the appropriate fallback model before forwarding the request.

Supported media types: `image`, `audio`, `video`, `pdf`.

- Plain string models (no `ModelEntry`) default to `multimodality: []` — no media support.
- The fallback model generates a textual description, which replaces the original block.
- Results are cached (URL: 1 hour; base64/file: 30 days) to avoid redundant API calls.
- **PDF special case**: if a model doesn't support `pdf` and no `pdf_fallback`/`pdf_model` is configured, binary PDF blocks are stripped and replaced with a placeholder (rather than forwarded, which would cause an upstream error).

```yaml
server:
  image_fallback: "google/gemini-2.5-flash"   # global default for images
  audio_fallback: "openai/gpt-4o-audio-preview"
  video_fallback: "google/gemini-2.5-flash"
  pdf_fallback: "anthropic/claude-opus-4-5"   # global default for PDFs

providers:
  local:
    endpoints:
      openai_chat:
        base_url: "http://localhost:11434"
        models:
          - name: llava
            multimodality: [image]             # natively supports images
          - name: deepseek-r1                  # uses server.image_fallback for images
          - name: qwen2.5-coder:32b
            image_fallback: "local/llava"      # per-model override — use local llava
  anthropic:
    endpoints:
      anthropic:
        models:
          - name: claude-opus-4-5
            multimodality: [image, pdf]        # Claude natively supports images and PDFs
```

Priority: **per-model `*_fallback`** > **`server.*_fallback`** > **`server.*_model` MCP placeholder** > strip with placeholder (pdf) / forward as-is with warning (other types).

### `pricing`

Pricing can be set **inline** on each model entry (recommended) or in a top-level `pricing` section (fallback). Prices are per million tokens.

```yaml
providers:
  anthropic:
    endpoints:
      anthropic:
        models:
          - name: claude-opus-4-5
            multimodality: [image]
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
            multimodality: [image, video]
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
  claude-opus-4-5                                image pdf         $15.00in $75.00out $1.50cr $18.75cw  /MTok
  claude-sonnet-4-5                              image pdf         $3.00in $15.00out $0.30cr $3.75cw  /MTok

openai  [openai_chat]
  gpt-4o                                         image
  gpt-4o-mini                                    image

openai  [openai_responses]
  o3                                             text
  o4-mini                                        text
```

### `simple-api-router usage`

Show API usage statistics from the SQLite usage database.

```
simple-api-router usage [--last N] [--period day|week|month]
                        [--daily] [--model PATTERN] [--provider NAME]
                        [--format table|json] [--logs]
                        [--import-jsonl PATH] [--config PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--last N` | `7` | Number of days to include |
| `--period` | — | Preset: `day` (1), `week` (7), `month` (30) |
| `--daily` | off | Show per-day breakdown instead of summary |
| `--model PATTERN` | — | Filter by model name (substring) |
| `--provider NAME` | — | Filter by provider name |
| `--format` | `table` | `table` or `json` |
| `--logs` | off | Print each raw record as a JSON line (like the old JSONL file) |
| `--import-jsonl PATH` | — | Import records from a legacy JSONL file into the usage database |

The `¥ Cost` and `$ Cost` columns show costs in CNY and USD respectively; a `-` means no pricing is configured for that model. Cost is computed at query time from the current config — pricing changes apply retroactively.

### `simple-api-router test`

Quickly test whether a model is correctly configured and reachable.

```
simple-api-router test <provider/model> [--config PATH]
```

Sends a minimal "Hello" message and prints the response. Exits with a non-zero status on any error.

### `simple-api-router auto-config`

Automatically generate config entries for online providers by fetching model metadata from [models.dev](https://models.dev).

```
simple-api-router auto-config <online-provider> [<online-model-id>]
                              [--provider <local-provider>]
                              [--model <local-model-id>]
                              [--dry-run]
                              [--config PATH]
```

| Argument | Description |
|----------|-------------|
| `<online-provider>` | Provider name as listed on models.dev (e.g. `anthropic`, `openai`, `openrouter`) |
| `<online-model-id>` | Specific model to add; if omitted, adds all models for the provider |
| `--provider` | Local provider name to write to (defaults to the online provider name) |
| `--model` | Local model ID to use in config (defaults to the online model ID) |
| `--dry-run` | Show a git-style diff of what would change without writing to disk |

When adding new models, `auto-config`:
- Infers the endpoint format from the provider's SDK metadata
- Detects supported modalities (image, audio, video, pdf) from the model's capabilities
- Sets pricing from models.dev (formatted to ≥ 2 decimal places)
- Backs up `config.yaml` to `config.yaml.bak` before writing

When updating existing models, only `multimodality` and `pricing` are overwritten — user-set fields like `max_reasoning_effort` and per-model fallbacks are preserved.

**Examples:**

```bash
# Preview all Anthropic models (dry run)
simple-api-router auto-config anthropic --dry-run

# Add a single model from OpenRouter under a local provider name
simple-api-router auto-config openrouter qwen3.7-max \
    --provider openrouter --model qwen/qwen3.7-max

# Add all Gemini models to a provider named "google"
simple-api-router auto-config google --provider google
```

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

Returns a minimal HTML usage dashboard with:

- period controls for **last N days** or a **date range** (`from` / `to`; a single day is `from == to`)
- model/day aggregate tables
- recent requests filtered to the selected period
- recent request filters for **provider** and **model**
- quick links to `/stats/data` and `/config`

### `GET /stats/data`

Returns the same usage data as JSON for programmatic access. Supports the same
date-range query parameters for recent requests, plus `provider` / `model`
filters on the `recent` section.

---

## Media MCP Server

The router mounts an [MCP](https://modelcontextprotocol.io/) server at `/mcp` on the **same port** (Streamable HTTP transport). The exposed tool list is driven by `server.image_model`, `server.audio_model`, `server.video_model`, and `server.pdf_model`, so MCP-capable clients (e.g. Claude Code) will only see the understanding tools whose backing models are currently configured.

Requests from the MCP tools go through the router's own `/v1/messages` endpoint, so they appear in usage logs and are subject to the same routing rules.

### Setup

**1. Configure any media models you want in `config.yaml`:**

```yaml
server:
  port: 8080
  image_model: "google/gemini-2.5-flash"        # enables image_understanding
  audio_model: "openai/gpt-4o-audio-preview"    # enables audio_understanding
  video_model: "google/gemini-2.5-flash"        # enables video_understanding
  pdf_model: "anthropic/claude-opus-4-5"        # enables pdf_understanding
```

`/mcp` is always mounted. Adding, removing, or changing any of those four `server.*_model` values hot-reloads the available MCP tools without restarting the router.

**2. Add to Claude Code (`~/.claude/settings.json`):**

```json
{
  "mcpServers": {
    "media": {
      "type": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### Tool signatures

All four tools share the same input schema: provide **exactly one** of `path`, `url`, or `base64_data`.

```
image_understanding(
    path?        — absolute or relative path to a local image file
    url?         — HTTPS URL of an image
    base64_data? — raw base64-encoded bytes (no data: prefix)
    media_type?  — MIME type for base64 input (default: "image/jpeg")
    question?    — what to ask (default: "Please describe this image in detail.")
    max_tokens?  — response length limit (default: 16384)
)

audio_understanding(
    path?        — absolute or relative path to a local audio file
    url?         — HTTPS URL of an audio file
    base64_data? — raw base64-encoded bytes (no data: prefix)
    media_type?  — MIME type for base64 input (default: "audio/mp3")
    question?    — what to ask (default: "Please transcribe and describe this audio in detail.")
    max_tokens?  — response length limit (default: 16384)
)

video_understanding(
    path?        — absolute or relative path to a local video file
    url?         — HTTPS URL of a video
    base64_data? — raw base64-encoded bytes (no data: prefix)
    media_type?  — MIME type for base64 input (default: "video/mp4")
    question?    — what to ask (default: "Please describe this video in detail.")
    max_tokens?  — response length limit (default: 16384)
)

pdf_understanding(
    path?        — absolute or relative path to a local PDF file
    url?         — HTTPS URL of a PDF
    base64_data? — raw base64-encoded bytes (no data: prefix)
    question?    — what to ask (default: "Please read and summarize this PDF document in detail.")
    max_tokens?  — response length limit (default: 16384)
)
```

### Standalone mode

If you want the MCP server on a separate port (or without the main router):

```bash
python -m simple_api_router.mcp_media \
    --image-model google/gemini-2.5-flash \
    --audio-model openai/gpt-4o-audio-preview \
    --video-model google/gemini-2.5-flash \
    --pdf-model anthropic/claude-opus-4-5 \
    --router-url http://localhost:8080 \
    --port 8081
```

Claude Code config for standalone mode:

```json
{
  "mcpServers": {
    "media": {
      "type": "http",
      "url": "http://localhost:8081/mcp"
    }
  }
}
```

---

## Advanced Examples

### Multiple Anthropic Accounts

```yaml
providers:
  ant1:
    api_key: "${ANTHROPIC_KEY_1}"
    endpoints:
      anthropic:
        models:
          - name: claude-opus-4-5
            multimodality: [image, pdf]
          - name: claude-sonnet-4-5
            multimodality: [image, pdf]
  ant2:
    api_key: "${ANTHROPIC_KEY_2}"
    endpoints:
      anthropic:
        models:
          - name: claude-opus-4-5
            multimodality: [image, pdf]
          - name: claude-sonnet-4-5
            multimodality: [image, pdf]
```

Use `model: "ant1/claude-opus-4-5"` or `model: "ant2/claude-opus-4-5"`.

### Local / Self-Hosted (Ollama, vLLM, LM Studio)

```yaml
server:
  image_fallback: "local/llava"   # use local llava for images when needed

providers:
  local:
    endpoints:
      openai_chat:
        base_url: "http://localhost:11434"
        models:
          - name: llava
            multimodality: [image]   # natively handles images
          - name: qwen2.5-coder      # text-only; images auto-described via server.image_fallback
          - deepseek-r1              # plain string = no media support
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
        models:
          - name: claude-opus-4-5
            multimodality: [image, pdf]
      openai_chat:
        models:
          - name: gpt-4o
            multimodality: [image]
      google:
        models:
          - name: gemini-2.5-pro
            multimodality: [image, video]
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

### Mid-conversation system messages

Anthropic's Messages API supports **mid-conversation system messages** —
`{"role": "system"}` entries placed *inside* the `messages` array (not just the
top-level `system` field). Claude Code uses these to relay a message the user
typed **while the model was still working** (it appears right after a tool
result as *"The user sent a new message while you were working: …"*).

- **Anthropic backend (passthrough):** these are forwarded **verbatim** — the
  feature works natively on models that support it (e.g. Claude Opus 4.8).
- **Converted backends (`openai_chat` / `openai_responses` / `google`):** the
  target models have no equivalent, and a non-leading `system` message is
  unreliable there — weaker models silently ignore it, and some strict
  OpenAI-compatible providers (e.g. MiniMax) **reject the whole request**
  (`invalid message role: system`). So the router **folds each mid-conversation
  `system` message into the adjacent `user` turn** as plain text, preserving its
  order and full content.

  Concretely, a `system` message is merged into the **preceding** `user`/tool-result
  turn (where Anthropic requires it to sit); if there is no preceding user turn it
  is prepended to the next one. Merging (rather than emitting a standalone `user`
  message) guarantees the request never contains two consecutive same-role turns,
  which backends like `deepseek-reasoner` and Gemini reject.

  **Effect you will observe:** with a converted backend, an operator/queued
  message keeps full **content** but loses its system-level **priority** (it is
  seen as user input). The model still receives it at the correct position in the
  conversation; whether it acts on it immediately or after finishing the current
  task depends on the model. The top-level `system` field is never affected.

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
  mcp_media.py        — Media MCP server (image/audio/video/PDF understanding tools, mounted at /mcp)
  usage_db.py         — Per-request SQLite usage logging and query helpers
  usage_cli.py        — `usage` subcommand: load/aggregate/display usage stats
  service.py          — Service management (launchd / systemd install/start/stop/…)
  logger.py           — Logging setup
  cli.py              — CLI entry point
```

---

## License

MIT
