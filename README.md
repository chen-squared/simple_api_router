# simple_api_router

A lightweight FastAPI proxy that routes OpenAI- and Anthropic-format requests across multiple backend AI API keys/providers, with automatic fallback, rate limiting, request-quota enforcement, and dollar-based budget control.

---

## Features

- **Unified endpoint** — accepts both OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) request formats and auto-converts between them as needed
- **Sequential fallback** — try APIs in order; move to the next one on error, rate-limit, or quota exhaustion
- **Load balancing** — distribute requests across backends weighted by their RPM capacity
- **Nested groups** — mix sequential and load_balance groups arbitrarily deep
- **RPM sliding window** — atomic per-API requests-per-minute enforcement (concurrent-safe)
- **Period request quotas** — daily / per-5h / weekly request count limits (not token counts)
- **Dollar budget** — estimate cost from input/output token prices; enforce daily / weekly / monthly USD caps
- **Retry & cooldown** — per-error-code retry limits, consecutive-failure cooldown, configurable no-retry duration after quota/budget exceeded
- **Streaming support** — SSE pass-through and cross-format stream conversion with safe resource cleanup
- **Environment variable expansion** — use `${VAR}` in config values
- **Stats & health endpoints** — `/health` and `/stats` for observability

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Copy and edit the provided `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080

default_group: main

apis:
  openai-primary:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    type: openai
    usage:
      rpm: 60
      daily_requests: 1000
      budget:
        input_price_per_1m: 2.50
        output_price_per_1m: 10.00
        daily: 5.0

  anthropic-fallback:
    base_url: "https://api.anthropic.com"
    api_key: "${ANTHROPIC_API_KEY}"
    type: anthropic

groups:
  main:
    strategy: sequential
    members:
      - api: openai-primary
      - api: anthropic-fallback

default_group: main
```

### 3. Run

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --config config.yaml
```

Or with a custom port:

```bash
python main.py --config config.yaml --port 9090
```

### 4. Make requests

The router accepts standard OpenAI and Anthropic payloads at the same host:

```bash
# OpenAI format
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'

# Anthropic format
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "Hello"}]}'
```

The router returns the response in the **same format the client used**, regardless of which backend handled the request.

---

## Configuration Reference

### `server`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Bind address |
| `port` | int | `8080` | Listen port |
| `log_level` | string | `"INFO"` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `log_file` | string | `"router.log"` | Log file path (rotating, 10 MB × 5 files). `null` to disable file logging |

### `apis.<name>`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_url` | string | ✓ | Backend base URL. For OpenAI-type APIs, include `/v1` (e.g. `https://api.openai.com/v1`). For Anthropic-type, omit it (e.g. `https://api.anthropic.com`) |
| `api_key` | string | ✓ | API key; supports `${ENV_VAR}` expansion |
| `type` | string | ✓ | `"openai"` or `"anthropic"` |
| `model` | string | — | Override model name sent to the backend (useful for provider-specific model names) |
| `endpoint_path` | string | — | Override the full request path (default: `/chat/completions` for OpenAI, `/v1/messages` for Anthropic) |

#### `apis.<name>.retry`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | int | `3` | Max total retry attempts per request |
| `cooldown_after` | int | `5` | Consecutive failures before triggering cooldown |
| `cooldown_duration` | int | `300` | Cooldown duration in seconds |
| `error_limits` | map(int→int) | `{}` | Per-HTTP-status max retries, e.g. `{429: 1, 500: 3}` |

#### `apis.<name>.usage`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rpm` | int | — | Max requests per minute (sliding 60-second window). Atomic enforcement under concurrency |
| `daily_requests` | int | — | Max requests per 24-hour rolling window |
| `per_5h_requests` | int | — | Max requests per 5-hour rolling window |
| `weekly_requests` | int | — | Max requests per 7-day rolling window |
| `no_retry_duration` | int | `3600` | Seconds to skip this API after request quota is exceeded **and** a 429 is received |

#### `apis.<name>.usage.budget`

Dollar-based budget estimation using token counts from API responses.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_price_per_1m` | float | `0.0` | USD per 1 million input/prompt tokens |
| `output_price_per_1m` | float | `0.0` | USD per 1 million output/completion tokens |
| `daily` | float | — | Max USD spend per 24-hour rolling window |
| `weekly` | float | — | Max USD spend per 7-day rolling window |
| `monthly` | float | — | Max USD spend per 30-day rolling window |
| `no_retry_duration` | int | `3600` | Seconds to skip this API after budget is exceeded **and** a 429 is received |

> **Note:** Budget tracking requires the upstream API to return token usage in its response. If usage data is missing (e.g. streaming without usage chunks), that request contributes zero to budget.

### `groups.<name>`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | string | `"sequential"` | `"sequential"` — try members in order until one succeeds. `"load_balance"` — pick a member randomly weighted by RPM capacity |
| `members` | list | `[]` | Ordered list of `{api: <name>}` or `{group: <name>}` entries |

### `default_group`

Name of the group that handles all incoming requests.

---

## Routing Logic

### Sequential strategy

1. Try each member in order.
2. If a member is **unavailable** (cooldown, RPM limit exceeded, request quota exceeded, budget exceeded), skip it immediately.
3. If a member **returns an error**, retry up to `max_retries` times, then move to the next member.
4. If all members fail or are unavailable, return HTTP 503.

### Load balance strategy

1. Compute weights from each member's `rpm` value (default weight = 1).
2. Randomly pick one member proportional to weight, skipping unavailable ones.
3. If selected member fails, fall back to sequential order through remaining members.

### Availability checks (order of precedence)

1. **Cooldown**: too many consecutive errors → skip for `cooldown_duration` seconds
2. **RPM limit**: sliding 60-second window full → skip
3. **Request quota exceeded**: daily / per-5h / weekly request count full → skip (or enter `no_retry_duration` block after 429)
4. **Budget exceeded**: estimated dollar spend over limit → skip (or enter budget `no_retry_duration` block after 429)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-format chat completions |
| `POST` | `/v1/messages` | Anthropic-format messages |
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/stats` | Per-API usage statistics |

### Stats response example

```json
{
  "openai-primary": {
    "total_requests": 150,
    "total_success": 148,
    "total_errors": 2,
    "daily_requests": 42,
    "per_5h_requests": 12,
    "weekly_requests": 150,
    "budget_spend": {
      "daily": 0.043,
      "weekly": 0.21,
      "monthly": 0.21
    }
  }
}
```

---

## Format Conversion

| Client sends | Backend type | Conversion |
|---|---|---|
| OpenAI | OpenAI | Pass-through |
| OpenAI | Anthropic | Request + response converted |
| Anthropic | Anthropic | Pass-through |
| Anthropic | OpenAI | Request + response converted |

Streaming responses are also converted in the SSE stream format (chunked). The client always receives a response in the format it requested.

---

## Development

### Run tests

```bash
PYTHONPATH=. venv/bin/python3.13 -m pytest tests/ -v
```

The test suite starts a local mock API server (port 19999) and the router itself (port 18080). **159 tests** covering unit, integration, edge cases, and real-world usage scenarios.

### Mock API server (for local testing)

```bash
python -m uvicorn mock_api.server:app --port 19999
```

Endpoints simulated:
- `POST /v1/chat/completions` — success with token usage
- `POST /v1/chat/completions/error/{code}` — returns HTTP `{code}`
- `POST /v1/chat/completions/slow` — 2-second delay then success
- `POST /v1/chat/completions/flaky/{n}` — fails first `n-1` times, then succeeds
- Anthropic equivalents at `/v1/messages/...`

### Project structure

```
simple_api_router/
├── main.py                 # CLI entry point
├── config.yaml             # Example configuration
├── requirements.txt
├── router/
│   ├── app.py              # FastAPI app factory
│   ├── config.py           # Pydantic config models + YAML loader
│   ├── converter.py        # OpenAI ↔ Anthropic format conversion
│   ├── endpoint.py         # Single-backend proxy with retry loop
│   ├── group.py            # Group routing (sequential / load_balance)
│   ├── logger.py           # Rotating file + console logging
│   ├── retry.py            # Error-count and cooldown tracking
│   └── usage.py            # RPM, request quota, and budget tracking
├── mock_api/
│   └── server.py           # Local mock backend for testing
└── tests/
    ├── test_router.py       # Core routing and usage tests
    ├── test_edge_cases.py   # Format conversion, RPM, streaming edge cases
    ├── test_qa_round2.py    # Model override, streaming cleanup, load balance
    ├── test_qa_round3.py    # Concurrent requests, non-JSON errors, content blocks
    └── test_scenarios.py    # Real-world usage scenario tests
```

---

## License

MIT
