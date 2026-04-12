# simple_api_router — Progress Log

---

## Iteration 1 — 2026-04-12T17:34:01Z

### What was done
Initial implementation from scratch. Built the full API router with:
- FastAPI server accepting OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) requests
- YAML config loading with env var expansion (`${VAR}`)
- RPM sliding-window rate limiting and daily/5h/weekly token quota tracking (`router/usage.py`)
- Per-error retry limits and consecutive-failure cooldown (`router/retry.py`)
- Full OpenAI ↔ Anthropic request/response format conversion including streaming SSE (`router/converter.py`)
- Single backend proxy with retry loop (`router/endpoint.py`)
- Sequential and load_balance group routing with nested group support (`router/group.py`)
- Mock API server simulating success, errors (429/500/502/503), slow responses, and flaky endpoints (`mock_api/server.py`)
- 35 tests covering unit and integration scenarios

### Bugs fixed
N/A (initial implementation)

### Test count
**35 tests passing**

---

## Iteration 2 — 2026-04-12T17:55:00Z

### What was done
QA Round 1 — sub-agent deep testing. Verified: cooldown recovery, RPM recovery, load balance weight distribution (2:1 ratio confirmed), circular group detection, streaming e2e, cross-format conversion, multiple system message merging. Added 49 new edge-case tests.

### Bugs fixed
1. **CRITICAL — Streaming errors were silent**: `_call_streaming()` created a lazy async generator that only entered the HTTP context when iterated. On a non-200 response, `UpstreamError` was raised *after* FastAPI had sent HTTP 200 headers, making fallback impossible and leaving clients with a broken stream. Fixed by eagerly connecting inside `call()` and checking status before returning the stream to the caller.
2. **MEDIUM — SSE blank lines stripped**: Pass-through streaming had `if line:` guard that stripped the blank lines separating SSE events, breaking the `\n\n` event boundary for EventSource clients. Fixed by removing the guard.

### Test count
**84 tests passing** (+49 new tests)

---

## Iteration 3 — 2026-04-12T18:24:33Z

### What was done
Final QA pass (Round 3). Verified existing behavior, added comprehensive edge-case tests, and added mock-API endpoints for HTML/non-JSON error responses.

### Investigation results
All five investigated areas were confirmed correct with no code bugs found:

| Area | Finding |
|---|---|
| Config self-loop / circular deps | Caught by `build_routing_tree` → `ValueError` ✓ |
| Non-JSON / HTML upstream errors | `_call_once` wraps non-JSON as `{"error": text}`; status-code-based retry still works ✓ |
| Anthropic list-type content blocks | Converter correctly flattens (→ OpenAI) or passes through (→ Anthropic) ✓ |
| Concurrent requests | `asyncio.Lock` in `UsageTracker` / `RetryTracker` prevents races ✓ |
| Slow backend (2 s delay) | Succeeds well within 120 s client timeout ✓ |

### Changes made
- **`mock_api/server.py`**: Added `POST /v1/chat/completions/html/{code}`, `/v1/chat/completions/text/{code}`, and `/v1/messages/html/{code}` endpoints that return HTML and plain-text bodies with arbitrary status codes (to exercise non-JSON upstream handling).
- **`tests/test_qa_round3.py`** (new, 27 tests):
  - `TestConfigValidation` (8 unit tests): self-loop, A→B→A circular group, missing `base_url`, unknown API reference, invalid strategy, both/neither api+group on GroupMember, default_group not found.
  - `TestContentBlocksUnit` (6 unit tests): list-content Anthropic→OpenAI, multi-block concat, OpenAI user list passthrough to Anthropic, system list merge, assistant list flatten, `_flatten_content` with missing `text` key.
  - `TestNonJSONUpstreamHTMLError` (2 integration tests): HTML 502 from upstream → router returns 5xx JSON (no crash).
  - `TestNonJSONUpstreamPlainText` (1 integration test): plain-text 503 → router returns 5xx JSON.
  - `TestNonJSONUpstreamFallback` (1 integration test): HTML error on primary → falls back to healthy secondary → 200.
  - `TestContentBlocksIntegration` (3 integration tests): end-to-end OpenAI & Anthropic requests with list-type content blocks.
  - `TestConcurrentRequests` (3 integration tests): 10 concurrent OpenAI requests, 10 concurrent Anthropic requests, stats tracking across concurrent load.
  - `TestSlowBackend` (3 integration tests): 2 s delay endpoint succeeds, timing assertion, Anthropic-format request to slow endpoint.

### Bugs fixed
None — all investigated areas were already correctly implemented.

### Test count
**129 tests passing** (was 102; +27 new tests)

---

---

## Iteration 4 (Round 2 QA) — 2026-04-12T18:48:00Z

### What was done
QA Round 2 — focused on model override, load balance with unavailable members, streaming conversion (Anthropic→OpenAI format), and streaming resource leaks. Added 18 new tests.

### Bugs fixed
1. **Empty sequential group — unhelpful error**: `_call_sequential()` with 0 members set `last_error = None` and raised `"...: None"`. Fixed with early check in `router/group.py` → now raises `"group has no members configured"`.
2. **CRITICAL — Streaming resource leak**: HTTP connections were not closed when clients disconnected mid-stream. Two root causes: (a) `_stream_body()` async generator's `finally` block was never entered if the generator was abandoned before iteration; (b) converter functions had `yield` outside `try/finally`. Fixed by replacing the generator with a `_ResponseStream` class with explicit `aclose()`, and moving all converter yields inside `try/finally`.

### Test count
**102 tests passing** (+18 new tests)

---

## Iteration 5 (Round 3 QA) — 2026-04-12T18:24:33Z

*(See earlier Iteration 3 entry — same round — the sub-agent wrote this entry directly.)*
