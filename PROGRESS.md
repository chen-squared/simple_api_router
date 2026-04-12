# simple_api_router — Progress Log

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
