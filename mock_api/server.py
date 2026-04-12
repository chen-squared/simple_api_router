"""
Mock API server simulating OpenAI and Anthropic backends.

Supports controlling behavior via URL paths:
  /v1/chat/completions           → success
  /v1/chat/completions/error/429 → always 429
  /v1/chat/completions/error/500 → always 500
  /v1/chat/completions/error/503 → always 503
  /v1/chat/completions/slow      → 2s delay then success
  /v1/chat/completions/flaky/N   → fail first N times per client, then succeed
  /v1/messages                   → Anthropic success
  (same error paths for /v1/messages)

Environment vars:
  MOCK_PORT  (default 9999)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock API")

# Track flaky call counts per path
_flaky_counts: Dict[str, int] = defaultdict(int)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _openai_success(model: str = "gpt-4", stream: bool = False) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock response from OpenAI."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


def _anthropic_success(model: str = "claude-3-5-sonnet-20241022") -> dict:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Mock response from Anthropic."}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }


def _error_body(code: int, msg: str) -> dict:
    return {"error": {"type": "api_error", "code": code, "message": msg}}


async def _openai_stream(model: str = "gpt-4"):
    """Yield SSE chunks for OpenAI streaming."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    words = ["Mock", " streaming", " response", " from", " OpenAI", "."]
    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None}, 'finish_reason': None}]})}\n\n".encode()
    for word in words:
        await asyncio.sleep(0.05)
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': word}, 'finish_reason': None}]})}\n\n".encode()
    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _anthropic_stream(model: str = "claude-3-5-sonnet-20241022"):
    """Yield SSE events for Anthropic streaming."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 10, 'output_tokens': 0}}})}\n\n".encode()
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()
    words = ["Mock", " streaming", " response", " from", " Anthropic", "."]
    for word in words:
        await asyncio.sleep(0.05)
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': word}})}\n\n".encode()
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 8}})}\n\n".encode()
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()


# -----------------------------------------------------------------------
# Route handlers
# -----------------------------------------------------------------------


async def _handle_openai(request: Request, behavior: str, param: str = ""):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    model = body.get("model", "gpt-4")
    is_stream = body.get("stream", False)

    if behavior == "success":
        if is_stream:
            return StreamingResponse(_openai_stream(model), media_type="text/event-stream")
        return JSONResponse(_openai_success(model))

    if behavior == "error":
        code = int(param) if param else 500
        return JSONResponse(_error_body(code, f"Mock {code} error"), status_code=code)

    if behavior == "slow":
        await asyncio.sleep(2.0)
        if is_stream:
            return StreamingResponse(_openai_stream(model), media_type="text/event-stream")
        return JSONResponse(_openai_success(model))

    if behavior == "flaky":
        n = int(param) if param else 2
        key = f"openai_flaky"
        _flaky_counts[key] += 1
        if _flaky_counts[key] <= n:
            return JSONResponse(_error_body(500, "flaky error"), status_code=500)
        _flaky_counts[key] = 0
        if is_stream:
            return StreamingResponse(_openai_stream(model), media_type="text/event-stream")
        return JSONResponse(_openai_success(model))

    return JSONResponse(_openai_success(model))


async def _handle_anthropic(request: Request, behavior: str, param: str = ""):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    model = body.get("model", "claude-3-5-sonnet-20241022")
    is_stream = body.get("stream", False)

    if behavior == "success":
        if is_stream:
            return StreamingResponse(_anthropic_stream(model), media_type="text/event-stream")
        return JSONResponse(_anthropic_success(model))

    if behavior == "error":
        code = int(param) if param else 500
        return JSONResponse(_error_body(code, f"Mock {code} error"), status_code=code)

    if behavior == "slow":
        await asyncio.sleep(2.0)
        if is_stream:
            return StreamingResponse(_anthropic_stream(model), media_type="text/event-stream")
        return JSONResponse(_anthropic_success(model))

    if behavior == "flaky":
        n = int(param) if param else 2
        key = f"anthropic_flaky"
        _flaky_counts[key] += 1
        if _flaky_counts[key] <= n:
            return JSONResponse(_error_body(500, "flaky error"), status_code=500)
        _flaky_counts[key] = 0
        if is_stream:
            return StreamingResponse(_anthropic_stream(model), media_type="text/event-stream")
        return JSONResponse(_anthropic_success(model))

    return JSONResponse(_anthropic_success(model))


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def openai_success(request: Request):
    return await _handle_openai(request, "success")

@app.post("/v1/chat/completions/error/{code}")
async def openai_error(request: Request, code: int):
    return await _handle_openai(request, "error", str(code))

@app.post("/v1/chat/completions/slow")
async def openai_slow(request: Request):
    return await _handle_openai(request, "slow")

@app.post("/v1/chat/completions/flaky/{n}")
async def openai_flaky(request: Request, n: int):
    return await _handle_openai(request, "flaky", str(n))

@app.post("/v1/messages")
async def anthropic_success(request: Request):
    return await _handle_anthropic(request, "success")

@app.post("/v1/messages/error/{code}")
async def anthropic_error(request: Request, code: int):
    return await _handle_anthropic(request, "error", str(code))

@app.post("/v1/messages/slow")
async def anthropic_slow(request: Request):
    return await _handle_anthropic(request, "slow")

@app.post("/v1/messages/flaky/{n}")
async def anthropic_flaky(request: Request, n: int):
    return await _handle_anthropic(request, "flaky", str(n))

@app.post("/v1/chat/completions/html/{code}")
async def openai_html_error(request: Request, code: int):
    """Return an HTML body (like an nginx error page) with the given status code."""
    from fastapi.responses import HTMLResponse
    html = f"<html><body><h1>{code} Error</h1><p>Upstream gateway error</p></body></html>"
    return HTMLResponse(content=html, status_code=code)


@app.post("/v1/chat/completions/text/{code}")
async def openai_text_error(request: Request, code: int):
    """Return a plain-text body with the given status code."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=f"upstream error {code}", status_code=code)


@app.post("/v1/messages/html/{code}")
async def anthropic_html_error(request: Request, code: int):
    from fastapi.responses import HTMLResponse
    html = f"<html><body><h1>{code} Error</h1></body></html>"
    return HTMLResponse(content=html, status_code=code)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "9999"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
