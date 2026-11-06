import copy
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

from app.api.schemas import ChatCompletionRequest
from app.core import config
from app.services import cache_service
from app.services.log_service import write_log
from app.services.pii_service import PIIScrubber

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# shortcut: cost is logged as 0.0 until per-model pricing tables are added.
_COST_NOT_TRACKED = 0.0


def _provider_for(model: str) -> str:
    return "openrouter" if "deepseek" in model.lower() else "groq"


def _client_for(model: str) -> AsyncOpenAI:
    if _provider_for(model) == "openrouter":
        return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)
    return AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=config.GROQ_API_KEY)


def _scrub_messages(messages: list[dict]) -> list[dict]:
    """Deep copy the messages and strip PII from user-authored text only."""
    scrubbed = copy.deepcopy(messages)
    for message in scrubbed:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = PIIScrubber.scrub_text(message["content"])
    return scrubbed


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _content_of(body: dict[str, Any]) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _completion_body(answer: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-cache-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _cached_stream(answer: str, model: str) -> AsyncIterator[str]:
    chunk = {
        "id": f"chatcmpl-cache-{uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def route_chat_completion(
    request: ChatCompletionRequest,
) -> dict[str, Any] | AsyncIterator[str]:
    """Scrub, check cache, then hit upstream; return a body or an SSE iterator."""
    started = time.perf_counter()
    messages = _scrub_messages(request.messages)
    prompt = _last_user_text(messages)

    cached = await cache_service.check_cache(prompt) if prompt else None
    if cached is not None:
        await write_log(
            provider_used="cache",
            cache_hit=True,
            ttft_ms=(time.perf_counter() - started) * 1000,
            total_cost=_COST_NOT_TRACKED,
        )
        if request.stream:
            return _cached_stream(cached, request.model)
        return _completion_body(cached, request.model)

    provider = _provider_for(request.model)
    client = _client_for(request.model)
    try:
        # Created eagerly so upstream status errors reach the route's handler.
        upstream = await client.chat.completions.create(
            model=request.model, messages=messages, stream=request.stream
        )
    except BaseException:
        await client.close()
        raise

    if request.stream:
        return _stream(upstream, client, prompt, provider, started)

    try:
        body = upstream.model_dump()
    finally:
        await client.close()

    answer = _content_of(body)
    if prompt and answer:
        await cache_service.store_cache(prompt, answer)
    await write_log(
        provider_used=provider,
        cache_hit=False,
        ttft_ms=(time.perf_counter() - started) * 1000,
        total_cost=_COST_NOT_TRACKED,
    )
    return body


async def _stream(
    upstream: AsyncIterator[Any],
    client: AsyncOpenAI,
    prompt: str,
    provider: str,
    started: float,
) -> AsyncIterator[str]:
    parts: list[str] = []
    ttft_ms: float | None = None
    try:
        async for chunk in upstream:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            yield f"data: {chunk.model_dump_json()}\n\n"
    finally:
        await upstream.close()
        await client.close()

    yield "data: [DONE]\n\n"

    answer = "".join(parts)
    if prompt and answer:
        await cache_service.store_cache(prompt, answer)
    await write_log(
        provider_used=provider,
        cache_hit=False,
        ttft_ms=ttft_ms or 0.0,
        total_cost=_COST_NOT_TRACKED,
    )
