import copy
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.api.schemas import ChatCompletionRequest
from app.core import config
from app.services import cache_service
from app.services.log_service import estimate_cost, write_log
from app.services.pii_service import PIIScrubber, Vault

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model ids are provider-specific, so the fallback switches model and endpoint.
FALLBACK_MODELS = {"groq": "openai/gpt-oss-20b", "openrouter": "deepseek/deepseek-chat"}

# Longest a partial "{{" may be held back before it is emitted unrewritten.
_MAX_TAG_LOOKAHEAD = 64


class Rehydrator:
    """Swap {{TAG}} tokens back to raw values across chunk boundaries.

    A tag arrives split across arbitrary chunk boundaries ("{{EMA" then
    "IL_1}}"), so the last chunk of text cannot always be emitted: it may be
    the start of a tag whose ending has not arrived yet. The buffer holds text
    from the last unmatched "{{" onwards and releases it on the next feed, once
    the tag closes or the lookahead cap is hit.
    """

    def __init__(self, vault: dict[str, str]) -> None:
        self._vault = vault
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        idx = self._buffer.rfind("{{")
        if idx != -1:
            tail = self._buffer[idx:]
            if "}}" not in tail[2:] and len(tail) <= _MAX_TAG_LOOKAHEAD:
                return self._emit(self._buffer[:idx], keep=tail)

        # Nothing pending, but a trailing "{" may still grow into "{{".
        emit = self._buffer
        keep = ""
        if emit.endswith("{"):
            emit, keep = emit[:-1], "{"
        return self._emit(emit, keep)

    def flush(self) -> str:
        """Release whatever is still buffered at end of stream."""
        return self._emit(self._buffer, keep="")

    def _emit(self, text: str, keep: str) -> str:
        self._buffer = keep
        return _rehydrate_text(text, self._vault)


def _rehydrate_text(text: str, vault: dict[str, str]) -> str:
    for tag, raw in vault.items():
        if tag in text:
            text = text.replace(tag, raw)
    return text


def _provider_for(model: str) -> str:
    return "openrouter" if "deepseek" in model.lower() else "groq"


def _other_provider(provider: str) -> str:
    return "groq" if provider == "openrouter" else "openrouter"


def _credentials(provider: str) -> tuple[str, str]:
    if provider == "openrouter":
        return OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY
    return GROQ_BASE_URL, config.GROQ_API_KEY


def _client(provider: str) -> AsyncOpenAI:
    base_url, api_key = _credentials(provider)
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _is_fallback_worthy(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    return isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)


async def _create(
    model: str, messages: list[dict], stream: bool
) -> tuple[AsyncIterator[Any], AsyncOpenAI, str]:
    """Call the primary provider, retrying once on the secondary if it fails."""
    primary = _provider_for(model)
    secondary = _other_provider(primary)
    attempts = [(primary, model), (secondary, FALLBACK_MODELS[secondary])]

    last_error: Exception | None = None
    for provider, candidate_model in attempts:
        client = _client(provider)
        try:
            upstream = await client.chat.completions.create(
                model=candidate_model, messages=messages, stream=stream
            )
            return upstream, client, provider
        except (APIStatusError, APIConnectionError) as exc:
            await client.close()
            if not _is_fallback_worthy(exc):
                raise
            logger.warning("provider %s failed (%s); retrying on %s", provider, exc, secondary)
            last_error = exc
    raise last_error  # type: ignore[misc]


def _scrub_messages(messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Deep copy the messages, tokenize PII in user text, and return the mapping."""
    scrubbed = copy.deepcopy(messages)
    vault = Vault()
    for message in scrubbed:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"], vault = PIIScrubber.scrub_text(message["content"], vault)
    return scrubbed, vault.load()


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


def _chunk_json(chunk: Any, text: str, model: str = "") -> str:
    if chunk is not None and chunk.choices:
        body = chunk.model_dump()
        body["choices"][0]["delta"]["content"] = text
    else:
        body = {
            "id": f"chatcmpl-vanguard-{uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}}],
        }
    return json.dumps(body)


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
    """Tokenize PII, check cache, then hit upstream; return a body or an SSE iterator."""
    started = time.perf_counter()
    messages, vault = _scrub_messages(request.messages)
    prompt = _last_user_text(messages)

    cached = await cache_service.check_cache(prompt) if prompt else None
    if cached is not None:
        answer = _rehydrate_text(cached, vault)
        await write_log(
            provider_used="cache",
            cache_hit=True,
            ttft_ms=(time.perf_counter() - started) * 1000,
            total_cost=0.0,
        )
        if request.stream:
            return _cached_stream(answer, request.model)
        return _completion_body(answer, request.model)

    upstream, client, provider = await _create(request.model, messages, request.stream)

    if request.stream:
        return _stream(upstream, client, prompt, provider, started, vault, request.model)

    try:
        body = upstream.model_dump()
    finally:
        await client.close()

    raw_answer = _content_of(body)
    if raw_answer:
        body["choices"][0]["message"]["content"] = _rehydrate_text(raw_answer, vault)
    if prompt and raw_answer:
        # Only the tokenized text is cached, so later requests rehydrate with
        # their own vault instead of leaking this caller's raw values.
        await cache_service.store_cache(prompt, raw_answer)
    await write_log(
        provider_used=provider,
        cache_hit=False,
        ttft_ms=(time.perf_counter() - started) * 1000,
        total_cost=estimate_cost(prompt, raw_answer),
    )
    return body


async def _stream(
    upstream: AsyncIterator[Any],
    client: AsyncOpenAI,
    prompt: str,
    provider: str,
    started: float,
    vault: dict[str, str],
    model: str,
) -> AsyncIterator[str]:
    rehydrator = Rehydrator(vault)
    parts: list[str] = []
    ttft_ms: float | None = None
    last_chunk = None

    try:
        async for chunk in upstream:
            last_chunk = chunk
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)

            visible = rehydrator.feed(delta or "")
            if not visible:
                # Keep terminal metadata even when the visible text is held back.
                if chunk.choices and chunk.choices[0].finish_reason:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                continue

            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            yield f"data: {_chunk_json(chunk, visible)}\n\n"

        tail = rehydrator.flush()
        if tail:
            yield f"data: {_chunk_json(None, tail, model)}\n\n"
    finally:
        await upstream.close()
        await client.close()

    yield "data: [DONE]\n\n"

    raw_answer = "".join(parts)
    if prompt and raw_answer:
        await cache_service.store_cache(prompt, raw_answer)
    await write_log(
        provider_used=provider,
        cache_hit=False,
        ttft_ms=ttft_ms or 0.0,
        total_cost=estimate_cost(prompt, raw_answer),
    )
