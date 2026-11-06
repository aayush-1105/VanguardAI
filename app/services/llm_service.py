from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.api.schemas import ChatCompletionRequest
from app.core import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _client_for(model: str) -> AsyncOpenAI:
    if "deepseek" in model.lower():
        return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)
    return AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=config.GROQ_API_KEY)


async def route_chat_completion(
    request: ChatCompletionRequest,
) -> dict[str, Any] | AsyncIterator[str]:
    """Return the JSON body, or an SSE iterator when streaming.

    Streaming returns an iterator rather than yielding directly so the upstream
    client stays open for the lifetime of the stream, not just this call.
    """
    if request.stream:
        return _stream(request)

    async with _client_for(request.model) as client:
        response = await client.chat.completions.create(
            model=request.model, messages=request.messages, stream=False
        )
        return response.model_dump()


async def _stream(request: ChatCompletionRequest) -> AsyncIterator[str]:
    async with _client_for(request.model) as client:
        stream = await client.chat.completions.create(
            model=request.model, messages=request.messages, stream=True
        )
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
