from collections.abc import AsyncIterator

import openai
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas import ChatCompletionRequest
from app.services.llm_service import route_chat_completion

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        result = await route_chat_completion(request)
    except (openai.APIStatusError, openai.APIConnectionError) as exc:
        # Connection failures have no upstream status, so fall back to Bad Gateway.
        status = getattr(exc, "status_code", None) or 502
        return JSONResponse(
            status_code=status,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )

    if isinstance(result, AsyncIterator):
        return StreamingResponse(result, media_type="text/event-stream")
    return result
