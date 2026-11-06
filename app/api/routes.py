from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatCompletionRequest
from app.services.llm_service import route_chat_completion

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    result = await route_chat_completion(request)
    if isinstance(result, AsyncIterator):
        return StreamingResponse(result, media_type="text/event-stream")
    return result
