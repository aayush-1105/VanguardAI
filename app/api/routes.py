import asyncio
import logging
from collections.abc import AsyncIterator

import openai
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas import ChatCompletionRequest
from app.services.llm_service import route_chat_completion
from app.services.security_service import PromptGuard
from vanguard.rag.ingestion import ingest_document
from vanguard.routing.circuit_breaker import AllProvidersOpen

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/knowledge/ingest")
async def knowledge_ingest(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.",
                    "type": "payload_too_large",
                }
            },
        )

    filename = file.filename or "upload.txt"
    try:
        chunks = await asyncio.to_thread(ingest_document, filename, data)
    except Exception:
        logger.exception("ingestion failed for %s", filename)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Could not parse the uploaded document.",
                    "type": "ingest_error",
                }
            },
        )

    if chunks == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "The document contains no extractable text.",
                    "type": "ingest_error",
                }
            },
        )
    return {"filename": filename, "chunks_indexed": chunks}


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    for message in request.messages:
        content = message.get("content")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and PromptGuard.check_injection(content)
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": "Prompt rejected: injection attempt detected.",
                        "type": "policy_violation",
                    }
                },
            )

    try:
        result = await route_chat_completion(request)
    except AllProvidersOpen as exc:
        # Fail fast: every circuit is open, so no provider was dialled.
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(exc), "type": "circuit_open"}},
        )
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
