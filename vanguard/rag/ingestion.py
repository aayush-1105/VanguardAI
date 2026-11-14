"""Server-side ingestion: uploaded bytes -> chunked, embedded points in Qdrant."""

import io
import logging
from datetime import datetime, timezone
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from app.services import cache_service

logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "vanguard_knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # Markdown is text; decoding is the whole parser.
    return data.decode("utf-8", errors="replace")


def _ensure_collection() -> None:
    cache_service._ensure_ready()  # loads the embedding model and the Qdrant client
    client = cache_service._client
    if not client.collection_exists(KNOWLEDGE_COLLECTION):
        client.create_collection(
            collection_name=KNOWLEDGE_COLLECTION,
            vectors_config=models.VectorParams(
                size=cache_service._model.get_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )


def ingest_document(filename: str, data: bytes) -> int:
    """Chunk, embed, and upsert one document; returns the number of chunks stored.

    Blocking: callers run this through a thread.
    """
    text = _extract_text(filename, data)
    chunks = [chunk for chunk in _splitter.split_text(text) if chunk.strip()]
    if not chunks:
        return 0

    _ensure_collection()
    timestamp = datetime.now(timezone.utc).isoformat()
    cache_service._client.upsert(
        collection_name=KNOWLEDGE_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=cache_service._embed(chunk),
                payload={
                    "text": chunk,
                    "filename": filename,
                    "timestamp": timestamp,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ],
    )
    return len(chunks)
