import asyncio
import logging
import threading
from uuid import uuid4

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION = "semantic_cache"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Calibrated to this model: true paraphrases score ~0.76-0.79, so 0.92 never hit.
# shortcut: 0.75 also admits same-topic-different-intent prompts (~0.80); add a
# cross-encoder reranker before trusting this cache in production.
SIMILARITY_THRESHOLD = 0.75

_client = QdrantClient(location=":memory:")
_init_lock = threading.Lock()
_model = None
_ready = False


def _ensure_ready():
    """Load the embedding model and create the collection on first use.

    Deferred so importing the app does not block on a model download.
    """
    global _model, _ready
    if _ready:
        return _model
    with _init_lock:
        if not _ready:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL)
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=_model.get_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
            _ready = True
    return _model


def _embed(text: str) -> list[float]:
    # Blocking CPU work: always called through asyncio.to_thread.
    return _ensure_ready().encode(text, normalize_embeddings=True).tolist()


async def check_cache(prompt: str) -> str | None:
    """Return the cached answer when a stored prompt is similar enough."""
    try:
        vector = await asyncio.to_thread(_embed, prompt)
        hits = await asyncio.to_thread(
            lambda: _client.query_points(
                collection_name=COLLECTION, query=vector, limit=1, with_payload=True
            ).points
        )
    except Exception:
        # The cache is an optimisation; a failure must not fail the request.
        logger.exception("semantic cache lookup failed")
        return None

    if hits and hits[0].score >= SIMILARITY_THRESHOLD:
        return (hits[0].payload or {}).get("response")
    return None


async def store_cache(prompt: str, response: str) -> None:
    try:
        vector = await asyncio.to_thread(_embed, prompt)
        await asyncio.to_thread(
            lambda: _client.upsert(
                collection_name=COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={"prompt": prompt, "response": response},
                    )
                ],
            )
        )
    except Exception:
        logger.exception("semantic cache store failed")
