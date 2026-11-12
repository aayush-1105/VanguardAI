import asyncio
import logging
import math
import os
import threading
from uuid import uuid4

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION = "semantic_cache"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# The bi-encoder is only a recall gate for the top-K candidates: this model
# scores true paraphrases ~0.76-0.79 and same-topic-different-intent prompts
# ~0.80, so on its own it cannot tell them apart.
SIMILARITY_THRESHOLD = 0.75

# Cross-encoder logits are unbounded, so they are squashed through a sigmoid to
# 0..1 before comparing. MS MARCO is a query->passage reranker, so it scores two
# user questions lower than a question and its answer. Measured on this model:
# identical prompts 0.996, a genuine paraphrase 0.31, another phrasing 0.014,
# same-topic-different-intent 0.0001-0.007, unrelated 0.000. 0.05 therefore
# favours precision: a valid paraphrase may miss, which only costs one provider
# call, while a different intent must never return another prompt's answer.
CROSS_ENCODER_THRESHOLD = 0.05

RERANK_CANDIDATES = 3

_qdrant_url = os.getenv("QDRANT_URL")
_client = QdrantClient(url=_qdrant_url) if _qdrant_url else QdrantClient(location=":memory:")
_init_lock = threading.Lock()
_model = None
_reranker = None
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
            # A remote Qdrant keeps the collection across restarts; creating an
            # existing one raises, so check first.
            if not _client.collection_exists(COLLECTION):
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


def _rerank(query: str, candidates: list[str]) -> list[float]:
    """Score each (query, candidate) pair, returning probabilities in 0..1."""
    global _reranker
    with _init_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        reranker = _reranker  # read under the lock, predict outside it
    logits = reranker.predict([(query, candidate) for candidate in candidates])
    return [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]


async def check_cache(prompt: str) -> str | None:
    """Return the cached answer only when the cross-encoder confirms the intent."""
    try:
        vector = await asyncio.to_thread(_embed, prompt)
        hits = await asyncio.to_thread(
            lambda: _client.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=RERANK_CANDIDATES,
                with_payload=True,
            ).points
        )
        candidates = [hit for hit in hits if hit.score >= SIMILARITY_THRESHOLD]
        if not candidates:
            return None
        scores = await asyncio.to_thread(
            _rerank, prompt, [(hit.payload or {}).get("prompt") or "" for hit in candidates]
        )
    except Exception:
        # The cache is an optimisation; a failure must not fail the request.
        logger.exception("semantic cache lookup failed")
        return None

    best = max(range(len(candidates)), key=lambda index: scores[index])
    if scores[best] < CROSS_ENCODER_THRESHOLD:
        logger.info("reranker rejected every candidate (best %.4f)", scores[best])
        return None
    return (candidates[best].payload or {}).get("response")


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
