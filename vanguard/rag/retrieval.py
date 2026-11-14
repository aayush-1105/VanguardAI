"""Hybrid retrieval over the knowledge base: dense + BM25, fused by RRF, reranked."""

import asyncio
import logging
import re

from rank_bm25 import BM25Okapi

from app.services import cache_service
from vanguard.rag.ingestion import KNOWLEDGE_COLLECTION

logger = logging.getLogger(__name__)

CANDIDATES = 10
FINAL_CHUNKS = 3
RRF_K = 60
# ms-marco is a query->passage reranker, so a relevant chunk scores well above
# this while an unrelated one sits near zero. Calibrated in test_rag.py.
RERANK_THRESHOLD = 0.05

# shortcut: BM25 loads at most this many chunks per query and is rebuilt each
# time, so cost grows with the corpus; page or index it once the KB is large.
_MAX_SCROLL = 10_000


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _dense(query: str) -> list[tuple[str, str]]:
    points = cache_service._client.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=cache_service._embed(query),
        limit=CANDIDATES,
        with_payload=True,
    ).points
    return [(str(point.id), (point.payload or {}).get("text", "")) for point in points]


def _sparse(query: str) -> list[tuple[str, str]]:
    points, _ = cache_service._client.scroll(
        collection_name=KNOWLEDGE_COLLECTION,
        limit=_MAX_SCROLL,
        with_payload=True,
        with_vectors=False,
    )
    corpus = [(str(point.id), (point.payload or {}).get("text", "")) for point in points]
    if not corpus:
        return []

    bm25 = BM25Okapi([_tokens(text) for _, text in corpus])
    scores = bm25.get_scores(_tokens(query))
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return [corpus[index] for index in ranked[:CANDIDATES] if scores[index] > 0]


def _fuse(dense: list[tuple[str, str]], sparse: list[tuple[str, str]]) -> list[str]:
    """Reciprocal rank fusion: a chunk ranked by both retrievers rises to the top."""
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    for ranked in (dense, sparse):
        for rank, (point_id, text) in enumerate(ranked):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            texts[point_id] = text
    return [texts[point_id] for point_id in sorted(scores, key=scores.get, reverse=True)]


async def retrieve(query: str) -> list[str]:
    """Return the chunks that best answer `query`, or [] when the KB cannot help."""
    try:
        dense, sparse = await asyncio.gather(
            asyncio.to_thread(_dense, query), asyncio.to_thread(_sparse, query)
        )
        candidates = _fuse(dense, sparse)
        if not candidates:
            return []
        scores = await asyncio.to_thread(cache_service._rerank, query, candidates)
    except Exception:
        # Retrieval is an enhancement; an empty knowledge base must not fail a chat.
        logger.exception("knowledge base retrieval failed")
        return []

    ranked = sorted(
        (
            (score, text)
            for score, text in zip(scores, candidates)
            if score >= RERANK_THRESHOLD
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [text for _, text in ranked[:FINAL_CHUNKS]]
