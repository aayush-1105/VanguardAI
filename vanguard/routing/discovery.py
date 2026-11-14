"""Live model discovery across the enabled providers.

Every enabled provider is asked for its model list concurrently, each request
bounded by a strict 5 second timeout, so one slow or dead endpoint delays the
refresh by its own timeout at worst rather than by the sum of all of them.
Results are cached in Redis (falling back to this process) and served at
GET /v1/models.
"""

import asyncio
import json
import logging
import os
import time

import httpx
import redis

from vanguard.routing import providers

logger = logging.getLogger(__name__)

MODELS_KEY = "vanguard:models"
MODELS_TTL_SECONDS = 3600
REQUEST_TIMEOUT = 5.0

_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)
_catalog: dict | None = None
# Test seam: an httpx transport can be injected instead of dialling the network.
_transport: httpx.AsyncBaseTransport | None = None


async def _fetch(
    http: httpx.AsyncClient, provider_id: str, provider: dict
) -> tuple[str, str, list[str]]:
    """Return (provider_id, "ok" | "unreachable", model ids)."""
    base_url = str(provider.get("base_url") or "").rstrip("/")
    api_key = provider.get("api_key") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = await http.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        body = response.json()
        model_ids = [
            entry["id"]
            for entry in body.get("data", [])
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        return provider_id, "ok", model_ids
    except Exception as exc:  # one bad provider must not abort the refresh
        logger.warning("model discovery failed for %s: %s", provider_id, exc)
        return provider_id, "unreachable", []


async def refresh_models() -> dict:
    """Query every enabled provider and cache the merged catalog."""
    known = providers.list_providers()
    enabled = [
        (provider_id, provider)
        for provider_id, provider in known.items()
        if provider.get("enabled") and provider.get("base_url")
    ]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, transport=_transport) as http:
        results = await asyncio.gather(
            *[_fetch(http, provider_id, provider) for provider_id, provider in enabled]
        )

    data: list[dict] = []
    statuses: dict[str, dict] = {}
    for provider_id, status, model_ids in results:
        statuses[provider_id] = {"status": status, "model_count": len(model_ids)}
        data.extend(
            {
                "id": f"{provider_id}/{model_id}",
                "object": "model",
                "created": 0,
                "owned_by": provider_id,
            }
            for model_id in model_ids
        )

    payload = {
        "object": "list",
        "data": data,
        "providers": statuses,
        "refreshed_at": int(time.time()),
    }
    _save(payload)
    return payload


def _save(payload: dict) -> None:
    global _catalog
    _catalog = payload
    if _client is None:
        return
    try:
        _client.set(MODELS_KEY, json.dumps(payload), ex=MODELS_TTL_SECONDS)
    except redis.RedisError:
        logger.warning("redis unavailable; discovered models kept in process only")


def load_catalog() -> dict | None:
    """Return the cached catalog, preferring Redis so every worker shares it."""
    global _catalog
    if _catalog is not None:
        return _catalog
    if _client is None:
        return None
    try:
        raw = _client.get(MODELS_KEY)
    except redis.RedisError:
        logger.warning("redis unavailable; model discovery cache is process-local")
        return None
    if not raw:
        return None
    try:
        _catalog = json.loads(raw)
    except ValueError:
        return None
    return _catalog


def reset_catalog() -> None:
    """Forget the cached catalog; the next refresh repopulates it."""
    global _catalog
    _catalog = None
    if _client is None:
        return
    try:
        _client.delete(MODELS_KEY)
    except redis.RedisError:
        logger.warning("redis unavailable; discovery cache was cleared in process only")
