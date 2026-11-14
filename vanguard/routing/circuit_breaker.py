"""Per-provider circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

State lives in Redis so every worker sees the same circuit. If Redis is
unreachable, state is mirrored in this process instead, matching the PII
vault's fallback: a Redis outage downgrades sharing, not availability.
"""

import logging
import os
import threading
import time

import redis

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

FAILURE_THRESHOLD = 3
RECOVERY_TIMEOUT = 30.0

_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)
_lock = threading.Lock()
_memory: dict[str, dict] = {}


class AllProvidersOpen(RuntimeError):
    """Every provider circuit is open, so no upstream call is attempted."""


def _now() -> float:
    # Tests replace this to drive the cooldown clock.
    return time.time()


def _key(provider: str) -> str:
    return f"vanguard:circuit:{provider}"


def _fresh() -> dict:
    return {"state": CLOSED, "failures": 0, "opened_at": 0.0, "probe_at": 0.0}


def _decode(raw: dict) -> dict:
    return {
        "state": raw.get("state", CLOSED),
        "failures": int(raw.get("failures", 0)),
        "opened_at": float(raw.get("opened_at", 0.0)),
        "probe_at": float(raw.get("probe_at", 0.0)),
    }


def _load(provider: str) -> dict:
    if _client is not None:
        try:
            raw = _client.hgetall(_key(provider))
            if raw:
                return _decode(raw)
            return _fresh()
        except redis.RedisError:
            logger.warning("redis unavailable; circuit state is process-local")
    with _lock:
        return dict(_memory.get(provider, _fresh()))


def _save(provider: str, state: dict) -> None:
    with _lock:
        _memory[provider] = dict(state)
    if _client is not None:
        try:
            _client.hset(_key(provider), mapping=state)
        except redis.RedisError:
            logger.warning("redis unavailable; circuit state kept in process only")


def allow(provider: str) -> bool:
    """True when a request may call this provider, claiming the probe if due."""
    state = _load(provider)
    now = _now()

    if state["state"] == CLOSED:
        return True

    if state["state"] == OPEN:
        if now - state["opened_at"] < RECOVERY_TIMEOUT:
            return False
        # Cooldown elapsed: hand exactly one request through as the probe.
        state.update(state=HALF_OPEN, probe_at=now)
        _save(provider, state)
        return True

    # HALF_OPEN: one probe at a time. A probe that never reported back (a
    # cancelled request) is assumed lost after another cooldown, so it cannot
    # wedge the circuit shut forever.
    if state["probe_at"] and now - state["probe_at"] < RECOVERY_TIMEOUT:
        return False
    state["probe_at"] = now
    _save(provider, state)
    return True


def record_success(provider: str) -> None:
    """A usable response closes the circuit and clears consecutive failures."""
    _save(provider, _fresh())


def record_failure(provider: str) -> None:
    """Count a failure, tripping or re-tripping the circuit at the threshold."""
    state = _load(provider)
    now = _now()

    if state["state"] == HALF_OPEN:
        # The probe failed: reopen immediately and restart the cooldown.
        state.update(state=OPEN, opened_at=now, probe_at=0.0)
    else:
        state["failures"] += 1
        if state["failures"] >= FAILURE_THRESHOLD:
            state.update(state=OPEN, opened_at=now, probe_at=0.0)
    _save(provider, state)


def state_of(provider: str) -> str:
    return _load(provider)["state"]


def reset(*providers: str) -> None:
    """Clear circuit state for the given providers, or for all of them."""
    with _lock:
        if providers:
            for provider in providers:
                _memory.pop(provider, None)
        else:
            _memory.clear()
    if _client is None:
        return
    try:
        if providers:
            _client.delete(*[_key(provider) for provider in providers])
        else:
            keys = list(_client.scan_iter("vanguard:circuit:*"))
            if keys:
                _client.delete(*keys)
    except redis.RedisError:
        logger.warning("redis unavailable; circuit state was cleared in process only")
