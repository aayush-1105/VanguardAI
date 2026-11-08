"""End-to-end verification of the VanguardAI gateway.

Drives the real FastAPI lifespan in-process via TestClient. Upstream providers
are replaced with fakes, so the suite is deterministic and never touches the
network.
"""

import py_compile
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import cache_service, llm_service
from app.services.pii_service import PIIScrubber

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "vanguard.db"
MODEL = "openai/gpt-oss-20b"

PROMPT_A = "How do I optimize Docker container sizes?"
PROMPT_B = "Best ways to reduce Docker image size"
PROMPT_C = "What is the capital of France?"
KNOWN_DOCKER_ANSWER = "Use multi-stage builds and slim base images to cut image size."
PARIS_ANSWER = "The capital of France is Paris."

METRICS: dict[str, float] = {}


class _FakeClient:
    """Stands in for AsyncOpenAI and records every dispatch the gateway makes."""

    def __init__(self, answers=None, error=None):
        self.answers = list(answers or [])
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, stream):
        self.calls.append({"model": model, "messages": messages, "stream": stream})
        if self.error is not None:
            raise self.error
        content = self.answers.pop(0) if self.answers else ""
        return _FakeCompletion(content, model)

    async def close(self):
        pass


class _FakeCompletion:
    def __init__(self, content, model):
        self._body = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def model_dump(self):
        return self._body


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def _warm_model(client):
    """Load the embedding model once so cache latency is measured warm."""
    cache_service._ensure_ready()


def _post(client, content, model=MODEL):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
    )


def _max_log_id() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM request_logs").fetchone()[0]


def _log_after(log_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM request_logs WHERE id > ? ORDER BY id LIMIT 1", (log_id,)
        ).fetchone()


# ---------------------------------------------------------------- A. health


def test_health_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "VanguardAI Gateway"}


def test_chat_completions_route_registered():
    paths = app.openapi()["paths"]
    assert "post" in paths.get("/v1/chat/completions", {})


# ------------------------------------------------------------- B. PII shield

PII_CASES = [
    ("test@enterprise.com", "[REDACTED_EMAIL]"),
    ("+1 415 555 0199", "[REDACTED_PHONE]"),
    ("9876543210", "[REDACTED_PHONE]"),
    ("415.555.0199", "[REDACTED_PHONE]"),
    ("4111111111111111", "[REDACTED_CREDIT_CARD]"),
    ("4111 1111 1111 1111", "[REDACTED_CREDIT_CARD]"),
    ("123456789012", "[REDACTED_AADHAAR]"),
    ("1234 5678 9012", "[REDACTED_AADHAAR]"),
    ("sk-or-v1-" + "a" * 64, "[REDACTED_API_KEY]"),
    ("gsk_" + "b" * 52, "[REDACTED_API_KEY]"),
]


@pytest.mark.parametrize("raw,tag", PII_CASES)
def test_pii_scrubber_patterns(raw, tag):
    assert PIIScrubber.scrub_text(raw) == tag


def test_pii_scrubbed_before_dispatch(client, monkeypatch):
    fake = _FakeClient(answers=["acknowledged"])
    monkeypatch.setattr(llm_service, "_client_for", lambda model: fake)

    raw = (
        "Email test@enterprise.com, call +1 415 555 0199, "
        "card 4111 1111 1111 1111, aadhaar 1234 5678 9012, "
        "key sk-or-v1-" + "a" * 64
    )
    response = _post(client, raw)
    assert response.status_code == 200
    assert len(fake.calls) == 1

    dispatched = fake.calls[0]["messages"][0]["content"]
    for secret in (
        "test@enterprise.com",
        "415 555 0199",
        "4111 1111 1111 1111",
        "1234 5678 9012",
        "sk-or-v1-",
    ):
        assert secret not in dispatched
    for tag in (
        "[REDACTED_EMAIL]",
        "[REDACTED_PHONE]",
        "[REDACTED_CREDIT_CARD]",
        "[REDACTED_AADHAAR]",
        "[REDACTED_API_KEY]",
    ):
        assert tag in dispatched


# ---------------------------------------------------- C. semantic cache


def test_semantic_cache_hit_miss_and_no_upstream(client, monkeypatch):
    fake = _FakeClient(answers=[KNOWN_DOCKER_ANSWER, PARIS_ANSWER])
    monkeypatch.setattr(llm_service, "_client_for", lambda model: fake)

    # Seed A through the real request path.
    seeded = _post(client, PROMPT_A)
    assert seeded.status_code == 200
    assert seeded.json()["choices"][0]["message"]["content"] == KNOWN_DOCKER_ANSWER
    assert len(fake.calls) == 1

    # Cosine between A and B, measured with the configured model.
    vector_a = cache_service._embed(PROMPT_A)
    vector_b = cache_service._embed(PROMPT_B)
    cosine = sum(x * y for x, y in zip(vector_a, vector_b))
    METRICS["cosine_a_b"] = cosine
    assert cosine >= cache_service.SIMILARITY_THRESHOLD

    # B is a semantic hit: no upstream dispatch, cached answer, zero cost.
    started = time.perf_counter()
    hit = _post(client, PROMPT_B)
    METRICS["cache_hit_ms"] = (time.perf_counter() - started) * 1000
    print(f"[METRIC] cosine_a_b={cosine:.4f} cache_hit_ms={METRICS['cache_hit_ms']:.1f}")

    assert hit.status_code == 200
    assert hit.json()["choices"][0]["message"]["content"] == KNOWN_DOCKER_ANSWER
    assert len(fake.calls) == 1, "cache hit must not reach a provider"

    row = _latest_log()
    assert row["cache_hit"] == 1
    assert row["provider_used"] == "cache"
    assert row["total_cost"] == 0.0

    # Unrelated C misses and reaches the provider.
    before = _max_log_id()
    miss = _post(client, PROMPT_C)
    assert miss.status_code == 200
    assert miss.json()["choices"][0]["message"]["content"] == PARIS_ANSWER
    assert len(fake.calls) == 2

    row = _log_after(before)
    assert row["cache_hit"] == 0
    assert row["provider_used"] == "groq"


def _latest_log():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM request_logs ORDER BY id DESC LIMIT 1").fetchone()


# ------------------------------------------ D. provider dispatch and errors


def test_model_routing():
    assert llm_service._provider_for("deepseek/deepseek-chat") == "openrouter"
    assert llm_service._provider_for("DeepSeek-R1") == "openrouter"
    assert llm_service._provider_for("openai/gpt-oss-20b") == "groq"
    assert llm_service._provider_for("llama-3.1-8b-instant") == "groq"


def _api_status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "boom"}})
    return openai.APIStatusError(
        f"{status} upstream failure", response=response, body={"error": {"message": "boom"}}
    )


@pytest.mark.parametrize("status", [429, 404])
def test_upstream_error_mapping(client, monkeypatch, status):
    fake = _FakeClient(error=_api_status_error(status))
    monkeypatch.setattr(llm_service, "_client_for", lambda model: fake)

    response = _post(client, "What are the symptoms of the common cold?")
    assert response.status_code == status
    payload = response.json()
    assert payload["error"]["type"] == "upstream_error"
    assert payload["error"]["message"]


# ------------------------------------------------ E. telemetry persistence


def test_completion_writes_request_log(client, monkeypatch):
    fake = _FakeClient(answers=["SYN, SYN-ACK, ACK."])
    monkeypatch.setattr(llm_service, "_client_for", lambda model: fake)

    before = _max_log_id()
    response = _post(client, "Explain the TCP three-way handshake.")
    assert response.status_code == 200

    row = _log_after(before)
    assert row is not None
    for field in ("id", "timestamp", "provider_used", "cache_hit", "ttft_ms"):
        assert row[field] is not None, f"{field} was NULL"
    assert row["provider_used"] == "groq"
    assert row["cache_hit"] == 0
    assert row["ttft_ms"] > 0


# ------------------------------------------------ F. dashboard integrity


def test_dashboard_compiles_and_queries_request_logs():
    py_compile.compile(str(ROOT / "dashboard.py"), doraise=True)

    import dashboard

    with sqlite3.connect(DB_PATH) as conn:
        logs = pd.read_sql_query(dashboard.LOGS_QUERY, conn)

    assert not logs.empty
    assert {"id", "timestamp", "provider_used", "cache_hit", "ttft_ms"}.issubset(logs.columns)
