# VanguardAI - Enterprise LLM Gateway

An OpenAI-compatible gateway that sits between your application and hosted LLM providers, adding the controls that enterprises need before a prompt leaves the network: reversible PII tokenization, semantic response caching, automatic provider failover, prompt-injection screening, and per-request cost telemetry.

Any OpenAI SDK client can point at `http://localhost:8000/v1` and talk to Groq or OpenRouter without changing a line of application code. Prompts are scrubbed of personal data on the way out, the raw values are put back on the way in, and every request is measured and logged.

```
POST /v1/chat/completions   ->   scrub   ->   cache   ->   provider   ->   rehydrate   ->   client
```

## Key Features

### Bidirectional PII Tokenization

Emails, phone numbers, credit cards, Aadhaar numbers, and API keys are replaced with reversible tokens (`{{EMAIL_1}}`, `{{PHONE_1}}`, ...) before a prompt reaches a provider. The provider never sees the raw value; the caller always does. Tokens are numbered per category across a conversation and identical values reuse the same token, so a vault is consistent for the whole request. The tag-to-raw map is stored in Redis under a 300-second TTL, so any worker can rehydrate the response and the raw values leave storage on their own.

### Sliding-Window Stream Rehydration

Restoring tokens in a stream is not a simple string replace, because a tag regularly arrives split across chunk boundaries (`"...is {{EMA"` + `"IL_1}} and"`). A lookahead buffer holds back any text from the last unmatched `{{` until the tag closes, then substitutes and emits it with the correct chunk. A 64-character cap guarantees that a literal `{{` in model prose can never stall the stream. Non-streaming responses use the same substitution applied once.

### Two-Stage Semantic Caching

Prompts are embedded with `all-MiniLM-L6-v2` and the three nearest neighbours are retrieved from Qdrant. A bi-encoder cannot separate a paraphrase from a same-topic question with a different intent (both score around `0.8`), so each candidate is then scored pairwise by `cross-encoder/ms-marco-MiniLM-L6-v2`. Only a candidate whose cross-encoder score clears `0.05` after a sigmoid is served; anything else falls through to the provider. Measured on this pair of models, identical prompts score `0.996` and same-topic-different-intent prompts stay below `0.007`, so the cutoff trades a few missed paraphrases for never returning the wrong answer. Only tokenized text is stored, so a cached answer is rehydrated with the requesting client's vault rather than leaking another caller's values.

### Provider Fallback Cascade

Requests route to Groq by default and to OpenRouter for `deepseek` models. A `429`, any `5xx`, or a connection error transparently retries once on the other provider with the same scrubbed payload; client errors such as `401` or `404` are surfaced immediately instead of being retried somewhere else. The client only sees an error if both providers fail.

### Prompt Injection Guardrails

A regex heuristic screens user messages before anything else happens - before scrubbing, before the cache, before any provider call. Detected jailbreak phrasing (`ignore all previous instructions`, `DAN`, `system prompt`, `bypass`, `developer mode`, and similar) is rejected with HTTP `403` and a `policy_violation` body.

### SQLite Telemetry

Every request - cache hit or miss - is written to `request_logs` in `vanguard.db`: provider used, cache-hit flag, time to first token, and estimated cost. The Streamlit dashboard turns those rows into request counts, hit rate, and average latency.

## Architecture

```
  +-----------------------+
  |  Streamlit UI         |   dashboard.py
  |  chat + telemetry     |
  +-----------+-----------+
              |  POST /v1/chat/completions  (SSE when stream=true)
              v
  +-----------------------+
  |  FastAPI gateway      |   app/main.py -> app/api/routes.py
  +-----------+-----------+
              |
              v
  +-----------------------+        rejected -> 403 policy_violation
  |  PromptGuard          |  -----------------------------> client
  +-----------+-----------+
              |
              v
  +-----------------------+        {{EMAIL_1}} -> raw on the way back
  |  PIIScrubber          |        vault in Redis, 300s TTL
  +-----------+-----------+
              |
              v
  +-----------------------+        hit  -> rehydrate -> client (0 cost)
  |  Semantic cache       |  -----------------------------> client
  |  Qdrant + cross-encoder    |
  +-----------+-----------+
              |  miss
              v
  +-----------------------+        Groq  <--fallback-->  OpenRouter
  |  Provider call        |
  +-----------+-----------+
              |
              v
  +-----------------------+        tokens swapped back as chunks arrive
  |  Rehydration buffer   |
  +-----------+-----------+
              |
              v
          SSE stream -> client
              |
              v
  +-----------------------+
  |  SQLite telemetry     |   vanguard.db / request_logs
  +-----------------------+
```

Request lifecycle:

1. `PromptGuard` screens every user message and short-circuits to `403` on a suspected injection.
2. `PIIScrubber` tokenizes user text and accumulates the vault for this request.
3. The tokenized prompt is embedded and looked up in Qdrant. A hit is rehydrated with the current vault, logged at zero cost, and returned.
4. On a miss, the request goes to the routed provider, falling back to the other one on `429`/`5xx`/connection errors.
5. The response is rehydrated - chunk by chunk with the sliding buffer when streaming - and the tokenized exchange is cached.
6. Telemetry is written for every request.

## Quick Start

### Prerequisites

Python 3.11 or newer, and API keys for at least one provider. Installing `sentence-transformers` pulls PyTorch, so expect a few hundred MB on first install and roughly 200 MB across the two model downloads.

### 1. Install

```bash
git clone https://github.com/aayush-1105/VanguardAI.git
cd VanguardAI
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure the environment

Create a `.env` file in the project root. It is ignored by git and must never be committed.

```dotenv
GOOGLE_API_KEY=your_google_key
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
```

All three are read at import time by `app/core/config.py`; the application will not start if one is missing. `GOOGLE_API_KEY` is loaded and reserved for a future provider route and is not yet used for any call.

### 3. Run the gateway (terminal 1)

```bash
uvicorn app.main:app --reload
```

Health check: `curl http://localhost:8000/health` -> `{"status":"ok","service":"VanguardAI Gateway"}`

### 4. Run the dashboard (terminal 2)

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501`. The chat tab talks to the gateway on port 8000, so the gateway must be running first. The telemetry tab reads `vanguard.db` from the gateway's working directory.

### 5. Send a request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Streaming uses the standard OpenAI shape:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}'
```

Model ids are provider-specific and change over time. `openai/gpt-oss-20b` and `openai/gpt-oss-120b` are valid Groq ids today; `llama-3.3-70b-versatile` may be retired on your account. DeepSeek ids such as `deepseek/deepseek-chat` route to OpenRouter.

### 6. Or run the whole stack with Docker

```bash
docker compose up --build
```

Three services start: `api` on `:8000`, `redis` on `:6379`, and `qdrant` on `:6333`. The API reads `REDIS_URL` and `QDRANT_URL` from the compose environment, so the PII vault and the semantic cache are shared between workers instead of living in one process. Model weights are cached in the `hf_cache` volume; `vanguard.db` stays inside the container, so attach a volume if the telemetry must survive a rebuild.

Docker is not required for local development. Without `QDRANT_URL` the cache uses an in-memory Qdrant, and a Redis connection is only attempted when a vault is used - if it is unreachable the vault falls back to an in-process dict and logs a warning.

## Load Testing

```bash
pip install -r requirements.txt
locust -f locustfile.py --host http://localhost:8000
```

Open `http://localhost:8089`, choose the number of users, and start. The test posts to `/v1/chat/completions`; since every user sends the same prompt, most requests become cache hits after the first one, which measures the gateway path rather than provider latency. Run it against the Docker stack to exercise Redis and a networked Qdrant.

## Project Layout

```
app/
  api/
    routes.py          POST /v1/chat/completions, guardrail, error mapping
    schemas.py         ChatCompletionRequest
  core/
    config.py          .env loading and required keys
  db/
    database.py        async SQLAlchemy engine, session factory, table creation
    models.py          RequestLog
  services/
    llm_service.py     routing, cache integration, rehydration buffer, fallback
    pii_service.py     PIIScrubber tokenization and vault
    security_service.py PromptGuard
    cache_service.py   Qdrant collection and embeddings
    log_service.py     telemetry writes and cost estimation
  main.py              application factory and lifespan
dashboard.py           Streamlit chat UI and telemetry view
Dockerfile             API image
docker-compose.yml     api + redis + qdrant
locustfile.py          load test for /v1/chat/completions
tests/                 28-test end-to-end suite
requirements.txt
```

Runtime artifacts: `vanguard.db` (SQLite telemetry), created on first start. It is ignored by git.

## Testing

The repository ships a 28-test end-to-end suite powered by pytest. Providers are replaced with fakes and Redis with `fakeredis`, so the suite is deterministic, runs offline, and never spends API credits. The cross-encoder is also stubbed, so the tests download no model beyond the bi-encoder.

```bash
python -m pytest tests -q
```

```
28 passed in 20.11s
```

Coverage includes PII pattern detection and token numbering, the Redis vault and its TTL, scrub-before-dispatch, cache hit/miss behavior with no upstream call on a hit, cross-encoder rejection and top-K rerank selection, model routing, upstream error mapping, log persistence, non-stream rehydration, streaming rehydration of tags split across chunk boundaries, the fallback cascade on `429`, prompt-injection rejection, and cost telemetry (miss charges, hit is free).

## Configuration Reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Default provider for all non-DeepSeek models |
| `OPENROUTER_API_KEY` | yes | DeepSeek models, and the fallback target when Groq fails |
| `GOOGLE_API_KEY` | yes | Loaded at startup; reserved for a future provider route |
| `REDIS_URL` | no | PII vault store; defaults to `redis://localhost:6379/0` |
| `QDRANT_URL` | no | Remote Qdrant; defaults to an in-memory collection |

Tunable constants live next to the code that uses them: `SIMILARITY_THRESHOLD`, `CROSS_ENCODER_THRESHOLD`, and `RERANK_CANDIDATES` in `app/services/cache_service.py`, `VAULT_TTL_SECONDS` in `app/services/pii_service.py`, `FALLBACK_MODELS` in `app/services/llm_service.py`, pricing in `app/services/log_service.py`, and the guard patterns in `app/services/security_service.py`.

## Known Limitations

- **No authentication on the gateway.** Anyone who can reach `/v1` can spend the configured provider keys. Put it behind a reverse proxy with auth, or add an API-key dependency, before exposing it.
- **Telemetry is still per process.** The PII vault is shared through Redis and the cache through Qdrant, but `request_logs` lives in the container's SQLite file.
- **Cost is an estimate.** Prices are derived from character counts at 4 characters per token, using a flat rate, rather than from provider-reported usage.
- **The injection guard is a heuristic.** It flags the literal phrase "system prompt", so a legitimate question about prompts is rejected as well.
- **The cache favours precision.** MS MARCO is a query-to-passage reranker, so it under-scores question pairs: a valid paraphrase can miss the cache and reach the provider. A wrong answer is never served in its place.
- **Failover changes the model.** A DeepSeek request that falls back is answered by the secondary provider's model, which is reflected in the response's `model` field.
