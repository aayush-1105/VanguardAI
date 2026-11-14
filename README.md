# VanguardAI

An OpenAI-compatible LLM gateway that adds the controls an enterprise needs before a prompt leaves the network: reversible PII tokenization, two-stage semantic caching, automatic provider failover, prompt-injection screening, server-side hybrid RAG over your own documents, and per-request cost telemetry.

Point any OpenAI SDK client at `http://localhost:8000/v1` and talk to Groq or OpenRouter without changing application code. Prompts are scrubbed of personal data on the way out, the raw values are restored on the way in, and every request is measured and logged.

VanguardAI v2 moves the gateway out of a single process: the PII vault is shared through Redis with a TTL, the semantic cache and the knowledge base are shared through Qdrant, and the whole stack runs from one `docker compose up`.

```
client -> PromptGuard -> PIIScrubber -> semantic cache -> provider -> rehydrate -> client
                                            |  hit                       |
                                            +--> rehydrate ---------------+
```

## v2 Architecture

### Distributed State

The tag-to-raw mapping for a request is written to Redis as a hash at `vanguard:vault:<uuid>` with a 300-second TTL. Raw values therefore leave storage on their own, and any worker can rehydrate a response - a request is not pinned to the process that scrubbed it. The gateway holds no per-request state in local memory, so it can run behind a load balancer with multiple workers or replicas.

Redis is not a hard dependency: if it is unreachable, the vault falls back to an in-process dict, logs a warning, and the request still completes correctly on that worker. The cache points at Qdrant via `QDRANT_URL` (in-memory collection when unset).

### Two-Stage Semantic Caching

A single bi-encoder cannot do this job alone. `all-MiniLM-L6-v2` scores true paraphrases around `0.76-0.79` and same-topic questions with a different intent around `0.80`, so any threshold that admits paraphrases also admits prompts that deserve a different answer.

v2 splits the decision:

1. **Recall:** embed the prompt and pull the Top-K = 3 nearest entries from Qdrant, keeping only those at or above `SIMILARITY_THRESHOLD = 0.75`.
2. **Precision:** score each candidate against the prompt with `cross-encoder/ms-marco-MiniLM-L6-v2`, squash the logits through a sigmoid, and serve the best one only if it clears `CROSS_ENCODER_THRESHOLD = 0.05`.

The cutoff was calibrated on the real model rather than guessed:

| Pair | Sigmoid score |
| --- | --- |
| Identical prompt | 0.996 |
| Genuine paraphrase | 0.31 |
| Another phrasing of the same question | 0.014 |
| Same topic, different intent | 0.0001 - 0.007 |
| Unrelated | 0.000 |

The threshold favours precision deliberately: a missed paraphrase costs one provider call, while a false hit returns another prompt's answer. Only tokenized text is cached, so a hit is rehydrated with the requesting client's own vault.

### Server-Side Hybrid RAG

Upload a document and the gateway answers questions from it without the client assembling any context. `POST /v1/knowledge/ingest` accepts PDF or Markdown, extracts the text, splits it into 500-character chunks with 50 characters of overlap, embeds each chunk with the same `all-MiniLM-L6-v2` model, and stores it in a Qdrant collection (`vanguard_knowledge_base`) together with its filename, timestamp, and chunk index.

A chat request with `"rag_enabled": true` runs hybrid retrieval before dispatch:

1. **Dense:** the 10 nearest chunks by cosine similarity in Qdrant.
2. **Sparse:** BM25 over the collection, which catches exact identifiers an embedding can blur, such as a launch code.
3. **Fusion:** Reciprocal Rank Fusion (`k = 60`) merges both rankings, so a chunk that both retrievers rank rises to the top.
4. **Rerank:** the cross-encoder scores each fused candidate against the question; chunks below `0.05` are dropped and the best 3 are injected as a system message.

Retrieved text is treated as untrusted. It passes through the same PII scrubber as user input, continuing the request's token numbering, so only tokens reach the provider and the caller receives the raw values back. A RAG answer is neither served from nor written to the prompt-keyed semantic cache, because it depends on the knowledge base rather than the prompt alone. If retrieval fails or the knowledge base is empty, the request proceeds without context instead of failing.

Ingestion is bounded: an upload above 10 MB is rejected with `413`, and an empty or unparseable document, including a malformed PDF, returns `400`.

### Resilience and Testing

- **Failover cascade:** a `429`, any `5xx`, or a connection error retries once on the other provider with the same scrubbed payload. Client errors such as `401` or `404` surface immediately.
- **Docker Compose:** `api`, `redis`, and `qdrant` start together, wired through `REDIS_URL` and `QDRANT_URL`.
- **Locust:** `locustfile.py` drives `/v1/chat/completions` for throughput and latency measurements.
- **43-test end-to-end suite:** runs offline, with providers, Redis (`fakeredis`), and the cross-encoder stubbed. `43 passed in 23.08s`.

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
  |  Qdrant Top-3         |
  |  + cross-encoder      |
  +-----------+-----------+
              |  miss
              v
  +-----------------------+        Groq  <--failover-->  OpenRouter
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
2. `PIIScrubber` tokenizes user text and writes the tag-to-raw mapping to Redis under a 300-second TTL.
3. With `rag_enabled`, the knowledge base is searched (dense + BM25, fused by RRF, reranked), the surviving chunks are scrubbed into the same vault, and they are prepended as a system message. RAG requests skip the prompt cache.
4. The tokenized prompt is looked up in Qdrant; the Top-3 candidates are reranked by the cross-encoder. A confirmed hit is rehydrated, logged at zero cost, and returned.
5. On a miss, the request goes to the routed provider, falling back to the other one on `429`/`5xx`/connection errors.
6. The response is rehydrated - chunk by chunk with the sliding buffer when streaming - and the tokenized exchange is cached when it was not a RAG request.
7. Telemetry is written for every request.

## Sliding-Window Stream Rehydration

Restoring tokens in a stream is not a string replace, because a tag regularly arrives split across chunks (`"...is {{EMA"` + `"IL_1}} and"`). A lookahead buffer holds back text from the last unmatched `{{` until the tag closes, then substitutes and emits it with the right chunk. A 64-character cap guarantees a literal `{{` in model prose can never stall the stream. Non-streaming responses use the same substitution applied once.

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

### 6. Index a document and ask about it

```bash
curl -s -X POST http://localhost:8000/v1/knowledge/ingest \
  -F "file=@handbook.pdf"
# {"filename":"handbook.pdf","chunks_indexed":12}

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "rag_enabled": true,
       "messages": [{"role": "user", "content": "What does the handbook say about leave?"}]}'
```

The knowledge base is shared through `QDRANT_URL`, so every worker sees the same documents.

### 7. Or run the whole stack with Docker

```bash
docker compose up --build
```

Three services start: `api` on `:8000`, `redis` on `:6379`, and `qdrant` on `:6333`. The API reads `REDIS_URL` and `QDRANT_URL` from the compose environment, so the vault and the cache are shared between workers instead of living in one process. Model weights are cached in the `hf_cache` volume; `vanguard.db` stays inside the container, so attach a volume if the telemetry must survive a rebuild.

Docker is not required for local development. Without `QDRANT_URL` the cache uses an in-memory Qdrant, and a Redis connection is attempted only when a vault is used.

## Load Testing

```bash
locust -f locustfile.py --host http://localhost:8000
```

Open `http://localhost:8089`, choose the number of users, and start. The test posts to `/v1/chat/completions`; since every user sends the same prompt, most requests become cache hits after the first one, which measures the gateway path rather than provider latency. Run it against the Docker stack to exercise Redis and a networked Qdrant.

## Testing

The repository ships a 43-test end-to-end suite powered by pytest. Providers are replaced with fakes, Redis with `fakeredis`, and the cross-encoder is stubbed, so the suite is deterministic, runs offline, and never spends API credits or downloads a second model.

```bash
python -m pytest tests -q
```

```
43 passed in 23.08s
```

Coverage includes PII pattern detection and token numbering, the Redis vault and its TTL, scrub-before-dispatch, cache hit/miss behavior with no upstream call on a hit, cross-encoder rejection and Top-K selection, model routing, upstream error mapping, log persistence, non-stream rehydration, streaming rehydration of tags split across chunk boundaries, the failover cascade on `429`, prompt-injection rejection, cost telemetry (miss charges, hit is free), dashboard queries, and the Locust workload definition.

The RAG tests cover Markdown and PDF ingestion (the PDF is built in memory), the `400`/`413` ingest guardrails, retrieval through both dense search and BM25, RRF fusion, the cross-encoder cutoff and chunk cap, context injection with the cache bypassed, PII scrubbed from retrieved context and rehydrated in the answer, and a RAG SSE stream whose tag is split across chunk boundaries.

## Project Layout

```
app/
  api/
    routes.py          POST /v1/chat/completions, POST /v1/knowledge/ingest, guardrail, error mapping
    schemas.py         ChatCompletionRequest
  core/
    config.py          .env loading and required keys
  db/
    database.py        async SQLAlchemy engine, session factory, table creation
    models.py          RequestLog
  services/
    llm_service.py     routing, RAG context injection, cache integration, rehydration, failover
    pii_service.py     PIIScrubber tokenization and Redis vault
    security_service.py PromptGuard
    cache_service.py   Qdrant retrieval, cross-encoder reranking
    log_service.py     telemetry writes and cost estimation
  main.py              application factory and lifespan
vanguard/
  rag/
    ingestion.py       PDF/Markdown extraction, chunking, embedding, upsert
    retrieval.py       dense + BM25 retrieval, RRF fusion, cross-encoder rerank
dashboard.py           Streamlit chat UI and telemetry view
Dockerfile             API image
docker-compose.yml     api + redis + qdrant
locustfile.py          load test for /v1/chat/completions
tests/                 43-test end-to-end suite
requirements.txt
```

Runtime artifacts: `vanguard.db` (SQLite telemetry), created on first start. It is ignored by git.

## Configuration Reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Default provider for all non-DeepSeek models |
| `OPENROUTER_API_KEY` | yes | DeepSeek models, and the failover target when Groq fails |
| `GOOGLE_API_KEY` | yes | Loaded at startup; reserved for a future provider route |
| `REDIS_URL` | no | PII vault store; defaults to `redis://localhost:6379/0` |
| `QDRANT_URL` | no | Remote Qdrant; defaults to an in-memory collection |

Tunable constants live next to the code that uses them: `SIMILARITY_THRESHOLD`, `CROSS_ENCODER_THRESHOLD`, and `RERANK_CANDIDATES` in `app/services/cache_service.py`, `VAULT_TTL_SECONDS` in `app/services/pii_service.py`, `FALLBACK_MODELS` in `app/services/llm_service.py`, pricing in `app/services/log_service.py`, and the guard patterns in `app/services/security_service.py`.

## Known Limitations

- **No authentication on the gateway.** Anyone who can reach `/v1` can spend the configured provider keys. Put it behind a reverse proxy with auth, or add an API-key dependency, before exposing it.
- **Telemetry is still per process.** The PII vault is shared through Redis and the cache through Qdrant, but `request_logs` lives in the container's SQLite file.
- **The vault degrades to process-local without Redis.** Requests still succeed on that worker, but they cannot be rehydrated by another one.
- **Cost is an estimate.** Prices are derived from character counts at 4 characters per token, using a flat rate, rather than from provider-reported usage.
- **The injection guard is a heuristic.** It flags the literal phrase "system prompt", so a legitimate question about prompts is rejected as well.
- **The cache favours precision.** MS MARCO is a query-to-passage reranker, so it under-scores question pairs: a valid paraphrase can miss the cache and reach the provider. A wrong answer is never served in its place.
- **Failover changes the model.** A DeepSeek request that falls back is answered by the secondary provider's model, which is reflected in the response's `model` field.
- **The BM25 side of RAG is rebuilt on every query.** It scrolls up to 10,000 chunks and constructs the index in memory, so query cost grows with the corpus; index it once or page it when the knowledge base is large.
- **Ingested documents are stored verbatim.** PII is tokenized on the way to the provider, for both user prompts and retrieved context, but the Qdrant payload keeps the original document text. Encrypt or redact the collection at ingest time if documents are sensitive at rest.
- **The prompt cache and RAG do not mix.** A `rag_enabled` request always calls a provider, so it is never cheaper than a cache hit.
