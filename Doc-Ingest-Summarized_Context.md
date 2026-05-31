# Claude Project Context: Doc-Ingestion

Use this file as the single memory attachment for AI help in this repository. It reflects the current architecture, runtime behavior, and operational conventions.

## Project Summary

Doc-Ingestion is a local-first, citation-aware RAG system for private document collections. It ingests files, builds hybrid retrieval indexes (BM25 + vectors), fuses and optionally reranks results, generates answers from retrieved context, and returns citations plus truthfulness signals.

It is a full application surface, not a prompt-only demo:

- React UI served by FastAPI static mount (`/`)
- FastAPI endpoints for config, query, streaming query, health, and metrics
- Optional Streamlit legacy UI
- CLI entry points for ingestion and query
- Docker Compose stack with API, Streamlit, Redis, and Qdrant
- Offline evaluation harness and test suites

## Why This Exists

Knowledge is usually spread across PDFs, DOCX, markdown notes, text files, and HTML exports. Keyword search alone is not enough for synthesis, and generic LLM answers can hallucinate without corpus grounding.

Doc-Ingestion addresses this by enforcing a retrieval-first workflow and exposing citations and support signals so users can inspect why an answer is trustworthy.

## Functional Capabilities

- Ingests local folders and uploaded files.
- Supports `.pdf`, `.docx`, `.txt`, `.md`, `.html`.
- Supports configurable chunking strategies: `tiktoken`, `spacy`, `nltk`, `medical`, `legal`.
- Supports configurable embedding profiles (Ollama + sentence-transformers profiles).
- Persists sparse index + vector index for hybrid retrieval.
- Uses weighted RRF fusion with optional cross-encoder reranking.
- Routes generation to Ollama, OpenAI, Anthropic, or Gemini per request.
- Returns answer text, retrieved evidence previews, citations, and `truthfulness` metrics.

## User-Facing Surfaces

- **Primary UI:** React app served by FastAPI on `:8000`.
- **API:** FastAPI for app integration and automation.
- **Legacy UI:** Streamlit on `:8501` (optional, still supported).
- **CLI:** `src/ingest.py` and `src/query.py`.

Primary endpoints:

- `GET /health`
- `GET /metrics`
- `GET /config/llm`
- `GET /config/runtime`
- `GET /observability/dashboard`
- `POST /query`
- `POST /query/stream` (SSE)

Demo-session endpoints (enabled only when `DOC_PROFILE=demo` and `DOC_DEMO_UPLOADS=1`):

- `POST /sessions`
- `GET /sessions/{sid}`
- `POST /sessions/{sid}/documents`
- `DELETE /sessions/{sid}`

## Technical Architecture

### Ingestion lifecycle

1. Files are loaded from `data/documents/` or UI uploads.
2. `DocumentProcessor` parses/normalizes source formats.
3. Chunks are generated with configured chunking strategy.
4. Chunks are indexed into BM25.
5. Embeddings are generated with selected embedding profile.
6. Vectors + metadata are stored in the vector DB (Chroma by default; Qdrant in Docker stack).

### Query lifecycle

1. Request enters via API/UI/CLI.
2. Query is sent to sparse + dense retrievers.
3. Results are fused with weighted RRF.
4. Optional cross-encoder reranking refines top context.
5. Context optimizer packs chunks under token budget.
6. Provider router invokes selected model/provider.
7. Citation tracker maps and verifies references.
8. Truthfulness scoring computes support signals.
9. Structured response is returned.

## Grounding And Quality Contract

The project follows this contract:

1. Retrieve evidence first.
2. Generate from retrieved evidence.
3. Attach citations to source chunks.
4. Verify citations and report support metrics.

Key response quality fields:

- `nli_faithfulness`
- `citation_groundedness`
- `uncited_claims`
- `score`

## Provider Strategy

- Default provider is local Ollama in local environments.
- Cloud providers (OpenAI, Anthropic, Gemini) are optional and key-driven.
- Provider/model can be selected per request.
- Runtime may hide Ollama automatically in hosted Spaces environments.

`SPACE_ID` and `DOC_OLLAMA_ENABLED` influence runtime provider availability via `src/utils/config.py`.

## Config And Runtime Behavior

Central config is `config.yaml` (plus optional environment-specific overlays like `config.dev.yaml`).

Important sections:

- `chunking` for default and allowed strategies
- `embeddings` for profiles and defaults
- `llm` for provider/model allowlists and defaults
- `api` for auth and rate limiting
- `evaluation` for inline truthfulness scoring

Notable runtime behavior:

- If `SPACE_ID` is set, runtime disables Ollama provider and prefers non-Ollama embedding defaults.
- If Redis rate limiting is enabled but Redis fails, API falls back to in-memory limiter.
- Local Ollama query requests can run without API key; non-local providers still require auth when enabled.

## Security And Operations

- API auth: `api.api_keys` or `DOC_API_KEYS`.
- Cloud provider keys come from environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) or per-request/session input in UI paths.
- Structured audit events are logged for auth/query/stream success/failure.
- CORS origins come from `DOC_FRONTEND_ORIGINS` or project defaults.

## Important Code Areas

- `src/api/main.py`: API app, routing, auth, rate limiting, streaming, session uploads, React static mount.
- `src/core/`: retrieval, reranking, orchestration, generation, citations, provider routing.
- `src/utils/config.py`: config schema, env overrides, runtime provider/profile adaptation.
- `src/web/`: Streamlit app and ingest service used by legacy/demo paths.
- `frontend/`: React UI.
- `src/evaluation/` and `evals/`: online truthfulness + offline eval harness.
- `config.yaml`: runtime defaults and allowlists.

## Evaluation And Validation

Standard test commands:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit -q
PYTHONPATH=. .venv/bin/python -m pytest tests/integration -q
```

Typical eval command:

```bash
PYTHONPATH=. python -m evals.run_evals --dataset evals/datasets/smoke.jsonl --mock --no-nli --output evals/reports/
```

## Current Project State

Core phases are implemented end-to-end: ingestion, hybrid retrieval, reranking, grounded generation, citations, truthfulness, API auth/rate limiting, React UI integration, and deployment paths.

Active areas for continued improvement:

- Production hardening and observability depth
- Auth/session isolation maturity
- Broader regression/eval automation across providers and embedding/chunking combinations

## Assistant Guidance

When helping on this codebase, preserve these principles unless explicitly asked to redesign:

- local-first behavior
- retrieval-grounded answers
- citation-aware responses
- config-driven provider/model/profile controls
- hybrid retrieval flow (BM25 + vectors + weighted RRF + optional rerank)

Prefer extending existing modules over creating parallel abstractions, and keep API contracts stable where practical.
