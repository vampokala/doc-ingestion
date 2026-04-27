"""FastAPI application for citation-aware querying."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from redis import Redis
from redis.exceptions import RedisError

from src.api.models import (
    HealthModel,
    MetricsModel,
    QueryRequestModel,
    QueryResponseModel,
    RetrievedChunkModel,
    TruthfulnessModel,
)
from src.core.rag_orchestrator import COLLECTION_NAME, QueryRequest, RAGOrchestrator
from src.utils.config import load_config

app = FastAPI(title="Doc Ingestion Citation API", version="0.1.0")
_cfg = load_config("config.yaml")
_orchestrator = RAGOrchestrator(_cfg)
_rate_window: Dict[str, Deque[float]] = defaultdict(deque)
_redis_client: Redis | None = None
_logger = logging.getLogger("api.audit")


def _resolve_client_key(request: Request, api_key: str | None) -> str:
    if api_key:
        return f"key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def _audit_log(event: str, request: Request, **fields: object) -> None:
    payload = {
        "event": event,
        "path": request.url.path,
        "method": request.method,
        "client_ip": request.client.host if request.client else "unknown",
        "ts": int(time.time()),
    }
    payload.update(fields)
    _logger.info(json.dumps(payload, separators=(",", ":")))


def _get_redis() -> Redis | None:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _cfg.api.redis_rate_limit_enabled:
        return None
    try:
        _redis_client = Redis.from_url(_cfg.api.redis_url, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except RedisError:
        return None


def _enforce_rate_limit(client_key: str) -> None:
    limit = int(_cfg.api.rate_limit_per_minute)
    redis_client = _get_redis()
    if redis_client is not None:
        now_bucket = int(time.time() // 60)
        redis_key = f"ratelimit:{client_key}:{now_bucket}"
        try:
            current = int(redis_client.incr(redis_key))
            if current == 1:
                redis_client.expire(redis_key, 120)
            if current > limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            return
        except RedisError:
            # fallback to in-memory limiter when Redis has transient issues
            pass
    now = time.time()
    q = _rate_window[client_key]
    while q and (now - q[0]) > 60.0:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    q.append(now)


def _is_local_provider(provider: str | None) -> bool:
    selected = (provider or _cfg.llm.default_provider or "").strip().lower()
    if selected == "claude":
        selected = "anthropic"
    return selected == "ollama"


def _verify_auth(api_key: str | None, provider: str | None = None) -> None:
    if not _cfg.api.auth_enabled:
        return
    # Local Ollama queries do not require API auth by default.
    if _is_local_provider(provider):
        return
    valid_keys = _cfg.api.resolved_api_keys()
    if not valid_keys:
        # secure-by-default for production paths
        raise HTTPException(status_code=503, detail="API auth enabled but no keys configured")
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=HealthModel)
def health() -> HealthModel:
    return HealthModel(status="ok", collection=COLLECTION_NAME)


@app.get("/metrics", response_model=MetricsModel)
def metrics(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> MetricsModel:
    # no request object here; audit logs are captured in protected endpoints with request context
    _verify_auth(x_api_key, provider="metrics")
    return MetricsModel(
        cache_ttl_seconds=int(_cfg.generation.cache_ttl),
        available_providers=sorted(_cfg.llm.allowed_models_by_provider.keys()),
    )


@app.post("/query", response_model=QueryResponseModel)
def query(
    req: QueryRequestModel,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> QueryResponseModel:
    try:
        _verify_auth(x_api_key, req.provider)
    except HTTPException as exc:
        _audit_log("auth_failed", request, status=exc.status_code, reason=exc.detail)
        raise
    client_key = _resolve_client_key(request, x_api_key)
    _enforce_rate_limit(client_key)
    _audit_log("auth_success", request, client_key=client_key)
    try:
        out = _orchestrator.run(
            QueryRequest(
                query_text=req.query,
                top_k=req.top_k,
                use_llm=req.use_llm,
                use_rerank=req.use_rerank,
                stream=req.stream,
                provider=req.provider,
                model=req.model,
                provider_api_key=req.provider_api_key,
                reranker_model=req.reranker_model,
                include_citations=req.include_citations,
            )
        )
    except Exception as exc:  # keep API error shape stable
        _audit_log("query_failed", request, client_key=client_key, error=str(exc))
        _logger.exception("query_failed provider=%s model=%s client_key=%s", req.provider, req.model, client_key)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_log(
        "query_success",
        request,
        client_key=client_key,
        provider=out.provider,
        model=out.model,
        latency_ms=round(out.processing_time_ms, 2),
        cached=out.cached,
    )

    retrieved = []
    for item in out.retrieved:
        legacy = item.to_legacy_dict()
        retrieved.append(
            RetrievedChunkModel(
                id=legacy["id"],
                score=float(legacy.get("score") or 0.0),
                source=str(legacy.get("source") or "hybrid"),
                confidence=float(legacy.get("confidence") or 0.0),
                metadata=dict(legacy.get("metadata") or {}),
                preview=(legacy.get("text") or "")[:240],
            )
        )

    truthfulness_model = None
    if out.truthfulness is not None:
        t = out.truthfulness
        truthfulness_model = TruthfulnessModel(
            nli_faithfulness=t.nli_faithfulness,
            citation_groundedness=t.citation_groundedness,
            uncited_claims=t.uncited_claims,
            score=t.score,
        )

    return QueryResponseModel(
        query=out.query,
        provider=out.provider,
        model=out.model,
        answer=out.answer,
        processing_time_ms=out.processing_time_ms,
        cached=out.cached,
        validation_issues=out.validation_issues,
        citations=out.citations,
        retrieved=retrieved,
        truthfulness=truthfulness_model,
    )


@app.post("/query/stream")
def query_stream(
    req: QueryRequestModel,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> StreamingResponse:
    try:
        _verify_auth(x_api_key, req.provider)
    except HTTPException as exc:
        _audit_log("auth_failed", request, status=exc.status_code, reason=exc.detail)
        raise
    client_key = _resolve_client_key(request, x_api_key)
    _enforce_rate_limit(client_key)
    _audit_log("stream_auth_success", request, client_key=client_key)

    def _gen() -> Iterator[str]:
        try:
            stream_req = QueryRequest(
                query_text=req.query,
                top_k=req.top_k,
                use_llm=req.use_llm,
                use_rerank=req.use_rerank,
                stream=True,
                provider=req.provider,
                model=req.model,
                provider_api_key=req.provider_api_key,
                reranker_model=req.reranker_model,
                include_citations=req.include_citations,
            )
            for piece in _orchestrator.stream(stream_req):
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
            final = _orchestrator.run(stream_req)
            _audit_log(
                "stream_success",
                request,
                client_key=client_key,
                provider=final.provider,
                model=final.model,
            )
            yield f"data: {json.dumps({'type': 'final', 'citations': final.citations, 'provider': final.provider, 'model': final.model})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            _audit_log("stream_failed", request, client_key=client_key, error=str(exc))
            _logger.exception("stream_failed provider=%s model=%s client_key=%s", req.provider, req.model, client_key)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
