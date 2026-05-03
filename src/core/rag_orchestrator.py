"""Shared query orchestration for CLI, API, and Streamlit."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.bm25_index import BM25Index
from src.core.bm25_search import BM25Search
from src.core.citation_tracker import CitationTracker
from src.core.citation_verifier import CitationVerifier
from src.core.context_optimizer import ContextOptimizer
from src.core.generator import GenerationResult, RAGGenerator
from src.core.hybrid_retriever import HybridRetriever
from src.core.llm_provider import LLMProviderRouter
from src.core.observability import get_observer
from src.core.prompt_manager import PromptManager
from src.core.query_processor import QueryProcessor
from src.core.reranker import CrossEncoderReranker, RankedResult
from src.core.response_cache import ResponseCache, cache_key
from src.core.response_processor import ResponseProcessor
from src.core.retrieval_result import RetrievalResult
from src.core.vector_search import VectorSearch
from src.evaluation.truthfulness import TruthfulnessResult, TruthfulnessScorer
from src.utils.config import Config
from src.utils.database import VectorDatabase

BM25_INDEX_PATH = "data/embeddings/bm25_index.json"
CHROMA_PATH = "data/embeddings/chroma"
COLLECTION_NAME = "documents"
logger = logging.getLogger(__name__)


@dataclass
class QueryRequest:
    query_text: str
    top_k: int = 5
    use_llm: bool = True
    use_rerank: bool = True
    stream: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    provider_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    include_citations: bool = True
    session_bm25_index_path: Optional[str] = None
    session_collection_name: Optional[str] = None
    session_chroma_path: Optional[str] = None
    knowledge_scope: str = "global"


@dataclass
class QueryResponse:
    query: str
    provider: str
    model: str
    answer: str = ""
    retrieved: List[RetrievalResult] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    cached: bool = False
    validation_issues: List[str] = field(default_factory=list)
    truthfulness: Optional[TruthfulnessResult] = None
    # Per-step latencies (retrieval, reranking, generation, etc.)
    step_latencies: Dict[str, float] = field(default_factory=dict)


class RAGOrchestrator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.prompt_manager = PromptManager()
        self.context_optimizer = ContextOptimizer(
            max_context_tokens=cfg.context.max_tokens,
            tokenizer_name=cfg.context.tokenizer,
        )
        self.provider_router = LLMProviderRouter(cfg.llm)
        self.response_processor = ResponseProcessor()
        self.citation_tracker = CitationTracker()
        self.citation_verifier = CitationVerifier()
        self.cache = ResponseCache(ttl_seconds=int(cfg.generation.cache_ttl))
        self._truthfulness_scorer: Optional[TruthfulnessScorer] = None
        self.observer = get_observer()

    def _get_truthfulness_scorer(self) -> Optional[TruthfulnessScorer]:
        if not self.cfg.evaluation.inline_enabled:
            return None
        if self._truthfulness_scorer is None:
            self._truthfulness_scorer = TruthfulnessScorer()
        return self._truthfulness_scorer

    def _load_components(self, req: QueryRequest) -> tuple[
        BM25Index,
        VectorDatabase,
        QueryProcessor,
        Optional[tuple[BM25Index, VectorDatabase]],
        str,
    ]:
        qp = QueryProcessor()
        requested_scope = (req.knowledge_scope or "global").strip().lower()
        session_pair: Optional[tuple[BM25Index, VectorDatabase]] = None
        effective_scope = requested_scope
        if requested_scope in {"session", "both"}:
            has_paths = bool(
                req.session_bm25_index_path and req.session_collection_name and req.session_chroma_path
            )
            if has_paths and os.path.exists(str(req.session_bm25_index_path)):
                try:
                    s_index = BM25Index.load(str(req.session_bm25_index_path))
                    s_db = VectorDatabase(mode="dev", chroma_path=str(req.session_chroma_path))
                    session_pair = (s_index, s_db)
                except Exception:
                    effective_scope = "global"
            else:
                effective_scope = "global"

        # Session-only: never touch the global corpus (HF Spaces upload-only demos).
        if effective_scope == "session" and session_pair is not None:
            placeholder_db = VectorDatabase(mode="dev", chroma_path=CHROMA_PATH)
            return BM25Index(), placeholder_db, qp, session_pair, effective_scope

        # Both: if the global BM25 file is absent, fall back to session-only rather than failing.
        if (
            effective_scope == "both"
            and session_pair is not None
            and not os.path.isfile(BM25_INDEX_PATH)
        ):
            logger.warning(
                "Global BM25 index missing at %s; using session corpus only (knowledge_scope=both)",
                BM25_INDEX_PATH,
            )
            effective_scope = "session"
            placeholder_db = VectorDatabase(mode="dev", chroma_path=CHROMA_PATH)
            return BM25Index(), placeholder_db, qp, session_pair, effective_scope

        index = BM25Index.load(BM25_INDEX_PATH)
        db = VectorDatabase(mode="dev", chroma_path=CHROMA_PATH)
        return index, db, qp, session_pair, effective_scope

    def _retrieve(
        self,
        query_text: str,
        index: BM25Index,
        db: VectorDatabase,
        qp: QueryProcessor,
        top_k: int,
        *,
        collection_name: str = COLLECTION_NAME,
    ) -> List[RetrievalResult]:
        processed = qp.process_query(query_text)
        bm25_query = " ".join(processed.all_terms)
        hybrid = HybridRetriever(BM25Search(index), VectorSearch(db, collection_name))
        return hybrid.retrieve(
            bm25_query,
            query_text,
            k=top_k,
            collection_name_for_cache=collection_name,
        )

    @staticmethod
    def _dedup_results(items: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        out: List[RetrievalResult] = []
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def run(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        step_latencies: Dict[str, float] = {}

        selection = self.provider_router.resolve_selection(
            req.provider,
            req.model or os.environ.get("OLLAMA_QUERY_MODEL"),
            has_api_key_override=bool((req.provider_api_key or "").strip()),
        )
        key = cache_key(
            req.query_text,
            selection.model,
            req.top_k,
            provider=selection.provider,
            use_rerank=req.use_rerank,
            reranker_model=req.reranker_model or self.cfg.reranker.model,
            corpus_fingerprint=(
                f"{COLLECTION_NAME}:{BM25_INDEX_PATH}|{req.knowledge_scope}|"
                f"{req.session_collection_name or '-'}:{req.session_bm25_index_path or '-'}"
            ),
        )
        cached = self.cache.get(key) if req.use_llm else None
        if cached is not None:
            with self.observer.trace_request("rag_query_cached", query=req.query_text) as trace:
                pass  # trace the cache hit
            return QueryResponse(
                query=req.query_text,
                provider=cached.provider,
                model=cached.model_name,
                answer=cached.response_text,
                citations=cached.citations,
                processing_time_ms=cached.latency_ms,
                cached=True,
                step_latencies={},
            )

        with self.observer.trace_request("rag_query", query=req.query_text) as trace:
            index, db, qp, session_pair, effective_scope = self._load_components(req)
            retrieve_k = max(req.top_k, 20) if req.use_rerank else req.top_k

            with self.observer.trace_step(trace, "retrieval", {"top_k": retrieve_k}) as s:
                t_retrieval = time.perf_counter()
                if effective_scope == "session" and session_pair is not None:
                    s_index, s_db = session_pair
                    fused = self._retrieve(
                        req.query_text,
                        s_index,
                        s_db,
                        qp,
                        top_k=retrieve_k,
                        collection_name=req.session_collection_name or COLLECTION_NAME,
                    )
                elif effective_scope == "both" and session_pair is not None:
                    global_results = self._retrieve(req.query_text, index, db, qp, top_k=retrieve_k)
                    s_index, s_db = session_pair
                    session_results = self._retrieve(
                        req.query_text,
                        s_index,
                        s_db,
                        qp,
                        top_k=retrieve_k,
                        collection_name=req.session_collection_name or COLLECTION_NAME,
                    )
                    fused = self._dedup_results(global_results + session_results, retrieve_k)
                else:
                    fused = self._retrieve(req.query_text, index, db, qp, top_k=retrieve_k)
                step_latencies["retrieval"] = (time.perf_counter() - t_retrieval) * 1000.0
                s["chunks_retrieved"] = len(fused)

            ranked: List[RankedResult] | None = None
            docs_for_gen: List[RetrievalResult] | List[RankedResult]
            display_items: List[RetrievalResult]

            if req.use_rerank:
                with self.observer.trace_step(trace, "reranking", {"input_chunks": len(fused)}) as s:
                    t_rerank = time.perf_counter()
                    try:
                        reranker = CrossEncoderReranker(
                            model_name=req.reranker_model or self.cfg.reranker.model,
                            batch_size=self.cfg.reranker.batch_size,
                            score_threshold=self.cfg.reranker.score_threshold,
                        )
                        ranked = reranker.rerank(req.query_text, fused, top_k=req.top_k)
                    except Exception as exc:
                        # Keep query path available even when model download/init fails.
                        logger.warning("Reranker unavailable; falling back to retrieval order: %s", exc)
                        ranked = None
                    step_latencies["reranking"] = (time.perf_counter() - t_rerank) * 1000.0
                    if ranked is not None:
                        docs_for_gen = ranked
                        display_items = [r.result for r in ranked]
                        s["output_chunks"] = len(ranked)
                    else:
                        docs_for_gen = fused[: req.top_k]
                        display_items = list(docs_for_gen)
                        s["output_chunks"] = len(display_items)
                        s["fallback"] = "retrieval_order"
            else:
                docs_for_gen = fused[: req.top_k]
                display_items = list(docs_for_gen)
                step_latencies["reranking"] = 0.0

            if not req.use_llm:
                return QueryResponse(
                    query=req.query_text,
                    provider=selection.provider,
                    model=selection.model,
                    retrieved=display_items,
                    processing_time_ms=(time.perf_counter() - t0) * 1000.0,
                    step_latencies=step_latencies,
                )

            query_type = PromptManager.intent_to_query_type(qp.process_query(req.query_text).intent)
            generator = RAGGenerator(
                model_name=selection.model,
                provider=selection.provider,
                prompt_manager=self.prompt_manager,
                context_optimizer=self.context_optimizer,
                provider_router=self.provider_router,
            )

            with self.observer.trace_step(
                trace, "generation",
                {"provider": selection.provider, "model": selection.model}
            ) as s:
                t_gen = time.perf_counter()
                if req.stream:
                    buf: List[str] = []
                    for piece in generator.generate_stream(
                        req.query_text,
                        docs_for_gen,
                        query_type=query_type,
                        provider=selection.provider,
                        model=selection.model,
                        provider_api_key=req.provider_api_key,
                    ):
                        buf.append(piece)
                    full = self.response_processor.format_response("".join(buf))
                    opt = self.context_optimizer.optimize_context(req.query_text, docs_for_gen)
                    raw_citations = self.citation_tracker.map_citations(full, opt.documents)
                    citations = self.citation_verifier.verify(full, raw_citations, opt.documents)
                    gen_result = GenerationResult(
                        response_text=full,
                        citations=citations,
                        model_name=selection.model,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        streamed=True,
                        optimized_context=opt,
                        provider=selection.provider,
                    )
                else:
                    gen_result = generator.generate(
                        req.query_text,
                        docs_for_gen,
                        stream=False,
                        query_type=query_type,
                        provider=selection.provider,
                        model=selection.model,
                        provider_api_key=req.provider_api_key,
                    )
                    docs = gen_result.optimized_context.documents if gen_result.optimized_context else []
                    tracked = self.citation_tracker.map_citations(gen_result.response_text, docs)
                    gen_result.citations = self.citation_verifier.verify(gen_result.response_text, tracked, docs)

                step_latencies["generation"] = (time.perf_counter() - t_gen) * 1000.0
                s["latency_ms"] = step_latencies["generation"]

            val = generator.validate_response(
                gen_result.response_text,
                gen_result.optimized_context or self.context_optimizer.optimize_context(req.query_text, docs_for_gen),
            )
            self.cache.set(key, gen_result)

            with self.observer.trace_step(trace, "citation_verification") as s:
                t_cit = time.perf_counter()
                truthfulness: Optional[TruthfulnessResult] = None
                scorer = self._get_truthfulness_scorer()
                if scorer is not None and gen_result.response_text.strip():
                    ctx = gen_result.optimized_context or self.context_optimizer.optimize_context(
                        req.query_text,
                        docs_for_gen,
                    )
                    source_texts = [str(d.get("text", "")) for d in ctx.documents if d.get("text")]
                    try:
                        truthfulness = scorer.score(
                            gen_result.response_text,
                            source_texts,
                            gen_result.citations,
                        )
                    except Exception:
                        pass  # never let truthfulness scoring break a response
                step_latencies["citation_verification"] = (time.perf_counter() - t_cit) * 1000.0
                s["citations_count"] = len(gen_result.citations)
                if truthfulness:
                    s["nli_faithfulness"] = truthfulness.nli_faithfulness
                    s["citation_groundedness"] = truthfulness.citation_groundedness

        self.observer.flush_async()

        return QueryResponse(
            query=req.query_text,
            provider=selection.provider,
            model=selection.model,
            answer=gen_result.response_text,
            citations=gen_result.citations if req.include_citations else [],
            retrieved=display_items,
            processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            validation_issues=val.issues,
            truthfulness=truthfulness,
            step_latencies=step_latencies,
        )

    def stream(self, req: QueryRequest):
        selection = self.provider_router.resolve_selection(
            req.provider,
            req.model or os.environ.get("OLLAMA_QUERY_MODEL"),
            has_api_key_override=bool((req.provider_api_key or "").strip()),
        )
        index, db, qp, session_pair, effective_scope = self._load_components(req)
        retrieve_k = max(req.top_k, 20) if req.use_rerank else req.top_k
        if effective_scope == "session" and session_pair is not None:
            s_index, s_db = session_pair
            fused = self._retrieve(
                req.query_text,
                s_index,
                s_db,
                qp,
                top_k=retrieve_k,
                collection_name=req.session_collection_name or COLLECTION_NAME,
            )
        elif effective_scope == "both" and session_pair is not None:
            global_results = self._retrieve(req.query_text, index, db, qp, top_k=retrieve_k)
            s_index, s_db = session_pair
            session_results = self._retrieve(
                req.query_text,
                s_index,
                s_db,
                qp,
                top_k=retrieve_k,
                collection_name=req.session_collection_name or COLLECTION_NAME,
            )
            fused = self._dedup_results(global_results + session_results, retrieve_k)
        else:
            fused = self._retrieve(req.query_text, index, db, qp, top_k=retrieve_k)
        if req.use_rerank:
            try:
                reranker = CrossEncoderReranker(
                    model_name=req.reranker_model or self.cfg.reranker.model,
                    batch_size=self.cfg.reranker.batch_size,
                    score_threshold=self.cfg.reranker.score_threshold,
                )
                ranked = reranker.rerank(req.query_text, fused, top_k=req.top_k)
                docs_for_gen: List[RetrievalResult] | List[RankedResult] = ranked
            except Exception as exc:
                logger.warning("Reranker unavailable in stream; falling back to retrieval order: %s", exc)
                docs_for_gen = fused[: req.top_k]
        else:
            docs_for_gen = fused[: req.top_k]

        query_type = PromptManager.intent_to_query_type(qp.process_query(req.query_text).intent)
        generator = RAGGenerator(
            model_name=selection.model,
            provider=selection.provider,
            prompt_manager=self.prompt_manager,
            context_optimizer=self.context_optimizer,
            provider_router=self.provider_router,
        )
        yield from generator.generate_stream(
            req.query_text,
            docs_for_gen,
            query_type=query_type,
            provider=selection.provider,
            model=selection.model,
            provider_api_key=req.provider_api_key,
        )
