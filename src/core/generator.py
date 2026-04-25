"""RAG answer generation via Ollama with optional streaming and light validation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Union

import ollama

from src.core.context_optimizer import ContextOptimizer, OptimizedContext
from src.core.prompt_manager import PromptManager
from src.core.reranker import RankedResult
from src.core.response_processor import ResponseProcessor
from src.core.retrieval_result import RetrievalResult


@dataclass
class GenerationResult:
    response_text: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    latency_ms: float = 0.0
    streamed: bool = False
    prompt: str = ""
    optimized_context: OptimizedContext | None = None


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)


class RAGGenerator:
    """Build prompt from optimized context and call Ollama chat."""

    def __init__(
        self,
        model_name: str,
        prompt_manager: PromptManager,
        context_optimizer: ContextOptimizer,
    ) -> None:
        self.model_name = model_name
        self.prompt_manager = prompt_manager
        self.context_optimizer = context_optimizer
        self.response_processor = ResponseProcessor()

    def _docs_for_citations(
        self,
        optimized: OptimizedContext,
    ) -> List[Dict[str, Any]]:
        return list(optimized.documents)

    def validate_response(self, response: str, context: OptimizedContext) -> ValidationResult:
        issues: List[str] = []
        text = (response or "").strip()
        if len(text) < 15:
            issues.append("response_too_short")
        corpus = " ".join(d.get("text", "") for d in context.documents).lower()
        words = {w for w in text.lower().split() if len(w) > 4}
        src_words = {w for w in corpus.split() if len(w) > 4}
        overlap = len(words & src_words) / max(len(words), 1) if words else 0.0
        if overlap < 0.05 and len(text) > 80:
            issues.append("low_lexical_overlap_with_context")
        confidence = min(1.0, 0.35 + overlap)
        is_valid = len(issues) == 0 or (len(issues) == 1 and "low_lexical_overlap_with_context" in issues)
        return ValidationResult(is_valid=is_valid, confidence=confidence, issues=issues)

    def generate(
        self,
        query: str,
        documents: Sequence[Union[RetrievalResult, RankedResult]],
        stream: bool = False,
        query_type: str = "factual",
    ) -> GenerationResult:
        optimized = self.context_optimizer.optimize_context(query, list(documents))
        prompt = self.prompt_manager.build_prompt(query, optimized, query_type=query_type)
        t0 = time.perf_counter()
        if stream:
            parts: List[str] = []
            for piece in self.generate_stream_from_prompt(prompt):
                parts.append(piece)
            response_text = "".join(parts)
        else:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = str(resp.get("message", {}).get("content") or "")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        response_text = self.response_processor.format_response(response_text)
        cites = self.response_processor.extract_citations(response_text, self._docs_for_citations(optimized))
        return GenerationResult(
            response_text=response_text,
            citations=cites,
            model_name=self.model_name,
            latency_ms=latency_ms,
            streamed=stream,
            prompt=prompt,
            optimized_context=optimized,
        )

    def generate_stream_from_prompt(self, prompt: str) -> Iterator[str]:
        stream = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:  # type: ignore[assignment]
            msg = chunk.get("message") or {}
            piece = msg.get("content") or ""
            if piece:
                yield piece

    def generate_stream(
        self,
        query: str,
        documents: Sequence[Union[RetrievalResult, RankedResult]],
        query_type: str = "factual",
    ) -> Iterator[str]:
        optimized = self.context_optimizer.optimize_context(query, list(documents))
        prompt = self.prompt_manager.build_prompt(query, optimized, query_type=query_type)
        yield from self.generate_stream_from_prompt(prompt)
