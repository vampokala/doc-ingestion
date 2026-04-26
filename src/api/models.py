"""FastAPI request/response models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequestModel(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    use_llm: bool = True
    use_rerank: bool = True
    stream: bool = False
    include_citations: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    reranker_model: Optional[str] = None
    provider_api_key: Optional[str] = None


class CitationModel(BaseModel):
    raw_id: str
    chunk_id: str
    resolved: bool
    title: Optional[str] = None
    source: Optional[str] = None
    verification_score: float = 0.0
    verification: str = "unresolved"


class RetrievedChunkModel(BaseModel):
    id: str
    score: float = 0.0
    source: str = "hybrid"
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    preview: str = ""


class QueryResponseModel(BaseModel):
    query: str
    provider: str
    model: str
    answer: str = ""
    processing_time_ms: float = 0.0
    cached: bool = False
    validation_issues: List[str] = Field(default_factory=list)
    citations: List[CitationModel] = Field(default_factory=list)
    retrieved: List[RetrievedChunkModel] = Field(default_factory=list)


class HealthModel(BaseModel):
    status: str
    collection: str


class MetricsModel(BaseModel):
    cache_ttl_seconds: int
    available_providers: List[str]
