"""DTOs da API. Separados dos tipos de domínio para que o contrato HTTP evolua sem
acoplar ao core."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kit.types import ExtractionResult, IngestReport, QueryResult, Source, Usage


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    @classmethod
    def of(cls, usage: Usage) -> UsageOut:
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )


class SourceOut(BaseModel):
    ref: int
    source: str
    page: int | None
    chunk_id: str
    score: float
    text: str

    @classmethod
    def of(cls, s: Source) -> SourceOut:
        return cls(
            ref=s.ref, source=s.source, page=s.page, chunk_id=s.chunk_id, score=s.score, text=s.text
        )


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryOut(BaseModel):
    answer: str
    sources: list[SourceOut]
    usage: UsageOut

    @classmethod
    def of(cls, r: QueryResult) -> QueryOut:
        return cls(
            answer=r.answer,
            sources=[SourceOut.of(s) for s in r.sources],
            usage=UsageOut.of(r.usage),
        )


class IngestOut(BaseModel):
    files: list[str]
    documents: int
    chunks: int
    skipped: list[str]

    @classmethod
    def of(cls, r: IngestReport) -> IngestOut:
        return cls(files=r.files, documents=r.documents, chunks=r.chunks, skipped=r.skipped)


class ExtractOut(BaseModel):
    schema_name: str
    data: dict[str, Any] | None
    confidence: Literal["high", "low"]
    issues: list[str]
    attempts: int
    usage: UsageOut

    @classmethod
    def of(cls, name: str, r: ExtractionResult[Any]) -> ExtractOut:
        return cls(
            schema_name=name,
            data=r.data.model_dump(mode="json") if r.data is not None else None,
            confidence=r.confidence,
            issues=r.issues,
            attempts=r.attempts,
            usage=UsageOut.of(r.usage),
        )


class HealthOut(BaseModel):
    status: Literal["ok"]
    chunks: int
    llm: str
    embedder: str
    store: str


class ErrorOut(BaseModel):
    error: str
    detail: str
