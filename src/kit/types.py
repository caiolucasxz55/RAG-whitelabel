"""Tipos de domínio compartilhados pelo core e pelos adapters.

Nenhum tipo aqui conhece provedor, cliente ou domínio específico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Document:
    """Unidade produzida por um Loader (ex.: um arquivo .md ou uma página de PDF)."""

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Trecho de um Document, a unidade indexada no VectorStore."""

    id: str
    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Usage:
    """Consumo agregado de uma operação; o custo vem do pricing da config."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            cost,
        )


@dataclass(frozen=True)
class Source:
    """Citação devolvida ao usuário: qual documento e qual trecho sustentam a resposta."""

    ref: int
    source: str
    chunk_id: str
    text: str
    score: float
    page: int | None = None


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[Source]
    retrieved: list[ScoredChunk]
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class IngestReport:
    files: list[str]
    documents: int
    chunks: int
    skipped: list[str] = field(default_factory=list)


Confidence = Literal["high", "low"]


@dataclass(frozen=True)
class ExtractionResult(Generic[T]):
    """Resultado de kit.extract(): objeto validado (ou None) e o sinal de confiança."""

    data: T | None
    confidence: Confidence
    issues: list[str]
    attempts: int
    raw: str
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return self.data is not None
