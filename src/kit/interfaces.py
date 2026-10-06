"""Os quatro contratos que o core conhece.

São `typing.Protocol`: qualquer classe com os métodos certos é um adapter válido,
sem herdar de nada do kit. Isso permite plugar código de terceiros sem acoplamento.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kit.types import Chunk, Document, LLMResponse, Message, ScoredChunk


@runtime_checkable
class Loader(Protocol):
    """Lê um arquivo e devolve um ou mais Documents (ex.: uma página de PDF por Document)."""

    extensions: frozenset[str]

    def load(self, path: Path) -> list[Document]: ...


@runtime_checkable
class Embedder(Protocol):
    """Transforma texto em vetores. Separa query de passagem porque vários modelos
    (e5, bge) usam prefixos diferentes para cada lado."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def delete_source(self, source: str) -> int: ...

    def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...


@runtime_checkable
class LLM(Protocol):
    """Gera texto a partir de mensagens.

    `json_schema`, quando presente, pede uma resposta JSON nesse formato. O adapter pode
    usar o recurso nativo do provedor (structured outputs) ou só instruir o modelo;
    a validação final é sempre feita pelo core.
    """

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
