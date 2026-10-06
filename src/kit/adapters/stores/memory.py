"""Vector store em memória, busca exata por cosseno. Para testes, demos e uso como biblioteca
sem banco. Não persiste entre processos."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

from kit.errors import StoreError
from kit.types import Chunk, ScoredChunk


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)


class InMemoryVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self._lock = threading.Lock()

    def _check(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimension:
            raise StoreError(
                f"embedding com dimensão {len(vector)}, store configurado com {self.dimension}"
            )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise StoreError("chunks e embeddings com tamanhos diferentes")
        with self._lock:
            for chunk, vector in zip(chunks, embeddings, strict=True):
                self._check(vector)
                self._items[chunk.id] = (chunk, _normalize(vector))

    def delete_source(self, source: str) -> int:
        with self._lock:
            ids = [cid for cid, (chunk, _) in self._items.items() if chunk.source == source]
            for cid in ids:
                del self._items[cid]
            return len(ids)

    def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]:
        self._check(embedding)
        query = _normalize(embedding)
        with self._lock:
            scored = [
                ScoredChunk(chunk, sum(a * b for a, b in zip(query, vec, strict=True)))
                for chunk, vec in self._items.values()
            ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)
