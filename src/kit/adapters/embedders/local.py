"""Embeddings locais com sentence-transformers (extra `kit-rag[local]`). Sem chave de API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Modelos que foram treinados com prefixos distintos para consulta e passagem.
_KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
}


def _default_prefixes(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, prefixes in _KNOWN_PREFIXES.items():
        if key in name:
            return prefixes
    return "", ""


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-small",
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # import pesado, só aqui

        self.model_name = model
        self._model: Any = SentenceTransformer(model, device=device)
        default_query, default_passage = _default_prefixes(model)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.passage_prefix = default_passage if passage_prefix is None else passage_prefix
        self.batch_size = batch_size
        self.normalize = normalize
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self._dimension = int(get_dim())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
