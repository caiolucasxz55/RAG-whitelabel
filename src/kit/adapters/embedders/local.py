"""Embeddings locais com sentence-transformers + torch (extra `kit-rag[local]`).

Mais flexível (qualquer modelo do Hub), porém pesado: o torch sozinho ocupa ~800 MB.
Para imagens enxutas, prefira o adapter `onnx`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kit.adapters.embedders.prefixes import default_prefixes


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
        default_query, default_passage = default_prefixes(model)
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
