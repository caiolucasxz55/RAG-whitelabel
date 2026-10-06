"""Fachada pública. Liga config -> adapters -> serviços, e expõe ingest/query/extract."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from kit import registry
from kit.chunking import Chunker
from kit.config import KitConfig, load_config
from kit.ingestion import IngestionService, LoaderSet
from kit.interfaces import LLM, Embedder, Loader, VectorStore
from kit.observability import timed
from kit.rag import RAGService
from kit.types import IngestReport, QueryResult, ScoredChunk

logger = logging.getLogger("kit")

_PARTS = ("loaders", "embedder", "store", "llm")


def build_components(config: KitConfig, **overrides: Any) -> dict[str, Any]:
    """Instancia só os adapters que não vieram prontos em `overrides`."""
    unknown = set(overrides) - set(_PARTS)
    if unknown:
        raise TypeError(f"overrides desconhecidos: {sorted(unknown)}")
    parts = dict(overrides)
    if "loaders" not in parts:
        parts["loaders"] = [registry.build("loader", c) for c in config.loader]
    if "embedder" not in parts:
        parts["embedder"] = registry.build("embedder", config.embedder)
    if "store" not in parts:
        # O store recebe a dimensão do embedder para criar o índice com o tamanho certo.
        parts["store"] = registry.build(
            "store", config.store, dimension=parts["embedder"].dimension
        )
    if "llm" not in parts:
        parts["llm"] = registry.build("llm", config.llm)
    return parts


class Kit:
    def __init__(
        self,
        *,
        config: KitConfig,
        loaders: Sequence[Loader],
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
    ) -> None:
        self.config = config
        self.loaders = list(loaders)
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self._loader_set = LoaderSet(self.loaders)
        self._ingestion = IngestionService(
            self._loader_set,
            Chunker(config.chunking.size, config.chunking.overlap),
            embedder,
            store,
        )
        self._rag = RAGService(
            embedder, store, llm, top_k=config.retrieval.top_k, pricing=config.llm.pricing
        )

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> Kit:
        """Cria o Kit a partir de kit.yaml (ou $KIT_CONFIG). `overrides` substitui adapters
        já instanciados (loaders=, embedder=, store=, llm=), útil em testes e composição."""
        return cls.from_settings(load_config(path), **overrides)

    @classmethod
    def from_settings(cls, config: KitConfig, **overrides: Any) -> Kit:
        return cls(config=config, **build_components(config, **overrides))

    def ingest(self, paths: str | Path | Iterable[str | Path]) -> IngestReport:
        """Indexa arquivos ou pastas (recursivo). Reingerir um arquivo substitui seus chunks."""
        with timed(logger, "ingest") as log:
            report = self._ingestion.ingest(paths)
            log.update(files=len(report.files), chunks=report.chunks, skipped=len(report.skipped))
        return report

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Só a busca vetorial, sem LLM (usada pelo eval de hit rate)."""
        return self._rag.retrieve(question, top_k)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """Responde com base nos documentos indexados, citando documento e trecho."""
        with timed(logger, "query", top_k=top_k or self.config.retrieval.top_k) as log:
            result = self._rag.query(question, top_k)
            log.update(
                sources=len(result.sources),
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
        return result
