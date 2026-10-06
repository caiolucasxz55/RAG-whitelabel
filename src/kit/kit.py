"""Fachada pública. Liga config -> adapters -> serviços, e expõe ingest/query/extract."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from kit import registry
from kit.config import KitConfig, load_config
from kit.interfaces import LLM, Embedder, Loader, VectorStore

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

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> Kit:
        """Cria o Kit a partir de kit.yaml. `overrides` substitui adapters já instanciados
        (loaders=, embedder=, store=, llm=), útil em testes e composição manual."""
        config = path if isinstance(path, KitConfig) else load_config(path)
        return cls.from_settings(config, **overrides)

    @classmethod
    def from_settings(cls, config: KitConfig, **overrides: Any) -> Kit:
        return cls(config=config, **build_components(config, **overrides))
