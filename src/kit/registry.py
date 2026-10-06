"""Resolve o nome do provedor no YAML para a classe do adapter.

Três formas de apontar um adapter, em ordem de resolução:
1. nome registrado em tempo de execução com `register()` (útil em testes e plugins);
2. nome embutido (tabela `_BUILTINS`), importado de forma preguiçosa para que
   dependências pesadas (torch, psycopg...) só carreguem quando usadas;
3. caminho `pacote.modulo:Classe`, para adapters de terceiros sem fork do kit.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Literal

from kit.config import ProviderConfig
from kit.errors import ConfigError

Kind = Literal["loader", "embedder", "store", "llm"]
Factory = Callable[..., Any]

_BUILTINS: dict[Kind, dict[str, str]] = {
    "loader": {
        "pdf": "kit.adapters.loaders.pdf:PdfLoader",
        "text": "kit.adapters.loaders.text:TextLoader",
    },
    "embedder": {
        "local": "kit.adapters.embedders.local:SentenceTransformerEmbedder",
        "hashing": "kit.adapters.embedders.hashing:HashingEmbedder",
    },
    "store": {
        "pgvector": "kit.adapters.stores.pgvector:PgVectorStore",
        "memory": "kit.adapters.stores.memory:InMemoryVectorStore",
    },
    "llm": {
        "anthropic": "kit.adapters.llms.anthropic:AnthropicLLM",
        "extractive": "kit.adapters.llms.extractive:ExtractiveLLM",
    },
}

_registered: dict[Kind, dict[str, Factory]] = {kind: {} for kind in _BUILTINS}


def register(kind: Kind, name: str, factory: Factory) -> None:
    """Registra um adapter (classe ou função fábrica) sob um nome usável no kit.yaml."""
    _registered[kind][name] = factory


def unregister(kind: Kind, name: str) -> None:
    _registered[kind].pop(name, None)


def available(kind: Kind) -> list[str]:
    return sorted({*_BUILTINS[kind], *_registered[kind]})


def _import_target(target: str) -> Factory:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ConfigError(f"caminho de adapter inválido '{target}', use 'pacote.modulo:Classe'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"não foi possível importar '{module_name}' ({exc}). "
            "Instale o extra correspondente, ex.: pip install 'kit-rag[all]'."
        ) from exc
    try:
        factory: Factory = getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(f"'{module_name}' não define '{attr}'") from exc
    return factory


def resolve(kind: Kind, name: str) -> Factory:
    if name in _registered[kind]:
        return _registered[kind][name]
    if name in _BUILTINS[kind]:
        return _import_target(_BUILTINS[kind][name])
    if ":" in name:
        return _import_target(name)
    raise ConfigError(
        f"{kind} desconhecido: '{name}'. Disponíveis: {', '.join(available(kind))} "
        "(ou use 'pacote.modulo:Classe')."
    )


def build(kind: Kind, config: ProviderConfig, **extra: Any) -> Any:
    """Instancia o adapter com as opções do YAML (+ dependências injetadas em `extra`)."""
    factory = resolve(kind, config.provider)
    try:
        return factory(**config.options, **extra)
    except TypeError as exc:
        raise ConfigError(f"opções inválidas para {kind} '{config.provider}': {exc}") from exc
