from __future__ import annotations

import pytest

from kit import registry
from kit.adapters.embedders.hashing import HashingEmbedder
from kit.adapters.stores.memory import InMemoryVectorStore
from kit.config import ProviderConfig
from kit.errors import ConfigError
from kit.interfaces import Embedder, VectorStore


def test_resolves_builtin_lazily() -> None:
    assert registry.resolve("embedder", "hashing") is HashingEmbedder


def test_build_passes_yaml_options_and_injected_dependencies() -> None:
    store = registry.build("store", ProviderConfig(provider="memory"), dimension=8)
    assert isinstance(store, InMemoryVectorStore) and store.dimension == 8


def test_resolves_dotted_path_for_third_party_adapters() -> None:
    target = "kit.adapters.embedders.hashing:HashingEmbedder"
    assert registry.resolve("embedder", target) is HashingEmbedder


def test_runtime_registration_wins() -> None:
    sentinel = object()
    registry.register("store", "custom", lambda **_: sentinel)
    try:
        assert registry.build("store", ProviderConfig(provider="custom")) is sentinel
        assert "custom" in registry.available("store")
    finally:
        registry.unregister("store", "custom")


def test_unknown_provider_lists_available() -> None:
    with pytest.raises(ConfigError, match="memory"):
        registry.resolve("store", "nao-existe")


def test_bad_options_become_config_error() -> None:
    with pytest.raises(ConfigError, match="opções inválidas"):
        registry.build("embedder", ProviderConfig(provider="hashing", foo=1))


def test_builtins_satisfy_protocols() -> None:
    assert isinstance(HashingEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(4), VectorStore)
