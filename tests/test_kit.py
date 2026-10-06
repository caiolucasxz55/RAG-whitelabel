from __future__ import annotations

from pathlib import Path

import pytest

from kit import Kit
from kit.adapters.stores.memory import InMemoryVectorStore
from kit.config import KitConfig
from tests.fakes import FakeLLM


def test_from_config_builds_adapters_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "kit.yaml"
    path.write_text(
        "loader: [text]\n"
        "embedder: {provider: hashing, dimension: 64}\n"
        "store: {provider: memory}\n"
        "llm: {provider: fake}\n",
        encoding="utf-8",
    )
    kit = Kit.from_config(path)
    assert isinstance(kit.llm, FakeLLM)
    assert isinstance(kit.store, InMemoryVectorStore)
    assert kit.store.dimension == kit.embedder.dimension == 64
    assert len(kit.loaders) == 1


def test_overrides_skip_building_that_adapter(config: KitConfig, fake_llm: FakeLLM) -> None:
    kit = Kit.from_settings(config, llm=fake_llm)
    assert kit.llm is fake_llm


def test_unknown_override_is_rejected(config: KitConfig) -> None:
    with pytest.raises(TypeError):
        Kit.from_settings(config, banana=1)
