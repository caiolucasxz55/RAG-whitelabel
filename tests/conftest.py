from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kit import registry
from kit.config import KitConfig, parse_config
from tests.fakes import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture(autouse=True)
def _register_fake_llm() -> Iterator[None]:
    registry.register("llm", "fake", lambda **_: FakeLLM())
    yield
    registry.unregister("llm", "fake")


@pytest.fixture
def config(tmp_path: Path) -> KitConfig:
    """Config 100% offline: hashing + memória + LLM falso."""
    return parse_config(
        {
            "loader": ["pdf", "text"],
            "embedder": {"provider": "hashing", "dimension": 256},
            "store": {"provider": "memory"},
            "llm": {
                "provider": "fake",
                "pricing": {"input_per_mtok": 4.0, "output_per_mtok": 20.0},
            },
            "chunking": {"size": 200, "overlap": 40},
            "retrieval": {"top_k": 3},
        },
        base_dir=tmp_path,
    )
