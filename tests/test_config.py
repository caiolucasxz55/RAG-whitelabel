from __future__ import annotations

from pathlib import Path

import pytest

from kit.config import interpolate_env, load_config, parse_config
from kit.errors import ConfigError

MINIMAL = {
    "embedder": {"provider": "hashing"},
    "store": {"provider": "memory"},
    "llm": {"provider": "fake"},
}


def test_interpolates_required_and_default_values() -> None:
    env = {"DB": "postgresql://x", "EMPTY": ""}
    raw = {"a": "${DB}", "b": ["${MISSING:-padrao}", "${EMPTY:-vazio}"], "c": 3}
    assert interpolate_env(raw, env) == {"a": "postgresql://x", "b": ["padrao", "vazio"], "c": 3}


def test_missing_required_variable_raises() -> None:
    with pytest.raises(ConfigError, match="NAO_EXISTE"):
        interpolate_env("${NAO_EXISTE}", {})


def test_loader_shorthand_and_provider_options() -> None:
    cfg = parse_config(
        {**MINIMAL, "loader": ["pdf", "text"], "embedder": {"provider": "local", "model": "m"}}
    )
    assert [c.provider for c in cfg.loader] == ["pdf", "text"]
    assert cfg.embedder.options == {"model": "m"}


def test_defaults() -> None:
    cfg = parse_config(MINIMAL)
    assert cfg.chunking.size == 800 and cfg.chunking.overlap == 100
    assert cfg.retrieval.top_k == 5
    assert cfg.extraction.max_retries == 1


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ConfigError, match="overlap"):
        parse_config({**MINIMAL, "chunking": {"size": 100, "overlap": 100}})


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config({**MINIMAL, "typo_key": 1})


def test_pricing_cost() -> None:
    cfg = parse_config(
        {
            **MINIMAL,
            "llm": {"provider": "x", "pricing": {"input_per_mtok": 4, "output_per_mtok": 20}},
        }
    )
    assert cfg.llm.pricing is not None
    assert cfg.llm.pricing.cost(1_000_000, 500_000) == pytest.approx(14.0)
    assert "pricing" not in cfg.llm.options


def test_load_config_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIT_TEST_MODEL", "modelo-x")
    path = tmp_path / "kit.yaml"
    path.write_text(
        "embedder: {provider: hashing}\n"
        "store: {provider: memory}\n"
        "llm: {provider: fake, model: '${KIT_TEST_MODEL}'}\n",
        encoding="utf-8",
    )
    cfg = load_config(path, dotenv=False)
    assert cfg.llm.options["model"] == "modelo-x"
    assert cfg.base_dir == tmp_path.resolve()


def test_missing_file_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"kit\.example\.yaml"):
        load_config(tmp_path / "nao_existe.yaml", dotenv=False)


def test_example_config_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DATABASE_URL", "LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "kit.example.yaml", dotenv=False)
    assert cfg.llm.provider == "auto"
    assert cfg.llm.options["model"] == "claude-opus-5-5"
    assert cfg.store.options["dsn"].startswith("postgresql://")
    assert set(cfg.schemas) == {"nota_fiscal", "contrato"}
