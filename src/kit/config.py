"""Carrega kit.yaml + .env e valida com Pydantic.

Interpolação de variáveis de ambiente em qualquer valor string:
  ${VAR}            obrigatória (erro se ausente)
  ${VAR:-padrao}    usa `padrao` se VAR estiver ausente ou vazia
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kit.errors import ConfigError

DEFAULT_CONFIG_ENV = "KIT_CONFIG"
DEFAULT_CONFIG_FILE = "kit.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ProviderConfig(BaseModel):
    """`provider` escolhe o adapter; os demais campos viram kwargs do construtor dele."""

    model_config = ConfigDict(extra="allow")

    provider: str

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class Pricing(BaseModel):
    """Preço do LLM em US$ por milhão de tokens. Opcional: sem ele o custo fica None."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class LLMConfig(ProviderConfig):
    pricing: Pricing | None = None


class ChunkingConfig(BaseModel):
    size: int = Field(default=800, gt=0, description="Tamanho máximo do trecho, em caracteres.")
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap precisa ser menor que chunking.size")
        return self


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, gt=0)


class ExtractionConfig(BaseModel):
    max_retries: int = Field(default=1, ge=0)
    max_chars: int = Field(default=200_000, gt=0)


class KitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loader: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig(provider="pdf"), ProviderConfig(provider="text")]
    )
    embedder: ProviderConfig
    store: ProviderConfig
    llm: LLMConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    schemas: dict[str, str] = Field(
        default_factory=dict,
        description="Schemas de extração expostos por nome na API/CLI: nome -> 'modulo:Classe'.",
    )
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @field_validator("loader", mode="before")
    @classmethod
    def _loader_shorthand(cls, value: Any) -> Any:
        # Permite `loader: [pdf, text]` além da forma completa `- provider: pdf`.
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [{"provider": v} if isinstance(v, str) else v for v in value]
        return value


def interpolate_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Substitui ${VAR} e ${VAR:-padrao} recursivamente em strings de dicts/listas."""
    env = dict(os.environ) if env is None else env

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = env.get(name)
        if current:
            return current
        if default is not None:
            return default
        raise ConfigError(f"variável de ambiente obrigatória não definida: {name}")

    if isinstance(value, str):
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    return value


def parse_config(raw: dict[str, Any], *, base_dir: Path | None = None) -> KitConfig:
    try:
        data = interpolate_env(raw)
        if base_dir is not None:
            data["base_dir"] = base_dir
        return KitConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"configuração inválida:\n{exc}") from exc


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    candidate = Path(path or os.environ.get(DEFAULT_CONFIG_ENV) or DEFAULT_CONFIG_FILE)
    if not candidate.is_file():
        raise ConfigError(
            f"arquivo de configuração não encontrado: {candidate}. "
            "Copie kit.example.yaml para kit.yaml ou defina KIT_CONFIG."
        )
    return candidate


def load_config(path: str | os.PathLike[str] | None = None, *, dotenv: bool = True) -> KitConfig:
    """Lê o YAML (path, $KIT_CONFIG ou ./kit.yaml), carregando antes o .env do diretório atual."""
    if dotenv:
        load_dotenv(override=False)
    config_path = resolve_config_path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido em {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} deve conter um mapeamento no nível raiz")
    return parse_config(raw, base_dir=config_path.resolve().parent)
