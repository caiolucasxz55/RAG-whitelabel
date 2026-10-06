"""Resolve schemas de extração declarados no kit.yaml (`schemas: {nome: 'modulo:Classe'}`).

É o que permite à API e à CLI extrair com modelos do projeto cliente sem alterar o kit.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from pydantic import BaseModel

from kit.errors import ConfigError


def import_schema(target: str, base_dir: Path | None = None) -> type[BaseModel]:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ConfigError(f"schema inválido '{target}', use 'modulo:Classe'")
    # O diretório do kit.yaml entra no path para importar módulos do projeto cliente.
    if base_dir is not None and str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"não foi possível importar o schema '{target}': {exc}") from exc
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ConfigError(f"'{target}' não é um modelo Pydantic (BaseModel)")
    return obj


class SchemaRegistry:
    def __init__(self, targets: dict[str, str], base_dir: Path | None = None) -> None:
        self._targets = dict(targets)
        self._base_dir = base_dir
        self._cache: dict[str, type[BaseModel]] = {}

    def names(self) -> list[str]:
        return sorted(self._targets)

    def get(self, name: str) -> type[BaseModel]:
        if name not in self._targets:
            available = ", ".join(self.names()) or "nenhum (defina `schemas` no kit.yaml)"
            raise KeyError(f"schema '{name}' não configurado. Disponíveis: {available}")
        if name not in self._cache:
            self._cache[name] = import_schema(self._targets[name], self._base_dir)
        return self._cache[name]
