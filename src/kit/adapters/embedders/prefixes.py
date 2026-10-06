"""Prefixos de consulta/passagem por família de modelo, compartilhados pelos embedders."""

from __future__ import annotations

# Modelos que foram treinados com prefixos distintos para consulta e passagem.
_KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
}


def default_prefixes(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, prefixes in _KNOWN_PREFIXES.items():
        if key in name:
            return prefixes
    return "", ""
