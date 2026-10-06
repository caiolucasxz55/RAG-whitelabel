"""Embedder léxico por feature hashing: sem download, sem GPU, determinístico.

Não captura semântica como um modelo neural, mas serve para testes, CI e como
linha de base nos evals.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class HashingEmbedder:
    def __init__(self, dimension: int = 512, ngrams: int = 2) -> None:
        if dimension <= 0:
            raise ValueError("dimension precisa ser positivo")
        self._dimension = dimension
        self._ngrams = max(1, ngrams)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(_normalize(text))
        features = list(tokens)
        for n in range(2, self._ngrams + 1):
            features += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self._dimension] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
