"""Chunking por tamanho com sobreposição.

O tamanho é medido em caracteres (independe de tokenizer, então funciona igual para
qualquer embedder). O corte procura a melhor fronteira na segunda metade da janela:
parágrafo > linha > fim de frase > espaço, para não partir palavras ao meio.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from kit.types import Chunk, Document

_SPACES = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{index:05d}"


class Chunker:
    def __init__(self, size: int = 800, overlap: int = 100) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("exige size > 0 e 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def _cut(self, text: str, start: int) -> int:
        end = start + self.size
        if end >= len(text):
            return len(text)
        window = text[start:end]
        floor = self.size // 2
        for sep in _BREAKS:
            pos = window.rfind(sep, floor)
            if pos != -1:
                return start + pos + len(sep)
        return end

    def _next_start(self, text: str, start: int, end: int) -> int:
        nxt = max(end - self.overlap, start + 1)
        # Começa a sobreposição numa fronteira de palavra, se houver uma por perto.
        if 0 < nxt < end and not text[nxt - 1].isspace():
            space = text.find(" ", nxt, end)
            if space != -1:
                nxt = space + 1
        return nxt

    def split(self, text: str) -> list[str]:
        text = normalize_whitespace(text)
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = self._cut(text, start)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = self._next_start(text, start, end)
        return pieces

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """Numera os chunks sequencialmente por `source` (um PDF gera vários Documents)."""
        chunks: list[Chunk] = []
        counters: dict[str, int] = {}
        for doc in documents:
            for piece in self.split(doc.text):
                index = counters.get(doc.source, 0)
                counters[doc.source] = index + 1
                chunks.append(
                    Chunk(
                        id=chunk_id(doc.source, index),
                        text=piece,
                        source=doc.source,
                        index=index,
                        metadata=dict(doc.metadata),
                    )
                )
        return chunks
