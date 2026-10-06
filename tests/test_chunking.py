from __future__ import annotations

from itertools import pairwise

import pytest

from kit.chunking import Chunker, chunk_id, normalize_whitespace
from kit.types import Document


def test_short_text_is_single_chunk() -> None:
    assert Chunker(100, 10).split("  Olá   mundo \n\n\n fim ") == ["Olá mundo\n\nfim"]


def test_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"palavra{i}" for i in range(400))
    chunks = Chunker(size=200, overlap=50).split(text)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # Sobreposição: o início de cada chunk aparece no fim do anterior.
    for prev, nxt in pairwise(chunks):
        assert nxt.split()[0] in prev


def test_never_splits_words_when_there_are_spaces() -> None:
    words = [f"w{i:04d}" for i in range(300)]
    for chunk in Chunker(size=120, overlap=30).split(" ".join(words)):
        assert all(token in words for token in chunk.split())


def test_prefers_paragraph_boundaries() -> None:
    para = "frase curta. " * 6
    chunks = Chunker(size=120, overlap=10).split(f"{para}\n\n{para}")
    assert chunks[0].endswith("frase curta.")


def test_covers_whole_text() -> None:
    text = " ".join(f"t{i}" for i in range(500))
    joined = " ".join(Chunker(150, 30).split(text))
    assert all(f"t{i}" in joined for i in range(500))


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        Chunker(100, 100)


def test_chunk_documents_numbers_per_source_and_keeps_metadata() -> None:
    docs = [
        Document("a " * 150, "x.pdf", {"page": 1}),
        Document("b " * 150, "x.pdf", {"page": 2}),
        Document("c", "y.md"),
    ]
    chunks = Chunker(100, 10).chunk_documents(docs)
    x = [c for c in chunks if c.source == "x.pdf"]
    assert [c.index for c in x] == list(range(len(x)))
    assert {c.metadata["page"] for c in x} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[-1].id == chunk_id("y.md", 0)


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("a\r\n\tb  c\n\n\n\nd") == "a\nb c\n\nd"
