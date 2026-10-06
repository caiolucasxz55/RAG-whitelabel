"""Testes de integração do PgVectorStore. Rodam só com KIT_TEST_DATABASE_URL definida
(o CI sobe um Postgres com pgvector; localmente: docker compose up db)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from kit.errors import StoreError
from kit.types import Chunk

DSN = os.environ.get("KIT_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="KIT_TEST_DATABASE_URL não definida"),
]


@pytest.fixture
def store() -> Iterator[object]:
    from kit.adapters.stores.pgvector import PgVectorStore

    table = f"kit_test_{uuid.uuid4().hex[:8]}"
    s = PgVectorStore(DSN or "", dimension=3, table=table)
    yield s
    with s._pool.connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    s.close()


def _chunk(i: int, source: str = "a.md") -> Chunk:
    return Chunk(id=f"{source}-{i}", text=f"t{i}", source=source, index=i, metadata={"page": i})


def test_add_search_count_delete(store) -> None:  # type: ignore[no-untyped-def]
    store.add([_chunk(0), _chunk(1), _chunk(0, "b.md")], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert store.count() == 3
    hits = store.search([0.9, 0.1, 0], top_k=2)
    assert hits[0].chunk.id == "a.md-0" and hits[0].chunk.metadata == {"page": 0}
    assert hits[0].score > hits[1].score
    assert store.delete_source("a.md") == 2
    assert store.count() == 1


def test_upsert_is_idempotent(store) -> None:  # type: ignore[no-untyped-def]
    store.add([_chunk(0)], [[1, 0, 0]])
    store.add([_chunk(0)], [[0, 1, 0]])
    assert store.count() == 1


def test_dimension_mismatch(store) -> None:  # type: ignore[no-untyped-def]
    from kit.adapters.stores.pgvector import PgVectorStore

    with pytest.raises(StoreError, match="dimensão"):
        PgVectorStore(DSN or "", dimension=5, table=store.table)
    with pytest.raises(StoreError):
        store.search([1, 0], top_k=1)


def test_rejects_unsafe_table_name() -> None:
    from kit.adapters.stores.pgvector import PgVectorStore

    with pytest.raises(StoreError, match="tabela"):
        PgVectorStore(DSN or "", dimension=3, table="x; drop table y")
