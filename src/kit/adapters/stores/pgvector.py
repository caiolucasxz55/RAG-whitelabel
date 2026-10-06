"""Vector store em Postgres + pgvector (extra `kit-rag[pgvector]`).

Cria extensão, tabela e índices na primeira conexão. Distância de cosseno com índice HNSW.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from kit.errors import StoreError
from kit.types import Chunk, ScoredChunk

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dimension: int,
        *,
        table: str = "kit_chunks",
        pool_size: int = 5,
        connect_timeout: float = 10.0,
    ) -> None:
        if not _IDENTIFIER.match(table):
            raise StoreError(f"nome de tabela inválido: {table!r}")
        self.dimension = dimension
        self.table = table
        self._t = sql.Identifier(table)
        try:
            self._setup_schema(dsn, connect_timeout)
            self._pool = ConnectionPool(
                dsn,
                min_size=1,
                max_size=pool_size,
                configure=register_vector,
                kwargs={"autocommit": True},
                open=True,
            )
            self._pool.wait(timeout=connect_timeout)
        except psycopg.Error as exc:
            raise StoreError(f"falha ao conectar no Postgres: {exc}") from exc

    def _setup_schema(self, dsn: str, timeout: float) -> None:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=int(timeout)) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t} (
                        id text PRIMARY KEY,
                        source text NOT NULL,
                        chunk_index integer NOT NULL,
                        content text NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dim}) NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                ).format(t=self._t, dim=sql.Literal(self.dimension))
            )
            existing = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (self.table,),
            ).fetchone()
            if existing and existing[0] != self.dimension:
                raise StoreError(
                    f"tabela '{self.table}' tem embeddings de dimensão {existing[0]}, mas o "
                    f"embedder gera {self.dimension}. Use outra tabela (store.table) ou reingira."
                )
            conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} (source)").format(
                    i=sql.Identifier(f"{self.table}_source_idx"), t=self._t
                )
            )
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)"
                ).format(i=sql.Identifier(f"{self.table}_embedding_idx"), t=self._t)
            )

    def _vector(self, values: Sequence[float]) -> Any:
        if len(values) != self.dimension:
            raise StoreError(f"embedding com dimensão {len(values)}, esperado {self.dimension}")
        return np.asarray(values, dtype=np.float32)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise StoreError("chunks e embeddings com tamanhos diferentes")
        if not chunks:
            return
        rows = [
            (c.id, c.source, c.index, c.text, Jsonb(c.metadata), self._vector(e))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        query = sql.SQL(
            """
            INSERT INTO {t} (id, source, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                source = EXCLUDED.source, chunk_index = EXCLUDED.chunk_index,
                content = EXCLUDED.content, metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """
        ).format(t=self._t)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(query, rows)

    def delete_source(self, source: str) -> int:
        with self._pool.connection() as conn:
            cur = conn.execute(
                sql.SQL("DELETE FROM {t} WHERE source = %s").format(t=self._t), (source,)
            )
            return cur.rowcount

    def search(self, embedding: Sequence[float], top_k: int) -> list[ScoredChunk]:
        query = sql.SQL(
            """
            SELECT id, source, chunk_index, content, metadata, 1 - (embedding <=> %(q)s) AS score
            FROM {t} ORDER BY embedding <=> %(q)s LIMIT %(k)s
            """
        ).format(t=self._t)
        with self._pool.connection() as conn:
            rows = conn.execute(query, {"q": self._vector(embedding), "k": top_k}).fetchall()
        return [
            ScoredChunk(
                Chunk(id=r[0], source=r[1], index=r[2], text=r[3], metadata=dict(r[4])),
                float(r[5]),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(sql.SQL("SELECT count(*) FROM {t}").format(t=self._t)).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._pool.close()
