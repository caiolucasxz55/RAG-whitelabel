"""Busca vetorial + montagem do prompt + extração das citações.

O contexto vai numerado ([1], [2]...) e o modelo é instruído a citar esses números.
As citações da resposta viram `Source`s apontando documento, página e trecho.
"""

from __future__ import annotations

import html
import re

from kit.config import Pricing
from kit.interfaces import LLM, Embedder, VectorStore
from kit.types import Message, QueryResult, ScoredChunk, Source, Usage
from kit.usage import usage_of

SYSTEM_PROMPT = """\
You answer questions using only the documents provided inside <documents>.

Rules:
- Base every statement on the documents. Do not use outside knowledge.
- After each statement, cite the supporting document by its index in square brackets, \
e.g. [1] or [1][3].
- If the documents do not contain the answer, say so plainly instead of guessing.
- Answer in the same language as the question, concisely."""

NO_CONTEXT_ANSWER = "Não há documentos indexados relevantes para responder a esta pergunta."

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_context(question: str, retrieved: list[ScoredChunk]) -> str:
    parts = ["<documents>"]
    for ref, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        page = chunk.metadata.get("page")
        page_attr = f' page="{page}"' if page is not None else ""
        parts.append(
            f'<document index="{ref}" source="{html.escape(chunk.source)}"{page_attr}>\n'
            f"{chunk.text}\n</document>"
        )
    parts.append("</documents>")
    parts.append(f"<question>\n{question}\n</question>")
    return "\n".join(parts)


def parse_citations(answer: str, max_ref: int) -> list[int]:
    """Índices citados, na ordem da primeira aparição, ignorando números fora do contexto."""
    seen: list[int] = []
    for group in _CITATION_GROUP.findall(answer):
        for number in group.split(","):
            ref = int(number)
            if 1 <= ref <= max_ref and ref not in seen:
                seen.append(ref)
    return seen


def to_source(ref: int, item: ScoredChunk) -> Source:
    page = item.chunk.metadata.get("page")
    return Source(
        ref=ref,
        source=item.chunk.source,
        chunk_id=item.chunk.id,
        text=item.chunk.text,
        score=item.score,
        page=int(page) if page is not None else None,
    )


class RAGService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        *,
        top_k: int = 5,
        pricing: Pricing | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.pricing = pricing
        self.system_prompt = system_prompt

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        return self.store.search(self.embedder.embed_query(question), top_k or self.top_k)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("a pergunta não pode ser vazia")
        retrieved = self.retrieve(question, top_k)
        if not retrieved:
            # Sem contexto não há o que citar: evita uma chamada paga ao LLM.
            return QueryResult(answer=NO_CONTEXT_ANSWER, sources=[], retrieved=[], usage=Usage())
        response = self.llm.generate(
            [Message("user", format_context(question, retrieved))], system=self.system_prompt
        )
        refs = parse_citations(response.text, len(retrieved))
        return QueryResult(
            answer=response.text.strip(),
            sources=[to_source(ref, retrieved[ref - 1]) for ref in refs],
            retrieved=retrieved,
            usage=usage_of(response, self.pricing),
        )
