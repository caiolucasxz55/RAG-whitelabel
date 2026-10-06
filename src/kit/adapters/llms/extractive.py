"""LLM "extrativo" offline: responde copiando as frases do contexto mais parecidas com a
pergunta, já com citação. Não gera texto nem faz extração estruturada.

Existe para o kit rodar sem chave de API (docker compose up e testar na hora). Ele entende
o formato de prompt do RAG do kit (<documents>/<document index=...>/<question>).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from kit.errors import CapabilityError
from kit.types import LLMResponse, Message

_DOC = re.compile(r'<document index="(\d+)"[^>]*>\n?(.*?)\n?</document>', re.DOTALL)
_QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")
_WORD = re.compile(r"\w+")
# Palavras muito frequentes em pt/en que não ajudam a ranquear frases.
_STOPWORDS = frozenset(
    [
        "a",
        "o",
        "as",
        "os",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "para",
        "por",
        "com",
        "que",
        "qual",
        "quais",
        "quando",
        "como",
        "onde",
        "se",
        "ao",
        "aos",
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "are",
        "what",
        "which",
        "how",
        "when",
        "for",
        "on",
    ]
)


def _terms(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return {w for w in _WORD.findall(folded) if w not in _STOPWORDS and len(w) > 1}


class ExtractiveLLM:
    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences
        self.model = "extractive"

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if json_schema is not None:
            raise CapabilityError(
                "o LLM 'extractive' não faz extração estruturada; configure um LLM gerativo "
                "(ex.: llm.provider: anthropic)"
            )
        prompt = messages[-1].content if messages else ""
        question_match = _QUESTION.search(prompt)
        question = _terms(question_match.group(1)) if question_match else set()

        candidates: list[tuple[float, int, int, str]] = []
        for doc_index, body in _DOC.findall(prompt):
            for position, sentence in enumerate(_SENTENCE.split(body)):
                sentence = sentence.strip()
                if len(sentence) < 3:
                    continue
                overlap = len(question & _terms(sentence))
                if overlap:
                    # Empate: privilegia documentos melhor ranqueados e frases mais cedo.
                    candidates.append((overlap, -int(doc_index), -position, sentence))

        if not candidates:
            text = "Não encontrei a resposta nos documentos recuperados."
        else:
            best = sorted(candidates, reverse=True)[: self.max_sentences]
            text = " ".join(f"{sentence} [{-doc}]" for _, doc, _, sentence in best)
        return LLMResponse(text=text, model=self.model)
