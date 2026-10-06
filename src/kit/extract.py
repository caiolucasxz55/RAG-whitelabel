"""Extração estruturada: documento + modelo Pydantic -> objeto validado + sinal de confiança.

Fluxo:
1. Gera o JSON Schema a partir do modelo e pede ao LLM um JSON nesse formato.
2. Valida com Pydantic (tipos + validators do próprio modelo).
3. Se falhar, reenvia a conversa com os erros de validação (até `max_retries` vezes).
4. Marca confiança baixa se a validação não passou ou se há campos vazios.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from kit.chunking import normalize_whitespace
from kit.config import Pricing
from kit.errors import DocumentTooLargeError
from kit.ingestion import LoaderSet
from kit.interfaces import LLM
from kit.types import ExtractionResult, Message, T, Usage
from kit.usage import usage_of

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You extract structured data from documents.

Rules:
- Return only a JSON object that matches the provided JSON Schema.
- Copy values exactly as they appear in the document, converting them to the schema's types \
(numbers without currency symbols or thousand separators, dates as YYYY-MM-DD).
- If a field is not present in the document, use null. Never invent values."""

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def build_prompt(text: str, schema: dict[str, Any], instructions: str | None = None) -> str:
    parts = [
        f"<document>\n{text}\n</document>",
        f"<schema>\n{json.dumps(schema, ensure_ascii=False)}\n</schema>",
    ]
    if instructions:
        parts.append(f"<instructions>\n{instructions}\n</instructions>")
    parts.append("Extract the data from the document as JSON following the schema.")
    return "\n\n".join(parts)


def strip_fences(raw: str) -> str:
    """Alguns provedores sem structured outputs embrulham o JSON em ```json ... ```."""
    text = raw.strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def format_errors(exc: ValidationError | json.JSONDecodeError) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    return "\n".join(
        f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def empty_fields(value: Any, prefix: str = "") -> Iterator[str]:
    """Caminhos (a.b[0].c) de campos None, string em branco ou coleção vazia."""
    if isinstance(value, BaseModel):
        value = {name: getattr(value, name) for name in type(value).model_fields}
    if isinstance(value, dict):
        if not value and prefix:
            yield prefix
        for key, item in value.items():
            yield from empty_fields(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        if not value:
            yield prefix
        for i, item in enumerate(value):
            yield from empty_fields(item, f"{prefix}[{i}]")
    elif value is None or (isinstance(value, str) and not value.strip()):
        yield prefix


class Extractor:
    def __init__(
        self,
        loaders: LoaderSet,
        llm: LLM,
        *,
        max_retries: int = 1,
        max_chars: int = 200_000,
        pricing: Pricing | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.loaders = loaders
        self.llm = llm
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.pricing = pricing
        self.system_prompt = system_prompt

    def read(self, path: Path) -> str:
        documents = self.loaders.load(path)
        return "\n\n".join(normalize_whitespace(d.text) for d in documents)

    def extract(
        self,
        path: str | Path,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"arquivo não encontrado: {path}")
        return self.extract_text(
            self.read(path), schema, instructions=instructions, max_retries=max_retries
        )

    def extract_text(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str | None = None,
        max_retries: int | None = None,
    ) -> ExtractionResult[T]:
        if not text.strip():
            return ExtractionResult(None, "low", ["documento sem texto extraível"], 0, "")
        if len(text) > self.max_chars:
            # Não truncamos em silêncio: um campo cortado viraria um dado errado.
            raise DocumentTooLargeError(
                f"documento com {len(text)} caracteres excede extraction.max_chars={self.max_chars}"
            )
        retries = self.max_retries if max_retries is None else max_retries
        json_schema = schema.model_json_schema()
        messages = [Message("user", build_prompt(text, json_schema, instructions))]
        usage = Usage()
        issues: list[str] = []
        raw = ""

        for attempt in range(1, retries + 2):
            response = self.llm.generate(
                messages, system=self.system_prompt, json_schema=json_schema
            )
            usage = usage + usage_of(response, self.pricing)
            raw = response.text
            try:
                data = schema.model_validate_json(strip_fences(raw))
            except (ValidationError, json.JSONDecodeError) as exc:
                errors = format_errors(exc)
                issues.append(f"tentativa {attempt}: validação falhou\n{errors}")
                logger.info(
                    "extração inválida", extra={"attempt": attempt, "schema": schema.__name__}
                )
                messages += [
                    Message("assistant", raw),
                    Message(
                        "user",
                        "Your JSON failed validation:\n"
                        f"{errors}\n"
                        "Fix these problems and return the full corrected JSON object only.",
                    ),
                ]
                continue

            missing = list(empty_fields(data))
            if missing:
                issues.append(f"campos vazios: {', '.join(missing)}")
            return ExtractionResult(
                data=data,
                confidence="low" if missing else "high",
                issues=issues,
                attempts=attempt,
                raw=raw,
                usage=usage,
            )

        return ExtractionResult(None, "low", issues, retries + 1, raw, usage)
