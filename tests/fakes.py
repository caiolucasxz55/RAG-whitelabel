"""Adapters falsos usados nos testes. Não herdam de nada: só cumprem os Protocols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kit.types import LLMResponse, Message

Responder = Callable[[Sequence[Message], str | None, dict[str, Any] | None], str]


@dataclass
class Call:
    messages: list[Message]
    system: str | None
    json_schema: dict[str, Any] | None


@dataclass
class FakeLLM:
    """Devolve respostas roteirizadas (lista) ou calculadas (função) e grava as chamadas."""

    responses: list[str] | Responder = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    model: str = "fake-llm"

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(Call(list(messages), system, json_schema))
        if callable(self.responses):
            text = self.responses(messages, system, json_schema)
        else:
            if not self.responses:
                raise AssertionError("FakeLLM sem respostas roteirizadas")
            text = self.responses.pop(0)
        return LLMResponse(text=text, model=self.model, input_tokens=100, output_tokens=10)
