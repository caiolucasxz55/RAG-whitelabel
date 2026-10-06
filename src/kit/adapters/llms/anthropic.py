"""Adapter do Claude via SDK oficial da Anthropic (extra `kit-rag[anthropic]`).

- Com `json_schema`, usa structured outputs (`output_config.format`), que restringe a
  resposta ao schema; o core ainda valida com Pydantic.
- `fallbacks: default` liga o fallback do servidor: se o modelo recusar por política,
  a API refaz a mesma chamada num modelo alternativo dentro da mesma requisição.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import anthropic

from kit.errors import ConfigError, LLMError, LLMRefusalError
from kit.types import LLMResponse, Message

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        fallbacks: Literal["default"] | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort or None
        if fallbacks not in (None, "", "default"):
            raise ConfigError("llm.fallbacks aceita apenas 'default' (ou vazio para desligar)")
        self.fallbacks: Literal["default"] | None = fallbacks or None
        try:
            self._client = client or anthropic.Anthropic(
                api_key=api_key or None, timeout=timeout, max_retries=max_retries
            )
        except anthropic.AnthropicError as exc:
            raise ConfigError(
                f"credenciais da Anthropic ausentes ({exc}). Defina ANTHROPIC_API_KEY "
                "ou use llm.provider: extractive para rodar sem chave."
            ) from exc

    def _request(
        self, messages: Sequence[Message], system: str | None, json_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            request["system"] = system
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(json_schema),
            }
        if output_config:
            request["output_config"] = output_config
        return request

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        request = self._request(messages, system, json_schema)
        response: Any
        try:
            if self.fallbacks:
                response = self._client.beta.messages.create(
                    **request, betas=[FALLBACK_BETA], fallbacks=self.fallbacks
                )
            else:
                response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LLMError("chave da Anthropic inválida (ANTHROPIC_API_KEY)") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"requisição rejeitada pela Anthropic: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("limite de taxa da Anthropic atingido; tente novamente") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"erro da API Anthropic ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"falha de conexão com a Anthropic: {exc}") from exc
        except anthropic.AnthropicError as exc:  # ex.: credenciais ausentes
            raise LLMError(
                f"falha ao chamar a Anthropic: {exc}. Defina ANTHROPIC_API_KEY "
                "ou use llm.provider: extractive para rodar sem chave."
            ) from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusalError(f"o modelo recusou a requisição (categoria: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"resposta truncada em max_tokens={self.max_tokens}")

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
