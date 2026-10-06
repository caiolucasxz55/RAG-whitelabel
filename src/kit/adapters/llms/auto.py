"""`llm.provider: auto`: usa a Anthropic se houver credencial no ambiente; senão cai para o
LLM extrativo offline. É o padrão do docker compose, para a demo rodar sem chave."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def auto_llm(**options: Any) -> Any:
    if any(os.environ.get(var) for var in _CREDENTIAL_VARS):
        from kit.adapters.llms.anthropic import AnthropicLLM

        return AnthropicLLM(**options)

    from kit.adapters.llms.extractive import ExtractiveLLM

    logger.warning(
        "ANTHROPIC_API_KEY ausente: usando o LLM 'extractive' (sem geração e sem extração "
        "estruturada). Defina a chave para usar o Claude."
    )
    return ExtractiveLLM()
