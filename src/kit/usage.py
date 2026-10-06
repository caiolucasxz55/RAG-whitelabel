from __future__ import annotations

from kit.config import Pricing
from kit.types import LLMResponse, Usage


def usage_of(response: LLMResponse, pricing: Pricing | None) -> Usage:
    cost = pricing.cost(response.input_tokens, response.output_tokens) if pricing else None
    return Usage(response.input_tokens, response.output_tokens, cost)
