from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from kit.adapters.embedders.hashing import HashingEmbedder
from kit.adapters.llms.anthropic import FALLBACK_BETA, AnthropicLLM
from kit.adapters.llms.extractive import ExtractiveLLM
from kit.errors import CapabilityError, LLMError, LLMRefusalError
from kit.interfaces import LLM, Embedder
from kit.rag import format_context
from kit.types import Chunk, Message, ScoredChunk

# ---------- hashing embedder ----------


def test_hashing_is_deterministic_normalized_and_accent_insensitive() -> None:
    emb = HashingEmbedder(dimension=64)
    a, b = emb.embed_query("Rescisão"), emb.embed_query("rescisao")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert isinstance(emb, Embedder)


# ---------- extractive LLM ----------


def _prompt(question: str, texts: list[str]) -> str:
    retrieved = [ScoredChunk(Chunk(str(i), t, f"d{i}.md", 0), 1.0) for i, t in enumerate(texts)]
    return format_context(question, retrieved)


def test_extractive_picks_best_sentence_with_citation() -> None:
    llm = ExtractiveLLM(max_sentences=1)
    prompt = _prompt(
        "Qual o prazo de aviso prévio?",
        ["O foro é São Paulo.", "Introdução. O aviso prévio é de 30 dias. Outra frase."],
    )
    out = llm.generate([Message("user", prompt)])
    assert out.text == "O aviso prévio é de 30 dias. [2]"
    assert isinstance(llm, LLM)


def test_extractive_without_match_and_without_json_support() -> None:
    llm = ExtractiveLLM()
    assert "Não encontrei" in llm.generate([Message("user", _prompt("xyz", ["abc"]))]).text
    with pytest.raises(CapabilityError):
        llm.generate([Message("user", "x")], json_schema={"type": "object"})


# ---------- Anthropic adapter (cliente falso, sem rede) ----------


class _Messages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _response(text: str = "ok", stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber"),
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )


def _client(response: Any) -> Any:
    std, beta = _Messages(response), _Messages(response)
    return SimpleNamespace(messages=std, beta=SimpleNamespace(messages=beta))


class _Nota(BaseModel):
    cnpj: str
    valor_total: float


def test_anthropic_builds_request_with_structured_output() -> None:
    client = _client(_response('{"cnpj": "1", "valor_total": 2}'))
    llm = AnthropicLLM(model="claude-opus-5-5", effort="medium", client=client)
    out = llm.generate(
        [Message("user", "extraia")], system="sys", json_schema=_Nota.model_json_schema()
    )
    kwargs = client.messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "extraia"}]
    assert kwargs["output_config"]["effort"] == "medium"
    schema = kwargs["output_config"]["format"]["schema"]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert schema["additionalProperties"] is False
    assert out.text == '{"cnpj": "1", "valor_total": 2}'
    assert (out.input_tokens, out.output_tokens) == (12, 3)


def test_anthropic_fallbacks_use_beta_endpoint() -> None:
    client = _client(_response())
    AnthropicLLM(fallbacks="default", client=client).generate([Message("user", "oi")])
    assert client.beta.messages.kwargs["betas"] == [FALLBACK_BETA]
    assert client.beta.messages.kwargs["fallbacks"] == "default"
    assert "output_config" not in client.beta.messages.kwargs


def test_anthropic_refusal_and_truncation_raise() -> None:
    with pytest.raises(LLMRefusalError, match="cyber"):
        AnthropicLLM(client=_client(_response(stop_reason="refusal"))).generate(
            [Message("user", "x")]
        )
    with pytest.raises(LLMError, match="max_tokens"):
        AnthropicLLM(client=_client(_response(stop_reason="max_tokens"))).generate(
            [Message("user", "x")]
        )


# ---------- embedder local (só roda com o extra [local] instalado) ----------


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers não instalado",
)
def test_local_embedder_prefixes_for_e5() -> None:
    from kit.adapters.embedders.local import _default_prefixes

    assert _default_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert _default_prefixes("outro/modelo") == ("", "")
