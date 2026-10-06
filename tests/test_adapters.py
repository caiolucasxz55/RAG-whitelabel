from __future__ import annotations

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


def test_prefixes_for_e5() -> None:
    from kit.adapters.embedders.prefixes import default_prefixes as _default_prefixes

    assert _default_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert _default_prefixes("outro/modelo") == ("", "")


def test_auto_llm_picks_by_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from kit.adapters.llms.auto import auto_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert isinstance(auto_llm(model="claude-opus-5-5", effort="medium"), ExtractiveLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(auto_llm(model="claude-opus-5-5", effort="medium"), AnthropicLLM)


# ---------- embedder ONNX (sessão e tokenizer falsos, sem download) ----------


class _Encoding:
    def __init__(self, n_tokens: int, width: int) -> None:
        self.ids = [7] * n_tokens + [0] * (width - n_tokens)
        self.attention_mask = [1] * n_tokens + [0] * (width - n_tokens)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def enable_truncation(self, max_length: int) -> None: ...

    def enable_padding(self) -> None: ...

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        self.batches.append(list(texts))
        width = max(len(t.split()) for t in texts)
        return [_Encoding(len(t.split()), width) for t in texts]


class _FakeSession:
    """Hidden state = [n_tokens_do_texto, 1, 0] em cada posição real; padding vale 100."""

    def __init__(self, with_token_types: bool) -> None:
        names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_token_types else [])
        self._inputs = [SimpleNamespace(name=n) for n in names]
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[Any]:
        return self._inputs

    def run(self, _: Any, feeds: dict[str, Any]) -> list[Any]:
        import numpy as np

        self.feeds.append(feeds)
        mask = feeds["attention_mask"]
        n = mask.sum(axis=1, keepdims=True).astype(float)
        hidden = np.stack(
            [np.broadcast_to(n, mask.shape), np.ones(mask.shape), np.zeros(mask.shape)], -1
        )
        hidden[mask == 0] = 100.0  # se o padding entrar na média, o teste falha
        return [hidden]


def test_onnx_embedder_pools_ignoring_padding_and_uses_prefixes() -> None:
    from kit.adapters.embedders.onnx import OnnxEmbedder

    tok, session = _FakeTokenizer(), _FakeSession(with_token_types=True)
    emb = OnnxEmbedder(session=session, tokenizer=tok, batch_size=2, normalize=False)
    assert emb.dimension == 3
    vectors = emb.embed_documents(["a b c", "a", "a b"])
    # prefixo e5 ("passage: ") soma um token a cada texto
    assert [round(v[0]) for v in vectors] == [4, 2, 3]
    assert all(v[1] == 1.0 for v in vectors)
    assert tok.batches[1] == ["passage: a b c", "passage: a"]  # batch_size=2
    assert "token_type_ids" in session.feeds[-1]
    assert emb.embed_query("x y")[0] == 3.0  # "query: x y"
    assert isinstance(emb, Embedder)


def test_onnx_embedder_normalizes_and_skips_token_types_when_absent() -> None:
    import math

    from kit.adapters.embedders.onnx import OnnxEmbedder

    session = _FakeSession(with_token_types=False)
    emb = OnnxEmbedder(session=session, tokenizer=_FakeTokenizer(), query_prefix="")
    vector = emb.embed_query("um dois")
    assert math.isclose(sum(x * x for x in vector), 1.0)
    assert "token_type_ids" not in session.feeds[-1]
