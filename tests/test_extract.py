from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.schemas import NotaFiscal, normalize_cnpj
from pydantic import BaseModel

from kit import Kit
from kit.config import KitConfig
from kit.errors import DocumentTooLargeError
from kit.extract import empty_fields, strip_fences
from tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
LABELS = [
    json.loads(line)
    for line in (ROOT / "evals/datasets/extraction.jsonl").read_text(encoding="utf-8").splitlines()
]


class Simples(BaseModel):
    nome: str
    valor: float
    tags: list[str]


@pytest.fixture
def kit(config: KitConfig, fake_llm: FakeLLM) -> Kit:
    return Kit.from_settings(config, llm=fake_llm)


def test_valid_first_try_is_high_confidence(kit: Kit, fake_llm: FakeLLM) -> None:
    fake_llm.responses = ['{"nome": "ACME", "valor": 10.5, "tags": ["a"]}']
    result = kit.extract_text("ACME vendeu por 10,50", Simples)
    assert result.data == Simples(nome="ACME", valor=10.5, tags=["a"])
    assert (result.confidence, result.attempts, result.issues) == ("high", 1, [])
    call = fake_llm.calls[0]
    assert call.json_schema == Simples.model_json_schema()
    assert "<document>" in call.messages[0].content
    assert result.usage.input_tokens == 100


def test_retry_sends_validation_errors_back(kit: Kit, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [
        '{"nome": "ACME", "valor": "dez", "tags": []}',
        '{"nome": "ACME", "valor": 10, "tags": ["x"]}',
    ]
    result = kit.extract_text("doc", Simples)
    assert result.ok and result.attempts == 2
    assert result.confidence == "high"
    assert "tentativa 1" in result.issues[0]
    retry = fake_llm.calls[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "valor" in retry[2].content and "failed validation" in retry[2].content
    assert result.usage.input_tokens == 200  # soma as duas chamadas


def test_gives_up_after_retries_with_low_confidence(kit: Kit, fake_llm: FakeLLM) -> None:
    fake_llm.responses = ["não é json", '{"nome": 1}']
    result = kit.extract_text("doc", Simples)
    assert result.data is None and result.confidence == "low"
    assert result.attempts == 2 and len(result.issues) == 2
    assert result.raw == '{"nome": 1}'


def test_max_retries_zero_means_single_attempt(kit: Kit, fake_llm: FakeLLM) -> None:
    fake_llm.responses = ["{}"]
    result = kit.extract_text("doc", Simples, max_retries=0)
    assert result.attempts == 1 and not result.ok and len(fake_llm.calls) == 1


def test_empty_fields_lower_confidence(kit: Kit, fake_llm: FakeLLM) -> None:
    class ComOpcional(BaseModel):
        nome: str
        email: str | None

    fake_llm.responses = ['{"nome": "ACME", "email": null}']
    result = kit.extract_text("doc", ComOpcional)
    assert result.ok and result.confidence == "low"
    assert result.issues == ["campos vazios: email"]


def test_empty_document_does_not_call_llm(kit: Kit, fake_llm: FakeLLM) -> None:
    result = kit.extract_text("   ", Simples)
    assert result.confidence == "low" and fake_llm.calls == []


def test_too_large_document_is_rejected(kit: Kit) -> None:
    with pytest.raises(DocumentTooLargeError):
        kit.extract_text("x" * (kit.config.extraction.max_chars + 1), Simples)


def test_extracts_nota_fiscal_pdf_from_examples(kit: Kit, fake_llm: FakeLLM) -> None:
    """Critério da Etapa 3: o PDF de exemplo vira um NotaFiscal validado.
    O LLM falso devolve o rótulo, com CNPJ formatado como no documento."""
    label = LABELS[0]
    payload = dict(label["expected"])
    cnpj = payload["cnpj_emitente"]
    payload["cnpj_emitente"] = f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"
    fake_llm.responses = [json.dumps(payload)]

    result = kit.extract(ROOT / label["file"], NotaFiscal)
    assert result.confidence == "high"
    assert result.data is not None
    assert result.data.cnpj_emitente == cnpj  # normalizado pelo validator do schema
    prompt = fake_llm.calls[0].messages[0].content
    assert "DOCUMENTO FICTÍCIO" in prompt  # o texto do PDF chegou ao LLM


def test_schema_validator_triggers_retry_on_bad_cnpj(kit: Kit, fake_llm: FakeLLM) -> None:
    good = dict(LABELS[1]["expected"])
    bad = {**good, "cnpj_emitente": "11.111.111/1111-11"}
    fake_llm.responses = [json.dumps(bad), json.dumps(good)]
    result = kit.extract_text("doc", NotaFiscal)
    assert result.attempts == 2 and result.ok
    assert "CNPJ inválido" in fake_llm.calls[1].messages[2].content


def test_schema_registry_from_config(tmp_path: Path, fake_llm: FakeLLM) -> None:
    (tmp_path / "meus_schemas.py").write_text(
        "from pydantic import BaseModel\nclass Recibo(BaseModel):\n    total: float\n",
        encoding="utf-8",
    )
    from kit.config import parse_config

    cfg = parse_config(
        {
            "embedder": {"provider": "hashing", "dimension": 8},
            "store": {"provider": "memory"},
            "llm": {"provider": "fake"},
            "schemas": {"recibo": "meus_schemas:Recibo"},
        },
        base_dir=tmp_path,
    )
    kit = Kit.from_settings(cfg, llm=fake_llm)
    assert kit.schemas.names() == ["recibo"]
    assert kit.schemas.get("recibo").__name__ == "Recibo"
    with pytest.raises(KeyError, match="recibo"):
        kit.schemas.get("nao_existe")


def test_helpers() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert list(empty_fields({"a": "", "b": [], "c": {"d": None}, "e": [{"f": " "}]})) == [
        "a",
        "b",
        "c.d",
        "e[0].f",
    ]
    with pytest.raises(ValueError):
        normalize_cnpj("12.345.678/0001-00")
