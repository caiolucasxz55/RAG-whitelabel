from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kit import Kit
from kit.config import KitConfig
from kit.errors import LLMError
from tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
NF = ROOT / "examples/notas_fiscais/nf_001.pdf"
LABEL = json.loads((ROOT / "evals/datasets/extraction.jsonl").read_text("utf-8").splitlines()[0])


@pytest.fixture
def kit(config: KitConfig, fake_llm: FakeLLM) -> Kit:
    cfg = config.model_copy(
        update={"schemas": {"nota_fiscal": "examples.schemas:NotaFiscal"}, "base_dir": ROOT}
    )
    return Kit.from_settings(cfg, llm=fake_llm)


@pytest.fixture
def client(kit: Kit) -> Iterator[TestClient]:
    from kit.api import create_app

    with TestClient(create_app(kit)) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "chunks": 0,
        "llm": "FakeLLM",
        "embedder": "hashing",
        "store": "memory",
    }
    assert r.headers["x-request-id"]


def test_ingest_then_query(client: TestClient, fake_llm: FakeLLM) -> None:
    files = [
        ("files", ("contrato.md", b"Aviso previo de rescisao: 30 dias.", "text/markdown")),
        ("files", ("../../etc/nf.pdf", NF.read_bytes(), "application/pdf")),
    ]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["files"]) == ["contrato.md", "nf.pdf"]  # caminho do cliente descartado
    assert body["chunks"] > 0

    fake_llm.responses = ["São 30 dias [1]."]
    r = client.post("/query", json={"question": "prazo de aviso previo de rescisao?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "São 30 dias [1]."
    assert data["sources"][0]["source"] == "contrato.md"
    assert data["usage"]["input_tokens"] == 100


def test_query_validation(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422


def test_unsupported_upload_is_415(client: TestClient) -> None:
    r = client.post("/ingest", files=[("files", ("foto.png", b"\x89PNG", "image/png"))])
    assert r.status_code == 415
    assert r.json()["error"] == "UnsupportedFileError"


def test_extract_with_configured_schema(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [json.dumps(LABEL["expected"])]
    r = client.post(
        "/extract",
        files={"file": ("nf_001.pdf", NF.read_bytes(), "application/pdf")},
        data={"schema": "nota_fiscal"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["confidence"] == "high"
    assert body["data"]["valor_total"] == LABEL["expected"]["valor_total"]


def test_extract_unknown_schema_is_404(client: TestClient) -> None:
    r = client.post(
        "/extract", files={"file": ("a.txt", b"x", "text/plain")}, data={"schema": "nope"}
    )
    assert r.status_code == 404


def test_schemas_endpoint(client: TestClient) -> None:
    r = client.get("/schemas")
    assert r.status_code == 200
    assert "cnpj_emitente" in r.json()["nota_fiscal"]["properties"]


def test_llm_errors_map_to_502(client: TestClient, fake_llm: FakeLLM) -> None:
    client.post("/ingest", files=[("files", ("a.md", b"conteudo qualquer", "text/markdown"))])

    def boom(*_: object) -> str:
        raise LLMError("fora do ar")

    fake_llm.responses = boom
    r = client.post("/query", json={"question": "conteudo"})
    assert r.status_code == 502
    assert r.json() == {"error": "LLMError", "detail": "fora do ar"}
