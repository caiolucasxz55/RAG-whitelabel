from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kit import Kit, evals
from kit.cli import app
from kit.config import KitConfig
from tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "kit.yaml"
    path.write_text(
        "loader: [pdf, text]\n"
        "embedder: {provider: hashing, dimension: 512}\n"
        "store: {provider: memory}\n"
        "llm: {provider: extractive}\n"
        "chunking: {size: 400, overlap: 50}\n",
        encoding="utf-8",
    )
    return path


def test_cli_ingest_and_query_offline(config_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["ingest", str(ROOT / "examples/contratos"), "-c", str(config_file)])
    assert r.exit_code == 0, r.output
    assert "8 arquivo(s)" in r.output
    # Store em memória não persiste entre processos: query sem ingest responde "sem documentos".
    r = runner.invoke(app, ["query", "Qual o foro?", "-c", str(config_file)])
    assert r.exit_code == 0 and "Não há documentos" in r.output


def test_cli_eval_rag_writes_results(config_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    r = CliRunner().invoke(
        app,
        [
            "eval",
            "--rag-dataset", str(ROOT / "evals/datasets/rag.jsonl"),
            "--corpus", str(ROOT / "examples"),
            "--chunk-sizes", "300,600",
            "--limit", "6",
            "--output", str(out),
            "-c", str(config_file),
        ],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "Hit rate@5" in r.output
    saved = json.loads(next(out.glob("*.json")).read_text("utf-8"))
    assert [x["summary"]["chunk_size"] for x in saved["rag"]] == [300, 600]


def test_eval_extraction_metrics(config: KitConfig) -> None:
    dataset = evals.read_jsonl(ROOT / "evals/datasets/extraction.jsonl")[:3]
    answers = [json.dumps(row["expected"]) for row in dataset]
    wrong = dict(dataset[2]["expected"], valor_total=1.0)
    answers[2] = json.dumps(wrong)
    cfg = config.model_copy(
        update={"schemas": {"nota_fiscal": "examples.schemas:NotaFiscal"}, "base_dir": ROOT}
    )
    kit = Kit.from_settings(cfg, llm=FakeLLM(responses=answers))
    report = evals.eval_extraction(kit, dataset, root=ROOT)
    summary = report.summary()
    assert summary["documents"] == 3
    assert summary["exact_match"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["per_field"]["valor_total"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["per_field"]["cnpj_emitente"] == 1.0
    assert summary["cost_per_100_docs_usd"] == pytest.approx(100 * (100 * 4e-6 + 10 * 20e-6))
    assert "valor_total" in evals.markdown([], [report])


def test_percentile_nearest_rank() -> None:
    assert evals.percentile([1, 2, 3, 4], 50) == 2
    assert evals.percentile([1, 2, 3, 4], 95) == 4
