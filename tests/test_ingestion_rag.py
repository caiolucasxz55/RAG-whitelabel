from __future__ import annotations

from pathlib import Path

import pytest
from examples.minipdf import write_pdf

from kit import Kit
from kit.config import KitConfig
from kit.errors import UnsupportedFileError
from kit.rag import NO_CONTEXT_ANSWER, format_context, parse_citations
from tests.fakes import FakeLLM

CONTRATO = (
    "Contrato de prestação de serviços.\n\n"
    "Cláusula 7 - Rescisão. Qualquer parte pode rescindir este contrato mediante aviso "
    "prévio de 30 dias, por escrito.\n\n"
    "Cláusula 8 - Foro. Fica eleito o foro da comarca de São Paulo."
)
POLITICA = "Política de reembolso: despesas de viagem são reembolsadas em até 10 dias úteis."


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "contrato.md").write_text(CONTRATO, encoding="utf-8")
    (root / "sub" / "politica.txt").write_text(POLITICA, encoding="utf-8")
    (root / "imagem.png").write_bytes(b"\x89PNG")
    write_pdf(root / "manual.pdf", [["Manual do produto"], ["Garantia de 12 meses."]])
    return root


@pytest.fixture
def kit(config: KitConfig, fake_llm: FakeLLM) -> Kit:
    return Kit.from_settings(config, llm=fake_llm)


def test_ingest_folder_recursively_and_skip_unsupported(kit: Kit, docs: Path) -> None:
    report = kit.ingest(docs)
    assert sorted(report.files) == ["contrato.md", "manual.pdf", "sub/politica.txt"]
    assert report.skipped == ["imagem.png"]
    assert report.documents == 4  # 2 páginas do PDF + 2 textos
    assert kit.store.count() == report.chunks > 0


def test_reingest_replaces_chunks(kit: Kit, docs: Path) -> None:
    kit.ingest(docs)
    before = kit.store.count()
    kit.ingest(docs)
    assert kit.store.count() == before


def test_explicit_unsupported_file_raises(kit: Kit, docs: Path) -> None:
    with pytest.raises(UnsupportedFileError):
        kit.ingest(docs / "imagem.png")


def test_missing_path_raises(kit: Kit, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        kit.ingest(tmp_path / "nada")


def test_retrieve_finds_relevant_chunk(kit: Kit, docs: Path) -> None:
    kit.ingest(docs)
    top = kit.retrieve("aviso prévio para rescindir o contrato", top_k=2)
    assert top[0].chunk.source == "contrato.md"


def test_pdf_pages_are_cited(kit: Kit, docs: Path, fake_llm: FakeLLM) -> None:
    kit.ingest(docs)

    def answer(messages, system, schema):  # type: ignore[no-untyped-def]
        prompt = messages[-1].content
        ref = next(
            i for i, c in enumerate(kit.retrieve("garantia meses"), 1) if "Garantia" in c.chunk.text
        )
        assert "<question>" in prompt
        return f"A garantia é de 12 meses [{ref}]."

    fake_llm.responses = answer
    result = kit.query("garantia meses")
    assert result.sources[0].source == "manual.pdf"
    assert result.sources[0].page == 2


def test_query_returns_cited_sources_and_cost(kit: Kit, docs: Path, fake_llm: FakeLLM) -> None:
    kit.ingest(docs)
    fake_llm.responses = ["O aviso prévio é de 30 dias [1]. Detalhe extra [1, 9]."]
    result = kit.query("Qual o prazo de rescisão?")
    assert result.answer.startswith("O aviso prévio")
    assert [s.ref for s in result.sources] == [1]  # [9] está fora do contexto: ignorado
    assert result.sources[0].source == "contrato.md"
    assert "30 dias" in result.sources[0].text
    call = fake_llm.calls[0]
    assert call.system and "cite" in call.system
    # pricing do conftest: 100 tokens in * $4/M + 10 out * $20/M
    assert result.usage.cost_usd == pytest.approx(100 * 4e-6 + 10 * 20e-6)


def test_query_without_index_skips_llm(kit: Kit, fake_llm: FakeLLM) -> None:
    result = kit.query("qualquer coisa")
    assert result.answer == NO_CONTEXT_ANSWER
    assert fake_llm.calls == []


def test_empty_question_rejected(kit: Kit) -> None:
    with pytest.raises(ValueError):
        kit.query("   ")


def test_parse_citations_orders_dedups_and_bounds() -> None:
    assert parse_citations("a [2] b [1, 2] c [3][2] d [0] [7]", max_ref=3) == [2, 1, 3]


def test_format_context_escapes_source() -> None:
    from kit.types import Chunk, ScoredChunk

    ctx = format_context("q?", [ScoredChunk(Chunk("id", "texto", 'a"b.md', 0, {"page": 3}), 0.9)])
    assert 'source="a&quot;b.md" page="3"' in ctx


def test_contextual_header_uses_source_and_first_line() -> None:
    from kit.ingestion import document_header
    from kit.types import Document

    docs = [Document("\n\n# Contrato de limpeza\n\nCláusula 1", "c.md")]
    assert document_header("c.md", docs) == "c.md | Contrato de limpeza"
    assert document_header("vazio.md", [Document("  ", "vazio.md")]) == "vazio.md"


def test_contextual_header_improves_retrieval_of_templated_docs(
    config: KitConfig, fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """Documentos de mesmo modelo: só o título diferencia o trecho da cláusula."""
    root = tmp_path / "c"
    root.mkdir()
    filler = "Texto padrão da cláusula de vigência e renovação do contrato. " * 4
    for slug, servico, dias in [("a", "jardinagem", 15), ("b", "vigilância", 90)]:
        (root / f"{slug}.md").write_text(
            f"# Contrato de {servico}\n\n{filler}\n\nRescisão com aviso prévio de {dias} dias.",
            encoding="utf-8",
        )
    question = "aviso prévio de rescisão do contrato de vigilância"

    def top_source(headers: bool) -> str:
        chunking = config.chunking.model_copy(update={"contextual_headers": headers})
        cfg = config.model_copy(update={"chunking": chunking})
        kit = Kit.from_settings(cfg, llm=fake_llm)
        kit.ingest(root)
        hits = [h for h in kit.retrieve(question, top_k=5) if "aviso prévio" in h.chunk.text]
        return hits[0].chunk.source

    assert top_source(headers=True) == "b.md"
