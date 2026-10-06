"""CLI: kit ingest | query | extract | schemas | eval | serve."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("a CLI precisa do extra: pip install 'kit-rag[cli]'") from exc

from kit import Kit
from kit.errors import KitError
from kit.observability import configure_logging

app = typer.Typer(
    help="Kit de RAG com citações e extração estruturada.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

ConfigOpt = typer.Option(
    None, "--config", "-c", help="kit.yaml (padrão: $KIT_CONFIG ou ./kit.yaml)"
)


def _kit(config: Path | None) -> Kit:
    return Kit.from_config(config)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Logs JSON detalhados"),
) -> None:
    configure_logging("INFO" if verbose else "WARNING")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Arquivos ou pastas (recursivo)"),
    config: Path | None = ConfigOpt,
) -> None:
    """Indexa documentos no vector store configurado."""
    report = _kit(config).ingest(paths)
    typer.echo(
        f"{len(report.files)} arquivo(s), {report.documents} documento(s), {report.chunks} chunk(s)"
    )
    if report.skipped:
        typer.echo(f"ignorados (sem loader): {', '.join(report.skipped)}")


@app.command()
def query(
    question: str = typer.Argument(...),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
    as_json: bool = typer.Option(False, "--json"),
    config: Path | None = ConfigOpt,
) -> None:
    """Responde a uma pergunta citando documento e trecho."""
    result = _kit(config).query(question, top_k)
    if as_json:
        _echo_json(
            {
                "answer": result.answer,
                "sources": [s.__dict__ for s in result.sources],
                "usage": result.usage.__dict__,
            }
        )
        return
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\nFontes:")
        for s in result.sources:
            page = f", p. {s.page}" if s.page else ""
            snippet = s.text[:160].replace("\n", " ")
            typer.echo(f'  [{s.ref}] {s.source}{page} - "{snippet}..."')


@app.command()
def extract(
    file: Path = typer.Argument(..., exists=True, dir_okay=False),
    schema: str = typer.Option(..., "--schema", "-s", help="Nome do schema no kit.yaml"),
    instructions: str | None = typer.Option(None, "--instructions", "-i"),
    config: Path | None = ConfigOpt,
) -> None:
    """Extrai dados estruturados de um documento."""
    kit = _kit(config)
    result = kit.extract(file, kit.schemas.get(schema), instructions=instructions)
    _echo_json(
        {
            "data": result.data.model_dump(mode="json") if result.data else None,
            "confidence": result.confidence,
            "issues": result.issues,
            "attempts": result.attempts,
            "usage": result.usage.__dict__,
        }
    )
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def schemas(config: Path | None = ConfigOpt) -> None:
    """Lista os schemas de extração configurados."""
    for name in _kit(config).schemas.names():
        typer.echo(name)


@app.command("eval")
def run_eval(
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Avaliar recuperação (hit rate@k)"),
    extraction: bool = typer.Option(
        False, "--extraction/--no-extraction", help="Avaliar extração (chama o LLM: tem custo)"
    ),
    rag_dataset: Path = typer.Option(Path("evals/datasets/rag.jsonl")),
    extraction_dataset: Path = typer.Option(Path("evals/datasets/extraction.jsonl")),
    corpus: Path = typer.Option(Path("examples"), help="Pasta indexada para o eval de RAG"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    chunk_sizes: str | None = typer.Option(
        None, help="Comparar tamanhos de chunk, ex.: 500,800 (padrão: o do kit.yaml)"
    ),
    compare_headers: bool = typer.Option(
        False, help="Comparar busca com vs. sem cabeçalho contextual nos chunks"
    ),
    compare_retry: bool = typer.Option(False, help="Comparar extração sem retry vs. com retry"),
    limit: int | None = typer.Option(None, help="Usar só os N primeiros exemplos"),
    output: Path = typer.Option(Path("evals/results"), help="Pasta para salvar JSON e Markdown"),
    config: Path | None = ConfigOpt,
) -> None:
    """Roda os evals e imprime a tabela de métricas (a mesma do README)."""
    from kit import evals

    kit = _kit(config)
    progress = lambda msg: typer.echo(msg, err=True)  # noqa: E731
    rag_reports, extraction_reports = [], []

    if rag:
        dataset = evals.read_jsonl(rag_dataset)[:limit]
        sizes = [int(s) for s in chunk_sizes.split(",")] if chunk_sizes else [None]
        headers = [False, True] if compare_headers else [None]
        for size in sizes:
            overlap = None if size is None else min(kit.config.chunking.overlap, size // 4)
            for header in headers:
                rag_reports.append(
                    evals.eval_rag(
                        kit,
                        dataset,
                        corpus,
                        top_k=top_k,
                        chunk_size=size,
                        chunk_overlap=overlap,
                        contextual_headers=header,
                        progress=progress,
                    )
                )

    if extraction:
        dataset = evals.read_jsonl(extraction_dataset)[:limit]
        variants = [0, kit.config.extraction.max_retries or 1] if compare_retry else [None]
        for retries in variants:
            label = "" if retries is None else ("sem retry" if retries == 0 else "com retry")
            extraction_reports.append(
                evals.eval_extraction(
                    kit, dataset, max_retries=retries, label=label, progress=progress
                )
            )

    table = evals.markdown(rag_reports, extraction_reports)
    typer.echo("\n" + table)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (output / f"eval-{stamp}.md").write_text(table, encoding="utf-8")
    (output / f"eval-{stamp}.json").write_text(
        json.dumps(evals.to_json(rag_reports, extraction_reports), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"resultados salvos em {output}/eval-{stamp}.{{md,json}}", err=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    config: Path | None = ConfigOpt,
) -> None:
    """Sobe a API HTTP (FastAPI + uvicorn)."""
    import uvicorn

    if config:
        os.environ["KIT_CONFIG"] = str(config)
    uvicorn.run("kit.api.app:create_app", factory=True, host=host, port=port, log_config=None)


def main() -> None:
    try:
        app()
    except KitError as exc:
        typer.echo(f"erro: {exc}", err=True)
        sys.exit(2)
    except KeyError as exc:
        typer.echo(f"erro: {exc.args[0]}", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
