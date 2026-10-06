"""Evals do kit: números reproduzíveis para o README.

- RAG: hit rate@k (o trecho esperado aparece entre os k primeiros resultados) e latência.
- Extração: acurácia por campo, documentos 100% corretos, confiança baixa, latência e custo.

Cada execução indexa o corpus num store em memória novo, isolado do índice de produção,
usando o embedder configurado no kit.yaml.
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kit.adapters.stores.memory import InMemoryVectorStore
from kit.config import KitConfig
from kit.kit import Kit

Progress = Callable[[str], None]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    """Percentil pelo método nearest-rank (sem interpolação)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


# ---------------------------------------------------------------- RAG


@dataclass
class RagCase:
    question: str
    source: str
    contains: str
    hit: bool
    rank: int | None
    latency_s: float


@dataclass
class RagReport:
    label: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    contextual_headers: bool = True
    cases: list[RagCase] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return sum(c.hit for c in self.cases) / len(self.cases) if self.cases else math.nan

    @property
    def mrr(self) -> float:
        return sum(1 / c.rank for c in self.cases if c.rank) / len(self.cases) if self.cases else 0

    def latency(self, q: float) -> float:
        return percentile([c.latency_s for c in self.cases], q)

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "questions": len(self.cases),
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "contextual_headers": self.contextual_headers,
            f"hit_rate@{self.top_k}": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "retrieval_p50_s": round(self.latency(50), 4),
            "retrieval_p95_s": round(self.latency(95), 4),
        }


def _isolated_kit(base: Kit, config: KitConfig) -> Kit:
    """Mesmo embedder/loaders/LLM já carregados, com store em memória novo."""
    return Kit.from_settings(
        config,
        loaders=base.loaders,
        embedder=base.embedder,
        llm=base.llm,
        store=InMemoryVectorStore(base.embedder.dimension),
    )


def eval_rag(
    base: Kit,
    dataset: list[dict[str, Any]],
    corpus: str | Path,
    *,
    top_k: int = 5,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    contextual_headers: bool | None = None,
    label: str = "",
    progress: Progress | None = None,
) -> RagReport:
    chunking = base.config.chunking.model_copy(
        update={
            k: v
            for k, v in {
                "size": chunk_size,
                "overlap": chunk_overlap,
                "contextual_headers": contextual_headers,
            }.items()
            if v is not None
        }
    )
    config = base.config.model_copy(update={"chunking": chunking})
    kit = _isolated_kit(base, config)
    kit.ingest(corpus)
    default_label = f"chunk {chunking.size}" + (
        "" if chunking.contextual_headers else ", sem header"
    )
    report = RagReport(
        label or default_label,
        top_k,
        chunking.size,
        chunking.overlap,
        chunking.contextual_headers,
    )
    for row in dataset:
        start = time.perf_counter()
        retrieved = kit.retrieve(row["question"], top_k)
        latency = time.perf_counter() - start
        needle = _fold(row["contains"])
        rank = next(
            (
                i
                for i, item in enumerate(retrieved, start=1)
                if Path(item.chunk.source).name == row["source"]
                and needle in _fold(item.chunk.text)
            ),
            None,
        )
        report.cases.append(
            RagCase(
                row["question"], row["source"], row["contains"], rank is not None, rank, latency
            )
        )
        if progress:
            progress(f"[rag {report.label}] {'HIT ' if rank else 'MISS'} {row['question']}")
    return report


# ---------------------------------------------------------------- Extração


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return _fold(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def compare_fields(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, bool]:
    """Campo de primeiro nível correto? Listas e objetos aninhados contam como um campo."""
    return {
        name: predicted is not None and _normalize(predicted.get(name)) == _normalize(value)
        for name, value in expected.items()
    }


@dataclass
class ExtractionCase:
    file: str
    fields: dict[str, bool]
    confidence: str
    attempts: int
    latency_s: float
    cost_usd: float | None
    error: str | None = None

    @property
    def exact(self) -> bool:
        return bool(self.fields) and all(self.fields.values())


@dataclass
class ExtractionReport:
    label: str
    max_retries: int
    cases: list[ExtractionCase] = field(default_factory=list)

    def field_accuracy(self) -> dict[str, float]:
        names = sorted({n for c in self.cases for n in c.fields})
        return {n: sum(c.fields.get(n, False) for c in self.cases) / len(self.cases) for n in names}

    def summary(self) -> dict[str, Any]:
        n = len(self.cases) or 1
        per_field = self.field_accuracy()
        costs = [c.cost_usd for c in self.cases if c.cost_usd is not None]
        latencies = [c.latency_s for c in self.cases]
        return {
            "label": self.label,
            "documents": len(self.cases),
            "max_retries": self.max_retries,
            "field_accuracy": round(sum(per_field.values()) / len(per_field), 4)
            if per_field
            else math.nan,
            "exact_match": round(sum(c.exact for c in self.cases) / n, 4),
            "low_confidence": round(sum(c.confidence == "low" for c in self.cases) / n, 4),
            "errors": sum(c.error is not None for c in self.cases),
            "avg_attempts": round(sum(c.attempts for c in self.cases) / n, 3),
            "latency_p50_s": round(percentile(latencies, 50), 3),
            "latency_p95_s": round(percentile(latencies, 95), 3),
            "cost_per_100_docs_usd": round(sum(costs) / len(costs) * 100, 4) if costs else None,
            "per_field": {k: round(v, 4) for k, v in per_field.items()},
        }


def eval_extraction(
    kit: Kit,
    dataset: list[dict[str, Any]],
    *,
    root: str | Path = ".",
    max_retries: int | None = None,
    label: str = "",
    progress: Progress | None = None,
) -> ExtractionReport:
    retries = kit.config.extraction.max_retries if max_retries is None else max_retries
    report = ExtractionReport(label or f"retries={retries}", retries)
    for row in dataset:
        schema: type[BaseModel] = kit.schemas.get(row["schema"])
        # Normaliza o rótulo pelo próprio schema (ex.: CNPJ só com dígitos).
        expected = schema.model_validate(row["expected"]).model_dump(mode="json")
        start = time.perf_counter()
        error = None
        try:
            result = kit.extract(Path(root) / row["file"], schema, max_retries=retries)
            predicted = result.data.model_dump(mode="json") if result.data else None
            confidence, attempts, cost = result.confidence, result.attempts, result.usage.cost_usd
        except Exception as exc:  # um documento com erro não derruba o eval inteiro
            predicted, confidence, attempts, cost = None, "low", 0, None
            error = f"{type(exc).__name__}: {exc}"
        case = ExtractionCase(
            file=row["file"],
            fields=compare_fields(expected, predicted),
            confidence=confidence,
            attempts=attempts,
            latency_s=time.perf_counter() - start,
            cost_usd=cost,
            error=error,
        )
        report.cases.append(case)
        if progress:
            ok = sum(case.fields.values())
            progress(f"[extração {report.label}] {row['file']}: {ok}/{len(case.fields)} campos")
    return report


# ---------------------------------------------------------------- Saída


def _pct(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def markdown(rag: Sequence[RagReport], extraction: Sequence[ExtractionReport]) -> str:
    lines: list[str] = []
    if rag:
        k = rag[0].top_k
        lines += [
            "### RAG",
            "",
            f"| Config | Perguntas | Hit rate@{k} | MRR | Latência busca p50 | p95 |",
            "|---|---|---|---|---|---|",
        ]
        for r in rag:
            lines.append(
                f"| {r.label} (overlap {r.chunk_overlap}) | {len(r.cases)} | {_pct(r.hit_rate)} | "
                f"{r.mrr:.3f} | {_num(r.latency(50))} s | {_num(r.latency(95))} s |"
            )
        lines.append("")
    if extraction:
        lines += [
            "### Extração",
            "",
            "| Config | Docs | Acurácia por campo | Docs 100% corretos | Confiança baixa | "
            "Tentativas (média) | Latência p50 | p95 | Custo / 100 docs |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for e in extraction:
            s = e.summary()
            cost = s["cost_per_100_docs_usd"]
            lines.append(
                f"| {e.label} | {s['documents']} | {_pct(s['field_accuracy'])} | "
                f"{_pct(s['exact_match'])} | {_pct(s['low_confidence'])} | {s['avg_attempts']} | "
                f"{_num(s['latency_p50_s'], 2)} s | {_num(s['latency_p95_s'], 2)} s | "
                f"{'n/a' if cost is None else f'US$ {cost:.2f}'} |"
            )
        lines += ["", "Acurácia por campo (primeira configuração):", ""]
        lines += ["| Campo | Acurácia |", "|---|---|"]
        for name, acc in extraction[0].field_accuracy().items():
            lines.append(f"| `{name}` | {_pct(acc)} |")
        lines.append("")
    return "\n".join(lines)


def to_json(rag: Sequence[RagReport], extraction: Sequence[ExtractionReport]) -> dict[str, Any]:
    return {
        "rag": [{"summary": r.summary(), "cases": [asdict(c) for c in r.cases]} for r in rag],
        "extraction": [
            {"summary": e.summary(), "cases": [asdict(c) for c in e.cases]} for e in extraction
        ],
    }
