"""Gera o corpus sintético rotulado usado na demo e nos evals.

Todos os documentos são FICTÍCIOS (empresas, CNPJs e valores aleatórios) e trazem o aviso
"DOCUMENTO FICTÍCIO" no cabeçalho. A geração é determinística (seed fixa).

    python -m examples.generate
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from examples.minipdf import write_pdf

ROOT = Path(__file__).resolve().parent.parent
NF_DIR = ROOT / "examples" / "notas_fiscais"
CONTRACT_DIR = ROOT / "examples" / "contratos"
DATASETS = ROOT / "evals" / "datasets"
DISCLAIMER = "DOCUMENTO FICTÍCIO GERADO PARA TESTES - SEM VALOR FISCAL OU JURÍDICO"

NAMES = [
    "Alfa", "Boreal", "Cedro", "Delta Sul", "Estrela Azul", "Farol", "Granito", "Horizonte",
    "Ipê Amarelo", "Jatobá", "Lúmen", "Marítima", "Nascente", "Orvalho", "Pinheiral", "Quartzo",
]  # fmt: skip
SECTORS = [
    "Comércio de Papéis", "Tecnologia", "Alimentos", "Logística", "Ferragens", "Têxtil",
    "Serviços Gráficos", "Equipamentos Médicos",
]  # fmt: skip
SUFFIXES = ["Ltda", "S.A.", "ME", "Eireli"]
PRODUCTS = [
    ("Resma papel A4 75g", 18, 32), ("Toner impressora laser", 120, 380),
    ("Cabo de rede Cat6 (metro)", 2, 6), ("Parafuso sextavado M8", 0.3, 1.2),
    ("Café torrado 500g", 14, 29), ("Licença de software mensal", 90, 450),
    ("Luva nitrílica (caixa)", 25, 60), ("Tecido algodão (metro)", 12, 35),
    ("Frete rodoviário", 150, 900), ("Monitor 24 polegadas", 650, 1200),
]  # fmt: skip
CITIES = ["São Paulo", "Curitiba", "Belo Horizonte", "Recife", "Porto Alegre", "Salvador"]
SERVICES = [
    "manutenção preventiva de equipamentos de informática",
    "limpeza e conservação predial",
    "consultoria contábil e fiscal",
    "desenvolvimento e sustentação de software",
    "transporte de cargas fracionadas",
    "vigilância patrimonial desarmada",
    "fornecimento de refeições corporativas",
    "suporte técnico em telefonia",
]


def make_cnpj(rng: random.Random) -> str:
    base = [rng.randint(0, 9) for _ in range(8)] + [0, 0, 0, 1]
    for _ in range(2):
        weights = list(range(len(base) - 7, 1, -1)) + list(range(9, 1, -1))
        rest = sum(d * w for d, w in zip(base, weights, strict=True)) % 11
        base.append(0 if rest < 2 else 11 - rest)
    return "".join(map(str, base))


def fmt_cnpj(d: str) -> str:
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def brl(value: float) -> str:
    text = f"{value:,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def company(rng: random.Random, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(NAMES)} {rng.choice(SECTORS)} {rng.choice(SUFFIXES)}"
        if name not in used:
            used.add(name)
            return name


@dataclass
class Nota:
    numero: int
    serie: int
    data: date
    emitente: str
    cnpj_emitente: str
    destinatario: str
    cnpj_destinatario: str
    itens: list[tuple[str, int, float]]

    @property
    def total(self) -> float:
        return round(sum(q * v for _, q, v in self.itens), 2)

    def label(self) -> dict[str, object]:
        return {
            "numero": str(self.numero),
            "data_emissao": self.data.isoformat(),
            "cnpj_emitente": self.cnpj_emitente,
            "razao_social_emitente": self.emitente,
            "cnpj_destinatario": self.cnpj_destinatario,
            "valor_total": self.total,
            "itens": [
                {"descricao": d, "quantidade": q, "valor_unitario": v} for d, q, v in self.itens
            ],
        }


def _item_lines(nota: Nota, style: int) -> list[str]:
    lines = []
    for i, (desc, qtd, unit) in enumerate(nota.itens, start=1):
        if style == 0:
            lines.append(
                f"{i:02d}  {desc}  QTD {qtd}  UN R$ {brl(unit)}  TOTAL R$ {brl(qtd * unit)}"
            )
        else:
            lines.append(f"Item {i}: {desc} | quantidade: {qtd} | unitário: {brl(unit)}")
    return lines


def render_nota(nota: Nota, style: int) -> list[list[str]]:
    if style == 0:
        header = [
            DISCLAIMER,
            "DANFE - DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA",
            f"Nº {nota.numero:09d}   SÉRIE {nota.serie}",
            f"DATA DE EMISSÃO: {nota.data:%d/%m/%Y}",
            "",
            "EMITENTE",
            f"Razão social: {nota.emitente}",
            f"CNPJ: {fmt_cnpj(nota.cnpj_emitente)}",
            "",
            "DESTINATÁRIO / REMETENTE",
            f"Nome: {nota.destinatario}",
            f"CNPJ: {fmt_cnpj(nota.cnpj_destinatario)}",
            "",
            "DADOS DOS PRODUTOS / SERVIÇOS",
        ]
        footer = ["", f"VALOR TOTAL DA NOTA: R$ {brl(nota.total)}"]
    else:
        header = [
            DISCLAIMER,
            "NOTA FISCAL DE VENDA",
            f"Emitida em {nota.data:%d/%m/%Y} - número {nota.numero}, série {nota.serie}",
            f"Fornecedor: {nota.emitente} (CNPJ {fmt_cnpj(nota.cnpj_emitente)})",
            f"Cliente: {nota.destinatario} (CNPJ {fmt_cnpj(nota.cnpj_destinatario)})",
            "",
            "Itens:",
        ]
        footer = ["", f"Total a pagar: {brl(nota.total)} reais"]
    items = _item_lines(nota, style)
    if len(items) > 4:  # notas maiores quebram em duas páginas
        return [header + items[:3], [*items[3:], *footer]]
    return [header + items + footer]


def build_notas(rng: random.Random, count: int = 25) -> list[Nota]:
    used: set[str] = set()
    clients = [(company(rng, used), make_cnpj(rng)) for _ in range(5)]
    notas = []
    for _ in range(count):
        dest, cnpj_dest = rng.choice(clients)
        itens = []
        for desc, lo, hi in rng.sample(PRODUCTS, rng.randint(1, 6)):
            itens.append((desc, rng.randint(1, 40), round(rng.uniform(lo, hi), 2)))
        notas.append(
            Nota(
                numero=rng.randint(1000, 99999),
                serie=rng.randint(1, 3),
                data=date(2026, 1, 5) + timedelta(days=rng.randint(0, 250)),
                emitente=company(rng, used),
                cnpj_emitente=make_cnpj(rng),
                destinatario=dest,
                cnpj_destinatario=cnpj_dest,
                itens=itens,
            )
        )
    return notas


@dataclass
class Contrato:
    slug: str
    contratante: str
    contratada: str
    servico: str
    valor: float
    aviso: int
    multa: int
    foro: str
    assinatura: date
    vigencia: int

    def text(self) -> str:
        return f"""# Contrato de prestação de serviços de {self.servico}

> {DISCLAIMER}

**Contratante:** {self.contratante}
**Contratada:** {self.contratada}

## Cláusula 1 - Objeto
A contratada prestará à contratante serviços de {self.servico}, conforme escopo do anexo I.

## Cláusula 2 - Preço e pagamento
Pelos serviços, a contratante pagará o valor mensal de R$ {brl(self.valor)}, com vencimento \
todo dia 10 do mês seguinte à prestação, mediante emissão de nota fiscal.

## Cláusula 3 - Vigência
Este contrato vigora por {self.vigencia} meses a partir da assinatura, renovável por \
iguais períodos mediante termo aditivo.

## Cláusula 4 - Rescisão
Qualquer das partes poderá rescindir este contrato, sem justa causa, mediante aviso prévio \
de {self.aviso} dias, por escrito. A rescisão sem o aviso prévio sujeita a parte infratora a \
multa de {self.multa}% sobre o valor mensal.

## Cláusula 5 - Confidencialidade
As partes manterão sigilo sobre informações recebidas em razão deste contrato pelo prazo de \
cinco anos após o seu término.

## Cláusula 6 - Foro
Fica eleito o foro da comarca de {self.foro} para dirimir quaisquer dúvidas oriundas deste \
contrato.

Assinado em {self.assinatura:%d/%m/%Y}.
"""

    def label(self) -> dict[str, object]:
        return {
            "contratante": self.contratante,
            "contratada": self.contratada,
            "valor_mensal": self.valor,
            "prazo_aviso_previo_dias": self.aviso,
            "foro": self.foro,
            "data_assinatura": self.assinatura.isoformat(),
        }


def build_contratos(rng: random.Random) -> list[Contrato]:
    used: set[str] = set()
    contratos = []
    for i, servico in enumerate(SERVICES, start=1):
        contratos.append(
            Contrato(
                slug=f"contrato_{i:02d}",
                contratante=company(rng, used),
                contratada=company(rng, used),
                servico=servico,
                valor=round(rng.uniform(1500, 48000), 2),
                aviso=rng.choice([15, 30, 45, 60, 90]),
                multa=rng.choice([10, 20, 30, 50]),
                foro=rng.choice(CITIES),
                assinatura=date(2025, 3, 1) + timedelta(days=rng.randint(0, 400)),
                vigencia=rng.choice([12, 24, 36]),
            )
        )
    return contratos


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def main() -> None:
    rng = random.Random(20261006)
    notas = build_notas(rng)
    contratos = build_contratos(rng)

    NF_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    extraction_rows: list[dict[str, object]] = []
    rag_rows: list[dict[str, object]] = []

    for i, nota in enumerate(notas, start=1):
        name = f"nf_{i:03d}.pdf"
        write_pdf(NF_DIR / name, render_nota(nota, style=i % 2))
        extraction_rows.append(
            {
                "file": f"examples/notas_fiscais/{name}",
                "schema": "nota_fiscal",
                "expected": nota.label(),
            }
        )
        if i % 2 == 0:
            rag_rows.append(
                {
                    "question": f"Qual o valor total da nota fiscal número {nota.numero}?",
                    "source": name,
                    "contains": brl(nota.total),
                }
            )

    for c in contratos:
        # Dois contratos viram PDF para o corpus misturar formatos.
        if c.slug in {"contrato_03", "contrato_06"}:
            name = f"{c.slug}.pdf"
            write_pdf(CONTRACT_DIR / name, [c.text().replace("**", "").split("\n")])
        else:
            name = f"{c.slug}.md"
            (CONTRACT_DIR / name).write_text(c.text(), encoding="utf-8")
        rag_rows += [
            {
                "question": f"Qual o prazo de aviso prévio para rescindir o contrato de "
                f"{c.servico} com a {c.contratada}?",
                "source": name,
                "contains": f"{c.aviso} dias",
            },
            {
                "question": f"Qual o foro eleito no contrato de {c.servico}?",
                "source": name,
                "contains": f"comarca de {c.foro}",
            },
        ]

    write_jsonl(DATASETS / "extraction.jsonl", extraction_rows)
    write_jsonl(DATASETS / "rag.jsonl", rag_rows)
    print(
        f"{len(notas)} notas, {len(contratos)} contratos, "
        f"{len(extraction_rows)} rótulos de extração, {len(rag_rows)} perguntas de RAG"
    )


if __name__ == "__main__":
    main()
