# kit — RAG com citações + extração estruturada, white-label

Kit open source que qualquer projeto instala para ter **perguntas e respostas sobre documentos com citação de fonte** e **extração de dados estruturados validados**, trocando provedores (embeddings, vector store, LLM) **só por configuração**.

```
docs/ ──► kit.ingest() ──► pgvector          "Qual o prazo de rescisão?"
                                 │                      │
                                 └──► kit.query() ◄─────┘ ──► resposta + [1] contrato_01.md, trecho
nf_123.pdf ──► kit.extract(schema=NotaFiscal) ──► objeto Pydantic validado + confiança
```

- **Núcleo agnóstico:** o core só conhece 4 `Protocol`s (Loader, Embedder, VectorStore, LLM).
- **Config, não código:** um projeto adota o kit editando `kit.yaml` e `.env`, sem fork.
- **Duas formas de uso:** biblioteca Python ou serviço (imagem Docker com API REST).
- **Roda sem chave de API:** embeddings locais e um LLM extrativo offline permitem `docker compose up` e testar na hora.
- **Medido:** `kit eval` gera as métricas da seção [Resultados](#resultados): hit rate@5 de **82,1%** no corpus de exemplo, rodando em CPU.

## Quickstart (Docker)

```bash
cp .env.example .env            # opcional: coloque ANTHROPIC_API_KEY para usar o Claude
docker compose up -d --build    # API em :8000 + Postgres/pgvector em :5433
```

```bash
# 1. indexar documentos (multipart; vários arquivos por chamada)
curl -F files=@examples/contratos/contrato_01.md -F files=@examples/contratos/contrato_03.pdf \
     http://localhost:8000/ingest

# 2. perguntar, com citações
curl -H 'Content-Type: application/json' \
     -d '{"question": "Qual o prazo de aviso prévio para rescisão?"}' \
     http://localhost:8000/query

# 3. extrair uma nota fiscal para JSON validado (precisa de ANTHROPIC_API_KEY)
curl -F file=@examples/notas_fiscais/nf_001.pdf -F schema=nota_fiscal \
     http://localhost:8000/extract
```

Documentação interativa em `http://localhost:8000/docs`. Rotas: `POST /ingest`, `POST /query`, `POST /extract`, `GET /schemas`, `GET /health`.

Sem `ANTHROPIC_API_KEY`, o provedor `auto` usa o LLM **extrativo**: a resposta é a frase mais relevante dos trechos recuperados, já com citação. A busca, a ingestão e as citações funcionam igual; a extração estruturada exige um LLM gerativo e responde `501` até a chave ser configurada.

## Uso como biblioteca

```bash
pip install ".[all]"        # a partir do clone do repositório
```

```python
from pydantic import BaseModel
from kit import Kit

kit = Kit.from_config("kit.yaml")

kit.ingest("docs/")
resp = kit.query("Qual o prazo de rescisão?")
print(resp.answer, resp.sources)


class NotaFiscal(BaseModel):
    cnpj: str
    valor_total: float
    data_emissao: str


nota = kit.extract("nf_123.pdf", schema=NotaFiscal)
print(nota.data, nota.confidence)  # objeto validado (ou None) + "high" | "low"
print(nota.issues)  # por que a confiança é baixa, se for
```

Cada `Source` traz `source` (arquivo), `page` (PDF), `text` (trecho), `score` e `ref` (o `[n]` citado na resposta). `ExtractionResult` traz `data`, `confidence`, `issues`, `attempts` e `usage` (tokens e custo).

Extras: `anthropic`, `local` (sentence-transformers), `pgvector`, `api`, `cli`, `all`.

### CLI

```bash
kit ingest docs/
kit query "Qual o foro do contrato?"
kit extract nf_123.pdf --schema nota_fiscal
kit eval --extraction --compare-retry --chunk-sizes 500,800
kit serve --host 0.0.0.0
```

## Trocando provedores via `kit.yaml`

```yaml
loader: [pdf, text]
embedder:
  provider: local                         # sentence-transformers, sem chave
  model: intfloat/multilingual-e5-small
store:
  provider: pgvector
  dsn: ${DATABASE_URL}
llm:
  provider: anthropic                     # ou: auto | extractive
  model: ${LLM_MODEL:-claude-opus-5-5}
  effort: medium
chunking: {size: 800, overlap: 100}
schemas:
  nota_fiscal: meu_projeto.schemas:NotaFiscal   # exposto em /extract e na CLI
```

| Interface   | Embutidos                              | Para testes / offline     |
|-------------|----------------------------------------|---------------------------|
| Loader      | `pdf` (pypdf, uma página por documento), `text` (.txt/.md) | — |
| Embedder    | `local` (sentence-transformers)         | `hashing` (léxico, sem download) |
| VectorStore | `pgvector` (HNSW, cosseno)              | `memory`                  |
| LLM         | `anthropic` (structured outputs, fallback do servidor) | `extractive`, `auto` |

**Adapter próprio sem fork:** qualquer classe com os métodos do `Protocol` serve. Aponte pelo caminho de import:

```yaml
store:
  provider: meu_pacote.stores:QdrantStore   # recebe as demais chaves como kwargs
  url: http://qdrant:6333
```

ou registre em código: `kit.registry.register("store", "qdrant", QdrantStore)`.

## Resultados

<!-- RESULTS:START -->
Medido com `kit eval` sobre o corpus sintético de `examples/` (8 contratos + 25 notas fiscais, 33 arquivos, PDF e Markdown), embedder `intfloat/multilingual-e5-small` em CPU, dentro do container (`docker compose exec api kit eval --compare-headers --chunk-sizes 500,800`). Os datasets rotulados estão em `evals/datasets/`.

**RAG: hit rate@5** (o trecho que contém a resposta aparece entre os 5 primeiros resultados), 28 perguntas:

| Configuração | Hit rate@5 | MRR | Latência da busca p50 | p95 |
|---|---|---|---|---|
| chunk 500, sem header contextual | 25,0% | 0,129 | 0,013 s | 0,016 s |
| chunk 500, com header contextual | 67,9% | 0,536 | 0,013 s | 0,017 s |
| chunk 800, sem header contextual | 60,7% | 0,280 | 0,014 s | 0,017 s |
| **chunk 800, com header contextual (padrão)** | **82,1%** | **0,661** | **0,013 s** | **0,021 s** |

O que os números mostram:

- **Header contextual é o maior ganho.** Os contratos seguem o mesmo modelo, então o trecho "Cláusula 4 - Rescisão... aviso prévio de 45 dias" é quase idêntico em todos; o que diferencia é o título, que fica em outro chunk. Embutir `arquivo | título` junto com cada chunk (só no embedding, não no texto citado) leva o hit rate de 60,7% para 82,1%.
- **Chunks maiores ajudam neste corpus**, porque documentos curtos e com cláusulas pequenas cabem com mais contexto em 800 caracteres.
- **As 5 falhas restantes** são todas "valor total da nota fiscal número X". Embedding denso é fraco em casar números exatos, e nas notas de 2 páginas o total fica num chunk sem o número. Isso pede busca híbrida (BM25 + vetorial), o próximo item do roadmap.

**Extração: acurácia por campo, latência e custo.** Precisa de um LLM gerativo; ainda não medido aqui por falta de chave de API. Para gerar a tabela (25 notas fiscais rotuladas, comparando sem retry e com retry):

```bash
docker compose exec -e ANTHROPIC_API_KEY=sk-... api kit eval --no-rag --extraction --compare-retry
```

A saída traz acurácia por campo, % de documentos 100% corretos, % com confiança baixa, tentativas médias, latência p50/p95 e custo em US$ por 100 documentos (calculado com o `pricing` do `kit.yaml`), e é salva em `evals/results/`.
<!-- RESULTS:END -->

## Decisões e trade-offs

**Protocols em vez de herança.** `typing.Protocol` é tipagem estrutural: um adapter de terceiros não precisa importar nem herdar nada do kit, basta ter os métodos certos. Isso mantém o core desacoplado e torna os testes triviais (os fakes em `tests/fakes.py` são classes simples). O custo é perder implementações padrão compartilhadas via classe base, que aqui não faziam falta.

**pgvector como store padrão.** A maioria dos projetos já tem Postgres; um vector store dentro dele significa backup, transação e operação que o time já domina, sem mais um serviço. Com índice HNSW o desempenho cobre folgadamente o volume típico de um projeto (milhares a poucos milhões de chunks). Quando passar disso, um store dedicado entra como adapter.

**Embeddings locais.** `multilingual-e5-small` roda em CPU, entende português, não tem custo por chamada e não envia documentos para terceiros na ingestão, que é a etapa de maior volume. O modelo é baixado no build da imagem, então o container sobe offline. O preço é qualidade menor que modelos de API maiores; trocar é uma linha no `kit.yaml`.

**Chunk por caracteres.** Independe do tokenizer do embedder, então o mesmo `kit.yaml` vale para qualquer modelo. O corte procura parágrafo, linha, fim de frase ou espaço na segunda metade da janela, para não partir palavras.

**Extração: structured outputs + validação Pydantic + 1 retry.** O adapter Anthropic restringe a saída ao JSON Schema do modelo; mesmo assim o core valida com Pydantic, porque validators de domínio (ex.: dígitos verificadores de CNPJ em `examples/schemas.py`) não cabem num JSON Schema. Se a validação falhar, os erros voltam ao LLM numa segunda tentativa. A confiança fica `low` quando a validação não passa ou há campos vazios, que é o sinal para revisão humana. Documentos acima de `extraction.max_chars` são rejeitados, nunca truncados em silêncio.

**Citações por índice.** O contexto vai numerado (`<document index="n">`) e o modelo cita `[n]`; o kit converte cada índice citado de volta em arquivo, página e trecho. Índices inventados (fora do contexto) são descartados.

**Evals isolados.** `kit eval` indexa o corpus num store em memória novo, com o embedder configurado, para não poluir o índice de produção e para cada comparação (ex.: tamanho de chunk) partir do zero.

## Desenvolvimento

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest                                    # unitários (pgvector é pulado sem banco)
docker compose up -d db
KIT_TEST_DATABASE_URL=postgresql://kit:kit@localhost:5433/kit pytest   # + integração
ruff check . && ruff format --check . && mypy
python -m examples.generate               # regera o corpus sintético e os datasets
```

Os documentos em `examples/` são **fictícios** (empresas, CNPJs e valores gerados aleatoriamente) e servem de demo e de dataset rotulado para os evals.

Estrutura:

```
src/kit/
├── kit.py            fachada: ingest, query, extract
├── interfaces.py     Protocols: Loader, Embedder, VectorStore, LLM
├── config.py         kit.yaml + .env, validado com Pydantic
├── registry.py       nome no YAML -> adapter (import preguiçoso)
├── chunking.py  ingestion.py  rag.py  extract.py  schemas.py
├── evals.py          hit rate@k, acurácia por campo, latência, custo
├── observability.py  logs JSON + latência
├── api/              FastAPI
├── cli.py            Typer
└── adapters/         loaders/ embedders/ stores/ llms/
```
