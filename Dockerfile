# syntax=docker/dockerfile:1

# ---------- build: venv com dependências e modelo de embeddings ----------
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch só CPU: a imagem fica ~2 GB menor que com CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
# Instala as dependências antes do código para aproveitar o cache de camadas.
COPY pyproject.toml README.md ./
RUN mkdir -p src/kit && touch src/kit/__init__.py \
    && pip install ".[all]" \
    && pip uninstall -y kit-rag

COPY src ./src
RUN pip install --no-deps .

# Baixa o modelo no build: o container sobe offline e sem espera.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    KIT_CONFIG=/app/kit.yaml

RUN useradd --create-home --uid 1000 kit
WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
COPY kit.example.yaml ./kit.yaml
COPY examples ./examples
COPY evals ./evals

RUN chown -R kit:kit /app
USER kit

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["kit", "serve", "--host", "0.0.0.0", "--port", "8000"]
