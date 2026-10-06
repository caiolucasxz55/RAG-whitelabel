# syntax=docker/dockerfile:1

# ---------- build: venv com dependências e modelo de embeddings ----------
# Sem torch: embeddings via ONNX Runtime + modelo quantizado int8 (extra `server`).
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
# Instala as dependências antes do código para aproveitar o cache de camadas.
COPY pyproject.toml README.md ./
RUN mkdir -p src/kit && touch src/kit/__init__.py \
    && pip install ".[server]" \
    && pip uninstall -y kit-rag

COPY src ./src
RUN pip install --no-deps . && pip uninstall -y pip

# Baixa no build só o ONNX quantizado + tokenizer (~135 MB): o container sobe offline e
# sem espera. Para outro modelo, ajuste estes ARGs e o kit.yaml juntos.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG EMBEDDING_FILE=onnx/model_qint8_avx512_vnni.onnx
ARG EMBEDDING_TOKENIZER=onnx/tokenizer.json
RUN python -c "from huggingface_hub import hf_hub_download as get; \
[get('${EMBEDDING_MODEL}', f) for f in ('${EMBEDDING_FILE}', '${EMBEDDING_TOKENIZER}')]"

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
