"""API HTTP: camada fina sobre o Kit. Toda regra de negócio fica no core.

uvicorn kit.api.app:create_app --factory
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from kit import Kit
from kit.api.schemas import ErrorOut, ExtractOut, HealthOut, IngestOut, QueryIn, QueryOut
from kit.errors import (
    CapabilityError,
    ConfigError,
    DocumentTooLargeError,
    KitError,
    LLMError,
    LLMRefusalError,
    StoreError,
    UnsupportedFileError,
)
from kit.observability import configure_logging

logger = logging.getLogger("kit.api")

# Mais específico primeiro: LLMRefusalError é subclasse de LLMError.
_STATUS: list[tuple[type[KitError], int]] = [
    (UnsupportedFileError, 415),
    (DocumentTooLargeError, 413),
    (CapabilityError, 501),
    (LLMRefusalError, 422),
    (LLMError, 502),
    (StoreError, 503),
    (ConfigError, 500),
]
_SAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)
MAX_UPLOAD_BYTES = int(os.environ.get("KIT_MAX_UPLOAD_MB", "25")) * 1024 * 1024


def _status_for(exc: KitError) -> int:
    return next((code for cls, code in _STATUS if isinstance(exc, cls)), 500)


def _safe_filename(name: str | None) -> str:
    """Remove diretórios e caracteres estranhos do nome enviado pelo cliente."""
    base = _SAFE_NAME.sub("_", Path(name or "").name).strip(" .")
    if not base:
        raise HTTPException(400, "arquivo sem nome")
    return base


async def _save_upload(upload: UploadFile, directory: Path) -> Path:
    path = directory / _safe_filename(upload.filename)
    size = 0
    with path.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"arquivo excede {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            out.write(chunk)
    return path


def create_app(kit: Kit | None = None, *, config_path: str | None = None) -> FastAPI:
    """Fábrica da aplicação. Testes injetam um Kit pronto; em produção ele é criado no
    startup a partir do kit.yaml ($KIT_CONFIG)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(os.environ.get("KIT_LOG_LEVEL", "INFO"))
        app.state.kit = kit or await run_in_threadpool(Kit.from_config, config_path)
        logger.info("api pronta", extra={"chunks": app.state.kit.store.count()})
        yield
        close = getattr(app.state.kit.store, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="kit",
        version="0.1.0",
        description="RAG com citações e extração estruturada de documentos.",
        lifespan=lifespan,
        responses={500: {"model": ErrorOut}},
    )

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    @app.exception_handler(KitError)
    async def kit_error(_: Request, exc: KitError) -> JSONResponse:
        status = _status_for(exc)
        if status >= 500:
            logger.error("erro do kit", extra={"error": type(exc).__name__, "detail": str(exc)})
        body = ErrorOut(error=type(exc).__name__, detail=str(exc))
        return JSONResponse(body.model_dump(), status_code=status)

    def get_kit(request: Request) -> Kit:
        k: Kit = request.app.state.kit
        return k

    @app.get("/health", response_model=HealthOut)
    async def health(request: Request) -> HealthOut:
        k = get_kit(request)
        cfg = k.config
        return HealthOut(
            status="ok",
            chunks=await run_in_threadpool(k.store.count),
            llm=type(k.llm).__name__,
            embedder=cfg.embedder.provider,
            store=cfg.store.provider,
        )

    @app.post("/ingest", response_model=IngestOut)
    async def ingest(request: Request, files: list[UploadFile] = File(...)) -> IngestOut:
        k = get_kit(request)
        with tempfile.TemporaryDirectory(prefix="kit-upload-") as tmp:
            paths = [await _save_upload(f, Path(tmp)) for f in files]
            report = await run_in_threadpool(k.ingest, paths)
        return IngestOut.of(report)

    @app.post("/query", response_model=QueryOut)
    async def query(request: Request, body: QueryIn) -> QueryOut:
        try:
            result = await run_in_threadpool(get_kit(request).query, body.question, body.top_k)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return QueryOut.of(result)

    @app.get("/schemas")
    async def schemas(request: Request) -> dict[str, dict[str, object]]:
        registry = get_kit(request).schemas
        return {name: registry.get(name).model_json_schema() for name in registry.names()}

    @app.post("/extract", response_model=ExtractOut)
    async def extract(
        request: Request,
        file: UploadFile = File(...),
        schema_name: str = Form(
            ..., alias="schema", description="Nome de um schema configurado em kit.yaml"
        ),
        instructions: str | None = Form(None),
    ) -> ExtractOut:
        k = get_kit(request)
        try:
            model = k.schemas.get(schema_name)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0])) from exc
        with tempfile.TemporaryDirectory(prefix="kit-upload-") as tmp:
            path = await _save_upload(file, Path(tmp))
            result = await run_in_threadpool(k.extract, path, model, instructions=instructions)
        return ExtractOut.of(schema_name, result)

    return app
