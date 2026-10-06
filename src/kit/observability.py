"""Logs estruturados em JSON (uma linha por evento) e medição de latência."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, json_format: bool = True) -> None:
    """Configura o logger raiz. Chamado pela API/CLI; a biblioteca nunca configura sozinha."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if json_format else logging.Formatter("%(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Loga `event` com `duration_ms`. Campos extras podem ser adicionados ao dict retornado."""
    extra: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        extra["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        extra["error"] = type(exc).__name__
        logger.warning(event, extra=extra)
        raise
    extra["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
    logger.info(event, extra=extra)
