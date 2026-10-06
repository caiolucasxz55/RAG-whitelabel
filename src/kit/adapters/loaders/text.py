from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from kit.types import Document

DEFAULT_EXTENSIONS = (".txt", ".md", ".markdown")


class TextLoader:
    """Arquivos de texto puro e Markdown. Tenta UTF-8 e cai para Latin-1."""

    def __init__(self, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> None:
        self.extensions = frozenset(e.lower() for e in extensions)

    def load(self, path: Path) -> list[Document]:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        if not text.strip():
            return []
        return [Document(text=text, source=path.name, metadata={"path": str(path)})]
