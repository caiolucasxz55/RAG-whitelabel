from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from kit.errors import UnsupportedFileError
from kit.types import Document


class PdfLoader:
    """Um Document por página, com `page` (1-based) nos metadados para citar a página.

    Só extrai a camada de texto; PDFs escaneados (imagem) precisam de OCR, fora do MVP.
    """

    extensions = frozenset({".pdf"})

    def load(self, path: Path) -> list[Document]:
        try:
            reader = PdfReader(path)
        except PdfReadError as exc:
            raise UnsupportedFileError(f"PDF ilegível: {path.name} ({exc})") from exc
        documents = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        text=text,
                        source=path.name,
                        metadata={"path": str(path), "page": number},
                    )
                )
        return documents
