"""Pipeline de ingestão: arquivos -> Documents -> Chunks -> embeddings -> VectorStore."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from kit.chunking import Chunker
from kit.errors import UnsupportedFileError
from kit.interfaces import Embedder, Loader, VectorStore
from kit.types import Document, IngestReport

logger = logging.getLogger(__name__)

PathLike = str | Path


class LoaderSet:
    """Escolhe o loader pela extensão do arquivo."""

    def __init__(self, loaders: Sequence[Loader]) -> None:
        self._by_ext: dict[str, Loader] = {}
        for loader in loaders:
            for ext in loader.extensions:
                self._by_ext.setdefault(ext.lower(), loader)

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self._by_ext

    def load(self, path: Path, source: str | None = None) -> list[Document]:
        loader = self._by_ext.get(path.suffix.lower())
        if loader is None:
            supported = ", ".join(sorted(self._by_ext)) or "nenhuma"
            raise UnsupportedFileError(
                f"nenhum loader para '{path.suffix}' ({path.name}); suportadas: {supported}"
            )
        documents = loader.load(path)
        if source is not None:
            documents = [replace(d, source=source) for d in documents]
        return documents


def _walk(target: Path) -> list[tuple[Path, str]]:
    """Expande diretórios recursivamente. O `source` de cada arquivo é o caminho relativo
    à raiz informada, o que evita colisão entre arquivos homônimos em pastas diferentes."""
    if target.is_dir():
        return [
            (p, p.relative_to(target).as_posix())
            for p in sorted(target.rglob("*"))
            if p.is_file() and not p.name.startswith(".")
        ]
    if target.is_file():
        return [(target, target.name)]
    raise FileNotFoundError(f"caminho não encontrado: {target}")


class IngestionService:
    def __init__(
        self,
        loaders: LoaderSet,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        batch_size: int = 64,
    ) -> None:
        self.loaders = loaders
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    def ingest(self, paths: PathLike | Iterable[PathLike]) -> IngestReport:
        targets = [paths] if isinstance(paths, str | Path) else list(paths)
        files: list[str] = []
        skipped: list[str] = []
        total_docs = total_chunks = 0
        for target in map(Path, targets):
            explicit_file = target.is_file()
            for path, source in _walk(target):
                if not explicit_file and not self.loaders.supports(path):
                    skipped.append(source)
                    continue
                docs, chunks = self.ingest_file(path, source=source)
                files.append(source)
                total_docs += docs
                total_chunks += chunks
        return IngestReport(files=files, documents=total_docs, chunks=total_chunks, skipped=skipped)

    def ingest_file(self, path: Path, *, source: str | None = None) -> tuple[int, int]:
        """Reingerir o mesmo `source` substitui os chunks antigos (idempotente)."""
        documents = self.loaders.load(path, source=source)
        chunks = self.chunker.chunk_documents(documents)
        source_id = source or path.name
        self.store.delete_source(source_id)
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            self.store.add(batch, self.embedder.embed_documents([c.text for c in batch]))
        logger.info(
            "arquivo ingerido",
            extra={"source": source_id, "documents": len(documents), "chunks": len(chunks)},
        )
        return len(documents), len(chunks)
