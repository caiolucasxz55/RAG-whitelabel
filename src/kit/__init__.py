"""kit: RAG com citações e extração estruturada, com provedores trocáveis por configuração."""

from kit.errors import KitError
from kit.kit import Kit
from kit.types import ExtractionResult, IngestReport, QueryResult, Source

__all__ = ["ExtractionResult", "IngestReport", "Kit", "KitError", "QueryResult", "Source"]
__version__ = "0.1.0"
