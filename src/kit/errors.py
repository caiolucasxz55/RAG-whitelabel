"""Hierarquia de erros do kit. Adapters traduzem erros do provedor para estes tipos."""


class KitError(Exception):
    """Base de todos os erros do kit."""


class ConfigError(KitError):
    """kit.yaml inválido, variável de ambiente ausente ou provedor desconhecido."""


class UnsupportedFileError(KitError):
    """Nenhum loader configurado aceita o arquivo."""


class DocumentTooLargeError(KitError):
    """Documento excede o limite configurado para uma única chamada ao LLM."""


class LLMError(KitError):
    """Falha ao chamar o LLM (rede, autenticação, limite de taxa...)."""


class LLMRefusalError(LLMError):
    """O LLM recusou a requisição."""


class CapabilityError(KitError):
    """O adapter configurado não oferece a capacidade pedida (ex.: extração sem LLM gerativo)."""


class StoreError(KitError):
    """Falha no vector store (conexão, dimensão incompatível...)."""
