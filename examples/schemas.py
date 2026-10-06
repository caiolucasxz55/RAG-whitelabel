"""Schemas de exemplo. Um projeto cliente define os seus e os aponta no kit.yaml:

schemas:
  nota_fiscal: examples.schemas:NotaFiscal
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator


def cnpj_is_valid(digits: str) -> bool:
    if len(digits) != 14 or digits == digits[0] * 14:
        return False

    def check(base: str) -> int:
        weights = list(range(len(base) - 7, 1, -1)) + list(range(9, 1, -1))
        total = sum(int(d) * w for d, w in zip(base, weights, strict=True))
        rest = total % 11
        return 0 if rest < 2 else 11 - rest

    return check(digits[:12]) == int(digits[12]) and check(digits[:13]) == int(digits[13])


def normalize_cnpj(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if not cnpj_is_valid(digits):
        raise ValueError(f"CNPJ inválido: {value!r} (14 dígitos com verificadores corretos)")
    return digits


class ItemNota(BaseModel):
    descricao: str
    quantidade: float
    valor_unitario: float = Field(description="Valor unitário em reais")


class NotaFiscal(BaseModel):
    numero: str = Field(description="Número da nota, só dígitos, sem zeros à esquerda")
    data_emissao: date
    cnpj_emitente: str = Field(description="CNPJ do emitente (14 dígitos)")
    razao_social_emitente: str
    cnpj_destinatario: str = Field(description="CNPJ do destinatário (14 dígitos)")
    valor_total: float = Field(description="Valor total da nota em reais")
    itens: list[ItemNota]

    @field_validator("cnpj_emitente", "cnpj_destinatario")
    @classmethod
    def _cnpj(cls, value: str) -> str:
        return normalize_cnpj(value)

    @field_validator("numero")
    @classmethod
    def _numero(cls, value: str) -> str:
        digits = re.sub(r"\D", "", value).lstrip("0")
        if not digits:
            raise ValueError("número da nota precisa conter dígitos")
        return digits


class Contrato(BaseModel):
    contratante: str
    contratada: str
    valor_mensal: float = Field(description="Valor mensal em reais")
    prazo_aviso_previo_dias: int = Field(description="Dias de aviso prévio para rescisão")
    foro: str = Field(description="Cidade do foro eleito")
    data_assinatura: date
