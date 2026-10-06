"""Embeddings locais com ONNX Runtime (extra `kit-rag[onnx]`). Sem torch e sem chave de API.

Usa o export ONNX publicado no próprio repositório do modelo no Hugging Face (por padrão a
versão quantizada int8 do multilingual-e5-small, ~118 MB) e o `tokenizer.json` do mesmo
repositório. O pooling (média pela attention mask + normalização L2) é o mesmo do
sentence-transformers para a família e5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from kit.adapters.embedders.prefixes import default_prefixes

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_FILE = "onnx/model_qint8_avx512_vnni.onnx"
DEFAULT_TOKENIZER = "onnx/tokenizer.json"


def _fetch(repo: str, filename: str, revision: str | None) -> str:
    """Baixa (ou lê do cache) um arquivo do Hub. Respeita HF_HOME e HF_HUB_OFFLINE."""
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(repo, filename, revision=revision))


def mean_pool(hidden: np.ndarray, mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Média dos vetores de token ignorando padding; opcionalmente normaliza (L2)."""
    weights = mask[..., None].astype(np.float32)
    summed = (hidden * weights).sum(axis=1)
    pooled: np.ndarray = summed / np.clip(weights.sum(axis=1), 1e-9, None)
    if normalize:
        pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    return pooled


class OnnxEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        file: str = DEFAULT_FILE,
        tokenizer_file: str = DEFAULT_TOKENIZER,
        revision: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        threads: int | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        normalize: bool = True,
        session: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        default_query, default_passage = default_prefixes(model)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.passage_prefix = default_passage if passage_prefix is None else passage_prefix

        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(_fetch(model, tokenizer_file, revision))
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding()
        self._tokenizer = tokenizer

        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            if threads:
                options.intra_op_num_threads = threads
            session = ort.InferenceSession(
                _fetch(model, file, revision),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self._session = session
        self._input_names = {i.name for i in session.get_inputs()}
        self._dimension = int(self._encode(["dimensão"]).shape[1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            encodings = self._tokenizer.encode_batch(texts[start : start + self.batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]  # last_hidden_state
            outputs.append(mean_pool(hidden, mask, self.normalize))
        return np.concatenate(outputs) if outputs else np.empty((0, 0), dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._encode([self.passage_prefix + t for t in texts])
        return [row.tolist() for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        result: list[float] = self._encode([self.query_prefix + text])[0].tolist()
        return result
