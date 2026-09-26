"""Dense retrieval with a configurable Sentence Transformers bi-encoder.

The frozen encoder is multi-qa-mpnet-base-cos-v1. Query and passage vectors
are L2-normalised, so their dot product is cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from melbourne_mate.config import CONFIG
from melbourne_mate.retrieval.base import Hit


class Encoder(Protocol):
    """Interface for text encoders used by dense retrieval."""

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Sentence Transformers wrapper driven by the frozen dense parameters."""

    def __init__(self, model_name: str | None = None, device: str | None = None, params=None) -> None:
        from sentence_transformers import SentenceTransformer

        self.params = params or CONFIG.dense
        self.model_name = model_name or self.params.model
        self.revision = self.params.revision
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            revision=self.revision or None,
        )
        self._model.max_seq_length = self.params.max_seq_length

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prefixed = [f"{prefix}{text}" for text in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=self.params.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.params.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.params.passage_prefix)


class DenseRetriever:
    """Bi-encoder retriever with pre-computed passage vectors."""

    name = "dense"

    def __init__(self, passage_ids: list[str], texts: list[str], encoder: Encoder) -> None:
        if len(passage_ids) != len(texts):
            raise ValueError("passage_ids and texts must be the same length")
        if not passage_ids:
            raise ValueError("cannot index an empty collection")
        self.passage_ids = list(passage_ids)
        self.encoder = encoder
        self._matrix = np.asarray(encoder.encode_passages(list(texts)), dtype=np.float32)
        if self._matrix.shape[0] != len(passage_ids):
            raise ValueError("encoder returned the wrong number of passage vectors")

    def similarities(self, query: str) -> np.ndarray:
        vector = np.asarray(self.encoder.encode_queries([query]), dtype=np.float32)[0]
        return self._matrix @ vector

    def search(self, query: str, k: int) -> list[Hit]:
        k = max(1, min(k, len(self.passage_ids)))
        scores = self.similarities(query)
        # Break equal scores with the passage id.
        order = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), self.passage_ids[i]),
        )[:k]
        return [
            Hit(self.passage_ids[index], rank, float(scores[index]))
            for rank, index in enumerate(order, start=1)
        ]
