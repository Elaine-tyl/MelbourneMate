"""Dense encoder helpers, including an offline test encoder."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

DIMENSIONS = 256


def _vectorise(text: str, prefix: str) -> np.ndarray:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    normalised = f"{prefix}{text}".lower()
    for size in (3, 4):
        for i in range(max(0, len(normalised) - size + 1)):
            gram = normalised[i : i + size]
            index = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16) % DIMENSIONS
            vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class HashingEncoder:
    """Offline stand-in with the same interface as the dense encoder."""

    model_name = "hashing-offline"

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([_vectorise(text, "") for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([_vectorise(text, "") for text in texts])


def load_encoder(name: str):
    """Load an encoder; `hashing` is the smoke-test stand-in."""
    if name == "hashing":
        return HashingEncoder()
    models = {
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "multi-qa-mpnet": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    }
    if name in models:
        from melbourne_mate.config import CONFIG
        from melbourne_mate.retrieval.dense import SentenceTransformerEncoder

        revision = CONFIG.dense.revision if name == "multi-qa-mpnet" else ""
        return SentenceTransformerEncoder(
            params=replace(CONFIG.dense, model=models[name], revision=revision),
        )
    raise ValueError(f"unknown encoder {name!r}, expected one of {sorted([*models, 'hashing'])}")
