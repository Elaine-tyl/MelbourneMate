"""Dense encoder helpers, including an offline test encoder."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from melbourne_mate.config import CONFIG, DenseParams

DIMENSIONS = 256

ENCODER_SPECS = {
    "all-minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "multi-qa-mpnet": (
        "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "d51b22a1dfa8184e9258074e56e2875e50612dca",
    ),
}


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


def encoder_params(name: str) -> DenseParams:
    if name not in ENCODER_SPECS:
        raise ValueError(f"unknown encoder {name!r}")
    model, revision = ENCODER_SPECS[name]
    return replace(CONFIG.dense, model=model, revision=revision)


def load_encoder(name: str):
    """Load an encoder; `hashing` is the smoke-test stand-in."""
    if name == "hashing":
        return HashingEncoder()
    if name in ENCODER_SPECS:
        from melbourne_mate.retrieval.dense import SentenceTransformerEncoder

        return SentenceTransformerEncoder(params=encoder_params(name))
    choices = sorted([*ENCODER_SPECS, "hashing"])
    raise ValueError(f"unknown encoder {name!r}, expected one of {choices}")
