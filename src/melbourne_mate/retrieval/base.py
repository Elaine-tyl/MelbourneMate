"""Shared retrieval result and retriever interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    passage_id: str
    rank: int  # 1-based
    score: float


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]:
        """Return at most k hits, best first, ranks starting at 1."""
        ...
