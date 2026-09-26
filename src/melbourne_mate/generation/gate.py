"""Decide whether retrieved evidence is strong enough to answer.

The same dense scorer is used for both retrieval arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from melbourne_mate.config import CONFIG


class SufficiencyScorer(Protocol):
    def encode_queries(self, texts: Sequence[str]): ...

    def encode_passages(self, texts: Sequence[str]): ...


@dataclass(frozen=True)
class GateDecision:
    sufficient: bool
    top_score: float
    supporting: int
    reason: str


class EvidenceGate:
    def __init__(
        self,
        scorer: SufficiencyScorer,
        min_top_score: float | None = None,
        min_supporting_hits: int | None = None,
    ) -> None:
        self.scorer = scorer
        self.min_top_score = (
            CONFIG.gate.min_top_score if min_top_score is None else min_top_score
        )
        self.min_supporting_hits = (
            CONFIG.gate.min_supporting_hits
            if min_supporting_hits is None
            else min_supporting_hits
        )

    def decide(self, question: str, passage_texts: Sequence[str]) -> GateDecision:
        if not passage_texts:
            return GateDecision(False, 0.0, 0, "no evidence retrieved")

        import numpy as np

        query_vector = np.asarray(self.scorer.encode_queries([question]))[0]
        passage_matrix = np.asarray(self.scorer.encode_passages(list(passage_texts)))
        scores = passage_matrix @ query_vector

        top_score = float(scores.max())
        supporting = int((scores >= self.min_top_score).sum())

        if top_score < self.min_top_score:
            return GateDecision(
                False,
                top_score,
                supporting,
                f"best evidence scored {top_score:.3f} < {self.min_top_score:.3f}",
            )
        if supporting < self.min_supporting_hits:
            return GateDecision(
                False,
                top_score,
                supporting,
                f"{supporting} supporting passages < {self.min_supporting_hits}",
            )
        return GateDecision(True, top_score, supporting, "sufficient evidence")
