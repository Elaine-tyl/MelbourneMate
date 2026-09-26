"""Measure intrusion and displacement between easily confused topics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from melbourne_mate.corpus import Collection


@dataclass(frozen=True)
class PairResult:
    topic_a: str
    topic_b: str
    reason: str
    questions: int
    intrusions: int
    displacements: int

    @property
    def intrusion_rate(self) -> float:
        return self.intrusions / self.questions if self.questions else 0.0

    @property
    def displacement_rate(self) -> float:
        return self.displacements / self.questions if self.questions else 0.0


def _passages_by_topic(collection: Collection) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for passage in collection.passages.values():
        out.setdefault(passage.topic_id, set()).add(passage.passage_id)
    return out


def _one_direction(
    collection: Collection,
    rankings: Mapping[str, Sequence[str]],
    topic_a: str,
    topic_b: str,
    by_topic: Mapping[str, set[str]],
    k: int,
) -> tuple[int, int, int]:
    questions = intrusions = displacements = 0
    other = by_topic.get(topic_b, set())

    for question in collection.questions.values():
        if question.topic_id != topic_a or question.is_ookb:
            continue
        ranked = list(rankings.get(question.question_id, []))[:k]
        if not ranked:
            continue
        questions += 1

        relevant = {
            pid
            for pid, grade in collection.qrels.get(question.question_id, {}).items()
            if grade > 0
        }
        intruder_ranks = [i for i, pid in enumerate(ranked) if pid in other]
        if not intruder_ranks:
            continue
        intrusions += 1

        relevant_ranks = [i for i, pid in enumerate(ranked) if pid in relevant]
        # Displacement means the nearby topic ranked above all correct passages.
        if not relevant_ranks or min(intruder_ranks) < min(relevant_ranks):
            displacements += 1

    return questions, intrusions, displacements


def analyse_pairs(
    collection: Collection,
    rankings: Mapping[str, Sequence[str]],
    k: int = 5,
) -> list[PairResult]:
    """One result per pair, counting both directions together."""
    by_topic = _passages_by_topic(collection)
    results = []
    for pair in collection.pairs:
        totals = [0, 0, 0]
        for a, b in ((pair.topic_a, pair.topic_b), (pair.topic_b, pair.topic_a)):
            counts = _one_direction(collection, rankings, a, b, by_topic, k)
            totals = [total + count for total, count in zip(totals, counts)]
        results.append(
            PairResult(
                topic_a=pair.topic_a,
                topic_b=pair.topic_b,
                reason=pair.reason,
                questions=totals[0],
                intrusions=totals[1],
                displacements=totals[2],
            )
        )
    return results
