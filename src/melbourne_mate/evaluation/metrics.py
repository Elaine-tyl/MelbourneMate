"""Compute graded retrieval metrics. Missing results receive a zero score."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from melbourne_mate.config import CONFIG


def _gains(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int) -> list[int]:
    return [judgements.get(pid, 0) for pid in list(ranked_ids)[:k]]


def gain(grade: int, kind: str | None = None) -> float:
    """Return exponential or linear gain for one relevance grade."""
    kind = kind or CONFIG.evaluation.ndcg_gain
    if kind == "exponential":
        return float(2**grade - 1)
    if kind == "linear":
        return float(grade)
    raise ValueError(f"unknown ndcg gain {kind!r}, expected 'exponential' or 'linear'")


def dcg(gains: Sequence[int], kind: str | None = None) -> float:
    return sum(gain(g, kind) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    if not judgements:
        raise ValueError("ndcg is undefined for a question with no relevant passages")
    ideal = sorted(judgements.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    if denominator == 0:
        return 0.0
    return dcg(_gains(ranked_ids, judgements, k)) / denominator


def recall_at_k(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    if not judgements:
        raise ValueError("recall is undefined for a question with no relevant passages")
    retrieved = set(list(ranked_ids)[:k])
    found = sum(1 for pid in judgements if pid in retrieved)
    return found / len(judgements)


def mrr(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int | None = None) -> float:
    ids = list(ranked_ids) if k is None else list(ranked_ids)[:k]
    for rank, pid in enumerate(ids, start=1):
        if judgements.get(pid, 0) > 0:
            return 1.0 / rank
    return 0.0


def retrieval_failure(
    ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int
) -> float:
    """1.0 when nothing relevant appears in the top k."""
    return 0.0 if any(judgements.get(pid, 0) > 0 for pid in list(ranked_ids)[:k]) else 1.0


METRICS = ("ndcg@1", "ndcg@3", "ndcg@5", "recall@5", "mrr", "retrieval_failure@5")


def score_question(
    ranked_ids: Sequence[str], judgements: Mapping[str, int]
) -> dict[str, float]:
    return {
        "ndcg@1": ndcg_at_k(ranked_ids, judgements, 1),
        "ndcg@3": ndcg_at_k(ranked_ids, judgements, 3),
        "ndcg@5": ndcg_at_k(ranked_ids, judgements, 5),
        "recall@5": recall_at_k(ranked_ids, judgements, 5),
        "mrr": mrr(ranked_ids, judgements),
        "retrieval_failure@5": retrieval_failure(ranked_ids, judgements, 5),
    }


def score_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    question_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Score every requested question, including missing rankings."""
    scored: dict[str, dict[str, float]] = {}
    for question_id in question_ids:
        judgements = qrels.get(question_id)
        if not judgements:
            raise ValueError(f"{question_id}: no relevance judgements")
        scored[question_id] = score_question(run.get(question_id, []), judgements)
    return scored


def aggregate(per_question: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_question:
        return {metric: 0.0 for metric in METRICS}
    return {
        metric: sum(row[metric] for row in per_question.values()) / len(per_question)
        for metric in METRICS
    }
