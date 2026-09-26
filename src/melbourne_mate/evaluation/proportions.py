"""Estimate topic-clustered confidence intervals and paired rate differences."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from melbourne_mate.config import CONFIG


@dataclass(frozen=True)
class RateEstimate:
    rate: float
    ci_low: float
    ci_high: float
    n: int
    clusters: int

    def __str__(self) -> str:
        return f"{self.rate:.1%} [{self.ci_low:.1%}, {self.ci_high:.1%}] (n={self.n})"


@dataclass(frozen=True)
class RateDifference:
    left: float
    right: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    discordant: int  # questions where the two arms disagree
    n: int
    clusters: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _by_cluster(
    values: Mapping[str, bool], clusters: Mapping[str, str]
) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for question_id, value in values.items():
        grouped.setdefault(clusters.get(question_id, question_id), []).append(value)
    return grouped


def rate_ci(
    values: Mapping[str, bool],
    clusters: Mapping[str, str],
    resamples: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> RateEstimate:
    """Cluster bootstrap interval for the share of True values."""
    if not values:
        return RateEstimate(0.0, 0.0, 0.0, 0, 0)

    resamples = CONFIG.evaluation.bootstrap_resamples if resamples is None else resamples
    rng = random.Random(CONFIG.evaluation.seed if seed is None else seed)
    grouped = _by_cluster(values, clusters)
    keys = list(grouped)

    def rate_of(selected: Sequence[str]) -> float:
        flat = [v for key in selected for v in grouped[key]]
        return sum(flat) / len(flat) if flat else 0.0

    observed = rate_of(keys)
    draws = sorted(
        rate_of([keys[rng.randrange(len(keys))] for _ in range(len(keys))])
        for _ in range(resamples)
    )
    low = draws[max(0, int((alpha / 2) * resamples) - 1)]
    high = draws[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return RateEstimate(observed, low, high, len(values), len(keys))


def paired_rate_difference(
    left: Mapping[str, bool],
    right: Mapping[str, bool],
    clusters: Mapping[str, str],
    resamples: int | None = None,
    permutations: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> RateDifference:
    """Compare two arms on the questions both arms answered."""
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("the two arms share no questions")

    resamples = CONFIG.evaluation.bootstrap_resamples if resamples is None else resamples
    permutations = (
        CONFIG.evaluation.randomisation_permutations if permutations is None else permutations
    )
    rng = random.Random(CONFIG.evaluation.seed if seed is None else seed)

    differences = {q: float(right[q]) - float(left[q]) for q in shared}
    grouped: dict[str, list[float]] = {}
    for question_id, value in differences.items():
        grouped.setdefault(clusters.get(question_id, question_id), []).append(value)
    keys = list(grouped)

    def mean_of(selected: Sequence[str]) -> float:
        flat = [v for key in selected for v in grouped[key]]
        return sum(flat) / len(flat) if flat else 0.0

    observed = mean_of(keys)
    draws = sorted(
        mean_of([keys[rng.randrange(len(keys))] for _ in range(len(keys))])
        for _ in range(resamples)
    )
    low = draws[max(0, int((alpha / 2) * resamples) - 1)]
    high = draws[min(resamples - 1, int((1 - alpha / 2) * resamples))]

    total = len(shared)
    extreme = 0
    for _ in range(permutations):
        flipped = sum(
            (1 if rng.random() < 0.5 else -1) * sum(grouped[key]) for key in keys
        )
        if abs(flipped / total) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)

    left_rate = sum(left[q] for q in shared) / total
    right_rate = sum(right[q] for q in shared) / total
    return RateDifference(
        left=left_rate,
        right=right_rate,
        difference=observed,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        discordant=sum(1 for q in shared if left[q] != right[q]),
        n=total,
        clusters=len(keys),
    )
