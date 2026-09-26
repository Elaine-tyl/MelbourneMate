"""Compare systems with topic-clustered bootstrap and randomisation tests."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from melbourne_mate.config import CONFIG


@dataclass(frozen=True)
class ComparisonResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    clusters: int
    observations: int
    resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _cluster_means(
    differences: Mapping[str, float], clusters: Mapping[str, str]
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for question_id, value in differences.items():
        cluster = clusters.get(question_id, question_id)
        grouped.setdefault(cluster, []).append(value)
    return grouped


def cluster_bootstrap_ci(
    differences: Mapping[str, float],
    clusters: Mapping[str, str],
    resamples: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over clusters. Returns (mean, low, high)."""
    grouped = _cluster_means(differences, clusters)
    if not grouped:
        raise ValueError("no differences to bootstrap")

    resamples = CONFIG.evaluation.bootstrap_resamples if resamples is None else resamples
    rng = random.Random(CONFIG.evaluation.seed if seed is None else seed)
    keys = list(grouped)

    def mean_of(selected: Sequence[str]) -> float:
        values = [value for key in selected for value in grouped[key]]
        return sum(values) / len(values)

    observed = mean_of(keys)
    draws = sorted(
        mean_of([keys[rng.randrange(len(keys))] for _ in range(len(keys))])
        for _ in range(resamples)
    )
    low = draws[max(0, int((alpha / 2) * resamples) - 1)]
    high = draws[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return observed, low, high


def paired_randomisation_test(
    differences: Mapping[str, float],
    clusters: Mapping[str, str],
    permutations: int | None = None,
    seed: int | None = None,
) -> float:
    """Two-sided paired randomisation test, flipping the sign of whole clusters."""
    grouped = _cluster_means(differences, clusters)
    if not grouped:
        raise ValueError("no differences to test")

    permutations = (
        CONFIG.evaluation.randomisation_permutations if permutations is None else permutations
    )
    rng = random.Random(CONFIG.evaluation.seed if seed is None else seed)
    keys = list(grouped)
    total = sum(len(values) for values in grouped.values())
    observed = abs(sum(sum(grouped[key]) for key in keys) / total)

    extreme = 0
    for _ in range(permutations):
        flipped = sum(
            (1 if rng.random() < 0.5 else -1) * sum(grouped[key]) for key in keys
        )
        if abs(flipped / total) >= observed:
            extreme += 1
    # A finite test cannot support a p-value of exactly zero.
    return (extreme + 1) / (permutations + 1)


def compare(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    clusters: Mapping[str, str],
    metric: str | None = None,
    resamples: int | None = None,
    permutations: int | None = None,
    seed: int | None = None,
) -> ComparisonResult:
    """Compare two systems on one metric. Difference is right minus left."""
    metric = metric or CONFIG.evaluation.primary_metric
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("the two runs share no questions")
    if set(left) != set(right):
        raise ValueError("runs must cover exactly the same questions")

    differences = {qid: right[qid][metric] - left[qid][metric] for qid in shared}
    mean, low, high = cluster_bootstrap_ci(differences, clusters, resamples, seed)
    p_value = paired_randomisation_test(differences, clusters, permutations, seed)
    return ComparisonResult(
        mean_difference=mean,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        clusters=len({clusters.get(q, q) for q in shared}),
        observations=len(shared),
        resamples=resamples or CONFIG.evaluation.bootstrap_resamples,
        seed=seed if seed is not None else CONFIG.evaluation.seed,
    )
