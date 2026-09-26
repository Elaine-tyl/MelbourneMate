"""Pool top results from each retriever for independent relevance review."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from melbourne_mate.corpus import Collection
from melbourne_mate.retrieval.base import Retriever


@dataclass(frozen=True)
class PoolEntry:
    question_id: str
    passage_id: str
    arms: tuple[str, ...]  # which arms surfaced it
    best_rank: int

    @property
    def unique_to_one_arm(self) -> bool:
        return len(self.arms) == 1


def build_pool(
    collection: Collection,
    retrievers: Mapping[str, Retriever],
    depth: int = 10,
    question_ids: Sequence[str] | None = None,
) -> list[PoolEntry]:
    """Union the top `depth` results of every arm, per question."""
    if not retrievers:
        raise ValueError("pooling needs at least one retriever")

    ids = list(question_ids) if question_ids is not None else sorted(collection.questions)
    entries: list[PoolEntry] = []

    for question_id in ids:
        question = collection.questions[question_id]
        found: dict[str, tuple[set[str], int]] = {}
        for arm_name, retriever in retrievers.items():
            for hit in retriever.search(question.text, depth):
                arms, best = found.get(hit.passage_id, (set(), hit.rank))
                arms.add(arm_name)
                found[hit.passage_id] = (arms, min(best, hit.rank))
        for passage_id, (arms, best_rank) in sorted(
            found.items(), key=lambda item: (item[1][1], item[0])
        ):
            entries.append(
                PoolEntry(question_id, passage_id, tuple(sorted(arms)), best_rank)
            )
    return entries


def write_pool(path: str | Path, entries: Sequence[PoolEntry]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "passage_id", "arms", "best_rank"])
        for entry in entries:
            writer.writerow(
                [entry.question_id, entry.passage_id, "+".join(entry.arms), entry.best_rank]
            )
    return target


def read_pool(path: str | Path) -> list[PoolEntry]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            PoolEntry(
                row["question_id"],
                row["passage_id"],
                tuple(row["arms"].split("+")) if row["arms"] else (),
                int(row["best_rank"]),
            )
            for row in csv.DictReader(handle)
        ]


def pool_coverage(
    entries: Sequence[PoolEntry], qrels: Mapping[str, Mapping[str, int]]
) -> float:
    """Return the share of relevant passages included in the pool."""
    pooled = {(entry.question_id, entry.passage_id) for entry in entries}
    relevant = {
        (question_id, passage_id)
        for question_id, graded in qrels.items()
        for passage_id, grade in graded.items()
        if grade > 0
    }
    if not relevant:
        return 0.0
    return len(relevant & pooled) / len(relevant)
