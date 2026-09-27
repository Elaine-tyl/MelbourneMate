"""Load the frozen generation sample and checked qrels."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from melbourne_mate.corpus import (
    RISK_CATEGORIES,
    Collection,
    CollectionError,
    load_qrels,
)


class GenerationInputError(ValueError):
    """Raised when formal generation inputs do not match the collection."""


@dataclass(frozen=True)
class GenerationCase:
    question_id: str
    risk_category: str


def load_generation_cases(
    path: str | Path, collection: Collection
) -> tuple[GenerationCase, ...]:
    """Load and check the frozen question sample."""
    sample_path = Path(path)
    seen: set[str] = set()
    cases: list[GenerationCase] = []

    with sample_path.open(newline="", encoding="utf-8") as handle:
        for row_no, row in enumerate(csv.DictReader(handle), start=2):
            question_id = (row.get("question_id") or "").strip()
            risk = (row.get("risk_category") or "").strip()
            if question_id in seen:
                raise GenerationInputError(f"duplicate question_id {question_id}")
            if question_id not in collection.questions:
                raise GenerationInputError(f"unknown question_id {question_id}")
            if risk not in RISK_CATEGORIES:
                raise GenerationInputError(
                    f"invalid risk_category {risk} at row {row_no}"
                )
            seen.add(question_id)
            cases.append(GenerationCase(question_id, risk))

    if not cases:
        raise GenerationInputError("generation sample is empty")
    return tuple(cases)


def generation_sample_fingerprint(cases: tuple[GenerationCase, ...]) -> str:
    """Create a short fingerprint for the ordered sample."""
    payload = "".join(
        f"{case.question_id},{case.risk_category}\n" for case in cases
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_generation_qrels(
    path: str | Path, collection: Collection
) -> tuple[dict[str, dict[str, int]], str]:
    """Load qrels and reject unknown IDs."""
    qrels_path = Path(path)
    try:
        qrels = load_qrels(qrels_path)
    except (CollectionError, OSError) as exc:
        raise GenerationInputError(str(exc)) from exc

    for question_id, passages in qrels.items():
        if question_id not in collection.questions:
            raise GenerationInputError(f"unknown question_id {question_id}")
        for passage_id in passages:
            if passage_id not in collection.passages:
                raise GenerationInputError(f"unknown passage_id {passage_id}")

    fingerprint = hashlib.sha256(qrels_path.read_bytes()).hexdigest()[:16]
    return qrels, fingerprint
