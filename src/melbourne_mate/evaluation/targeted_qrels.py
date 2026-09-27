"""Build test qrels from curated ground truth and a small verified increment."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from melbourne_mate.corpus import Collection, load_qrels
from melbourne_mate.evaluation.runs import read_manifest, read_run_file


@dataclass(frozen=True)
class TargetedQrelsResult:
    qrels_path: Path
    seed_pairs: int
    checked_candidates: int
    added_positive_pairs: int


@dataclass(frozen=True)
class ReviewPoolResult:
    path: Path
    candidates: int


def build_review_pool(
    collection: Collection,
    run_dir: str | Path,
    output_path: str | Path,
) -> ReviewPoolResult:
    manifest = read_manifest(run_dir)
    if manifest.arm != "dense":
        raise ValueError("review pool requires a Dense run")
    if manifest.collection_fingerprint != collection.fingerprint():
        raise ValueError("run and collection fingerprints do not match")

    confusable = {
        frozenset((pair.topic_a, pair.topic_b)) for pair in collection.pairs
    }
    rows = []
    for qid, passage_ids in sorted(read_run_file(run_dir).items()):
        question = collection.questions.get(qid)
        if question is None or collection.splits.get(question.topic_id) != manifest.split:
            raise ValueError(f"{qid}: run does not match the recorded split")
        qtopic = collection.topics[question.topic_id]

        for rank, pid in enumerate(passage_ids[:5], start=1):
            if pid in collection.qrels.get(qid, {}):
                continue
            passage = collection.passages[pid]
            ptopic = collection.topics[passage.topic_id]
            same_category = qtopic.category == ptopic.category
            is_confusable = frozenset((qtopic.topic_id, ptopic.topic_id)) in confusable
            if not (same_category or is_confusable):
                continue

            source = collection.sources[passage.source_id]
            rows.append(
                {
                    "question_id": qid,
                    "passage_id": pid,
                    "rank": rank,
                    "reason": "same_category" if same_category else "confusable_topics",
                    "question_text": question.text,
                    "passage_text": passage.text,
                    "source_url": source.url,
                    "grade": "",
                    "primary_reviewer": "",
                    "needs_second_review": "",
                    "note": "",
                }
            )

    out = Path(output_path)
    if out.exists():
        raise ValueError(f"review pool already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return ReviewPoolResult(out, len(rows))


def _write_qrels(path: str | Path, qrels: dict[str, dict[str, int]]) -> Path:
    """Write positive graded mappings in the TREC qrels format used by Walert."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{question_id} 0 {passage_id} {grade}"
        for question_id in sorted(qrels)
        for passage_id, grade in sorted(qrels[question_id].items())
        if grade > 0
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _grade(value: str, label: str) -> int:
    value = (value or "").strip()
    if value not in {"0", "1", "2"}:
        raise ValueError(f"{label}: grade must be 0, 1, or 2")
    return int(value)


def build_targeted_qrels(
    base_qrels: str | Path,
    verification_csv: str | Path,
    output_path: str | Path,
    *,
    question_ids: Iterable[str],
) -> TargetedQrelsResult:
    """Restrict curated qrels to a test set, then add verified new candidates."""
    allowed = set(question_ids)
    if not allowed:
        raise ValueError("question_ids must not be empty")

    base = load_qrels(Path(base_qrels))
    qrels = {
        question_id: dict(passages)
        for question_id, passages in base.items()
        if question_id in allowed
    }
    seed_pairs = sum(len(passages) for passages in qrels.values())
    base_pairs = {
        (question_id, passage_id)
        for question_id, passages in qrels.items()
        for passage_id in passages
    }

    with Path(verification_csv).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("targeted verification is empty")

    required = {"question_id", "passage_id", "grade", "primary_reviewer"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("verification is missing columns: " + ", ".join(sorted(missing)))

    seen: set[tuple[str, str]] = set()
    added = 0
    for index, row in enumerate(rows, start=2):
        question_id = (row.get("question_id") or "").strip()
        passage_id = (row.get("passage_id") or "").strip()
        reviewer = (row.get("primary_reviewer") or "").strip()
        if question_id not in allowed:
            raise ValueError(f"row {index}: question is not in the held-out run")
        if not passage_id:
            raise ValueError(f"row {index}: passage_id is required")
        if not reviewer:
            raise ValueError(f"row {index}: primary_reviewer is required")

        pair = (question_id, passage_id)
        if pair in seen:
            raise ValueError(f"row {index}: duplicate candidate {question_id}/{passage_id}")
        if pair in base_pairs:
            raise ValueError(
                f"row {index}: {question_id}/{passage_id} already exists in the ground truth"
            )
        seen.add(pair)

        grade = _grade(row.get("grade", ""), f"row {index}")
        if grade > 0:
            qrels.setdefault(question_id, {})[passage_id] = grade
            added += 1

    path = _write_qrels(output_path, qrels)
    return TargetedQrelsResult(path, seed_pairs, len(rows), added)
