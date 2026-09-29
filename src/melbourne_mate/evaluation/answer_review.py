"""Prepare and score the shared blind answer review."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import Collection
from melbourne_mate.evaluation.generation_runs import read_traces

CRITERIA = ("correctness", "evidence_support", "fallback_appropriateness")
REVIEWERS = ("elaine", "sriporn")
ARMS = ("bm25", "mpnet", "no-context")
EXPECTED_RUN_ARMS = {"bm25": "bm25", "mpnet": "dense", "no-context": "none"}
SHEET_COLUMNS = (
    "item_id",
    "question",
    "reference_answer",
    "answer",
    "retrieved_evidence",
    *CRITERIA,
    "note",
)

INSTRUCTIONS = """# Blind answer review

Score each answer without opening `key.csv`.

- `correctness`: 0 wrong, 1 partly correct, 2 correct.
- `evidence_support`: 0 unsupported, 1 partly supported, 2 fully supported.
- `fallback_appropriateness`: 0 wrong action, 1 acceptable but weak, 2 appropriate.

Use 0 for evidence support when no retrieved evidence supports an answer. For an
out-of-knowledge-base question, score fallback appropriateness against the need
to refuse safely. Add a short note only when a score needs explanation.
"""


class AnswerReviewError(ValueError):
    """Raised when a blind review package is incomplete or inconsistent."""


def _read_manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_runs(
    collection: Collection, run_dirs: Mapping[str, str | Path]
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    if set(run_dirs) != set(ARMS):
        raise AnswerReviewError("runs must include bm25, mpnet and no-context")
    manifests = {arm: _read_manifest(Path(run_dirs[arm])) for arm in ARMS}
    traces = {arm: read_traces(run_dirs[arm]) for arm in ARMS}

    for arm, manifest in manifests.items():
        if manifest["arm"] != EXPECTED_RUN_ARMS[arm]:
            raise AnswerReviewError(f"{arm} points to the wrong generation arm")
    if {item["collection_fingerprint"] for item in manifests.values()} != {
        collection.fingerprint()
    }:
        raise AnswerReviewError("runs use a different collection")
    for field in (
        "config_fingerprint",
        "model_digest",
        "qrels_fingerprint",
        "sample_fingerprint",
    ):
        if len({item[field] for item in manifests.values()}) != 1:
            raise AnswerReviewError(f"runs use different {field}")

    question_sets = [{row["question_id"] for row in rows} for rows in traces.values()]
    if len({frozenset(ids) for ids in question_sets}) != 1:
        raise AnswerReviewError("runs contain different questions")
    return manifests, traces


def _pick(rows: Sequence[dict], count: int, seed: int) -> list[dict]:
    answerable = sorted(
        (row for row in rows if row["answerable"]), key=lambda row: row["question_id"]
    )
    ookb = sorted(
        (row for row in rows if not row["answerable"]),
        key=lambda row: row["question_id"],
    )
    answerable_n = round(count * 0.6)
    if len(answerable) < answerable_n or len(ookb) < count - answerable_n:
        raise AnswerReviewError("run does not contain enough answerable and OOKB items")
    rng = random.Random(seed)
    rng.shuffle(answerable)
    rng.shuffle(ookb)
    return answerable[:answerable_n] + ookb[: count - answerable_n]


def prepare_answer_review(
    directory: str | Path,
    collection: Collection,
    run_dirs: Mapping[str, str | Path],
    *,
    size: int = 30,
    seed: int = CONFIG.evaluation.seed,
) -> Path:
    """Write two identical blind sheets and a closed arm key."""
    if size < 3 or size % 3:
        raise AnswerReviewError("review size must be divisible by three")
    out = Path(directory)
    if out.exists() and any(out.iterdir()):
        raise AnswerReviewError(f"review directory {out} already exists")

    manifests, traces = _validate_runs(collection, run_dirs)
    per_arm = size // 3
    chosen: list[dict[str, object]] = []
    for offset, arm in enumerate(ARMS):
        for trace in _pick(traces[arm], per_arm, seed + offset):
            item_id = hashlib.sha256(
                f"{manifests[arm]['run_id']}:{trace['question_id']}".encode()
            ).hexdigest()[:10]
            chosen.append({"item_id": item_id, "arm": arm, "trace": trace})
    random.Random(seed).shuffle(chosen)

    sheet_rows = []
    key_rows = []
    for item in chosen:
        trace = item["trace"]
        question_id = str(trace["question_id"])
        reference = collection.gold.get(question_id)
        evidence = "; ".join(
            f"[{passage_id}] {collection.passages[passage_id].text}"
            for passage_id in trace.get("retrieved", [])
            if passage_id in collection.passages
        )
        sheet_rows.append(
            {
                "item_id": item["item_id"],
                "question": trace["question"],
                "reference_answer": reference.answer
                if reference
                else "Out of knowledge base: a cautious fallback is expected.",
                "answer": trace["answer"],
                "retrieved_evidence": evidence,
                "correctness": "",
                "evidence_support": "",
                "fallback_appropriateness": "",
                "note": "",
            }
        )
        key_rows.append(
            {
                "item_id": item["item_id"],
                "run_id": manifests[item["arm"]]["run_id"],
                "arm": item["arm"],
                "question_id": question_id,
                "answerable": trace["answerable"],
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    for reviewer in REVIEWERS:
        _write_csv(out / f"review-{reviewer}.csv", SHEET_COLUMNS, sheet_rows)
    _write_csv(
        out / "key.csv",
        ("item_id", "run_id", "arm", "question_id", "answerable"),
        key_rows,
    )
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "review_id": out.name,
                "reviewers": list(REVIEWERS),
                "sample_size": size,
                "seed": seed,
                "source_runs": {arm: manifests[arm]["run_id"] for arm in ARMS},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _score(value: str, item_id: str, criterion: str) -> int:
    value = value.strip()
    if value not in {"0", "1", "2"}:
        raise AnswerReviewError(f"unfilled score for {item_id}/{criterion}")
    return int(value)


def _weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    n = len(left)
    if not n:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    observed = sum((a - b) ** 2 / 4 for a, b in zip(left, right)) / n
    expected = sum(
        ((a - b) ** 2 / 4) * left_counts[a] * right_counts[b]
        for a in range(3)
        for b in range(3)
    ) / (n * n)
    return 1.0 if expected == 0 and observed == 0 else 1 - observed / expected


def score_answer_review(directory: str | Path) -> Path:
    """Check completed sheets and write agreement plus arm-level means."""
    path = Path(directory)
    sheets = {
        reviewer: {
            row["item_id"]: row
            for row in _read_csv(path / f"review-{reviewer}.csv")
        }
        for reviewer in REVIEWERS
    }
    key = {row["item_id"]: row for row in _read_csv(path / "key.csv")}
    if any(set(sheet) != set(key) for sheet in sheets.values()):
        raise AnswerReviewError("review sheets do not match the key")

    scores: dict[str, dict[str, dict[str, int]]] = {}
    for reviewer, sheet in sheets.items():
        scores[reviewer] = {
            item_id: {
                criterion: _score(row[criterion], item_id, criterion)
                for criterion in CRITERIA
            }
            for item_id, row in sheet.items()
        }

    adjudications: dict[tuple[str, str], int] = {}
    adjudication_path = path / "adjudication.csv"
    if adjudication_path.exists():
        for row in _read_csv(adjudication_path):
            item_id = row["item_id"]
            criterion = row["criterion"]
            if item_id not in key or criterion not in CRITERIA:
                raise AnswerReviewError("adjudication refers to an unknown score")
            score_key = (item_id, criterion)
            if score_key in adjudications:
                raise AnswerReviewError("adjudication contains a duplicate score")
            for reviewer in REVIEWERS:
                recorded = _score(row[f"{reviewer}_score"], item_id, criterion)
                if recorded != scores[reviewer][item_id][criterion]:
                    raise AnswerReviewError("adjudication does not match review sheets")
            adjudications[score_key] = _score(
                row["agreed_score"], item_id, criterion
            )

        differences = {
            (item_id, criterion)
            for item_id in key
            for criterion in CRITERIA
            if scores["elaine"][item_id][criterion]
            != scores["sriporn"][item_id][criterion]
        }
        if set(adjudications) != differences:
            raise AnswerReviewError("adjudication must cover every score difference")

    agreement_rows = []
    for criterion in CRITERIA:
        left = [scores["elaine"][item][criterion] for item in key]
        right = [scores["sriporn"][item][criterion] for item in key]
        agreement_rows.append(
            {
                "criterion": criterion,
                "items": len(key),
                "exact_agreement": sum(a == b for a, b in zip(left, right))
                / len(key),
                "weighted_kappa": _weighted_kappa(left, right),
            }
        )

    summary_rows = []
    agreed_summary_rows = []
    for arm in ARMS:
        item_ids = [item_id for item_id, row in key.items() if row["arm"] == arm]
        for criterion in CRITERIA:
            left = [scores["elaine"][item][criterion] for item in item_ids]
            right = [scores["sriporn"][item][criterion] for item in item_ids]
            summary_rows.append(
                {
                    "arm": arm,
                    "criterion": criterion,
                    "items": len(item_ids),
                    "elaine_mean": sum(left) / len(left),
                    "sriporn_mean": sum(right) / len(right),
                    "combined_mean": (sum(left) + sum(right))
                    / (2 * len(item_ids)),
                }
            )
            if adjudications:
                agreed = [
                    left_score
                    if left_score == right_score
                    else adjudications[(item_id, criterion)]
                    for item_id, left_score, right_score in zip(item_ids, left, right)
                ]
                agreed_summary_rows.append(
                    {
                        "arm": arm,
                        "criterion": criterion,
                        "items": len(item_ids),
                        "agreed_mean": sum(agreed) / len(agreed),
                    }
                )

    _write_csv(
        path / "agreement.csv",
        ("criterion", "items", "exact_agreement", "weighted_kappa"),
        agreement_rows,
    )
    _write_csv(
        path / "summary.csv",
        (
            "arm",
            "criterion",
            "items",
            "elaine_mean",
            "sriporn_mean",
            "combined_mean",
        ),
        summary_rows,
    )
    if agreed_summary_rows:
        _write_csv(
            path / "agreed-summary.csv",
            ("arm", "criterion", "items", "agreed_mean"),
            agreed_summary_rows,
        )
    lines = [
        "# Blind answer review",
        "",
        f"Both reviewers scored the same {len(key)} answers on a 0-2 scale.",
        "",
        "## Agreement",
        "",
    ]
    for row in agreement_rows:
        lines.append(
            f"- {row['criterion']}: exact {row['exact_agreement']:.1%}; "
            f"weighted kappa {row['weighted_kappa']:.3f}."
        )
    lines.extend(
        [
            "",
            "Arm-level means are in `summary.csv`. Agreement is calculated before any discussion.",
            "",
        ]
    )
    if agreed_summary_rows:
        lines.extend(
            [
                "## Adjudication",
                "",
                (
                    f"The reviewers resolved {len(adjudications)} score difference(s). "
                    "Final agreed means are in `agreed-summary.csv`."
                ),
                "",
            ]
        )
        before = {
            (row["arm"], row["criterion"]): row["combined_mean"]
            for row in summary_rows
        }
        for row in agreed_summary_rows:
            original = before[(row["arm"], row["criterion"])]
            if original != row["agreed_mean"]:
                lines.append(
                    f"- {row['arm']} {row['criterion']}: {original:.3f} before "
                    f"discussion; {row['agreed_mean']:.3f} agreed."
                )
        lines.extend(
            [
                "",
                (
                    "The original independent sheets remain unchanged. Final decisions "
                    "are saved in `adjudication.csv`."
                ),
                "",
            ]
        )
    (path / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return path
