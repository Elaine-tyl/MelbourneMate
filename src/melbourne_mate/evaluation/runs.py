"""Read and write immutable retrieval runs with full experiment details."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from melbourne_mate import __version__
from melbourne_mate.config import CONFIG
from melbourne_mate.evaluation.metrics import METRICS, aggregate


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    arm: str
    split: str
    encoder: str
    collection_fingerprint: str
    config_fingerprint: str
    protocol_version: str
    ndcg_gain: str
    questions: int
    created_utc: str
    code_version: str
    python_version: str
    platform: str
    note: str = ""


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run(
    directory: str | Path,
    run_id: str,
    arm: str,
    split: str,
    encoder: str,
    collection_fingerprint: str,
    rankings: Mapping[str, Sequence[str]],
    per_question: Mapping[str, Mapping[str, float]],
    question_meta: Mapping[str, Mapping[str, str]] | None = None,
    note: str = "",
) -> Path:
    """Write retrieval.run, per-question and aggregate metrics, and a manifest."""
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise RunError(f"run directory {path} already exists; choose a new run id")
    path.mkdir(parents=True, exist_ok=True)

    with (path / "retrieval.run").open("w", encoding="utf-8") as handle:
        for question_id in sorted(rankings):
            for rank, passage_id in enumerate(rankings[question_id], start=1):
                handle.write(f"{question_id} Q0 {passage_id} {rank} {1.0 / rank:.6f} {run_id}\n")

    meta = question_meta or {}
    rows = []
    for question_id in sorted(per_question):
        row: dict[str, object] = {"question_id": question_id}
        row.update(meta.get(question_id, {}))
        row.update({m: round(per_question[question_id][m], 6) for m in METRICS})
        rows.append(row)
    extra_columns = sorted({key for value in meta.values() for key in value})
    _write_csv(
        path / "per_question_metrics.csv",
        rows,
        ["question_id", *extra_columns, *METRICS],
    )

    totals = aggregate(per_question)
    _write_csv(
        path / "aggregate_metrics.csv",
        [{"metric": m, "value": round(totals[m], 6)} for m in METRICS],
        ["metric", "value"],
    )

    manifest = RunManifest(
        run_id=run_id,
        arm=arm,
        split=split,
        encoder=encoder,
        collection_fingerprint=collection_fingerprint,
        config_fingerprint=CONFIG.fingerprint(),
        protocol_version=CONFIG.protocol_version,
        ndcg_gain=CONFIG.evaluation.ndcg_gain,
        questions=len(per_question),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        note=note,
    )
    (path / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_manifest(directory: str | Path) -> RunManifest:
    payload = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    return RunManifest(**payload)


def read_per_question(directory: str | Path) -> dict[str, dict[str, float]]:
    path = Path(directory) / "per_question_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["question_id"]: {m: float(row[m]) for m in METRICS}
            for row in csv.DictReader(handle)
        }


def read_run_file(directory: str | Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line in (Path(directory) / "retrieval.run").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question_id, _, passage_id, _rank, _score, _tag = line.split()
        rankings.setdefault(question_id, []).append(passage_id)
    return rankings
