"""Check and summarise the small student study."""

from __future__ import annotations

import csv
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean, median

METHODS = ("melbournemate", "official-search")
COMPLETION = ("yes", "partial", "no")
MEASURES = ("completed", "time_seconds", "confidence", "trust")
RESPONSE_COLUMNS = ("participant_id", "position", "task_id", "method", *MEASURES, "note")
MAX_SECONDS = 600
MIN_PARTICIPANTS = 4
TEMPLATE = "responses-template.csv"
WARM_UP_QUESTION = "How do I get from Melbourne Airport to the city?"
RESPONSES = "responses.csv"


class StudyError(ValueError):
    """Raised when study responses do not match the schedule or the scales."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _whole_number(value: str, low: int, high: int, where: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise StudyError(f"{where}: {value!r} is not a whole number") from exc
    if not low <= number <= high:
        raise StudyError(f"{where}: {number} is outside {low}-{high}")
    return number


def responses_path(study_dir: str | Path) -> Path:
    """Local responses file, or the blank template before any session."""
    root = Path(study_dir)
    local = root / RESPONSES
    return local if local.exists() else root / TEMPLATE


def ensure_responses(study_dir: str | Path) -> Path:
    """Create the local responses file from the blank template once."""
    root = Path(study_dir)
    local = root / RESPONSES
    if not local.exists():
        shutil.copyfile(root / TEMPLATE, local)
    return local


def load_responses(study_dir: str | Path) -> list[dict[str, object]]:
    """Return completed attempts after checking them against the schedule."""
    root = Path(study_dir)
    tasks = {row["task_id"] for row in _read_csv(root / "tasks.csv")}
    schedule = {
        (row["participant_id"], row["position"]): row
        for row in _read_csv(root / "schedule.csv")
    }
    with responses_path(root).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
            raise StudyError(f"responses.csv columns must be {', '.join(RESPONSE_COLUMNS)}")
        rows = list(reader)

    attempts = []
    for row in rows:
        where = f"{row['participant_id']} position {row['position']}"
        planned = schedule.get((row["participant_id"], row["position"]))
        if planned is None or (planned["task_id"], planned["method"]) != (
            row["task_id"],
            row["method"],
        ):
            raise StudyError(f"{where}: row does not match schedule.csv")
        if row["task_id"] not in tasks or row["method"] not in METHODS:
            raise StudyError(f"{where}: unknown task or method")

        filled = [bool(row[column].strip()) for column in MEASURES]
        if not any(filled):
            continue  # not attempted yet
        if not all(filled):
            raise StudyError(f"{where}: fill all of {', '.join(MEASURES)} or none")
        if row["completed"] not in COMPLETION:
            raise StudyError(f"{where}: completed must be one of {', '.join(COMPLETION)}")
        attempts.append(
            {
                "participant_id": row["participant_id"],
                "task_id": row["task_id"],
                "method": row["method"],
                "completed": row["completed"],
                "time_seconds": _whole_number(row["time_seconds"], 1, MAX_SECONDS, where),
                "confidence": _whole_number(row["confidence"], 1, 5, where),
                "trust": _whole_number(row["trust"], 1, 5, where),
            }
        )
    return attempts


def participants(study_dir: str | Path) -> list[str]:
    """Participant codes in schedule order."""
    codes = [row["participant_id"] for row in _read_csv(Path(study_dir) / "schedule.csv")]
    return list(dict.fromkeys(codes))


def participant_tasks(study_dir: str | Path, participant_id: str) -> list[dict[str, str]]:
    """Scheduled tasks for one participant, with prompts and any saved answers."""
    root = Path(study_dir)
    tasks = {row["task_id"]: row for row in _read_csv(root / "tasks.csv")}
    saved = {
        row["position"]: row
        for row in _read_csv(responses_path(root))
        if row["participant_id"] == participant_id
    }
    return [
        {**tasks[row["task_id"]], **saved.get(row["position"], {}), **row}
        for row in _read_csv(root / "schedule.csv")
        if row["participant_id"] == participant_id
    ]


def save_response(
    study_dir: str | Path,
    participant_id: str,
    position: str,
    values: dict[str, object],
) -> None:
    """Write one task result into responses.csv after checking it."""
    root = Path(study_dir)
    if participant_id not in participants(root):
        raise StudyError(f"{participant_id} is not in schedule.csv")
    path = ensure_responses(root)
    rows = _read_csv(path)
    matches = [
        row
        for row in rows
        if row["participant_id"] == participant_id and row["position"] == str(position)
    ]
    if len(matches) != 1:
        raise StudyError(f"{participant_id} position {position} is not in responses.csv")

    updated = {**matches[0], **{key: str(values.get(key, "")).strip() for key in (*MEASURES, "note")}}
    where = f"{participant_id} position {position}"
    if updated["completed"] not in COMPLETION:
        raise StudyError(f"{where}: completed must be one of {', '.join(COMPLETION)}")
    _whole_number(updated["time_seconds"], 1, MAX_SECONDS, where)
    _whole_number(updated["confidence"], 1, 5, where)
    _whole_number(updated["trust"], 1, 5, where)

    matches[0].update(updated)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def warm_up(ask: Callable[[str], object]) -> int:
    """Run one fixed, untimed question through the full pipeline.

    Returns the seconds taken, so model and index loading never fall inside a
    timed task.
    """
    started = time.monotonic()
    ask(WARM_UP_QUESTION)
    return round(time.monotonic() - started)


def _summary_row(group: str, method: str, attempts: Sequence[dict]) -> dict[str, object]:
    return {
        "group": group,
        "method": method,
        "attempts": len(attempts),
        "participants": len({a["participant_id"] for a in attempts}),
        "completed_rate": round(
            sum(a["completed"] == "yes" for a in attempts) / len(attempts), 3
        ),
        "partial_rate": round(
            sum(a["completed"] == "partial" for a in attempts) / len(attempts), 3
        ),
        "median_seconds": median(a["time_seconds"] for a in attempts),
        "mean_confidence": round(mean(a["confidence"] for a in attempts), 2),
        "mean_trust": round(mean(a["trust"] for a in attempts), 2),
    }


def summarise_study(study_dir: str | Path) -> Path:
    """Write descriptive results by method and by task."""
    root = Path(study_dir)
    attempts = load_responses(root)
    if not attempts:
        raise StudyError("responses.csv has no completed attempts yet")

    rows = []
    groups = ["all", *sorted({a["task_id"] for a in attempts})]
    for group in groups:
        for method in METHODS:
            selected = [
                a
                for a in attempts
                if a["method"] == method and group in ("all", a["task_id"])
            ]
            if selected:
                rows.append(_summary_row(group, method, selected))

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    participants = len({a["participant_id"] for a in attempts})
    lines = [
        "# Student study summary",
        "",
        f"{participants} participants completed {len(attempts)} task attempts.",
        "Results are descriptive and are not tested for significance.",
    ]
    if participants < MIN_PARTICIPANTS:
        lines.append(
            f"Fewer than {MIN_PARTICIPANTS} participants took part, so the methods "
            "should only be described, not compared."
        )
    lines += ["", "| Method | Attempts | Completed | Median seconds | Confidence | Trust |"]
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        if row["group"] == "all":
            lines.append(
                f"| {row['method']} | {row['attempts']} | {row['completed_rate']:.0%} "
                f"| {row['median_seconds']} | {row['mean_confidence']} | {row['mean_trust']} |"
            )
    lines += ["", "Per-task results are in `summary.csv`."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
