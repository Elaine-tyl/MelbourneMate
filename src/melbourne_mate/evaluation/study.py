"""Check and summarise the small student study."""

from __future__ import annotations

import csv
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean, median

METHODS = ("melbournemate", "official-search")
COMPLETION = ("yes", "partial", "no")
MEASURES = ("completed", "time_seconds", "confidence", "trust")
# Saved during the session. Completion is judged afterwards in the review step.
SESSION_MEASURES = ("time_seconds", "confidence", "trust")
RESPONSE_COLUMNS = (
    "participant_id", "position", "task_id", "method", *MEASURES, "answer", "note"
)
MAX_SECONDS = 300
MIN_PARTICIPANTS = 4
TEMPLATE = "responses-template.csv"
WARM_UP_QUESTION = "How do I get from Melbourne Airport to the city?"
RESPONSES = "responses.csv"
FINAL_TEMPLATE = "final-template.csv"
FINAL = "final.csv"
FINAL_COLUMNS = ("participant_id", "would_use")
WOULD_USE_QUESTION = (
    "How likely are you to use MelbourneMate instead of searching multiple "
    "official websites? (1 = very unlikely, 5 = very likely)"
)


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

        filled = [bool(row[column].strip()) for column in SESSION_MEASURES]
        if not any(filled):
            if row["completed"].strip():
                raise StudyError(f"{where}: completion is judged only after an attempt")
            continue  # not attempted yet
        if not all(filled):
            raise StudyError(f"{where}: fill all of {', '.join(SESSION_MEASURES)} or none")
        if row["completed"] and row["completed"] not in COMPLETION:
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

    fields = (*SESSION_MEASURES, "answer", "note")
    entered = {key: str(values.get(key, "")).strip() for key in fields}
    updated = {**matches[0], **entered}  # keeps any completion judgement
    where = f"{participant_id} position {position}"
    _whole_number(updated["time_seconds"], 1, MAX_SECONDS, where)
    _whole_number(updated["confidence"], 1, 5, where)
    _whole_number(updated["trust"], 1, 5, where)

    matches[0].update(updated)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reset_response(study_dir: str | Path, participant_id: str, position: str) -> None:
    """Clear one saved task and any final rating for the participant."""
    root = Path(study_dir)
    path = root / RESPONSES
    rows = _read_csv(path) if path.exists() else []
    matches = [
        row
        for row in rows
        if row["participant_id"] == participant_id and row["position"] == str(position)
    ]
    where = f"{participant_id} position {position}"
    if len(matches) != 1 or not matches[0]["time_seconds"].strip():
        raise StudyError(f"{where}: no saved attempt to reset")
    for field in (*MEASURES, "answer", "note"):
        matches[0][field] = ""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    final_path = root / FINAL
    if final_path.exists():
        final_rows = _read_csv(final_path)
        for row in final_rows:
            if row["participant_id"] == participant_id:
                row["would_use"] = ""
        with final_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FINAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(final_rows)


def judge_completion(
    study_dir: str | Path, participant_id: str, position: str, completed: str
) -> None:
    """Record the researcher's completion judgement for one saved attempt."""
    where = f"{participant_id} position {position}"
    if completed not in COMPLETION:
        raise StudyError(f"{where}: completed must be one of {', '.join(COMPLETION)}")
    path = Path(study_dir) / RESPONSES
    rows = _read_csv(path) if path.exists() else []
    matches = [
        row
        for row in rows
        if row["participant_id"] == participant_id and row["position"] == str(position)
    ]
    if len(matches) != 1 or not matches[0]["time_seconds"].strip():
        raise StudyError(f"{where}: no saved attempt to judge")
    matches[0]["completed"] = completed
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_would_use(study_dir: str | Path) -> dict[str, int]:
    """Final answers saved so far, by participant code."""
    root = Path(study_dir)
    path = root / FINAL if (root / FINAL).exists() else root / FINAL_TEMPLATE
    return {
        row["participant_id"]: _whole_number(row["would_use"], 1, 5, row["participant_id"])
        for row in _read_csv(path)
        if row["would_use"].strip()
    }


def save_would_use(study_dir: str | Path, participant_id: str, value: object) -> None:
    """Save the participant's final would-use rating from 1 to 5."""
    root = Path(study_dir)
    rating = _whole_number(str(value), 1, 5, participant_id)
    path = root / FINAL
    if not path.exists():
        shutil.copyfile(root / FINAL_TEMPLATE, path)
    rows = _read_csv(path)
    match = [row for row in rows if row["participant_id"] == participant_id]
    if len(match) != 1:
        raise StudyError(f"{participant_id} is not in {FINAL}")
    match[0]["would_use"] = str(rating)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_COLUMNS, lineterminator="\n")
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
    scheduled = Counter(row["participant_id"] for row in _read_csv(root / "schedule.csv"))
    attempted = Counter(a["participant_id"] for a in attempts)
    finished = {code for code, count in attempted.items() if count == scheduled[code]}
    excluded = sorted(set(attempted) - finished)
    if not finished:
        raise StudyError("no participant has completed all four tasks yet")
    attempts = [a for a in attempts if a["participant_id"] in finished]
    pending = [a for a in attempts if not a["completed"]]
    if pending:
        names = ", ".join(f"{a['participant_id']} {a['task_id']}" for a in pending)
        raise StudyError(f"judge completion in the review step first ({names})")

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

    participants = len(finished)
    lines = [
        "# Student study summary",
        "",
        f"{participants} participants completed all four tasks ({len(attempts)} task attempts).",
        "Results are descriptive and are not tested for significance.",
    ]
    if excluded:
        lines.append(
            "Excluded because they did not finish all four tasks. " + ", ".join(excluded) + "."
        )
    if participants < MIN_PARTICIPANTS:
        lines.append(
            f"Fewer than {MIN_PARTICIPANTS} participants took part, so the methods "
            "should only be described, not compared."
        )
    overall = {row["method"]: row for row in rows if row["group"] == "all"}

    def cell(method: str, text: str) -> str:
        return text.format(**overall[method]) if method in overall else "n/a"

    lines += [
        "",
        "| Evidence | MelbourneMate | Official search |",
        "| --- | ---: | ---: |",
    ]
    for label, text in (
        ("Task attempts", "{attempts}"),
        ("Task completion", "{completed_rate:.0%}"),
        ("Median time", "{median_seconds:g} sec"),
        ("Mean confidence", "{mean_confidence}/5"),
        ("Mean trust", "{mean_trust}/5"),
    ):
        lines.append(
            f"| {label} | {cell('melbournemate', text)} | {cell('official-search', text)} |"
        )

    would_use = {
        code: rating for code, rating in load_would_use(root).items() if code in finished
    }
    if would_use:
        average = mean(would_use.values())
        sentence = (
            "Would use MelbourneMate instead of searching official websites. Mean "
            f"{average:.2f}/5 from {len(would_use)} participants."
        )
        lines += ["", sentence]
    lines += ["", "Per-task results are in `summary.csv`."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
