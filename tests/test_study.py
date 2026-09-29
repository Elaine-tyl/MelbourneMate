import csv
import shutil
from collections import Counter
from pathlib import Path

import pytest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.study import (
    RESPONSE_COLUMNS,
    StudyError,
    load_responses,
    summarise_study,
)

STUDY = Path("study")


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_study(tmp_path, answers):
    """Copy the study templates and fill rows as {(participant, position): values}."""
    target = tmp_path / "study"
    shutil.copytree(STUDY, target)
    rows = read(target / "responses.csv")
    for row in rows:
        values = answers.get((row["participant_id"], row["position"]))
        if values:
            row.update(zip(("completed", "time_seconds", "confidence", "trust"), values))
    with (target / "responses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target


def test_schedule_is_balanced_for_four_and_six_participants():
    schedule = read(STUDY / "schedule.csv")
    for participants in (4, 6):
        chosen = [r for r in schedule if int(r["participant_id"][1:]) <= participants]
        pairs = Counter((r["task_id"], r["method"]) for r in chosen)
        first = Counter(r["method"] for r in chosen if r["position"] == "1")

        assert set(pairs.values()) == {participants // 2}
        assert len(pairs) == 8
        assert set(first.values()) == {participants // 2}


def test_tasks_cite_sources_in_the_collection():
    sources = load_collection("data/v1").sources

    for task in read(STUDY / "tasks.csv"):
        assert set(task["source_ids"].split(";")) <= set(sources)


def test_blank_template_has_no_attempts():
    assert load_responses(STUDY) == []
    with pytest.raises(StudyError, match="no completed attempts"):
        summarise_study(STUDY)


def test_summary_reports_each_method(tmp_path):
    study = copy_study(
        tmp_path,
        {
            ("P01", "1"): ("yes", "120", "5", "4"),
            ("P01", "2"): ("partial", "300", "3", "4"),
            ("P01", "3"): ("no", "600", "2", "3"),
            ("P01", "4"): ("yes", "240", "4", "5"),
        },
    )

    summarise_study(study)
    rows = {(r["group"], r["method"]): r for r in read(study / "summary.csv")}
    report = (study / "summary.md").read_text(encoding="utf-8")

    assert rows[("all", "melbournemate")]["completed_rate"] == "0.5"
    assert rows[("all", "melbournemate")]["partial_rate"] == "0.5"
    assert rows[("all", "melbournemate")]["median_seconds"] == "210.0"
    assert rows[("all", "official-search")]["mean_trust"] == "4"
    assert rows[("TK1", "melbournemate")]["attempts"] == "1"
    assert "Fewer than 4 participants" in report


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("yes", "120", "", ""), "fill all"),
        (("done", "120", "4", "4"), "completed must be"),
        (("yes", "900", "4", "4"), "outside 1-600"),
        (("yes", "120", "6", "4"), "outside 1-5"),
    ],
)
def test_bad_responses_are_rejected(tmp_path, values, message):
    study = copy_study(tmp_path, {("P01", "1"): values})

    with pytest.raises(StudyError, match=message):
        load_responses(study)


def test_rows_must_match_the_schedule(tmp_path):
    study = copy_study(tmp_path, {})
    text = (study / "responses.csv").read_text(encoding="utf-8")
    (study / "responses.csv").write_text(
        text.replace("P01,1,TK1,melbournemate", "P01,1,TK1,official-search"),
        encoding="utf-8",
    )

    with pytest.raises(StudyError, match="does not match schedule"):
        load_responses(study)


def test_cli_writes_the_summary(tmp_path, capsys):
    study = copy_study(tmp_path, {("P02", "1"): ("yes", "90", "4", "4")})

    assert cli.main(["study-summary", "--study-dir", str(study)]) == 0
    assert (study / "summary.md").exists()
    assert "wrote" in capsys.readouterr().out
