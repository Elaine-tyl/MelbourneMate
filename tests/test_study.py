import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.study import (
    MEASURES,
    RESPONSE_COLUMNS,
    WARM_UP_QUESTION,
    StudyError,
    ensure_responses,
    load_responses,
    participant_tasks,
    save_response,
    summarise_study,
    warm_up,
)
from melbourne_mate.interface import app as chatbot

STUDY = Path("study")
STUDY_APP = Path(__file__).parents[1] / "src/melbourne_mate/interface/study_app.py"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_study(tmp_path, answers):
    """Copy the study templates and fill rows as {(participant, position): values}."""
    target = tmp_path / "study"
    shutil.copytree(STUDY, target, ignore=shutil.ignore_patterns("responses.csv"))
    rows = read(ensure_responses(target))
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


def test_blank_template_has_no_attempts(tmp_path):
    study = copy_study(tmp_path, {})
    (study / "responses.csv").unlink()

    assert load_responses(study) == []
    with pytest.raises(StudyError, match="no completed attempts"):
        summarise_study(study)
    template = read(STUDY / "responses-template.csv")
    assert not any(row[column] for row in template for column in (*MEASURES, "note"))


def test_participant_files_are_kept_out_of_git():
    ignored = subprocess.run(
        ["git", "check-ignore", "study/responses.csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tracked = subprocess.run(
        ["git", "ls-files", "study"], capture_output=True, text=True, check=True
    ).stdout.split()

    assert ignored == ["study/responses.csv"]
    assert sorted(tracked) == [
        "study/README.md",
        "study/responses-template.csv",
        "study/schedule.csv",
        "study/tasks.csv",
    ]


def test_warm_up_runs_one_fixed_question_through_the_pipeline():
    asked = []

    assert warm_up(asked.append) >= 0
    assert asked == [WARM_UP_QUESTION]
    assert all(WARM_UP_QUESTION != task["prompt"] for task in read(STUDY / "tasks.csv"))


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


def test_saved_response_updates_only_its_row(tmp_path):
    study = copy_study(tmp_path, {})
    (study / "responses.csv").unlink()  # the page creates it from the template

    save_response(
        study, "P03", "2", {"completed": "partial", "time_seconds": 240, "confidence": 3, "trust": 4}
    )

    attempts = load_responses(study)
    assert attempts == [
        {
            "participant_id": "P03",
            "task_id": "TK6",
            "method": "official-search",
            "completed": "partial",
            "time_seconds": 240,
            "confidence": 3,
            "trust": 4,
        }
    ]
    assert participant_tasks(study, "P03")[1]["completed"] == "partial"
    with pytest.raises(StudyError, match="outside 1-5"):
        save_response(
            study, "P03", "1", {"completed": "yes", "time_seconds": 60, "confidence": 9, "trust": 4}
        )
    with pytest.raises(StudyError, match="not in schedule"):
        save_response(
            study, "P09", "1", {"completed": "yes", "time_seconds": 60, "confidence": 3, "trust": 4}
        )


def test_study_page_warms_up_the_pipeline_then_saves_a_task(tmp_path, monkeypatch):
    study = copy_study(tmp_path, {})
    monkeypatch.setenv("MM_STUDY_DIR", str(study))
    asked = []
    monkeypatch.setattr(chatbot, "_build_pipeline", lambda: "pipeline")
    monkeypatch.setattr(chatbot, "answer_question", lambda pipeline, q: asked.append(q) or q)
    monkeypatch.setattr(
        chatbot,
        "answer_view",
        lambda answer: {"tone": "success", "label": "Answer ready", "text": "Use the RTBA.", "sources": ()},
    )
    app = AppTest.from_file(str(STUDY_APP)).run()
    assert not app.exception
    assert "voluntary" in app.info[0].value
    assert not app.checkbox  # no consent form

    app.selectbox[0].select("P02").run()
    button = {item.key: item for item in app.button}
    assert button["P02-1-start"].disabled  # P02 starts with MelbourneMate
    assert button["P02-1-ask"].disabled
    button["P02-warm-button"].click().run()
    assert asked == [WARM_UP_QUESTION]
    assert any("warmed up in" in item.value for item in app.success)

    button = {item.key: item for item in app.button}
    assert not button["P02-1-start"].disabled
    app.text_area(key="P02-1-question").input("Where do I lodge my bond?")
    button["P02-1-ask"].click().run()
    assert asked == [WARM_UP_QUESTION, "Where do I lodge my bond?"]
    assert any("Use the RTBA." in item.value for item in app.markdown)

    app.number_input(key="P02-1-time").set_value(75)
    app.radio(key="P02-1-done").set_value("yes")
    app.radio(key="P02-1-conf").set_value(4)
    app.radio(key="P02-1-trust").set_value(5)
    app.run()
    next(item for item in app.button if item.key == "P02-1-save").click().run()

    assert not app.exception
    saved = [row for row in read(study / "responses.csv") if row["completed"]]
    assert [(r["participant_id"], r["task_id"], r["time_seconds"]) for r in saved] == [
        ("P02", "TK4", "75")
    ]
