import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation import study as study_module
from melbourne_mate.evaluation.study import (
    CONSENT_ITEMS,
    MEASURES,
    RESPONSE_COLUMNS,
    WARM_UP_PROMPT,
    StudyError,
    ensure_responses,
    has_consent,
    load_responses,
    participant_tasks,
    record_consent,
    save_response,
    summarise_study,
)

STUDY = Path("study")
STUDY_APP = Path(__file__).parents[1] / "src/melbourne_mate/interface/study_app.py"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_study(tmp_path, answers):
    """Copy the study templates and fill rows as {(participant, position): values}."""
    target = tmp_path / "study"
    shutil.copytree(STUDY, target, ignore=shutil.ignore_patterns("responses.csv", "consent.csv"))
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
        ["git", "check-ignore", "study/responses.csv", "study/consent.csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tracked = subprocess.run(
        ["git", "ls-files", "study"], capture_output=True, text=True, check=True
    ).stdout.split()

    assert ignored == ["study/responses.csv", "study/consent.csv"]
    assert "study/responses.csv" not in tracked
    assert "study/consent.csv" not in tracked


def test_warm_up_sends_the_fixed_prompt():
    class FakeModel:
        def generate(self, request):
            self.request = request

    model = FakeModel()

    assert study_module.warm_up(model) >= 0
    assert model.request.prompt == WARM_UP_PROMPT
    assert model.request.max_output_tokens == 8


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


CONSENT = {item: True for item in CONSENT_ITEMS}


def test_consent_is_needed_before_saving_and_is_recorded_once(tmp_path):
    study = copy_study(tmp_path, {})
    values = {"completed": "yes", "time_seconds": 95, "confidence": 4, "trust": 5}

    with pytest.raises(StudyError, match="no recorded consent"):
        save_response(study, "P01", "1", values)
    with pytest.raises(StudyError, match="all consent items"):
        record_consent(study, "P01", {**CONSENT, "agrees_to_take_part": False})

    assert record_consent(study, "P01", CONSENT) is True
    assert record_consent(study, "P01", CONSENT) is False
    consent = read(study / "consent.csv")
    assert [row["participant_id"] for row in consent] == ["P01"]
    assert set(consent[0]) == {"participant_id", "consented_utc", *CONSENT_ITEMS}


def test_saved_response_updates_only_its_row(tmp_path):
    study = copy_study(tmp_path, {})
    record_consent(study, "P03", CONSENT)

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


def test_study_page_needs_consent_and_warm_up_then_saves_a_task(tmp_path, monkeypatch):
    study = copy_study(tmp_path, {})
    monkeypatch.setenv("MM_STUDY_DIR", str(study))
    app = AppTest.from_file(str(STUDY_APP)).run()
    assert not app.exception

    app.selectbox[0].select("P02").run()
    start = next(button for button in app.button if button.label == "Start session")
    assert start.disabled
    for box in app.checkbox:
        box.check()
    app.run()
    next(button for button in app.button if button.label == "Start session").click().run()
    assert has_consent(study, "P02")

    timer_start = next(button for button in app.button if button.key == "P02-1-start")
    assert timer_start.disabled  # P02 starts with MelbourneMate, so warm-up comes first
    monkeypatch.setattr(study_module, "warm_up", lambda: 3)
    next(button for button in app.button if button.key == "P02-warm-button").click().run()
    assert not next(button for button in app.button if button.key == "P02-1-start").disabled
    assert any("warmed up in 3 seconds" in item.value for item in app.success)

    app.number_input(key="P02-1-time").set_value(75)
    app.radio(key="P02-1-done").set_value("yes")
    app.radio(key="P02-1-conf").set_value(4)
    app.radio(key="P02-1-trust").set_value(5)
    app.run()
    next(button for button in app.button if button.key == "P02-1-save").click().run()

    assert not app.exception
    saved = [row for row in read(study / "responses.csv") if row["completed"]]
    assert [(r["participant_id"], r["task_id"], r["time_seconds"]) for r in saved] == [
        ("P02", "TK4", "75")
    ]
