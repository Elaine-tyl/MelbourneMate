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
    judge_completion,
    load_responses,
    load_would_use,
    participant_tasks,
    save_response,
    save_would_use,
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
    shutil.copytree(STUDY, target, ignore=shutil.ignore_patterns("responses.csv", "final.csv"))
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
        ["git", "check-ignore", "study/responses.csv", "study/final.csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tracked = subprocess.run(
        ["git", "ls-files", "study"], capture_output=True, text=True, check=True
    ).stdout.split()

    assert ignored == ["study/responses.csv", "study/final.csv"]
    assert sorted(tracked) == [
        "study/README.md",
        "study/final-template.csv",
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
            ("P01", "3"): ("no", "300", "2", "3"),
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
    assert "| Evidence | MelbourneMate | Official search |" in report
    assert "| Task completion | 50% | 50% |" in report
    assert "| Median time | 210 sec | 270 sec |" in report
    assert "Would use" not in report

    save_would_use(study, "P01", 4)
    summarise_study(study)
    report = (study / "summary.md").read_text(encoding="utf-8")
    assert "Mean 4.00/5 from 1 participants." in report


def test_would_use_rating_is_checked_and_saved(tmp_path):
    study = copy_study(tmp_path, {})

    save_would_use(study, "P02", "5")
    save_would_use(study, "P02", 3)

    assert load_would_use(study) == {"P02": 3}
    with pytest.raises(StudyError, match="outside 1-5"):
        save_would_use(study, "P02", 6)
    with pytest.raises(StudyError, match="not in final.csv"):
        save_would_use(study, "P09", 4)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("yes", "120", "", ""), "fill all"),
        (("done", "120", "4", "4"), "completed must be"),
        (("yes", "301", "4", "4"), "outside 1-300"),
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

    save_response(study, "P03", "2", {"time_seconds": 240, "confidence": 3, "trust": 4})

    attempts = load_responses(study)
    assert attempts == [
        {
            "participant_id": "P03",
            "task_id": "TK6",
            "method": "official-search",
            "completed": "",
            "time_seconds": 240,
            "confidence": 3,
            "trust": 4,
        }
    ]
    judge_completion(study, "P03", "2", "partial")
    save_response(study, "P03", "2", {"time_seconds": 250, "confidence": 3, "trust": 4})
    task = participant_tasks(study, "P03")[1]
    assert (task["completed"], task["time_seconds"]) == ("partial", "250")  # judgement kept
    with pytest.raises(StudyError, match="outside 1-5"):
        save_response(
            study, "P03", "1", {"completed": "yes", "time_seconds": 60, "confidence": 9, "trust": 4}
        )
    with pytest.raises(StudyError, match="not in schedule"):
        save_response(
            study, "P09", "1", {"completed": "yes", "time_seconds": 60, "confidence": 3, "trust": 4}
        )


def fake_chatbot(monkeypatch, asked):
    monkeypatch.setattr(chatbot, "_build_pipeline", lambda: "pipeline")
    monkeypatch.setattr(chatbot, "answer_question", lambda pipeline, q: asked.append(q) or q)
    monkeypatch.setattr(
        chatbot,
        "answer_view",
        lambda answer: {"tone": "success", "label": "Answer ready", "text": "Use the RTBA.", "sources": ()},
    )


def open_page(study, monkeypatch, participant=None):
    monkeypatch.setenv("MM_STUDY_DIR", str(study))
    app = AppTest.from_file(str(STUDY_APP)).run()
    assert not app.exception
    if participant:
        app.sidebar.selectbox[0].select(participant).run()
    return app


def button(app, key):
    return next(item for item in app.button if item.key == key)


def test_study_page_shows_the_participant_guide_and_waits_for_the_researcher(
    tmp_path, monkeypatch
):
    app = open_page(copy_study(tmp_path, {}), monkeypatch)

    assert "voluntary" in app.info[0].value
    assert not app.checkbox  # no consent form
    assert sorted(item.label for item in app.expander) == ["How it works", "Researcher steps"]
    assert any("one or two sentences" in item.value for item in app.markdown)
    assert any("in your own words" in item.value for item in app.markdown)
    assert any("researcher will start" in item.value for item in app.caption)


def test_study_page_warms_up_then_saves_a_task_and_moves_on(tmp_path, monkeypatch):
    study = copy_study(tmp_path, {})
    asked = []
    fake_chatbot(monkeypatch, asked)
    app = open_page(study, monkeypatch, "P02")
    task_text = participant_tasks(study, "P02")[0]["prompt"]

    assert any(item.value == "#### Task 1 of 4" for item in app.markdown)
    assert button(app, "P02-1-start").disabled  # P02 starts with MelbourneMate
    assert button(app, "P02-1-stop").disabled  # nothing to stop yet
    assert button(app, "P02-1-ask").disabled
    button(app, "P02-warm-button").click().run()
    assert asked == [WARM_UP_QUESTION]
    assert any("ready" in item.value for item in app.success)

    assert not button(app, "P02-1-start").disabled
    assert app.text_area(key="P02-1-question").value == task_text
    assert any("based on what MelbourneMate showed" in item.value for item in app.caption)
    button(app, "P02-1-ask").click().run()
    assert asked == [WARM_UP_QUESTION, task_text]
    assert any("Use the RTBA." in item.value for item in app.markdown)

    app.text_area(key="P02-1-answer").input("Lodge it with the RTBA within 14 days.")
    app.radio(key="P02-1-conf").set_value(4)
    app.radio(key="P02-1-trust").set_value(5)
    app.number_input(key="P02-1-time").set_value(75)
    app.run()
    button(app, "P02-1-save").click().run()

    assert not app.exception
    saved = [row for row in read(study / "responses.csv") if row["time_seconds"]]
    assert [(r["participant_id"], r["task_id"], r["time_seconds"]) for r in saved] == [
        ("P02", "TK4", "75")
    ]
    assert saved[0]["completed"] == ""  # judged later in the review step
    assert saved[0]["answer"] == "Lodge it with the RTBA within 14 days."
    assert any(item.value == "#### Task 2 of 4" for item in app.markdown)
    assert app.get("progress")[0].proto.value == 25


def test_completion_is_judged_after_the_session(tmp_path):
    study = copy_study(tmp_path, {("P01", "1"): ("", "120", "4", "4")})

    with pytest.raises(StudyError, match="review step first"):
        summarise_study(study)
    with pytest.raises(StudyError, match="no saved attempt"):
        judge_completion(study, "P01", "2", "yes")
    with pytest.raises(StudyError, match="completed must be"):
        judge_completion(study, "P01", "1", "done")

    judge_completion(study, "P01", "1", "yes")
    summarise_study(study)
    assert (study / "summary.md").exists()


def test_completion_without_an_attempt_is_rejected(tmp_path):
    study = copy_study(tmp_path, {("P01", "1"): ("yes", "", "", "")})

    with pytest.raises(StudyError, match="only after an attempt"):
        load_responses(study)


def test_task_card_never_shows_completion_rules(tmp_path, monkeypatch):
    app = open_page(copy_study(tmp_path, {}), monkeypatch, "P03")  # starts with official search
    assert any("based on the official websites you found" in item.value for item in app.caption)

    assert not any("Complete when" in item.value for item in [*app.markdown, *app.caption])
    assert not app.toggle
    assert not any(item.key == "P03-1-done" for item in app.radio)


def test_review_judges_saved_answers_out_of_view(tmp_path, monkeypatch):
    saved = ("", "120", "4", "4")
    study = copy_study(tmp_path, {("P01", "1"): saved, ("P01", "3"): saved})
    app = open_page(study, monkeypatch, "P01")
    app.sidebar.radio[0].set_value("review").run()

    assert app.subheader[0].value == "Review (researcher only)"
    assert sum("Complete when" in item.value for item in app.markdown) == 2
    assert button(app, "P01-judge-save").disabled
    app.radio(key="P01-1-judge").set_value("yes")
    app.radio(key="P01-3-judge").set_value("partial")
    app.run()
    button(app, "P01-judge-save").click().run()

    assert not app.exception
    judged = {r["position"]: r["completed"] for r in read(study / "responses.csv") if r["participant_id"] == "P01"}
    assert judged == {"1": "yes", "2": "", "3": "partial", "4": ""}
    assert any("All saved tasks are judged" in item.value for item in app.success)


def test_final_question_appears_after_all_tasks_and_is_saved(tmp_path, monkeypatch):
    done = ("yes", "120", "4", "4")
    study = copy_study(tmp_path, {("P03", str(n)): done for n in range(1, 5)})
    app = open_page(study, monkeypatch, "P03")

    assert app.subheader[-1].value == "Final question"
    app.radio(key="P03-use").set_value(5).run()
    button(app, "P03-use-save").click().run()

    assert not app.exception
    assert load_would_use(study) == {"P03": 5}
    assert any("Final answer saved (5/5)" in item.value for item in app.success)


def test_missing_retriever_shows_a_clear_message(tmp_path, monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'sentence_transformers'")

    monkeypatch.setattr(chatbot, "_build_pipeline", missing)
    app = open_page(copy_study(tmp_path, {}), monkeypatch, "P01")
    button(app, "P01-warm-button").click().run()

    assert not app.exception
    assert any("MPNet retriever is not installed" in item.value for item in app.error)
    assert button(app, "P01-1-start").disabled
