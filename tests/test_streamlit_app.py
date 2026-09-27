import csv
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from melbourne_mate.generation.citations import CitationReport
from melbourne_mate.generation.prompts import EvidenceItem
from melbourne_mate.interface.app import EXAMPLE_QUESTIONS, answer_view
from melbourne_mate.pipeline import Answer


def test_answer_view_deduplicates_official_sources():
    evidence = (
        EvidenceItem("P04-1", "Pay rights", "First", "Fair Work", "https://fw.example"),
        EvidenceItem("P04-2", "Get help", "Second", "Fair Work", "https://fw.example"),
    )
    citations = CitationReport(
        cited=("P04-1",),
        retrieved=("P04-1", "P04-2"),
        unsupported=(),
        irrelevant=(),
        uncited_urls=(),
        unsupported_numbers=(),
        has_citation=True,
    )

    view = answer_view(
        Answer("Where can I get pay help?", "answered", "Use Fair Work [P04-1].", evidence=evidence, citations=citations)
    )

    assert view["label"] == "Answer ready"
    assert view["tone"] == "success"
    assert view["citation_valid"] is True
    assert view["sources"] == (
        {
            "organisation": "Fair Work",
            "heading": "Pay rights",
            "url": "https://fw.example",
            "passage_ids": "P04-1, P04-2",
        },
    )


def test_answer_view_marks_refusal_for_an_official_check():
    view = answer_view(Answer("What is the cheapest rent today?", "refused", "Check an official source."))

    assert view["label"] == "Needs an official check"
    assert view["tone"] == "warning"
    assert view["citation_valid"] is None


def test_four_example_tasks_cover_the_required_cases():
    assert set(EXAMPLE_QUESTIONS) == {"Work", "Health", "Housing", "Outside our sources"}


def test_streamlit_page_loads_before_models_are_needed():
    app_path = Path(__file__).parents[1] / "src/melbourne_mate/interface/app.py"
    app = AppTest.from_file(app_path).run()

    assert not app.exception
    assert app.title[0].value == "🧭 MelbourneMate"
    labels = [button.label for button in app.button]
    assert set(labels[:4]) == set(EXAMPLE_QUESTIONS)
    assert labels[-1] == "Find an answer"


def test_four_task_record_matches_the_saved_mpnet_run():
    root = Path(__file__).parents[1]
    with (root / "review/chatbot-test-s9/results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        checks = list(csv.DictReader(handle))
    traces = {
        row["question_id"]: row
        for row in (
            json.loads(line)
            for line in (
                root
                / "runs/generation/s9-qwen25-20260927-mpnet/answers.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        )
    }

    assert len(checks) == 4
    for check in checks:
        trace = traces[check["question_id"]]
        citation_valid = (
            str(trace["citations"]["valid"]).lower()
            if trace["citations"]
            else "n.a."
        )
        assert check["observed_status"] == trace["status"]
        assert check["citation_valid"] == citation_valid
