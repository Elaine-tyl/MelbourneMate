import csv
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from melbourne_mate.corpus import load_collection
from melbourne_mate.generation.citations import CitationReport
from melbourne_mate.generation.prompts import EvidenceItem
from melbourne_mate.interface.app import (
    EXAMPLE_QUESTIONS,
    _load_app_collection,
    answer_question,
    answer_view,
)
from melbourne_mate.pipeline import Answer


def test_answer_view_lists_only_cited_official_sources():
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
            "passage_ids": "P04-1",
        },
    )


def test_answer_view_marks_refusal_for_an_official_check():
    evidence = (
        EvidenceItem("P20-1", "Housing", "Text", "Study Melbourne", "https://sm.example"),
    )
    view = answer_view(
        Answer(
            "What is the cheapest rent today?",
            "refused",
            "Check an official source.",
            evidence=evidence,
        )
    )

    assert view["label"] == "Needs an official check"
    assert view["tone"] == "warning"
    assert view["citation_valid"] is None
    assert view["sources"] == ()


def test_answer_question_uses_qrels_for_a_saved_example():
    class PipelineStub:
        collection = load_collection("data/v1")
        relevant_ids = ()

        def answer(self, question, relevant_ids=(), on_chunk=None):
            self.relevant_ids = relevant_ids
            return Answer(question, "answered", "Answer")

    pipeline = PipelineStub()
    answer_question(pipeline, EXAMPLE_QUESTIONS["Work"])

    assert pipeline.relevant_ids == tuple(
        sorted(pipeline.collection.qrels["Q04P1"])
    )


def test_answer_question_forwards_streamed_text_to_the_ui():
    class PipelineStub:
        collection = load_collection("data/v1")

        def answer(self, question, relevant_ids=(), on_chunk=None):
            assert on_chunk is not None
            on_chunk("First ")
            on_chunk("sentence.")
            return Answer(question, "answered", "First sentence.")

    pieces = []
    answer = answer_question(
        PipelineStub(),
        "A question not in the saved collection",
        on_chunk=pieces.append,
    )

    assert pieces == ["First ", "sentence."]
    assert answer.text == "First sentence."


def test_app_collection_uses_the_formal_qrels():
    root = Path(__file__).parents[1]

    collection = _load_app_collection(
        root / "data/v1",
        root / "review/targeted-qrels/qrels-final.txt",
    )

    assert "P03-1" in collection.qrels["Q04P1"]


def test_four_example_tasks_cover_the_required_cases():
    assert set(EXAMPLE_QUESTIONS) == {"Work", "Health", "Housing", "Outside our sources"}


def test_streamlit_page_loads_before_models_are_needed():
    app_path = Path(__file__).parents[1] / "src/melbourne_mate/interface/app.py"
    app = AppTest.from_file(app_path).run()

    assert not app.exception
    assert app.title[0].value == "MelbourneMate"
    assert app.segmented_control[0].options == [
        "Ask MelbourneMate",
        "Evaluation Results",
    ]
    assert app.segmented_control[0].value == "Ask MelbourneMate"
    assert not app.header
    labels = [button.label for button in app.button]
    assert set(labels[:4]) == set(EXAMPLE_QUESTIONS)
    assert labels[-1] == "Find an answer"


def test_streamlit_shows_saved_evaluation_results_before_models_are_needed():
    app_path = Path(__file__).parents[1] / "src/melbourne_mate/interface/app.py"
    app = AppTest.from_file(app_path).run()
    app.segmented_control[0].select("Evaluation Results").run()

    assert not app.expander
    assert [item.value for item in app.header] == ["Evaluation Results"]
    assert not app.text_area
    assert all(button.label != "Find an answer" for button in app.button)
    assert not app.tabs
    headings = [item.value for item in app.subheader]
    assert "Dense encoder selection" not in headings
    assert "Held-out retrieval performance" not in headings
    assert "Retrieval by question type" not in headings
    assert "Retrieval Metrics for Known and Inferred Questions" in headings
    assert "NDCG@5 differences and uncertainty" not in headings
    assert "OOKB Refusal Safety by Generation Arm" in headings
    charts = app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    assert len(charts) == 2
    sliced = json.loads(charts[0].proto.spec)
    safety = json.loads(charts[1].proto.spec)
    assert sliced["facet"]["column"]["field"] == "Metric"
    assert sliced["spec"]["layer"][0]["encoding"]["x"]["field"] == "Method"
    assert (
        sliced["spec"]["layer"][0]["encoding"]["xOffset"]["field"]
        == "Question type"
    )
    assert sliced["spec"]["layer"][0]["encoding"]["y"]["field"] == "Score"
    assert any(
        "Retrieval method" in item.value
        for item in app.markdown
    )
    assert safety["layer"][0]["encoding"]["x"]["field"] == "Arm"
    assert "xOffset" not in safety["layer"][0]["encoding"]
    assert safety["layer"][0]["encoding"]["y"]["field"] == "Rate"
    assert len(app.table) == 2


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
