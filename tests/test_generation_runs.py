import csv
import json

import pytest

from melbourne_mate.evaluation.generation_runs import (
    GenerationRunItem,
    write_generation_run,
)
from melbourne_mate.evaluation.runs import RunError
from melbourne_mate.generation.citations import CitationReport
from melbourne_mate.pipeline import Answer


def valid_citations():
    return CitationReport(
        cited=("P001",),
        retrieved=("P001",),
        unsupported=(),
        irrelevant=(),
        uncited_urls=(),
        unsupported_numbers=(),
        has_citation=True,
    )


def answered_item(arm="bm25", risk="visa", answerable=False):
    return GenerationRunItem(
        question_id="QX09" if not answerable else "Q04C",
        answer=Answer(
            question="Will my visa be approved?",
            status="answered",
            text="This cannot be guaranteed [P001].",
            citations=valid_citations(),
            arm=arm,
        ),
        answerable=answerable,
        retrieval_hit=answerable,
        risk_category=risk,
        language="en",
        question_form="ookb" if not answerable else "canonical",
    )


def write_run(path, arm="bm25"):
    return write_generation_run(
        path,
        run_id="test-run",
        arm=arm,
        split="test",
        encoder="mpnet@revision",
        collection_fingerprint="collection123",
        sample_fingerprint="sample123",
        qrels_fingerprint="qrels123",
        answers=[answered_item(arm=arm)],
        model="qwen2.5:7b-instruct",
        model_digest="845dbda0ea48",
    )


def read_summary(path):
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def test_run_saves_question_and_reproduction_metadata(tmp_path):
    path = tmp_path / "run"
    write_run(path)

    trace = json.loads((path / "answers.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    summary = read_summary(path)

    assert trace["risk_category"] == "visa"
    assert trace["language"] == "en"
    assert trace["question_form"] == "ookb"
    assert manifest["sample_fingerprint"] == "sample123"
    assert manifest["qrels_fingerprint"] == "qrels123"
    assert manifest["model_digest"] == "845dbda0ea48"
    assert summary["high_risk_unsupported_rate"] == "1.0"


def test_no_context_does_not_score_citations(tmp_path):
    path = tmp_path / "none"
    write_run(path, arm="none")

    summary = read_summary(path)

    assert summary["citation_validity_rate"] == ""
    assert summary["false_refusal_rate"] == ""


def test_existing_run_is_not_changed(tmp_path):
    path = tmp_path / "run"
    write_run(path)
    original = (path / "manifest.json").read_bytes()

    with pytest.raises(RunError, match="already exists"):
        write_run(path)

    assert (path / "manifest.json").read_bytes() == original


def test_run_csv_files_use_lf_line_endings(tmp_path):
    path = tmp_path / "run"
    write_run(path)

    assert b"\r\n" not in (path / "summary.csv").read_bytes()
    assert b"\r\n" not in (path / "contingency.csv").read_bytes()
