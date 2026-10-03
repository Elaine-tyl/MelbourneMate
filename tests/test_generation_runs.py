import csv
import json

import pytest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.generation_runs import (
    GenerationRunItem,
    rescore_generation_run,
    write_generation_run,
)
from melbourne_mate.evaluation.runs import RunError
from melbourne_mate.generation.citations import CitationReport
from melbourne_mate.pipeline import Answer
from melbourne_mate.retrieval.base import Hit


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


def test_rescore_generation_updates_labels_without_changing_answer(tmp_path):
    collection = load_collection("data/sample")
    source = tmp_path / "source"
    answer_text = "The official rule is explained here [P001]."
    # Simulate a citation judged irrelevant before the qrels correction.
    item = GenerationRunItem(
        question_id="Q001",
        answer=Answer(
            question=collection.questions["Q001"].text,
            status="answered",
            text=answer_text,
            hits=(Hit("P001", 1, 1.0),),
            citations=CitationReport(
                cited=("P001",),
                retrieved=("P001",),
                unsupported=(),
                irrelevant=("P001",),
                uncited_urls=(),
                unsupported_numbers=(),
                has_citation=True,
            ),
            arm="bm25",
        ),
        answerable=True,
        retrieval_hit=False,
    )
    write_generation_run(
        source,
        run_id="source-run",
        arm="bm25",
        split="test",
        encoder="bm25s@version",
        collection_fingerprint=collection.fingerprint(),
        sample_fingerprint="sample123",
        qrels_fingerprint="old-qrels",
        answers=[item],
        model="qwen2.5:7b-instruct",
    )
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["created_utc"] = "2026-09-27T13:28:40+00:00"
    source_manifest["source_run"] = "formal-source-run"
    source_manifest_path.write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )

    output = tmp_path / "rescored"
    # The corrected qrels should update labels while preserving generated text.
    rescore_generation_run(
        output,
        source_run=source,
        collection=collection,
        qrels={"Q001": {"P001": 2}},
        qrels_fingerprint="new-qrels",
        run_id="rescored-run",
    )

    trace = json.loads((output / "answers.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = read_summary(output)

    assert trace["answer"] == answer_text
    assert trace["citations"]["valid"] is True
    assert trace["citations"]["irrelevant"] == []
    assert summary["citation_validity_rate"] == "1.0"
    assert manifest["qrels_fingerprint"] == "new-qrels"
    assert manifest["created_utc"] == source_manifest["created_utc"]
    assert manifest["source_run"] == "formal-source-run"
    assert manifest["rescored_utc"]
    assert "model output unchanged" in manifest["note"]


def test_rescore_generation_cli_uses_saved_answers(tmp_path):
    collection = load_collection("data/sample")
    source = tmp_path / "source"
    item = GenerationRunItem(
        question_id="Q001",
        answer=Answer(
            question=collection.questions["Q001"].text,
            status="answered",
            text="See the official rule [P001].",
            hits=(Hit("P001", 1, 1.0),),
            citations=valid_citations(),
            arm="bm25",
        ),
        answerable=True,
        retrieval_hit=True,
    )
    write_generation_run(
        source,
        run_id="source-run",
        arm="bm25",
        split="test",
        encoder="bm25s@version",
        collection_fingerprint=collection.fingerprint(),
        qrels_fingerprint="old-qrels",
        answers=[item],
        model="qwen2.5:7b-instruct",
    )
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("Q001 0 P001 2\n", encoding="utf-8")

    # Exercise the public CLI path, not only the underlying Python function.
    result = cli.main(
        [
            "--data",
            "data/sample",
            "rescore-generation",
            "--source-run",
            str(source),
            "--qrels",
            str(qrels_path),
            "--run-id",
            "rescored",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert result == 0
    assert (tmp_path / "runs" / "rescored" / "manifest.json").exists()
