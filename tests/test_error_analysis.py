import csv
from pathlib import Path

import pytest

from melbourne_mate.cli import main
from melbourne_mate.evaluation.error_analysis import (
    ErrorAnalysisError,
    build_error_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def trace(
    question_id,
    *,
    answerable=True,
    risk="general",
    form="canonical",
    status="answered",
    retrieved=(),
    citation_valid=True,
):
    return {
        "question_id": question_id,
        "answerable": answerable,
        "risk_category": risk,
        "question_form": form,
        "status": status,
        "truncated": False,
        "retrieved": list(retrieved),
        "citations": {"valid": citation_valid},
    }


def test_builds_retrieval_and_generation_error_cases():
    bm25 = {
        "Q1": {"ndcg@5": 0.0, "recall@5": 0.0},
        "Q2": {"ndcg@5": 1.0, "recall@5": 1.0},
        "Q3": {"ndcg@5": 0.0, "recall@5": 0.0},
    }
    mpnet = {
        "Q1": {"ndcg@5": 1.0, "recall@5": 1.0},
        "Q2": {"ndcg@5": 0.0, "recall@5": 1.0},
        "Q3": {"ndcg@5": 0.0, "recall@5": 0.0},
    }
    generation = {
        "bm25": [
            trace(
                "Q1",
                risk="visa",
                form="paraphrased",
                status="refused",
                retrieved=("P1",),
            ),
            trace("Q2", retrieved=("P2",), citation_valid=False),
            trace("Q3"),
            trace("QX", answerable=False, risk="health", status="refused"),
        ],
        "mpnet": [
            trace("Q1", risk="visa", form="paraphrased", retrieved=("P1",)),
            trace("Q2", retrieved=("P2",)),
            trace("Q3"),
            trace("QX", answerable=False, risk="health"),
        ],
        "no-context": [
            trace("Q1", risk="visa", form="paraphrased"),
            trace("Q2"),
            trace("Q3"),
            trace("QX", answerable=False, risk="health"),
        ],
    }

    cases = build_error_cases(
        bm25,
        mpnet,
        generation,
        {"Q1": {"P1": 1}, "Q2": {"P2": 1}, "Q3": {"P3": 1}},
    )
    found = {(case.question_id, case.arm, case.error_type) for case in cases}

    assert ("Q1", "mpnet", "mpnet_better_paraphrase") in found
    assert ("Q2", "bm25", "bm25_better") in found
    assert ("Q3", "both", "both_retrieval_fail") in found
    assert ("Q1", "bm25", "retrieval_hit_but_refused") in found
    assert ("Q2", "bm25", "invalid_citation") in found
    assert ("Q3", "bm25", "retrieval_miss_but_answered") in found
    assert ("Q3", "mpnet", "retrieval_miss_but_answered") in found
    assert ("QX", "mpnet", "ookb_answered") in found
    assert ("QX", "no-context", "ookb_answered") in found
    assert next(case for case in cases if case.question_id == "Q1").high_risk
    assert not next(case for case in cases if case.question_id == "Q2").high_risk


def test_rejects_different_retrieval_question_sets():
    with pytest.raises(ErrorAnalysisError, match="different questions"):
        build_error_cases(
            {"Q1": {"ndcg@5": 1.0, "recall@5": 1.0}},
            {"Q2": {"ndcg@5": 1.0, "recall@5": 1.0}},
            {"bm25": [], "mpnet": [], "no-context": []},
            {},
        )


def test_cli_writes_error_analysis_from_saved_runs(tmp_path):
    out = tmp_path / "analysis"
    status = main(
        [
            "--data",
            str(ROOT / "data/v1"),
            "analyse-errors",
            "--qrels",
            str(ROOT / "review/targeted-qrels/qrels-final.txt"),
            "--bm25-retrieval",
            str(ROOT / "runs/retrieval/s9-final-bm25-test"),
            "--mpnet-retrieval",
            str(ROOT / "runs/retrieval/s9-final-mpnet-test"),
            "--bm25-generation",
            str(ROOT / "runs/generation/s9-qwen25-20260927-bm25"),
            "--mpnet-generation",
            str(ROOT / "runs/generation/s9-qwen25-20260927-mpnet"),
            "--no-context-generation",
            str(ROOT / "runs/generation/s9-qwen25-20260927-no-context"),
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert {path.name for path in out.iterdir()} == {
        "cases.csv",
        "manifest.json",
        "report.md",
        "summary.csv",
    }
    with (out / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "high_risk_failure" for row in rows)
