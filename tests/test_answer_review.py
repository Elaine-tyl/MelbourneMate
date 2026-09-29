import csv
import json

import pytest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.answer_review import (
    AnswerReviewError,
    prepare_answer_review,
    score_answer_review,
)


def write_run(path, arm, collection, rows):
    path.mkdir()
    manifest = {
        "arm": arm,
        "collection_fingerprint": collection.fingerprint(),
        "config_fingerprint": "cfg",
        "model_digest": "model",
        "qrels_fingerprint": "qrels",
        "run_id": path.name,
        "sample_fingerprint": "sample",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "answers.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def trace(question_id, question, answerable, arm):
    return {
        "question_id": question_id,
        "question": question,
        "answerable": answerable,
        "answer": f"Answer from {arm}",
        "retrieved": ["P04-1"] if arm != "none" else [],
    }


@pytest.fixture
def review_runs(tmp_path):
    collection = load_collection("data/v1")
    questions = [
        ("Q04C", True),
        ("Q04P1", True),
        ("QX01", False),
        ("QX02", False),
    ]
    runs = {}
    for label, arm in (("bm25", "bm25"), ("mpnet", "dense"), ("no-context", "none")):
        run = tmp_path / label
        write_run(
            run,
            arm,
            collection,
            [
                trace(question_id, collection.questions[question_id].text, answerable, arm)
                for question_id, answerable in questions
            ],
        )
        runs[label] = run
    return collection, runs


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_prepare_answer_review_is_balanced_and_blind(tmp_path, review_runs):
    collection, runs = review_runs
    out = tmp_path / "review"

    prepare_answer_review(out, collection, runs, size=6, seed=7)

    elaine = read_csv(out / "review-elaine.csv")
    sriporn = read_csv(out / "review-sriporn.csv")
    key = read_csv(out / "key.csv")
    assert len(elaine) == len(sriporn) == len(key) == 6
    assert [row["item_id"] for row in elaine] == [row["item_id"] for row in sriporn]
    assert "arm" not in elaine[0]
    assert {row["arm"] for row in key} == {"bm25", "mpnet", "no-context"}
    assert all(
        row[field] == ""
        for row in elaine
        for field in ("correctness", "evidence_support", "fallback_appropriateness")
    )


def test_score_answer_review_requires_completed_sheets(tmp_path, review_runs):
    collection, runs = review_runs
    out = tmp_path / "review"
    prepare_answer_review(out, collection, runs, size=6, seed=7)

    with pytest.raises(AnswerReviewError, match="unfilled score"):
        score_answer_review(out)


def test_score_answer_review_writes_agreement_and_arm_means(tmp_path, review_runs):
    collection, runs = review_runs
    out = tmp_path / "review"
    prepare_answer_review(out, collection, runs, size=6, seed=7)
    for reviewer in ("elaine", "sriporn"):
        path = out / f"review-{reviewer}.csv"
        rows = read_csv(path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                row.update(
                    correctness="2",
                    evidence_support="2",
                    fallback_appropriateness="2",
                )
                writer.writerow(row)

    score_answer_review(out)

    agreement = read_csv(out / "agreement.csv")
    summary = read_csv(out / "summary.csv")
    assert all(row["exact_agreement"] == "1.0" for row in agreement)
    assert all(row["weighted_kappa"] == "1.0" for row in agreement)
    assert {row["arm"] for row in summary} == {"bm25", "mpnet", "no-context"}
    assert (out / "report.md").exists()


def test_cli_prepares_current_answer_review(tmp_path, review_runs):
    _, runs = review_runs
    out = tmp_path / "review"

    status = cli.main(
        [
            "--data",
            "data/v1",
            "prepare-answer-review",
            "--bm25-run",
            str(runs["bm25"]),
            "--mpnet-run",
            str(runs["mpnet"]),
            "--no-context-run",
            str(runs["no-context"]),
            "--size",
            "6",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert (out / "review-elaine.csv").exists()
