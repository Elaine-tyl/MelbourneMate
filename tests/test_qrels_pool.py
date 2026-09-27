import csv
from types import SimpleNamespace

import pytest

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.metrics import METRICS
from melbourne_mate.evaluation.runs import write_run
from melbourne_mate.evaluation.targeted_qrels import build_review_pool


def _save_run(tmp_path, collection, rankings):
    scores = {qid: {metric: 0.0 for metric in METRICS} for qid in rankings}
    return write_run(
        tmp_path / "run",
        run_id="edge-case",
        arm="dense",
        split="test",
        encoder="test-encoder@fixed",
        collection_fingerprint=collection.fingerprint(),
        rankings=rankings,
        per_question=scores,
    )


def test_formal_mpnet_pool_keeps_only_plausible_new_pairs(tmp_path):
    data = "data/v1"
    run = "runs/retrieval/s9-formal-mpnet-test"
    out = tmp_path / "mpnet-pairs.csv"

    result = build_review_pool(load_collection(data), run, out)

    assert result.candidates == 37
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 37
    assert all(row["grade"] == "" for row in rows)
    assert all(row["primary_reviewer"] == "" for row in rows)
    assert {row["reason"] for row in rows} <= {
        "same_category",
        "confusable_topics",
    }


def test_review_pool_rejects_a_collection_mismatch(tmp_path):
    collection = load_collection("data/sample")

    try:
        build_review_pool(
            collection,
            "runs/retrieval/s9-formal-mpnet-test",
            tmp_path / "pairs.csv",
        )
    except ValueError as exc:
        assert "collection fingerprint" in str(exc)
    else:
        raise AssertionError("mismatched collection should fail")


def test_review_pool_rejects_an_unknown_passage(tmp_path):
    collection = load_collection("data/sample")
    run = _save_run(tmp_path, collection, {"Q007": ["P999"]})

    with pytest.raises(ValueError, match="unknown passage P999"):
        build_review_pool(collection, run, tmp_path / "pairs.csv")


def test_review_pool_rejects_an_empty_candidate_set(tmp_path):
    collection = load_collection("data/sample")
    run = _save_run(tmp_path, collection, {"Q007": ["P005"]})

    with pytest.raises(ValueError, match="no review candidates"):
        build_review_pool(collection, run, tmp_path / "pairs.csv")


def test_targeted_qrels_can_use_question_ids_from_the_saved_run(
    monkeypatch, tmp_path
):
    seen = {}

    monkeypatch.setattr(cli, "read_run_file", lambda _path: {"Q002": [], "Q001": []})

    def build(base, review, out, *, question_ids):
        seen["question_ids"] = list(question_ids)
        return SimpleNamespace(
            qrels_path=tmp_path / "qrels.txt",
            seed_pairs=2,
            checked_candidates=1,
            added_positive_pairs=1,
        )

    monkeypatch.setattr(cli, "build_targeted_qrels", build)
    (tmp_path / "qrels.txt").write_text("Q001 0 P001 2\n", encoding="utf-8")

    result = cli.main(
        [
            "targeted-qrels",
            "--base-qrels",
            "base.txt",
            "--verification",
            "review.csv",
            "--run",
            "saved-run",
            "--out",
            str(tmp_path / "qrels.txt"),
        ]
    )

    assert result == 0
    assert seen["question_ids"] == ["Q001", "Q002"]
