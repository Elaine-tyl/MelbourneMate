import csv
import hashlib
import json

from melbourne_mate import cli
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.metrics import METRICS
from melbourne_mate.evaluation.runs import read_run_file, write_run


def _source_run(tmp_path):
    collection = load_collection("data/sample")
    rankings = {
        question.question_id: ["P005", "P006", "P007", "P008"]
        for question in collection.questions_in_split("test")
    }
    scores = {
        question_id: {metric: 0.0 for metric in METRICS}
        for question_id in rankings
    }
    path = write_run(
        tmp_path / "source",
        run_id="source",
        arm="dense",
        split="test",
        encoder="fixed-encoder@revision",
        collection_fingerprint=collection.fingerprint(),
        rankings=rankings,
        per_question=scores,
    )
    return path, rankings


def test_rescore_retrieval_uses_saved_rankings_and_final_qrels(
    monkeypatch, tmp_path
):
    source, rankings = _source_run(tmp_path)
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["created_utc"] = "2026-09-27T11:18:58+00:00"
    source_manifest["source_run"] = "formal-source"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    qrels = tmp_path / "final-qrels.txt"
    qrels.write_text(
        "\n".join(
            f"{question_id} 0 P006 2" for question_id in sorted(rankings)
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_loaded(_name):
        raise AssertionError("rescoring must not load an encoder")

    monkeypatch.setattr(cli, "load_encoder", fail_if_loaded)
    out = tmp_path / "runs"
    result = cli.main(
        [
            "--data",
            "data/sample",
            "rescore-retrieval",
            "--source-run",
            str(source),
            "--qrels",
            str(qrels),
            "--run-id",
            "rescored",
            "--runs-dir",
            str(out),
        ]
    )

    assert result == 0
    rescored = out / "rescored"
    assert read_run_file(rescored) == rankings
    manifest = json.loads((rescored / "manifest.json").read_text())
    assert manifest["qrels_fingerprint"] == hashlib.sha256(
        qrels.read_bytes()
    ).hexdigest()[:16]
    assert manifest["created_utc"] == source_manifest["created_utc"]
    assert manifest["source_run"] == "formal-source"
    assert manifest["rescored_utc"]
    assert manifest["note"].startswith("Rescored from formal-source")
    assert b"\r\n" not in (rescored / "aggregate_metrics.csv").read_bytes()


def test_compare_rejects_different_qrels_fingerprints(tmp_path, capsys):
    left, _ = _source_run(tmp_path / "left")
    right, _ = _source_run(tmp_path / "right")
    for path, fingerprint in ((left, "qrels-a"), (right, "qrels-b")):
        manifest_path = path / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["qrels_fingerprint"] = fingerprint
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = cli.main(
        [
            "--data",
            "data/sample",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
        ]
    )

    assert result == 1
    assert "different qrels" in capsys.readouterr().err


def test_compare_rejects_collection_that_does_not_match_runs(tmp_path, capsys):
    left, _ = _source_run(tmp_path / "left")
    right, _ = _source_run(tmp_path / "right")

    result = cli.main(
        [
            "--data",
            "data/v1",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
        ]
    )

    assert result == 1
    assert "different collection" in capsys.readouterr().err


def test_compare_can_save_the_paired_result(tmp_path):
    left, _ = _source_run(tmp_path / "left")
    right, _ = _source_run(tmp_path / "right")
    out = tmp_path / "comparison.csv"

    result = cli.main(
        [
            "--data",
            "data/sample",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--out",
            str(out),
        ]
    )

    assert result == 0
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["metric"] == "ndcg@5"
    assert rows[0]["left_run"] == "source"
    assert rows[0]["right_run"] == "source"
    assert rows[0]["observations"] == "6"
    assert b"\r\n" not in out.read_bytes()
