import csv
import json
import shutil
from pathlib import Path

from melbourne_mate import cli


def write_sample(path, question_id="Q001", risk="visa"):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "risk_category"])
        writer.writeheader()
        writer.writerow({"question_id": question_id, "risk_category": risk})


def test_generate_uses_external_qrels_and_saves_fingerprints(tmp_path):
    sample = tmp_path / "sample.csv"
    qrels = tmp_path / "qrels.txt"
    runs = tmp_path / "runs"
    write_sample(sample)
    qrels.write_text("Q001 0 P001 2\n", encoding="utf-8")

    result = cli.main(
        [
            "--data",
            "data/sample",
            "generate",
            "--arm",
            "none",
            "--run-id",
            "one",
            "--sample",
            str(sample),
            "--qrels",
            str(qrels),
            "--offline",
            "--runs-dir",
            str(runs),
        ]
    )

    manifest = json.loads((runs / "one/manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((runs / "one/answers.jsonl").read_text(encoding="utf-8"))
    assert result == 0
    assert len(manifest["sample_fingerprint"]) == 16
    assert len(manifest["qrels_fingerprint"]) == 16
    assert trace["risk_category"] == "visa"


def test_evaluate_generation_creates_three_matching_runs(tmp_path):
    sample = tmp_path / "sample.csv"
    qrels = tmp_path / "qrels.txt"
    runs = tmp_path / "runs"
    write_sample(sample)
    qrels.write_text("Q001 0 P001 2\n", encoding="utf-8")

    result = cli.main(
        [
            "--data",
            "data/sample",
            "evaluate-generation",
            "--run-prefix",
            "quick",
            "--sample",
            str(sample),
            "--qrels",
            str(qrels),
            "--encoder",
            "hashing",
            "--offline",
            "--runs-dir",
            str(runs),
        ]
    )

    manifests = [
        json.loads((runs / f"quick-{arm}/manifest.json").read_text(encoding="utf-8"))
        for arm in ("bm25", "mpnet", "no-context")
    ]
    assert result == 0
    assert [item["arm"] for item in manifests] == ["bm25", "dense", "none"]
    assert {item["questions"] for item in manifests} == {1}
    assert len({item["sample_fingerprint"] for item in manifests}) == 1
    assert len({item["qrels_fingerprint"] for item in manifests}) == 1
    assert len({item["model"] for item in manifests}) == 1


def test_evaluate_generation_stops_when_a_target_run_exists(tmp_path, capsys):
    sample = tmp_path / "sample.csv"
    qrels = tmp_path / "qrels.txt"
    runs = tmp_path / "runs"
    existing = runs / "quick-mpnet"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    write_sample(sample)
    qrels.write_text("Q001 0 P001 2\n", encoding="utf-8")

    result = cli.main(
        [
            "--data",
            "data/sample",
            "evaluate-generation",
            "--run-prefix",
            "quick",
            "--sample",
            str(sample),
            "--qrels",
            str(qrels),
            "--offline",
            "--runs-dir",
            str(runs),
        ]
    )

    assert result == 1
    assert "target run already exists" in capsys.readouterr().err
    assert not (runs / "quick-bm25").exists()
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_evaluate_generation_accepts_sample_relative_to_data_dir(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    shutil.copytree(Path("data/sample").resolve(), data)
    sample = data / "generation-sample.csv"
    qrels = tmp_path / "qrels.txt"
    write_sample(sample)
    qrels.write_text("Q001 0 P001 2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "--data",
            "data",
            "evaluate-generation",
            "--run-prefix",
            "relative",
            "--sample",
            "generation-sample.csv",
            "--qrels",
            str(qrels),
            "--encoder",
            "hashing",
            "--offline",
            "--runs-dir",
            str(runs),
        ]
    )

    assert result == 0
    assert (runs / "relative-bm25/manifest.json").exists()
