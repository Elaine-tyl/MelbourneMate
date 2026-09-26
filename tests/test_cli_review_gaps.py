import argparse
import json

from melbourne_mate import cli
from melbourne_mate.config import CONFIG


def _manifest(run_id, gain):
    return {
        "run_id": run_id,
        "arm": "dense",
        "split": "test",
        "encoder": "multi-qa-mpnet",
        "collection_fingerprint": "collection-1",
        "config_fingerprint": "config-1",
        "protocol_version": "v1",
        "ndcg_gain": gain,
        "questions": 1,
        "created_utc": "2026-09-27T00:00:00+00:00",
        "code_version": "0.1.0",
        "python_version": "3.12.0",
        "platform": "test",
        "note": "",
    }


def test_generate_accepts_no_context_without_loading_an_encoder(
    monkeypatch, tmp_path
):
    sample = tmp_path / "questions.txt"
    sample.write_text("Q001\n", encoding="utf-8")

    def fail_if_loaded(_name):
        raise AssertionError("no-context generation must not load an encoder")

    monkeypatch.setattr(cli, "load_encoder", fail_if_loaded)

    result = cli.main(
        [
            "--data",
            "data/sample",
            "generate",
            "--arm",
            "none",
            "--run-id",
            "no-context-test",
            "--sample",
            str(sample),
            "--offline",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert result == 0
    manifest = json.loads(
        (tmp_path / "runs/no-context-test/manifest.json").read_text()
    )
    assert manifest["arm"] == "none"
    assert manifest["encoder"] == "none"


def test_compare_rejects_different_ndcg_gain(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "manifest.json").write_text(
        json.dumps(_manifest("left", "exponential")), encoding="utf-8"
    )
    (right / "manifest.json").write_text(
        json.dumps(_manifest("right", "linear")), encoding="utf-8"
    )

    result = cli.cmd_compare(
        argparse.Namespace(
            data="data/sample",
            left=left,
            right=right,
            metric="ndcg@5",
        )
    )

    assert result == 1
    assert "different NDCG gain" in capsys.readouterr().err


def test_config_has_only_the_retrieval_depth_that_is_used():
    assert not hasattr(CONFIG, "candidate_depth")
    assert "candidate_depth" not in CONFIG.to_yaml()
