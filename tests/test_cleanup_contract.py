import pytest

from melbourne_mate import cli
from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.collection_quality import build_report


def test_quality_report_matches_targeted_qrels_workflow():
    report = "\n".join(build_report(load_collection("data/sample")).lines())

    assert "judged (question, passage) pairs" in report
    assert "double-judged" not in report
    assert "Cohen's kappa" not in report
    assert "confusable topic pairs" not in report


def test_unused_multi_qa_minilm_encoder_is_not_a_cli_option(monkeypatch):
    monkeypatch.setattr(cli, "cmd_retrieve", lambda args: 0)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "retrieve",
                "--arm",
                "dense",
                "--run-id",
                "unused-encoder",
                "--encoder",
                "multi-qa-minilm",
            ]
        )


def test_config_snapshot_does_not_advertise_a_missing_command():
    assert "config --write" not in CONFIG.to_yaml()
