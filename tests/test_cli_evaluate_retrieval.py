from pathlib import Path

from melbourne_mate import cli


def test_evaluate_retrieval_runs_the_frozen_workflow(monkeypatch, tmp_path):
    calls = []

    def record(name):
        def command(args):
            calls.append((name, vars(args).copy()))
            return 0

        return command

    monkeypatch.setattr(cli, "cmd_validate", record("validate"))
    monkeypatch.setattr(cli, "cmd_quality", record("quality"))
    monkeypatch.setattr(cli, "cmd_retrieve", record("retrieve"))
    monkeypatch.setattr(cli, "cmd_compare", record("compare"))

    result = cli.main(
        [
            "--data",
            "data/v1",
            "evaluate-retrieval",
            "--split",
            "test",
            "--run-prefix",
            "formal-20260927",
            "--runs-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert [name for name, _ in calls] == [
        "validate",
        "quality",
        "retrieve",
        "retrieve",
        "compare",
    ]
    assert calls[2][1]["arm"] == "bm25"
    assert calls[2][1]["run_id"] == "formal-20260927-bm25-test"
    assert calls[3][1]["arm"] == "dense"
    assert calls[3][1]["encoder"] == "multi-qa-mpnet"
    assert calls[3][1]["run_id"] == "formal-20260927-mpnet-test"
    assert calls[4][1]["left"] == Path(tmp_path) / "formal-20260927-bm25-test"
    assert calls[4][1]["right"] == Path(tmp_path) / "formal-20260927-mpnet-test"


def test_evaluate_retrieval_stops_after_a_failed_step(monkeypatch, tmp_path):
    calls = []

    def validate(args):
        calls.append("validate")
        return 1

    monkeypatch.setattr(cli, "cmd_validate", validate)
    monkeypatch.setattr(cli, "cmd_quality", lambda args: calls.append("quality") or 0)

    result = cli.main(
        [
            "evaluate-retrieval",
            "--run-prefix",
            "failed-run",
            "--runs-dir",
            str(tmp_path),
        ]
    )

    assert result == 1
    assert calls == ["validate"]
