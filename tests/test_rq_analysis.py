import csv
import json
from pathlib import Path

import pytest

from melbourne_mate.cli import main
from melbourne_mate.corpus import load_collection
from melbourne_mate.evaluation.rq_analysis import (
    RQAnalysisError,
    rq2_rows,
    rq3a_rows,
    rq4_rows,
    write_rq_analysis,
)
from melbourne_mate.evaluation.stats import rate_interval

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "--qrels": ROOT / "review/targeted-qrels/qrels-final.txt",
    "--bm25-retrieval": ROOT / "runs/retrieval/s9-final-bm25-test",
    "--mpnet-retrieval": ROOT / "runs/retrieval/s9-final-mpnet-test",
    "--bm25-generation": ROOT / "runs/generation/s9-qwen25-20260927-bm25",
    "--mpnet-generation": ROOT / "runs/generation/s9-qwen25-20260927-mpnet",
    "--no-context-generation": ROOT / "runs/generation/s9-qwen25-20260927-no-context",
}


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_row(topic, bin_, form, value):
    return {"topic_id": topic, "containment_bin": bin_, "question_form": form, "ndcg@5": str(value)}


def test_rate_interval_gives_the_share_and_a_cluster_interval():
    outcomes = {"a": True, "b": False, "c": True, "d": True}
    clusters = {"a": "T1", "b": "T1", "c": "T2", "d": "T3"}

    rate, low, high = rate_interval(outcomes, clusters, resamples=200, seed=1)

    assert rate == 0.75
    assert 0.0 <= low <= rate <= high <= 1.0
    assert rate_interval({"a": True}, {"a": "T1"}, resamples=50) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="no outcomes"):
        rate_interval({}, {})


def test_rq2_reports_each_containment_bin_and_question_form():
    bm25 = {
        "q1": metric_row("T1", "low", "canonical", 0.5),
        "q2": metric_row("T1", "low", "paraphrased", 0.4),
        "q3": metric_row("T2", "high", "canonical", 0.9),
    }
    mpnet = {
        "q1": metric_row("T1", "low", "canonical", 0.9),
        "q2": metric_row("T1", "low", "paraphrased", 0.8),
        "q3": metric_row("T2", "high", "canonical", 0.9),
    }

    rows = {(r["dimension"], r["group"]): r for r in rq2_rows(bm25, mpnet)}

    assert set(rows) == {
        ("all", "all"),
        ("containment_bin", "low"),
        ("containment_bin", "high"),
        ("question_form", "canonical"),
        ("question_form", "paraphrased"),
    }
    assert rows[("containment_bin", "low")]["difference"] == 0.4
    assert rows[("containment_bin", "low")]["topics"] == 1
    assert rows[("containment_bin", "high")]["difference"] == 0.0
    with pytest.raises(RQAnalysisError, match="different questions"):
        rq2_rows(bm25, {"q1": mpnet["q1"]})


def outcomes(ookb_answered, hits=(), refused=()):
    """Facts for QX01.. out-of-KB questions and Q01C.. answerable questions."""
    facts = {
        f"QX{n:02d}": {"answerable": False, "refused": not answered, "retrieval_hit": False}
        for n, answered in enumerate(ookb_answered, start=1)
    }
    for question in ("Q01C", "Q01P1", "Q02C"):
        facts[question] = {
            "answerable": True,
            "refused": question in refused,
            "retrieval_hit": question in hits,
        }
    return facts


def test_rq3a_compares_the_same_out_of_kb_questions_and_flags_thin_evidence():
    collection = load_collection(ROOT / "data/v1")
    runs = {
        "no-context": outcomes([True] * 6),
        "bm25": outcomes([False] * 6),
        "mpnet": outcomes([True, False, False, False, False, False]),
    }

    rows = {r["comparison"]: r for r in rq3a_rows(collection, runs)}

    assert rows["bm25 minus no-context"]["difference"] == -1.0
    assert rows["bm25 minus no-context"]["discordant"] == 6
    assert rows["bm25 minus no-context"]["evidence"] == "sufficient"
    assert rows["mpnet minus bm25"]["discordant"] == 1
    assert rows["mpnet minus bm25"]["evidence"] == "insufficient"
    assert rows["mpnet minus bm25"]["p_value"] == ""  # thin comparisons quote no p-value
    assert rows["bm25 minus no-context"]["p_value"] != ""
    runs["mpnet"].pop("QX06")
    with pytest.raises(RQAnalysisError, match="different out-of-KB questions"):
        rq3a_rows(collection, runs)


def test_rq4_counts_false_refusal_only_when_evidence_was_retrieved():
    collection = load_collection(ROOT / "data/v1")
    runs = {
        "bm25": outcomes([False, False], hits=("Q01C", "Q02C"), refused=("Q01C", "Q01P1")),
        "mpnet": outcomes([False, True], hits=("Q01C", "Q01P1", "Q02C")),
        "no-context": outcomes([True, True]),
    }

    rows = {(r["arm"], r["metric"]): r for r in rq4_rows(collection, runs)}

    # Q01P1 was refused but its evidence was not retrieved, so it is not counted.
    assert rows[("bm25", "false_refusal")]["questions"] == 2
    assert rows[("bm25", "false_refusal")]["rate"] == 0.5
    assert rows[("bm25", "correct_refusal")]["rate"] == 1.0
    assert rows[("mpnet", "correct_refusal")]["rate"] == 0.5
    assert rows[("no-context", "false_refusal")]["note"] == "not applicable without retrieval"


def test_saved_runs_reproduce_the_published_results(tmp_path):
    out = tmp_path / "final-rq-analysis"
    args = [str(value) for pair in RUNS.items() for value in pair]

    assert main(["--data", str(ROOT / "data/v1"), "analyse-rqs", *args, "--out", str(out)]) == 0

    overall = next(r for r in read(out / "rq2-containment.csv") if r["group"] == "all")
    published = read(ROOT / "runs/retrieval/s9-final-comparison.csv")[0]
    assert float(overall["difference"]) == pytest.approx(float(published["mean_difference"]), abs=1e-4)
    assert float(overall["ci_low"]) == pytest.approx(float(published["ci_low"]), abs=1e-4)
    assert float(overall["p_value"]) == pytest.approx(float(published["p_value"]), abs=1e-4)

    grounding = {r["comparison"]: r for r in read(out / "rq3a-grounding.csv")}
    assert grounding["mpnet minus bm25"]["discordant"] == "3"
    assert grounding["mpnet minus bm25"]["p_value"] == ""
    assert "| 3 | insufficient |" in (out / "report.md").read_text(encoding="utf-8")
    assert "not reported" in (out / "report.md").read_text(encoding="utf-8")

    refusal = {(r["arm"], r["metric"]): r for r in read(out / "rq4-refusal.csv")}
    for arm in ("bm25", "mpnet"):
        saved = {
            r["metric"]: float(r["value"])
            for r in read(RUNS[f"--{arm}-generation"] / "summary.csv")
            if r["value"]
        }
        for metric in ("correct_refusal", "false_refusal"):
            assert float(refusal[(arm, metric)]["rate"]) == pytest.approx(saved[f"{metric}_rate"], abs=1e-3)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["input_fingerprints"]) == {
        "qrels",
        "bm25_retrieval",
        "mpnet_retrieval",
        "bm25_generation",
        "mpnet_generation",
        "no_context_generation",
    }
    assert manifest["seed"] == 20260923 and manifest["resamples"] == 10000
    assert {p.name for p in out.iterdir()} == {
        "manifest.json",
        "rq2-containment.csv",
        "rq3a-grounding.csv",
        "rq4-refusal.csv",
        "summary.csv",
        "report.md",
    }


def test_analysis_folder_is_write_once(tmp_path):
    collection = load_collection(ROOT / "data/v1")
    out = tmp_path / "done"
    out.mkdir()
    (out / "report.md").write_text("old", encoding="utf-8")

    with pytest.raises(RQAnalysisError, match="already exists"):
        write_rq_analysis(out, collection, *(RUNS[key] for key in RUNS))


def test_saved_analysis_quotes_no_p_value_for_thin_comparisons():
    saved = ROOT / "runs/analysis/final-rq-analysis"
    for row in read(saved / "rq3a-grounding.csv"):
        if row["evidence"] == "insufficient":
            assert int(row["discordant"]) < 5
            assert row["p_value"] == ""
