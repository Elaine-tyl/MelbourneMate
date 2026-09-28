from pathlib import Path

import pandas as pd
import pytest

from melbourne_mate.evaluation.visuals import (
    EvaluationVisualError,
    generation_chart_data,
    retrieval_chart_data,
)


def _write_metrics(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)


def test_retrieval_chart_data_uses_the_three_reported_metrics(tmp_path):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    _write_metrics(
        bm25,
        [("ndcg@5", 0.835286), ("recall@5", 0.875), ("mrr", 0.884259)],
    )
    _write_metrics(
        mpnet,
        [("ndcg@5", 0.970878), ("recall@5", 1.0), ("mrr", 0.981481)],
    )

    chart = retrieval_chart_data(bm25, mpnet)

    assert list(chart.index) == ["NDCG@5", "Recall@5", "MRR"]
    assert list(chart.columns) == ["BM25s", "MPNet"]
    assert chart.loc["NDCG@5"].to_dict() == {
        "BM25s": pytest.approx(0.835286),
        "MPNet": pytest.approx(0.970878),
    }


def test_generation_chart_data_uses_safety_rates(tmp_path):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    no_context = tmp_path / "none.csv"
    _write_metrics(
        bm25,
        [("correct_refusal_rate", 0.966667), ("unsupported_answer_rate", 0.033333)],
    )
    _write_metrics(
        mpnet,
        [("correct_refusal_rate", 0.866667), ("unsupported_answer_rate", 0.133333)],
    )
    _write_metrics(
        no_context,
        [("correct_refusal_rate", 0.0), ("unsupported_answer_rate", 1.0)],
    )

    chart = generation_chart_data(bm25, mpnet, no_context)

    assert list(chart.index) == ["Correct refusal", "Unsupported answer"]
    assert list(chart.columns) == ["BM25s", "MPNet", "No context"]
    assert chart.loc["Unsupported answer"].to_dict() == {
        "BM25s": pytest.approx(0.033333),
        "MPNet": pytest.approx(0.133333),
        "No context": pytest.approx(1.0),
    }


def test_chart_data_rejects_a_missing_required_metric(tmp_path):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    _write_metrics(bm25, [("ndcg@5", 0.8), ("recall@5", 0.9)])
    _write_metrics(
        mpnet,
        [("ndcg@5", 0.9), ("recall@5", 1.0), ("mrr", 0.95)],
    )

    with pytest.raises(EvaluationVisualError, match="mrr"):
        retrieval_chart_data(bm25, mpnet)
