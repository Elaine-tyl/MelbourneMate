from pathlib import Path

import pandas as pd
import pytest

from melbourne_mate.evaluation import visuals


def _write_metrics(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)


def test_retrieval_slice_chart_data_groups_known_and_inferred_questions(tmp_path):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    rows = [
        ("Q1", "known", 0.6),
        ("Q2", "known", 0.8),
        ("Q3", "inferred", 0.7),
        ("Q4", "inferred", 0.9),
    ]
    pd.DataFrame(rows, columns=["question_id", "knowledge_type", "ndcg@5"]).to_csv(
        bm25, index=False
    )
    pd.DataFrame(
        [(qid, kind, score + 0.1) for qid, kind, score in rows],
        columns=["question_id", "knowledge_type", "ndcg@5"],
    ).to_csv(mpnet, index=False)

    assert callable(getattr(visuals, "retrieval_slice_chart_data", None))
    chart = visuals.retrieval_slice_chart_data(bm25, mpnet)

    assert list(chart.columns) == [
        "Question type",
        "Method",
        "NDCG@5",
        "Questions",
    ]
    assert chart.to_dict("records") == [
        {
            "Question type": "Known (n=2)",
            "Method": "BM25s",
            "NDCG@5": pytest.approx(0.7),
            "Questions": 2,
        },
        {
            "Question type": "Known (n=2)",
            "Method": "MPNet",
            "NDCG@5": pytest.approx(0.8),
            "Questions": 2,
        },
        {
            "Question type": "Inferred (n=2)",
            "Method": "BM25s",
            "NDCG@5": pytest.approx(0.8),
            "Questions": 2,
        },
        {
            "Question type": "Inferred (n=2)",
            "Method": "MPNet",
            "NDCG@5": pytest.approx(0.9),
            "Questions": 2,
        },
    ]


def test_held_out_summary_data_formats_saved_statistical_result(tmp_path):
    held_out = tmp_path / "held-out.csv"
    columns = [
        "metric",
        "left_run",
        "right_run",
        "mean_difference",
        "ci_low",
        "ci_high",
        "p_value",
        "clusters",
        "observations",
    ]
    pd.DataFrame(
        [["ndcg@5", "bm25", "mpnet", 0.1356, 0.0486, 0.2519, 0.0065, 12, 36]],
        columns=columns,
    ).to_csv(held_out, index=False)

    data = visuals.held_out_summary_data(held_out)

    assert list(data.columns) == ["Overall held-out comparison", "Result"]
    assert data.to_dict("records") == [
        {
            "Overall held-out comparison": "Δ NDCG@5, MPNet − BM25s",
            "Result": "+0.1356",
        },
        {
            "Overall held-out comparison": "Topic-clustered 95% CI",
            "Result": "[0.0486, 0.2519]",
        },
        {
            "Overall held-out comparison": "Paired test p-value",
            "Result": "0.0065",
        },
    ]


def test_generation_chart_data_uses_unsupported_answer_rates(tmp_path):
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

    chart = visuals.generation_chart_data(bm25, mpnet, no_context)

    assert list(chart.columns) == ["Arm", "Rate"]
    assert chart.to_dict("records") == [
        {"Arm": "BM25s", "Rate": 0.033333},
        {"Arm": "MPNet", "Rate": 0.133333},
        {"Arm": "No context", "Rate": 1.0},
    ]


def test_retrieval_slice_chart_groups_methods_and_labels_values():
    data = pd.DataFrame(
        [
            ("Known (n=2)", "BM25s", 0.7, 2),
            ("Known (n=2)", "MPNet", 0.8, 2),
            ("Inferred (n=2)", "BM25s", 0.8, 2),
            ("Inferred (n=2)", "MPNet", 0.9, 2),
        ],
        columns=["Question type", "Method", "NDCG@5", "Questions"],
    )
    assert callable(getattr(visuals, "retrieval_slice_chart", None))
    spec = visuals.retrieval_slice_chart(data).to_dict()

    bar = spec["layer"][0]
    label = spec["layer"][1]
    assert bar["encoding"]["xOffset"]["field"] == "Method"
    assert bar["encoding"]["x"]["axis"]["title"] == "Question type"
    assert bar["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert bar["encoding"]["y"]["axis"]["title"] == "Mean NDCG@5"
    assert bar["mark"]["size"] == 48
    assert spec["height"] == 340
    assert spec["padding"]["left"] == 55
    assert label["encoding"]["y"]["field"] == "NDCG@5"
    assert label["mark"]["dy"] == -8
    assert label["mark"]["align"] == "center"
    assert label["mark"]["baseline"] == "bottom"
    assert label["mark"]["color"] == "black"
    assert label["encoding"]["text"]["format"] == ".3f"
    assert bar["encoding"]["color"]["scale"]["range"] == ["#4C78A8", "#F58518"]
    assert [{}] not in spec.get("datasets", {}).values()


def test_generation_chart_uses_vertical_risk_bars_and_percentage_labels():
    data = pd.DataFrame(
        [
            ("BM25s", 0.033333),
            ("MPNet", 0.133333),
            ("No context", 1.0),
        ],
        columns=["Arm", "Rate"],
    )
    assert callable(getattr(visuals, "generation_chart", None))
    spec = visuals.generation_chart(data).to_dict()

    bar = spec["layer"][0]
    label = spec["layer"][1]
    assert bar["encoding"]["x"]["field"] == "Arm"
    assert bar["encoding"]["x"]["axis"]["title"] == "Generation arm"
    assert bar["encoding"]["y"]["field"] == "Rate"
    assert bar["encoding"]["y"]["scale"]["domain"] == [0, 1.15]
    assert bar["encoding"]["y"]["axis"]["title"] == "Unsupported answer rate"
    assert bar["mark"]["size"] == 48
    assert spec["height"] == 340
    assert spec["padding"]["left"] == 55
    assert bar["encoding"]["color"]["scale"]["domain"] == [
        "BM25s",
        "MPNet",
        "No context",
    ]
    assert bar["encoding"]["color"]["scale"]["range"] == [
        "#4C78A8",
        "#F58518",
        "#6B7280",
    ]
    assert label["mark"]["color"] == "black"
    assert label["encoding"]["text"]["format"] == ".1%"
    assert "transform" not in spec
    assert [{}] not in spec.get("datasets", {}).values()


def test_chart_data_rejects_a_missing_required_metric(tmp_path):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    no_context = tmp_path / "none.csv"
    _write_metrics(bm25, [("correct_refusal_rate", 0.9)])
    _write_metrics(
        mpnet,
        [("correct_refusal_rate", 0.8), ("unsupported_answer_rate", 0.2)],
    )
    _write_metrics(
        no_context,
        [("correct_refusal_rate", 0.0), ("unsupported_answer_rate", 1.0)],
    )

    with pytest.raises(visuals.EvaluationVisualError, match="unsupported_answer_rate"):
        visuals.generation_chart_data(bm25, mpnet, no_context)
