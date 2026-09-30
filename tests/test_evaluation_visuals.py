import json
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


def test_retrieval_metric_chart_data_groups_three_metrics_by_question_type(
    tmp_path,
):
    bm25 = tmp_path / "bm25.csv"
    mpnet = tmp_path / "mpnet.csv"
    columns = ["question_id", "knowledge_type", "ndcg@5", "recall@5", "mrr"]
    pd.DataFrame(
        [
            ("Q1", "known", 0.6, 0.7, 0.8),
            ("Q2", "known", 0.8, 0.9, 1.0),
            ("Q3", "inferred", 0.4, 0.5, 0.6),
            ("Q4", "inferred", 0.6, 0.7, 0.8),
        ],
        columns=columns,
    ).to_csv(bm25, index=False)
    pd.DataFrame(
        [
            ("Q1", "known", 0.8, 0.9, 1.0),
            ("Q2", "known", 1.0, 1.0, 1.0),
            ("Q3", "inferred", 0.6, 0.7, 0.8),
            ("Q4", "inferred", 0.8, 0.9, 1.0),
        ],
        columns=columns,
    ).to_csv(mpnet, index=False)

    data = visuals.retrieval_metric_chart_data(bm25, mpnet)

    assert list(data.columns) == [
        "Metric",
        "Method",
        "Question type",
        "Score",
        "Questions",
    ]
    assert data.to_dict("records") == [
        {
            "Metric": "NDCG@5",
            "Method": "BM25s",
            "Question type": "Known",
            "Score": pytest.approx(0.7),
            "Questions": 2,
        },
        {
            "Metric": "Recall@5",
            "Method": "BM25s",
            "Question type": "Known",
            "Score": pytest.approx(0.8),
            "Questions": 2,
        },
        {
            "Metric": "MRR",
            "Method": "BM25s",
            "Question type": "Known",
            "Score": pytest.approx(0.9),
            "Questions": 2,
        },
        {
            "Metric": "NDCG@5",
            "Method": "BM25s",
            "Question type": "Inferred",
            "Score": pytest.approx(0.5),
            "Questions": 2,
        },
        {
            "Metric": "Recall@5",
            "Method": "BM25s",
            "Question type": "Inferred",
            "Score": pytest.approx(0.6),
            "Questions": 2,
        },
        {
            "Metric": "MRR",
            "Method": "BM25s",
            "Question type": "Inferred",
            "Score": pytest.approx(0.7),
            "Questions": 2,
        },
        {
            "Metric": "NDCG@5",
            "Method": "MPNet",
            "Question type": "Known",
            "Score": pytest.approx(0.9),
            "Questions": 2,
        },
        {
            "Metric": "Recall@5",
            "Method": "MPNet",
            "Question type": "Known",
            "Score": pytest.approx(0.95),
            "Questions": 2,
        },
        {
            "Metric": "MRR",
            "Method": "MPNet",
            "Question type": "Known",
            "Score": pytest.approx(1.0),
            "Questions": 2,
        },
        {
            "Metric": "NDCG@5",
            "Method": "MPNet",
            "Question type": "Inferred",
            "Score": pytest.approx(0.7),
            "Questions": 2,
        },
        {
            "Metric": "Recall@5",
            "Method": "MPNet",
            "Question type": "Inferred",
            "Score": pytest.approx(0.8),
            "Questions": 2,
        },
        {
            "Metric": "MRR",
            "Method": "MPNet",
            "Question type": "Inferred",
            "Score": pytest.approx(0.9),
            "Questions": 2,
        },
    ]


def _write_answers(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_generation_chart_data_groups_appropriate_behaviour_by_question_type(
    tmp_path,
):
    question_types = tmp_path / "question-types.csv"
    pd.DataFrame(
        [("Q1", "known"), ("Q2", "inferred")],
        columns=["question_id", "knowledge_type"],
    ).to_csv(question_types, index=False)
    common = [
        {"question_id": "Q1", "answerable": True, "status": "answered"},
        {"question_id": "Q2", "answerable": True, "status": "answered"},
        {"question_id": "QX1", "answerable": False, "status": "refused"},
    ]
    bm25 = tmp_path / "bm25.jsonl"
    mpnet = tmp_path / "mpnet.jsonl"
    no_context = tmp_path / "none.jsonl"
    _write_answers(bm25, common)
    _write_answers(
        mpnet,
        [
            common[0],
            {"question_id": "Q2", "answerable": True, "status": "refused"},
            common[2],
        ],
    )
    _write_answers(
        no_context,
        [
            common[0],
            common[1],
            {"question_id": "QX1", "answerable": False, "status": "answered"},
        ],
    )

    chart = visuals.generation_chart_data(
        bm25,
        mpnet,
        no_context,
        question_types,
    )

    assert list(chart.columns) == ["Arm", "Question type", "Rate", "Questions"]
    assert chart.to_dict("records") == [
        {"Arm": "BM25s", "Question type": "Known", "Rate": 1.0, "Questions": 1},
        {
            "Arm": "BM25s",
            "Question type": "Inferred",
            "Rate": 1.0,
            "Questions": 1,
        },
        {"Arm": "BM25s", "Question type": "OOKB", "Rate": 1.0, "Questions": 1},
        {"Arm": "MPNet", "Question type": "Known", "Rate": 1.0, "Questions": 1},
        {
            "Arm": "MPNet",
            "Question type": "Inferred",
            "Rate": 0.0,
            "Questions": 1,
        },
        {"Arm": "MPNet", "Question type": "OOKB", "Rate": 1.0, "Questions": 1},
        {
            "Arm": "No context",
            "Question type": "Known",
            "Rate": 1.0,
            "Questions": 1,
        },
        {
            "Arm": "No context",
            "Question type": "Inferred",
            "Rate": 1.0,
            "Questions": 1,
        },
        {
            "Arm": "No context",
            "Question type": "OOKB",
            "Rate": 0.0,
            "Questions": 1,
        },
    ]


def test_generation_summary_formats_safety_citation_and_truncation_rates(tmp_path):
    paths = [tmp_path / name for name in ("bm25.csv", "mpnet.csv", "none.csv")]
    for path, values in zip(
        paths,
        [
            (0.033333, 0.823529, 0.0),
            (0.133333, 0.787879, 0.0),
            (1.0, float("nan"), 0.0),
        ],
        strict=True,
    ):
        _write_metrics(
            path,
            list(
                zip(
                    (
                        "unsupported_answer_rate",
                        "citation_validity_rate",
                        "truncation_rate",
                    ),
                    values,
                    strict=True,
                )
            ),
        )

    summary = visuals.generation_summary_data(*paths)

    assert summary.to_dict("records") == [
        {
            "Generation arm": "BM25s",
            "OOKB unsupported answer": "3.3%",
            "Citation validity": "82.4%",
            "Truncation": "0.0%",
        },
        {
            "Generation arm": "MPNet",
            "OOKB unsupported answer": "13.3%",
            "Citation validity": "78.8%",
            "Truncation": "0.0%",
        },
        {
            "Generation arm": "No context",
            "OOKB unsupported answer": "100.0%",
            "Citation validity": "N/A",
            "Truncation": "0.0%",
        },
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
    assert spec["padding"]["left"] >= 55
    assert label["encoding"]["y"]["field"] == "NDCG@5"
    assert label["mark"]["dy"] == -8
    assert label["mark"]["align"] == "center"
    assert label["mark"]["baseline"] == "bottom"
    assert label["mark"]["color"] == "black"
    assert label["encoding"]["text"]["format"] == ".3f"
    assert bar["encoding"]["color"]["scale"]["range"] == ["#4C78A8", "#F58518"]
    assert [{}] not in spec.get("datasets", {}).values()


def test_retrieval_metric_chart_facets_metrics_and_groups_question_types():
    data = pd.DataFrame(
        [
            ("NDCG@5", "BM25s", "Known", 0.8, 15),
            ("NDCG@5", "BM25s", "Inferred", 0.7, 21),
            ("NDCG@5", "MPNet", "Known", 0.9, 15),
            ("NDCG@5", "MPNet", "Inferred", 0.85, 21),
        ],
        columns=["Metric", "Method", "Question type", "Score", "Questions"],
    )

    spec = visuals.retrieval_metric_chart(data).to_dict()

    bars, labels = spec["spec"]["layer"]
    assert spec["facet"]["column"]["field"] == "Metric"
    assert spec["spec"]["width"] >= 270
    assert bars["encoding"]["x"]["field"] == "Method"
    assert bars["encoding"]["x"]["axis"]["title"] is None
    assert bars["encoding"]["xOffset"]["field"] == "Question type"
    assert bars["encoding"]["y"]["field"] == "Score"
    assert bars["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert bars["encoding"]["y"]["axis"]["title"] == "Mean score"
    assert bars["encoding"]["color"]["scale"]["domain"] == [
        "Known",
        "Inferred",
    ]
    assert bars["encoding"]["color"]["scale"]["range"] == [
        "#4C78A8",
        "#F58518",
    ]
    assert bars["mark"]["size"] >= 40
    assert labels["encoding"]["text"]["format"] == ".2f"
    assert labels["mark"]["color"] == "black"
    assert labels["mark"]["fontSize"] >= 14
    assert spec["facet"]["column"]["header"]["labelFontSize"] >= 17


def test_generation_chart_groups_question_types_and_labels_rates():
    data = pd.DataFrame(
        [
            ("BM25s", "Known", 0.95, 15),
            ("BM25s", "Inferred", 0.90, 21),
            ("BM25s", "OOKB", 0.97, 30),
        ],
        columns=["Arm", "Question type", "Rate", "Questions"],
    )
    assert callable(getattr(visuals, "generation_chart", None))
    spec = visuals.generation_chart(data).to_dict()

    bar = spec["layer"][0]
    label = spec["layer"][1]
    assert bar["encoding"]["x"]["field"] == "Arm"
    assert bar["encoding"]["x"]["axis"]["title"] == "Generation arm"
    assert bar["encoding"]["xOffset"]["field"] == "Question type"
    assert bar["encoding"]["y"]["field"] == "Rate"
    assert bar["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert bar["encoding"]["y"]["axis"]["title"] == "Appropriate response rate"
    assert bar["mark"]["size"] >= 44
    assert spec["height"] >= 420
    assert spec["padding"]["left"] >= 55
    assert bar["encoding"]["color"]["scale"]["domain"] == [
        "Known",
        "Inferred",
        "OOKB",
    ]
    assert bar["encoding"]["color"]["scale"]["range"] == [
        "#4C78A8",
        "#F58518",
        "#6B7280",
    ]
    assert label["mark"]["color"] == "black"
    assert label["encoding"]["text"]["field"] == "Percent"
    assert label["encoding"]["text"]["format"] == ".0f"
    assert label["mark"]["fontSize"] >= 14
    assert bar["encoding"]["x"]["axis"]["labelFontSize"] >= 13
    assert bar["encoding"]["y"]["axis"]["labelFontSize"] >= 13
    assert bar["encoding"]["color"]["legend"]["labelFontSize"] >= 13
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

    with pytest.raises(visuals.EvaluationVisualError, match="citation_validity_rate"):
        visuals.generation_summary_data(bm25, mpnet, no_context)
