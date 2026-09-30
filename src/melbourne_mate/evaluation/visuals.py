"""Prepare saved evaluation results for the Streamlit charts."""

import json
from pathlib import Path

import altair as alt
import pandas as pd


class EvaluationVisualError(ValueError):
    """Raised when saved chart data is missing or incomplete."""


def _format_percent(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1%}"


def _metric_values(path: Path, required: tuple[str, ...]) -> dict[str, float]:
    """Read the required values from one saved metric table."""

    if not path.is_file():
        raise EvaluationVisualError(f"missing evaluation file: {path}")
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        raise EvaluationVisualError(f"invalid metric table: {path}")
    values = dict(zip(frame["metric"], frame["value"], strict=False))
    missing = [metric for metric in required if metric not in values]
    # Stop here rather than display an incomplete comparison.
    if missing:
        raise EvaluationVisualError(
            f"missing metric(s) in {path}: {', '.join(missing)}"
        )
    return {metric: float(values[metric]) for metric in required}


def _per_question_values(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise EvaluationVisualError(f"missing evaluation file: {path}")
    frame = pd.read_csv(path)
    required = {"question_id", "knowledge_type", "ndcg@5"}
    missing = required.difference(frame.columns)
    if missing:
        raise EvaluationVisualError(
            f"missing column(s) in {path}: {', '.join(sorted(missing))}"
        )
    return frame


def retrieval_slice_chart_data(bm25_path: Path, mpnet_path: Path) -> pd.DataFrame:
    """Summarise held-out NDCG@5 by question type and retrieval method."""

    frames = {
        "BM25s": _per_question_values(bm25_path),
        "MPNet": _per_question_values(mpnet_path),
    }
    rows: list[dict[str, object]] = []
    for kind, label in (("known", "Known"), ("inferred", "Inferred")):
        counts = {
            method: int((frame["knowledge_type"] == kind).sum())
            for method, frame in frames.items()
        }
        if not counts["BM25s"] or counts["BM25s"] != counts["MPNet"]:
            raise EvaluationVisualError(
                f"inconsistent {kind} question counts: {counts}"
            )
        group = f"{label} (n={counts['BM25s']})"
        for method, frame in frames.items():
            values = frame.loc[frame["knowledge_type"] == kind, "ndcg@5"]
            rows.append(
                {
                    "Question type": group,
                    "Method": method,
                    "NDCG@5": float(values.mean()),
                    "Questions": counts[method],
                }
            )
    return pd.DataFrame(rows)


def held_out_summary_data(path: Path) -> pd.DataFrame:
    """Format the saved held-out NDCG@5 comparison as a compact table."""

    if not path.is_file():
        raise EvaluationVisualError(f"missing evaluation file: {path}")
    frame = pd.read_csv(path)
    required = {"metric", "mean_difference", "ci_low", "ci_high", "p_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise EvaluationVisualError(
            f"missing column(s) in {path}: {', '.join(sorted(missing))}"
        )
    matches = frame.loc[frame["metric"] == "ndcg@5"]
    if len(matches) != 1:
        raise EvaluationVisualError(
            f"expected one ndcg@5 comparison in {path}, found {len(matches)}"
        )
    saved = matches.iloc[0]
    return pd.DataFrame(
        [
            ("Δ NDCG@5, MPNet − BM25s", f"{float(saved['mean_difference']):+.4f}"),
            (
                "Topic-clustered 95% CI",
                f"[{float(saved['ci_low']):.4f}, {float(saved['ci_high']):.4f}]",
            ),
            ("Paired test p-value", f"{float(saved['p_value']):.4f}"),
        ],
        columns=["Overall held-out comparison", "Result"],
    )


def retrieval_metric_chart_data(bm25_path: Path, mpnet_path: Path) -> pd.DataFrame:
    """Summarise three retrieval metrics by method and question type."""

    metrics = {
        "ndcg@5": "NDCG@5",
        "recall@5": "Recall@5",
        "mrr": "MRR",
    }
    frames: dict[str, pd.DataFrame] = {}
    required = {"question_id", "knowledge_type", *metrics}
    for method, path in (("BM25s", bm25_path), ("MPNet", mpnet_path)):
        if not path.is_file():
            raise EvaluationVisualError(f"missing evaluation file: {path}")
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise EvaluationVisualError(
                f"missing column(s) in {path}: {', '.join(sorted(missing))}"
            )
        frames[method] = frame

    rows: list[dict[str, object]] = []
    for method, frame in frames.items():
        for kind, label in (("known", "Known"), ("inferred", "Inferred")):
            group = frame.loc[frame["knowledge_type"] == kind]
            if group.empty:
                raise EvaluationVisualError(f"no {kind} questions in {method} run")
            for metric, metric_label in metrics.items():
                rows.append(
                    {
                        "Metric": metric_label,
                        "Method": method,
                        "Question type": label,
                        "Score": float(group[metric].mean()),
                        "Questions": len(group),
                    }
                )
    return pd.DataFrame(rows)


def retrieval_metric_chart(data: pd.DataFrame) -> alt.FacetChart:
    """Facet retrieval metrics and compare question types within each method."""

    position = {
        "x": alt.X(
            "Method:N",
            sort=["BM25s", "MPNet"],
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=14,
                labelPadding=8,
                title=None,
            ),
        ),
        "xOffset": alt.XOffset(
            "Question type:N",
            sort=["Known", "Inferred"],
        ),
        "y": alt.Y(
            "Score:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(
                title="Mean score",
                format=".1f",
                labelFontSize=13,
                titleFontSize=15,
                titlePadding=14,
            ),
        ),
    }
    base = alt.Chart(data).encode(**position)
    bars = base.mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        size=44,
    ).encode(
        color=alt.Color(
            "Question type:N",
            scale=alt.Scale(
                domain=["Known", "Inferred"],
                range=["#4C78A8", "#F58518"],
            ),
            legend=alt.Legend(
                orient="bottom",
                title=None,
                labelFontSize=14,
                symbolSize=190,
            ),
        ),
        tooltip=[
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Method:N"),
            alt.Tooltip("Question type:N"),
            alt.Tooltip("Score:Q", format=".4f"),
            alt.Tooltip("Questions:Q", format="d"),
        ],
    )
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        color="black",
        dy=-10,
        fontSize=14,
        fontWeight="bold",
    ).encode(
        text=alt.Text("Score:Q", format=".2f"),
        detail="Question type:N",
    )
    return (
        (bars + labels)
        .properties(width=280, height=400)
        .facet(
            column=alt.Column(
                "Metric:N",
                sort=["NDCG@5", "Recall@5", "MRR"],
                header=alt.Header(title=None, labelFontSize=18),
            ),
            spacing=34,
        )
        .resolve_scale(y="shared")
    )


def retrieval_slice_chart(data: pd.DataFrame) -> alt.LayerChart:
    """Compare retrieval methods within each answerable question type."""

    position = {
        "x": alt.X(
            "Question type:N",
            sort=["Known (n=15)", "Inferred (n=21)"],
            axis=alt.Axis(labelAngle=0, title="Question type"),
        ),
        "xOffset": alt.XOffset("Method:N", sort=["BM25s", "MPNet"]),
        "y": alt.Y(
            "NDCG@5:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title="Mean NDCG@5", format=".1f"),
        ),
    }
    base = alt.Chart(data).encode(**position)
    bars = base.mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        size=48,
    ).encode(
        color=alt.Color(
            "Method:N",
            scale=alt.Scale(
                domain=["BM25s", "MPNet"],
                range=["#4C78A8", "#F58518"],
            ),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            alt.Tooltip("Question type:N"),
            alt.Tooltip("Method:N"),
            alt.Tooltip("NDCG@5:Q", format=".3f"),
        ],
    )
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        color="black",
        dy=-8,
        fontSize=14,
        fontWeight="bold",
    ).encode(
        text=alt.Text("NDCG@5:Q", format=".3f"),
        detail="Method:N",
    )
    return (bars + labels).properties(
        height=340,
        padding={"left": 55, "right": 12, "top": 20, "bottom": 10},
    )


def generation_chart_data(
    bm25_path: Path,
    mpnet_path: Path,
    no_context_path: Path,
    question_types_path: Path,
) -> pd.DataFrame:
    """Calculate appropriate response rates by arm and question type."""

    if not question_types_path.is_file():
        raise EvaluationVisualError(
            f"missing evaluation file: {question_types_path}"
        )
    question_types = pd.read_csv(question_types_path)
    required_types = {"question_id", "knowledge_type"}
    missing_types = required_types.difference(question_types.columns)
    if missing_types:
        raise EvaluationVisualError(
            f"missing column(s) in {question_types_path}: "
            f"{', '.join(sorted(missing_types))}"
        )
    type_by_question = question_types.set_index("question_id")[
        "knowledge_type"
    ].to_dict()

    rows: list[dict[str, object]] = []
    paths = (
        ("BM25s", bm25_path),
        ("MPNet", mpnet_path),
        ("No context", no_context_path),
    )
    for arm, path in paths:
        if not path.is_file():
            raise EvaluationVisualError(f"missing evaluation file: {path}")
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (json.JSONDecodeError, OSError) as exc:
            raise EvaluationVisualError(f"invalid answer file: {path}") from exc
        frame = pd.DataFrame(records)
        required_answers = {"question_id", "answerable", "status"}
        missing_answers = required_answers.difference(frame.columns)
        if missing_answers:
            raise EvaluationVisualError(
                f"missing field(s) in {path}: {', '.join(sorted(missing_answers))}"
            )
        frame["Question type"] = frame["question_id"].map(type_by_question)
        frame.loc[~frame["answerable"], "Question type"] = "ookb"
        missing_answerable_types = frame.loc[
            frame["answerable"] & frame["Question type"].isna(), "question_id"
        ]
        if not missing_answerable_types.empty:
            raise EvaluationVisualError(
                f"missing question type(s) for {', '.join(missing_answerable_types)}"
            )

        for kind, label in (
            ("known", "Known"),
            ("inferred", "Inferred"),
            ("ookb", "OOKB"),
        ):
            group = frame.loc[frame["Question type"] == kind]
            if group.empty:
                raise EvaluationVisualError(f"no {kind} questions in {arm} run")
            desired_status = "refused" if kind == "ookb" else "answered"
            rows.append(
                {
                    "Arm": arm,
                    "Question type": label,
                    "Rate": float((group["status"] == desired_status).mean()),
                    "Questions": len(group),
                }
            )
    return pd.DataFrame(rows)


def generation_summary_data(
    bm25_path: Path,
    mpnet_path: Path,
    no_context_path: Path,
) -> pd.DataFrame:
    """Format saved generation safety and citation metrics for display."""

    metrics = (
        "unsupported_answer_rate",
        "citation_validity_rate",
        "truncation_rate",
    )
    rows = []
    for arm, path in (
        ("BM25s", bm25_path),
        ("MPNet", mpnet_path),
        ("No context", no_context_path),
    ):
        values = _metric_values(path, metrics)

        rows.append(
            {
                "Generation arm": arm,
                "OOKB unsupported answer": _format_percent(
                    values["unsupported_answer_rate"]
                ),
                "Citation validity": _format_percent(
                    values["citation_validity_rate"]
                ),
                "Truncation": _format_percent(values["truncation_rate"]),
            }
        )
    return pd.DataFrame(rows)


def generation_chart(data: pd.DataFrame) -> alt.LayerChart:
    """Compare appropriate response behaviour across question types."""

    chart_data = data.assign(Percent=data["Rate"] * 100)
    x = alt.X(
        "Arm:N",
        sort=["BM25s", "MPNet", "No context"],
        axis=alt.Axis(
            labelAngle=0,
            labelFontSize=14,
            labelPadding=10,
            title="Generation arm",
            titleFontSize=16,
            titlePadding=16,
        ),
    )
    y = alt.Y(
        "Rate:Q",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(
            title="Appropriate response rate",
            format=".0%",
            labelFontSize=14,
            titleFontSize=16,
            titlePadding=16,
        ),
    )
    base = alt.Chart(chart_data).encode(
        x=x,
        xOffset=alt.XOffset(
            "Question type:N",
            sort=["Known", "Inferred", "OOKB"],
        ),
        y=y,
    )
    bars = base.mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        size=48,
    ).encode(
        color=alt.Color(
            "Question type:N",
            scale=alt.Scale(
                domain=["Known", "Inferred", "OOKB"],
                range=["#4C78A8", "#F58518", "#6B7280"],
            ),
            legend=alt.Legend(
                orient="bottom",
                title=None,
                labelFontSize=14,
                symbolSize=200,
            ),
        ),
        tooltip=[
            alt.Tooltip("Arm:N"),
            alt.Tooltip("Question type:N"),
            alt.Tooltip("Rate:Q", title="Appropriate response", format=".1%"),
            alt.Tooltip("Questions:Q", format="d"),
        ],
    )
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        color="black",
        dy=-10,
        fontSize=14,
        fontWeight="bold",
    ).encode(
        text=alt.Text("Percent:Q", format=".0f"),
        detail="Question type:N",
    )
    return (bars + labels).properties(
        height=430,
        padding={"left": 65, "right": 18, "top": 30, "bottom": 12},
    )
