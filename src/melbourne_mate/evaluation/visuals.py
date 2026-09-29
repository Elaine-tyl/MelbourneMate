"""Prepare saved evaluation results for the Streamlit charts."""

from pathlib import Path

import altair as alt
import pandas as pd


class EvaluationVisualError(ValueError):
    """Raised when saved chart data is missing or incomplete."""


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
) -> pd.DataFrame:
    """Read unsupported-answer rates for the three saved generation arms."""

    metrics = ("unsupported_answer_rate",)
    conditions = {
        "BM25s": _metric_values(bm25_path, metrics),
        "MPNet": _metric_values(mpnet_path, metrics),
        "No context": _metric_values(no_context_path, metrics),
    }
    return pd.DataFrame(
        [
            {
                "Arm": condition,
                "Rate": values["unsupported_answer_rate"],
            }
            for condition, values in conditions.items()
        ]
    )


def generation_chart(data: pd.DataFrame) -> alt.LayerChart:
    """Compare unsupported-answer risk across the generation arms."""

    x = alt.X(
        "Arm:N",
        sort=["BM25s", "MPNet", "No context"],
        axis=alt.Axis(labelAngle=0, title="Generation arm"),
    )
    y = alt.Y(
        "Rate:Q",
        scale=alt.Scale(domain=[0, 1.15]),
        axis=alt.Axis(
            title="Unsupported answer rate",
            format=".0%",
        ),
    )
    base = alt.Chart(data).encode(x=x, y=y)
    bars = base.mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        size=48,
    ).encode(
        color=alt.Color(
            "Arm:N",
            scale=alt.Scale(
                domain=["BM25s", "MPNet", "No context"],
                range=["#4C78A8", "#F58518", "#6B7280"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Arm:N"),
            alt.Tooltip("Rate:Q", title="Unsupported answer", format=".1%"),
        ],
    )
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        color="black",
        dy=-8,
        fontWeight="bold",
    ).encode(
        text=alt.Text("Rate:Q", format=".1%"),
    )
    return (bars + labels).properties(
        height=340,
        padding={"left": 55, "right": 12, "top": 20, "bottom": 10},
    )
