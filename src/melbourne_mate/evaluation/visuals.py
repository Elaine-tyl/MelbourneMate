"""Prepare saved evaluation results for the Streamlit charts."""

from pathlib import Path

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


def retrieval_chart_data(bm25_path: Path, mpnet_path: Path) -> pd.DataFrame:
    """Build the held-out BM25s and MPNet retrieval comparison."""

    metrics = ("ndcg@5", "recall@5", "mrr")
    labels = {"ndcg@5": "NDCG@5", "recall@5": "Recall@5", "mrr": "MRR"}
    bm25 = _metric_values(bm25_path, metrics)
    mpnet = _metric_values(mpnet_path, metrics)
    return pd.DataFrame(
        {
            "BM25s": [bm25[metric] for metric in metrics],
            "MPNet": [mpnet[metric] for metric in metrics],
        },
        index=[labels[metric] for metric in metrics],
    )


def generation_chart_data(
    bm25_path: Path,
    mpnet_path: Path,
    no_context_path: Path,
) -> pd.DataFrame:
    """Build the saved three-condition generation safety comparison."""

    metrics = ("correct_refusal_rate", "unsupported_answer_rate")
    labels = {
        "correct_refusal_rate": "Correct refusal",
        "unsupported_answer_rate": "Unsupported answer",
    }
    conditions = {
        "BM25s": _metric_values(bm25_path, metrics),
        "MPNet": _metric_values(mpnet_path, metrics),
        "No context": _metric_values(no_context_path, metrics),
    }
    return pd.DataFrame(
        {
            condition: [values[metric] for metric in metrics]
            for condition, values in conditions.items()
        },
        index=[labels[metric] for metric in metrics],
    )
