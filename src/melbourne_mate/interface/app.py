"""Streamlit interface for the frozen MelbourneMate pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import streamlit as st

from melbourne_mate.corpus import CollectionError, load_collection
from melbourne_mate.evaluation.generation_inputs import load_generation_qrels
from melbourne_mate.evaluation.visuals import (
    EvaluationVisualError,
    generation_chart,
    generation_chart_data,
    generation_summary_data,
    held_out_summary_data,
    retrieval_metric_chart,
    retrieval_metric_chart_data,
)
from melbourne_mate.generation.gate import EvidenceGate
from melbourne_mate.generation.ollama import OllamaClient, OllamaError
from melbourne_mate.pipeline import Answer, RagPipeline, build_retriever
from melbourne_mate.retrieval.encoders import load_encoder

EXAMPLE_QUESTIONS = {
    "Work": (
        "My employer paid me below the rate I expected and gave no payslip. "
        "Where can I ask for help?"
    ),
    "Health": (
        "What should an international student do in a medical emergency in Victoria "
        "and what about ambulance costs?"
    ),
    "Housing": (
        "There is a dangerous electrical fault and the agent has not called back. "
        "What should I do first, and if they still do not arrange it, can I pay for "
        "the repair myself and what cost limit applies?"
    ),
    "Outside our sources": "What is the current exchange rate for Australian dollars?",
}

APP_VIEWS = ("Ask MelbourneMate", "Evaluation Results")


def answer_view(answer: Answer) -> dict[str, object]:
    """Turn a pipeline answer into the small view model used by Streamlit."""
    state = {
        "answered": ("Answer ready", "success"),
        "refused": ("Needs an official check", "warning"),
        "truncated": ("Answer stopped early", "warning"),
    }
    label, tone = state.get(answer.status, ("Unable to answer", "error"))

    cited = set(answer.citations.cited) if answer.citations and not answer.refused else set()
    sources: dict[str, dict[str, object]] = {}
    for item in answer.evidence:
        if item.passage_id not in cited:
            continue
        if item.url not in sources:
            sources[item.url] = {
                "organisation": item.organisation,
                "heading": item.heading,
                "url": item.url,
                "passage_ids": [],
            }
        sources[item.url]["passage_ids"].append(item.passage_id)

    source_rows = tuple(
        {
            **row,
            "passage_ids": ", ".join(row["passage_ids"]),
        }
        for row in sources.values()
    )
    return {
        "label": label,
        "tone": tone,
        "text": answer.text,
        "citation_valid": answer.citations.is_valid if answer.citations else None,
        "sources": source_rows,
        "latency_ms": answer.latency_ms,
    }


def answer_question(
    pipeline: RagPipeline,
    question: str,
    on_chunk: Callable[[str], None] | None = None,
) -> Answer:
    """Use saved qrels when the question is part of the test collection."""
    known = next(
        (item for item in pipeline.collection.questions.values() if item.text == question),
        None,
    )
    relevant = (
        tuple(sorted(pipeline.collection.qrels.get(known.question_id, {})))
        if known
        else ()
    )
    return pipeline.answer(question, relevant_ids=relevant, on_chunk=on_chunk)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_app_collection(data_dir: Path, qrels_path: Path):
    collection = load_collection(data_dir)
    qrels, _ = load_generation_qrels(qrels_path, collection)
    return replace(collection, qrels=qrels)


def _render_evaluation_results(root: Path) -> None:
    """Render saved evaluation evidence without loading the live RAG pipeline."""
    st.header("Evaluation Results")
    st.caption(
        "Frozen results from the held-out retrieval and generation evaluations. "
        "Use the tabs to compare ranking quality and answer behaviour."
    )

    try:
        retrieval_metrics = retrieval_metric_chart_data(
            root / "runs/retrieval/s9-final-bm25-test/per_question_metrics.csv",
            root / "runs/retrieval/s9-final-mpnet-test/per_question_metrics.csv",
        )
        held_out_summary = held_out_summary_data(
            root / "runs/retrieval/s9-final-comparison.csv"
        )
        generation = generation_chart_data(
            root / "runs/generation/s9-qwen25-20260927-bm25/answers.jsonl",
            root / "runs/generation/s9-qwen25-20260927-mpnet/answers.jsonl",
            root / "runs/generation/s9-qwen25-20260927-no-context/answers.jsonl",
            root / "runs/retrieval/s9-final-mpnet-test/per_question_metrics.csv",
        )
        generation_summary = generation_summary_data(
            root / "runs/generation/s9-qwen25-20260927-bm25/summary.csv",
            root / "runs/generation/s9-qwen25-20260927-mpnet/summary.csv",
            root / "runs/generation/s9-qwen25-20260927-no-context/summary.csv",
        )
    except EvaluationVisualError as exc:
        st.warning(f"Saved evaluation results are unavailable: {exc}")
        return

    with st.container(border=True):
        st.subheader("Retrieval Metrics for Known and Inferred Questions")
        st.caption(
            "Mean NDCG@5, Recall@5 and MRR across 15 known and 21 inferred "
            "held-out questions. Higher is better. MPNet was selected over "
            "MiniLM on validation data; that validation difference was not "
            "statistically significant."
        )
        retrieval_chart_column = st.columns([1, 10, 1])[1]
        with retrieval_chart_column:
            st.altair_chart(
                retrieval_metric_chart(retrieval_metrics),
                width="stretch",
            )
            st.markdown(
                '<div class="mm-axis-title">Retrieval method</div>',
                unsafe_allow_html=True,
            )
        st.table(
            held_out_summary,
            border="horizontal",
            width="stretch",
            hide_index=True,
        )
    with st.container(border=True):
        st.subheader("OOKB Refusal Safety by Generation Arm")
        st.caption(
            "Correct-refusal rate across the same 30 out-of-knowledge-base "
            "questions. Higher bars are better. The table adds unsupported "
            "answers, false refusals and citation validity; lower error rates "
            "are better."
        )
        st.altair_chart(generation_chart(generation), width="stretch")
        st.table(
            generation_summary,
            border="horizontal",
            width="stretch",
            hide_index=True,
        )


def _render_question_view() -> None:
    """Render the focused student question workflow."""
    st.subheader("Ask an official-source question")
    st.caption(
        "Choose an example or describe what you need help with. "
        "MelbourneMate will answer only when the checked sources provide support."
    )

    with st.container(border=True):
        st.markdown("#### Try an example")
        cols = st.columns(2)
        for index, (label, question) in enumerate(EXAMPLE_QUESTIONS.items()):
            if cols[index % 2].button(label, use_container_width=True):
                st.session_state["question"] = question

        question = st.text_area(
            "Your question",
            key="question",
            placeholder="For example: Where can I get help with underpayment?",
        )
        ask = st.button("Find an answer", type="primary", use_container_width=True)
        st.caption("Do not enter passport numbers, addresses or other personal details.")

    if not ask:
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return

    try:
        # Model loading starts only after the first question.
        pipeline = _build_pipeline()
        with st.spinner("Checking official sources..."):
            live_answer = st.empty()
            streamed: list[str] = []

            def show_chunk(piece: str) -> None:
                streamed.append(piece)
                live_answer.markdown(f"{''.join(streamed)} ▌")

            try:
                answer = answer_question(
                    pipeline,
                    question.strip(),
                    on_chunk=show_chunk,
                )
            finally:
                # Replace the temporary stream with the checked final answer below.
                live_answer.empty()
    except (CollectionError, OllamaError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"The local service is not ready: {exc}")
        return

    view = answer_view(answer)
    getattr(st, str(view["tone"]))(str(view["label"]))
    with st.container(border=True):
        st.markdown(str(view["text"]))

    sources = view["sources"]
    if sources:
        st.subheader("Official sources cited")
        for source in sources:
            st.markdown(
                f"- [{source['organisation']}: {source['heading']}]({source['url']}) "
                f"({source['passage_ids']})"
            )
    if view["citation_valid"] is False:
        st.warning("The answer needs a citation check before it is used.")
    st.caption(f"Local response time: {int(view['latency_ms']) / 1000:.1f}s")


@st.cache_resource(show_spinner=False)
def _build_pipeline() -> RagPipeline:
    # Read frozen CSV results only; opening the app does not rerun the models.
    root = _project_root()
    data_dir = Path(os.environ.get("MM_DATA", root / "data" / "v1"))
    qrels_path = Path(
        os.environ.get(
            "MM_QRELS",
            root / "review" / "targeted-qrels" / "qrels-final.txt",
        )
    )
    collection = _load_app_collection(data_dir, qrels_path)
    encoder = load_encoder("multi-qa-mpnet")
    retriever = build_retriever(collection, "dense", encoder)
    model = OllamaClient()
    model.resolve_digest()
    return RagPipeline(
        collection,
        retriever,
        model,
        gate=EvidenceGate(encoder),
        arm="dense",
    )


def main() -> None:
    st.set_page_config(
        page_title="MelbourneMate",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --mm-navy: #172b3d;
            --mm-blue: #246fa8;
            --mm-blue-soft: #eaf3fb;
            --mm-surface: #ffffff;
            --mm-canvas: #f5f7fa;
            --mm-border: #dbe3ea;
            --mm-muted: #607080;
        }
        .stApp {background: var(--mm-canvas);}
        .block-container {
            max-width: 1460px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {color: var(--mm-navy); letter-spacing: -0.025em;}
        h1 {margin-bottom: .3rem;}
        .mm-kicker {
            color: var(--mm-blue);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .12em;
            margin-bottom: -.35rem;
            text-transform: uppercase;
        }
        .mm-subtitle {
            color: var(--mm-muted);
            font-size: 1.08rem;
            line-height: 1.65;
            margin: 0 0 1rem;
            max-width: 760px;
        }
        .mm-trust-note {
            background: var(--mm-blue-soft);
            border: 1px solid #cfe1f1;
            border-radius: 14px;
            color: #294a63;
            line-height: 1.55;
            margin: .4rem 0 1.25rem;
            padding: .9rem 1.1rem;
        }
        .mm-axis-title {
            color: #737b8c;
            font-size: 1.05rem;
            font-weight: 500;
            margin: -.45rem 0 1rem;
            text-align: center;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--mm-surface);
            border-color: var(--mm-border);
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(30, 55, 75, .045);
        }
        div[data-testid="stSegmentedControl"] {margin: .2rem 0 1.7rem;}
        div[data-testid="stSegmentedControl"] button {
            min-height: 2.85rem;
            font-weight: 650;
        }
        div[data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--mm-border);
            gap: 1.25rem;
        }
        div[data-testid="stButton"] > button {
            border-color: #cbd7e1;
            border-radius: 11px;
            min-height: 3rem;
        }
        div[data-testid="stTextArea"] textarea {border-radius: 12px;}
        div[data-testid="stVegaLiteChart"] {overflow-x: auto;}
        div[data-testid="stTable"] {border-radius: 10px; overflow: hidden;}
        @media (min-width: 900px) {
            div[data-testid="stVegaLiteChart"] > div {
                margin-left: auto;
                margin-right: auto;
            }
        }
        @media (max-width: 700px) {
            h1 {font-size: 2.35rem;}
            h2 {font-size: 1.8rem;}
            h3 {font-size: 1.45rem;}
            .mm-subtitle {font-size: 1rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mm-kicker">Official-source student assistant</div>',
        unsafe_allow_html=True,
    )
    st.title("MelbourneMate")
    st.markdown(
        '<div class="mm-subtitle">Clear, source-grounded guidance for international '
        "students settling in Melbourne.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mm-trust-note"><strong>Designed to be cautious.</strong> '
        "MelbourneMate uses a checked set of official sources and directs you to "
        "an official source when the available evidence is not enough.</div>",
        unsafe_allow_html=True,
    )

    view = st.segmented_control(
        "Choose a view",
        APP_VIEWS,
        default=APP_VIEWS[0],
        key="app_view",
        label_visibility="collapsed",
        width="stretch",
    )
    if view == "Evaluation Results":
        _render_evaluation_results(_project_root())
    else:
        _render_question_view()


if __name__ == "__main__":
    main()
