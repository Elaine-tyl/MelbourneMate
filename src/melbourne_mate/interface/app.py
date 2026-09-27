"""Streamlit interface for the frozen MelbourneMate pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from melbourne_mate.corpus import CollectionError, load_collection
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


def answer_view(answer: Answer) -> dict[str, object]:
    """Turn a pipeline answer into the small view model used by Streamlit."""
    state = {
        "answered": ("Answer ready", "success"),
        "refused": ("Needs an official check", "warning"),
        "truncated": ("Answer stopped early", "warning"),
    }
    label, tone = state.get(answer.status, ("Unable to answer", "error"))

    sources: dict[str, dict[str, object]] = {}
    for item in answer.evidence:
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_resource(show_spinner=False)
def _build_pipeline() -> RagPipeline:
    data_dir = Path(os.environ.get("MM_DATA", _project_root() / "data" / "v1"))
    collection = load_collection(data_dir)
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
        page_icon="🧭",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 820px; padding-top: 2.2rem;}
        [data-testid="stAppViewContainer"] {background: #f5f8fb;}
        .mm-note {color: #4a5d70; font-size: .92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧭 MelbourneMate")
    st.write("Official information for international students settling in Melbourne.")
    st.info(
        "MelbourneMate uses a small set of checked official sources. "
        "It will ask you to check an official source when the evidence is not enough."
    )

    st.subheader("Try an example")
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
    if not ask:
        st.caption("Do not enter passport numbers, addresses or other personal details.")
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return

    try:
        # Model loading starts only after the first question.
        pipeline = _build_pipeline()
        with st.spinner("Checking official sources..."):
            answer = pipeline.answer(question.strip())
    except (CollectionError, OllamaError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"The local service is not ready: {exc}")
        return

    view = answer_view(answer)
    getattr(st, str(view["tone"]))(str(view["label"]))
    with st.container(border=True):
        st.markdown(str(view["text"]))

    sources = view["sources"]
    if sources:
        st.subheader("Official sources checked")
        for source in sources:
            st.markdown(
                f"- [{source['organisation']}: {source['heading']}]({source['url']}) "
                f"({source['passage_ids']})"
            )
    if view["citation_valid"] is False:
        st.warning("The answer needs a citation check before it is used.")
    st.caption(f"Local response time: {int(view['latency_ms']) / 1000:.1f}s")


if __name__ == "__main__":
    main()
