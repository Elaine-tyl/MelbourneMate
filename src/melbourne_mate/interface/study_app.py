"""Researcher page for running one student study session."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from melbourne_mate.corpus import CollectionError
from melbourne_mate.evaluation.study import (
    MAX_SECONDS,
    StudyError,
    participant_tasks,
    participants,
    save_response,
    warm_up,
)
from melbourne_mate.generation.ollama import OllamaError
from melbourne_mate.interface import app as chatbot

NOTICE = (
    "Taking part is voluntary. No personal information is recorded, and you can "
    "stop at any time."
)
METHOD_LABELS = {
    "melbournemate": "Use MelbourneMate",
    "official-search": "Use a search engine to open official websites",
}
OFFICIAL_SEARCH_RULE = (
    "Open official websites from a search engine. Do not use AI summaries, "
    "AI overviews or chat tools."
)
SERVICE_ERRORS = (CollectionError, OllamaError, OSError, RuntimeError, ValueError)
SCALE = [1, 2, 3, 4, 5]


def study_dir() -> Path:
    default = Path(__file__).resolve().parents[3] / "study"
    return Path(os.environ.get("MM_STUDY_DIR", default))


def ask_melbournemate(question: str):
    """Answer through the same cached pipeline as the MelbourneMate app."""
    return chatbot.answer_question(chatbot._build_pipeline(), question)


def warm_up_step(participant_id: str) -> None:
    """Untimed pipeline warm-up before the MelbourneMate tasks."""
    key = f"{participant_id}-warm"
    if key in st.session_state:
        st.success(f"MelbourneMate warmed up in {st.session_state[key]} seconds.")
        return
    st.info(
        "Warm up MelbourneMate before its timed tasks. One fixed question runs "
        "through retrieval and generation so loading time is not timed."
    )
    if st.button("Warm up MelbourneMate", key=f"{key}-button"):
        try:
            with st.spinner("Loading MelbourneMate..."):
                st.session_state[key] = warm_up(ask_melbournemate)
        except SERVICE_ERRORS as exc:
            st.error(f"Warm-up failed: {exc}")
        else:
            st.rerun()


def melbournemate_box(key: str, ready: bool) -> None:
    """Question box for MelbourneMate tasks, using the evaluated pipeline."""
    question = st.text_area("Question for MelbourneMate", key=f"{key}-question")
    if st.button("Ask MelbourneMate", key=f"{key}-ask", disabled=not ready):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            with st.spinner("Checking official sources..."):
                view = chatbot.answer_view(ask_melbournemate(question.strip()))
        except SERVICE_ERRORS as exc:
            st.error(f"The local service is not ready: {exc}")
            return
        getattr(st, str(view["tone"]))(str(view["label"]))
        with st.container(border=True):
            st.markdown(str(view["text"]))
        for source in view["sources"]:
            st.markdown(f"- [{source['organisation']}: {source['heading']}]({source['url']})")


def timer(key: str, ready: bool) -> None:
    cols = st.columns(2)
    if cols[0].button("Start timer", key=f"{key}-start", disabled=not ready):
        st.session_state[f"{key}-began"] = time.monotonic()
    if cols[1].button("Stop timer", key=f"{key}-stop") and f"{key}-began" in st.session_state:
        elapsed = time.monotonic() - st.session_state.pop(f"{key}-began")
        st.session_state[f"{key}-time"] = max(1, min(MAX_SECONDS, round(elapsed)))
    if f"{key}-began" in st.session_state:
        st.caption("Timer running.")


def task_step(root: Path, participant_id: str, task: dict[str, str]) -> None:
    key = f"{participant_id}-{task['position']}"
    saved = bool(task.get("completed"))
    uses_chatbot = task["method"] == "melbournemate"
    ready = f"{participant_id}-warm" in st.session_state or not uses_chatbot
    title = f"Task {task['position']} · {METHOD_LABELS[task['method']]}"
    with st.expander(title + (" · saved" if saved else ""), expanded=not saved):
        st.markdown(f"**{task['prompt']}**")
        st.caption(f"Complete when: {task['complete_when']}")
        if not uses_chatbot:
            st.caption(OFFICIAL_SEARCH_RULE)
        if saved:
            st.caption(
                f"Saved: {task['completed']}, {task['time_seconds']} seconds, "
                f"confidence {task['confidence']}, trust {task['trust']}. Save again to correct it."
            )
        timer(key, ready)
        if uses_chatbot:
            melbournemate_box(key, ready)
        st.session_state.setdefault(f"{key}-time", int(task.get("time_seconds") or 1))
        seconds = st.number_input(
            "Seconds taken", min_value=1, max_value=MAX_SECONDS, key=f"{key}-time"
        )
        completed = st.radio(
            "Completed", ["yes", "partial", "no"], index=None, horizontal=True, key=f"{key}-done"
        )
        confidence = st.radio(
            "Confidence in the answer", SCALE, index=None, horizontal=True, key=f"{key}-conf"
        )
        trust = st.radio(
            "Trust in the information", SCALE, index=None, horizontal=True, key=f"{key}-trust"
        )
        note = st.text_input("Note (no personal details)", key=f"{key}-note")
        if st.button("Save task", key=f"{key}-save", disabled=None in (completed, confidence, trust)):
            try:
                save_response(
                    root,
                    participant_id,
                    task["position"],
                    {
                        "completed": completed,
                        "time_seconds": seconds,
                        "confidence": confidence,
                        "trust": trust,
                        "note": note,
                    },
                )
            except StudyError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="MelbourneMate study", page_icon="📝", layout="centered")
    st.title("📝 MelbourneMate study session")
    st.info(NOTICE)
    root = study_dir()
    participant_id = st.selectbox("Participant code", participants(root), index=None)
    if participant_id is None:
        st.caption("Choose the participant code. Never enter a name.")
        return

    tasks = participant_tasks(root, participant_id)
    first_chatbot_task = next(
        (task["position"] for task in tasks if task["method"] == "melbournemate"), None
    )
    for task in tasks:
        if task["position"] == first_chatbot_task:
            warm_up_step(participant_id)
        task_step(root, participant_id, task)
    if all(task.get("completed") for task in tasks):
        st.success("All four tasks are saved. Thank the participant.")


if __name__ == "__main__":
    main()
