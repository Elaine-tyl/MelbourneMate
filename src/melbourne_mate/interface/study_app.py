"""Researcher page for running one student study session."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from melbourne_mate.evaluation.study import (
    CONSENT_ITEMS,
    MAX_SECONDS,
    StudyError,
    has_consent,
    participant_tasks,
    participants,
    record_consent,
    save_response,
)

CONSENT_LABELS = {
    "information_read": "I have read the information and my questions have been answered.",
    "agrees_to_take_part": "I agree to take part and understand I can stop at any time.",
    "agrees_to_anonymous_use": (
        "I agree that my anonymous results can be used in the course report and presentation."
    ),
}
METHOD_LABELS = {
    "melbournemate": "Use MelbourneMate",
    "official-search": "Use any search engine and official websites",
}
SCALE = [1, 2, 3, 4, 5]


def study_dir() -> Path:
    default = Path(__file__).resolve().parents[3] / "study"
    return Path(os.environ.get("MM_STUDY_DIR", default))


def information_text(root: Path) -> str:
    """Information sheet without its paper consent section."""
    text = (root / "information-and-consent.md").read_text(encoding="utf-8")
    return text.split("\n## Consent", 1)[0]


def consent_step(root: Path, participant_id: str) -> None:
    with st.expander("Participant information", expanded=True):
        st.markdown(information_text(root))
    answers = {item: st.checkbox(CONSENT_LABELS[item]) for item in CONSENT_ITEMS}
    ready = all(answers.values())
    if st.button("Start session", type="primary", disabled=not ready):
        record_consent(root, participant_id, answers)
        st.rerun()
    if not ready:
        st.caption("Tick all three boxes to start.")


def timer(key: str) -> None:
    cols = st.columns(2)
    if cols[0].button("Start timer", key=f"{key}-start"):
        st.session_state[f"{key}-began"] = time.monotonic()
    if cols[1].button("Stop timer", key=f"{key}-stop") and f"{key}-began" in st.session_state:
        elapsed = time.monotonic() - st.session_state.pop(f"{key}-began")
        st.session_state[f"{key}-time"] = max(1, min(MAX_SECONDS, round(elapsed)))
    if f"{key}-began" in st.session_state:
        st.caption("Timer running.")


def task_step(root: Path, participant_id: str, task: dict[str, str]) -> None:
    key = f"{participant_id}-{task['position']}"
    saved = bool(task.get("completed"))
    title = f"Task {task['position']} · {METHOD_LABELS[task['method']]}"
    with st.expander(title + (" · saved" if saved else ""), expanded=not saved):
        st.markdown(f"**{task['prompt']}**")
        st.caption(f"Complete when: {task['complete_when']}")
        if task["method"] == "melbournemate":
            st.caption("Open the chatbot in another tab for this task.")
        if saved:
            st.caption(
                f"Saved: {task['completed']}, {task['time_seconds']} seconds, "
                f"confidence {task['confidence']}, trust {task['trust']}. Save again to correct it."
            )
        timer(key)
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
    root = study_dir()
    participant_id = st.selectbox("Participant code", participants(root), index=None)
    if participant_id is None:
        st.caption("Choose the participant code. Never enter a name.")
        return

    if not has_consent(root, participant_id):
        consent_step(root, participant_id)
        return

    st.success(f"Consent recorded for {participant_id}.")
    tasks = participant_tasks(root, participant_id)
    for task in tasks:
        task_step(root, participant_id, task)
    if all(task.get("completed") for task in tasks):
        st.success("All four tasks are saved. Thank the participant.")


if __name__ == "__main__":
    main()
