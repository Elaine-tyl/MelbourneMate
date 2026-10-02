"""Study session page. The sidebar is for the researcher, the main area for the participant."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from melbourne_mate.corpus import CollectionError
from melbourne_mate.evaluation.study import (
    COMPLETION,
    MAX_SECONDS,
    WOULD_USE_QUESTION,
    StudyError,
    judge_completion,
    load_would_use,
    participant_tasks,
    participants,
    save_response,
    save_would_use,
    warm_up,
)
from melbourne_mate.generation.ollama import OllamaError
from melbourne_mate.interface import app as chatbot

NOTICE = (
    "Taking part is voluntary. No personal information is recorded, and you can "
    "stop at any time."
)
METHODS = {
    "melbournemate": ("MelbourneMate", "blue"),
    "official-search": ("Official search", "gray"),
}
OFFICIAL_SEARCH_RULE = (
    "Use a search engine and open **official websites only**. "
    "No AI overviews or chat tools."
)
PARTICIPANT_GUIDE = """
1. Read the task and use **only the method shown** on it.
   - **MelbourneMate**. Press **Ask MelbourneMate**, then read the answer and
     its sources.
   - **Official search**. Open official websites from a search engine. No AI
     overviews or chat tools.
2. Write your answer **in your own words** in **Your answer**,
   in one or two sentences. MelbourneMate's answer is not copied in for you.
3. Rate how **confident** you are and how much you **trust** the information.

You have up to **5 minutes** per task. Please do not write your name or other
personal details.
"""
RESEARCHER_GUIDE = """
1. Choose the participant code and press **Prepare MelbourneMate**.
2. Press **Start timer** when the participant starts reading a task.
3. Press **Stop timer** when they finish writing **Your answer** or give up.
4. After they rate the task, press **Save and continue**.
5. After four tasks, ask the final question.
6. When the participant has left, open **Review** and judge completion.
"""
SCALE = [1, 2, 3, 4, 5]
SCALE_HELP = "1 = not at all, 5 = fully"
WARM_MINUTES = 25  # Ollama keeps the model loaded for 30 minutes
FINAL = "final"
REVIEW = "review"
SERVICE_ERRORS = (CollectionError, ImportError, OllamaError, OSError, RuntimeError, ValueError)
MISSING_DENSE = (
    "MelbourneMate cannot load on this computer because the MPNet retriever is not "
    "installed. Run the session on Elaine's computer, or install it with "
    "`pip install -e \".[dense]\"`."
)
STYLE = """
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.mm-task {font-size: 1.2rem; line-height: 1.5; margin: .4rem 0 .8rem;}
.mm-muted {color: #607080; font-size: .95rem;}
</style>
"""


def study_dir() -> Path:
    default = Path(__file__).resolve().parents[3] / "study"
    return Path(os.environ.get("MM_STUDY_DIR", default))


def ask_melbournemate(question: str):
    """Answer through the same cached pipeline as the MelbourneMate app."""
    return chatbot.answer_question(chatbot._build_pipeline(), question)


def service_message(exc: Exception, lead: str) -> str:
    """Plain message for a local service problem."""
    if isinstance(exc, ImportError):
        return MISSING_DENSE
    return f"{lead}: {exc}"


def is_warm(participant_id: str) -> bool:
    started = st.session_state.get(f"{participant_id}-warm-at")
    return started is not None and time.monotonic() - started < WARM_MINUTES * 60


# ---------- researcher sidebar ----------


def warm_up_panel(participant_id: str) -> None:
    """Untimed full-pipeline warm-up, done by the researcher before the tasks."""
    if is_warm(participant_id):
        seconds = st.session_state[f"{participant_id}-warm"]
        st.success(f"MelbourneMate is ready (warmed up in {seconds} s).")
        return
    st.caption("Run once before the participant starts. It is not timed.")
    if st.button("Prepare MelbourneMate", key=f"{participant_id}-warm-button", type="primary"):
        try:
            with st.spinner("Loading MelbourneMate..."):
                st.session_state[f"{participant_id}-warm"] = warm_up(ask_melbournemate)
                st.session_state[f"{participant_id}-warm-at"] = time.monotonic()
        except SERVICE_ERRORS as exc:
            st.error(service_message(exc, "Warm-up failed"))
        else:
            st.rerun()


def is_saved(task: dict) -> bool:
    return bool(task.get("time_seconds"))


def task_picker(participant_id: str, tasks: list[dict], final_saved: bool) -> str:
    """Choose which step to show. Defaults to the first unfinished one."""
    key = f"{participant_id}-step"
    all_saved = all(is_saved(task) for task in tasks)
    # Final opens only after four saved tasks, and Review only after the final
    # answer, so the completion rules never appear while the participant works.
    options = [task["position"] for task in tasks]
    if all_saved:
        options.append(FINAL)
    if all_saved and final_saved:
        options.append(REVIEW)
    first_open = next((t["position"] for t in tasks if not is_saved(t)), FINAL)
    pending = st.session_state.pop(f"{key}-next", None)
    if pending in options:
        st.session_state[key] = pending
    if st.session_state.get(key) not in options:
        st.session_state[key] = first_open

    def label(option: str) -> str:
        if option == FINAL:
            return ("✅ " if final_saved else "◻️ ") + "Final question"
        if option == REVIEW:
            judged = all(t.get("completed") for t in tasks if is_saved(t))
            return ("✅ " if judged and any(map(is_saved, tasks)) else "🔍 ") + "Review (researcher)"
        task = tasks[int(option) - 1]
        name = METHODS[task["method"]][0]
        return ("✅ " if is_saved(task) else "◻️ ") + f"Task {option} · {name}"

    step = st.radio("Steps", options, format_func=label, key=key)
    if not all_saved:
        st.caption("Final question opens after all four tasks are saved.")
    elif not final_saved:
        st.caption("Review opens after the final answer is saved.")
    return step


# ---------- participant view ----------


def timer(key: str, ready: bool) -> None:
    left, right, status = st.columns([1, 1, 2], vertical_alignment="center")
    if left.button("Start timer", key=f"{key}-start", disabled=not ready, use_container_width=True):
        st.session_state[f"{key}-began"] = time.monotonic()
        st.session_state[f"{key}-began-clock"] = time.strftime("%H:%M:%S")
    running = f"{key}-began" in st.session_state
    stopped = right.button(
        "Stop timer", key=f"{key}-stop", disabled=not running, use_container_width=True
    )
    if stopped and running:
        elapsed = time.monotonic() - st.session_state.pop(f"{key}-began")
        st.session_state[f"{key}-time"] = max(1, min(MAX_SECONDS, round(elapsed)))
    if f"{key}-began" in st.session_state:
        status.markdown(f"⏱️ Running since {st.session_state[f'{key}-began-clock']}")
    elif stopped or st.session_state.get(f"{key}-time", 1) > 1:
        status.markdown(f"⏱️ {st.session_state.get(f'{key}-time', 1)} seconds")
    elif not ready:
        status.caption("Waiting for the researcher to prepare MelbourneMate.")


def melbournemate_box(key: str, prompt: str, ready: bool) -> None:
    """Ask MelbourneMate. The box starts with the task text."""
    st.session_state.setdefault(f"{key}-question", prompt)
    question = st.text_area("Question for MelbourneMate", key=f"{key}-question", height=90)
    if st.button("Ask MelbourneMate", key=f"{key}-ask", disabled=not ready, type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            with st.spinner("Checking official sources..."):
                view = chatbot.answer_view(ask_melbournemate(question.strip()))
        except SERVICE_ERRORS as exc:
            st.error(service_message(exc, "MelbourneMate is not ready"))
            return
        with st.container(border=True):
            getattr(st, str(view["tone"]))(str(view["label"]))
            st.markdown(str(view["text"]))
            if view["sources"]:
                st.markdown("**Sources**")
                for source in view["sources"]:
                    st.markdown(f"- [{source['organisation']}: {source['heading']}]({source['url']})")


def task_card(root: Path, participant_id: str, task: dict, last: bool) -> None:
    key = f"{participant_id}-{task['position']}"
    uses_chatbot = task["method"] == "melbournemate"
    ready = is_warm(participant_id) or not uses_chatbot
    name, colour = METHODS[task["method"]]

    with st.container(border=True):
        st.badge(name, color=colour)
        st.markdown(f"<div class='mm-task'>{task['prompt']}</div>", unsafe_allow_html=True)
        timer(key, ready)
        if uses_chatbot:
            melbournemate_box(key, task["prompt"], ready)
        else:
            st.info(OFFICIAL_SEARCH_RULE)

        st.session_state.setdefault(f"{key}-answer", task.get("answer", ""))
        answer = st.text_area("Your answer (one or two sentences)", key=f"{key}-answer", height=90)
        source = "what MelbourneMate showed" if uses_chatbot else "the official websites you found"
        st.caption(f"Write the answer in your own words, based on {source}.")
        left, right = st.columns(2)
        confidence = left.radio(
            "How confident are you in your answer?",
            SCALE,
            index=None,
            horizontal=True,
            key=f"{key}-conf",
        )
        trust = right.radio(
            "How much do you trust the information?",
            SCALE,
            index=None,
            horizontal=True,
            key=f"{key}-trust",
        )
        st.caption(SCALE_HELP)

    with st.container(border=True):
        st.markdown("**Researcher**")
        if is_saved(task):
            st.caption(
                f"Saved with {task['time_seconds']} s, confidence {task['confidence']} and "
                f"trust {task['trust']}. Saving again replaces it."
            )
        left, right = st.columns([1, 2])
        st.session_state.setdefault(f"{key}-time", int(task.get("time_seconds") or 1))
        seconds = left.number_input(
            "Seconds", min_value=1, max_value=MAX_SECONDS, key=f"{key}-time"
        )
        note = right.text_input("Note (optional, no personal details)", key=f"{key}-note")
        missing = None in (confidence, trust)
        label = "Save and go to the final question" if last else "Save and continue"
        if st.button(label, key=f"{key}-save", type="primary", disabled=missing):
            try:
                save_response(
                    root,
                    participant_id,
                    task["position"],
                    {
                        "time_seconds": seconds,
                        "confidence": confidence,
                        "trust": trust,
                        "answer": answer,
                        "note": note,
                    },
                )
            except StudyError as exc:
                st.error(str(exc))
            else:
                next_step = FINAL if last else str(int(task["position"]) + 1)
                st.session_state[f"{participant_id}-step-next"] = next_step
                st.rerun()
        if missing:
            st.caption("Confidence and trust are needed before saving.")


def final_card(root: Path, participant_id: str) -> None:
    saved = load_would_use(root).get(participant_id)
    with st.container(border=True):
        st.subheader("Final question")
        rating = st.radio(
            WOULD_USE_QUESTION, SCALE, index=None, horizontal=True, key=f"{participant_id}-use"
        )
        if st.button(
            "Save final answer",
            key=f"{participant_id}-use-save",
            type="primary",
            disabled=rating is None,
        ):
            try:
                save_would_use(root, participant_id, rating)
            except StudyError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    if saved:
        st.success(f"Final answer saved ({saved}/5). Thank you for taking part! 🎉")
        st.caption("Researcher. Open Review in the sidebar after the participant has left.")


def review_card(root: Path, participant_id: str, tasks: list[dict]) -> None:
    """Judge completion from the saved answers, out of the participant's view."""
    st.subheader("Review (researcher only)")
    st.caption("Judge each saved answer against its completion rule.")
    choices = {}
    for task in tasks:
        name, colour = METHODS[task["method"]]
        with st.container(border=True):
            st.badge(f"Task {task['position']} · {name}", color=colour)
            st.markdown(task["prompt"])
            if not is_saved(task):
                st.caption("Not attempted.")
                continue
            st.markdown(f"**Answer.** {task.get('answer') or '(no answer written)'}")
            st.markdown(f"**Complete when.** {task['complete_when']}")
            saved = task.get("completed")
            choices[task["position"]] = st.radio(
                "Completed?",
                COMPLETION,
                index=COMPLETION.index(saved) if saved in COMPLETION else None,
                horizontal=True,
                key=f"{participant_id}-{task['position']}-judge",
            )
    if not choices:
        st.info("No saved tasks to review yet.")
        return
    if st.button(
        "Save judgements",
        key=f"{participant_id}-judge-save",
        type="primary",
        disabled=None in choices.values(),
    ):
        try:
            for position, completed in choices.items():
                judge_completion(root, participant_id, position, completed)
        except StudyError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if all(task.get("completed") for task in tasks if is_saved(task)):
        st.success("All saved tasks are judged.")


def main() -> None:
    st.set_page_config(page_title="MelbourneMate study", page_icon="📋", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    root = study_dir()

    with st.sidebar:
        st.header("Researcher")
        participant_id = st.selectbox("Participant code", participants(root), index=None)
        with st.expander("Researcher steps"):
            st.markdown(RESEARCHER_GUIDE)

    st.title("📋 MelbourneMate study")
    st.markdown(
        "<div class='mm-muted'>Thank you for helping us test MelbourneMate.</div>",
        unsafe_allow_html=True,
    )
    st.info(NOTICE)
    tasks = participant_tasks(root, participant_id) if participant_id else []
    started = any(is_saved(task) for task in tasks)
    with st.expander("How it works", expanded=not started):
        st.markdown(PARTICIPANT_GUIDE)

    if participant_id is None:
        st.caption("The researcher will start the session.")
        return

    final_saved = participant_id in load_would_use(root)
    with st.sidebar:
        warm_up_panel(participant_id)
        step = task_picker(participant_id, tasks, final_saved)

    if step == REVIEW:
        review_card(root, participant_id, tasks)
        return
    done = sum(is_saved(task) for task in tasks)
    st.progress(done / len(tasks), text=f"{done} of {len(tasks)} tasks saved")
    if step == FINAL:
        final_card(root, participant_id)
        return
    position = int(step)
    st.markdown(f"#### Task {position} of {len(tasks)}")
    task_card(root, participant_id, tasks[position - 1], position == len(tasks))


if __name__ == "__main__":
    main()
