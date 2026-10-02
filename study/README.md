# Student study

Trello S10-04, GitHub #64. A small comparison of MelbourneMate with searching
official websites. It is impact evidence, not a research question, and it is
reported descriptively.

## Before recruiting

- Run sessions on Elaine's computer. MelbourneMate uses the final MPNet
  pipeline and the local `qwen2.5:7b-instruct` model, which is too slow on an
  8 GB CPU-only laptop for timed tasks.
- Run one internal pilot with a team member and record it outside this folder.
- After the first participant, do not change the model, prompts, tasks,
  collection or recording fields. If a change is unavoidable, stop, record why
  and restart with new participant codes.

## Materials

| File | Use |
| --- | --- |
| `tasks.csv` | The four tasks and the rule for judging completion |
| `schedule.csv` | Task order and method for P01 to P06 |
| `responses-template.csv` | Blank sheet with one row per scheduled task |

The study page copies the template to `responses.csv`. That file holds
participant-level data, so it is listed in `.gitignore` and stays on the
session computer. Only the blank template is committed.

The schedule is counterbalanced. Each participant does two tasks with
MelbourneMate and two with official websites. Across P01 to P04, and again
across P01 to P06, every task is done equally often with each method and half
the participants start with each method.

## Session

Start the study page.

```bash
streamlit run src/melbourne_mate/interface/study_app.py
```

The page shows this notice to every participant. Taking part is voluntary, no
personal information is recorded, and they can stop at any time. No separate
consent form is used for this classroom activity.

Official search means using a search engine to open official websites, such
as Home Affairs, Study Melbourne or Consumer Affairs Victoria. AI summaries, AI
overviews and chat tools are not allowed. Ask participants to scroll past any
AI overview the search engine shows.

1. Choose the participant code. Never enter a name.
2. The page lists the tasks in schedule order. For `official-search`, the
   participant follows the official search rule above.
3. Before the first `melbournemate` task, press **Warm up MelbourneMate**. It
   runs one fixed question through the full retrieval and generation pipeline,
   so loading MPNet, the index and the local model is not timed. The timer and
   the question box for MelbourneMate tasks stay locked until this is done.
4. For `melbournemate` tasks, the participant types into the question box on
   the same page. It uses the same pipeline as the MelbourneMate app.
5. Allow up to 10 minutes per task. Press **Start timer** when the task is
   read and **Stop timer** when the participant answers or gives up. The time
   can be corrected by hand.
6. After each task, ask how confident they are in their answer and how much
   they trust the information, each from 1 (not at all) to 5 (fully). Judge
   completion against the rule shown under the task, then press **Save task**.

## Recording

The study page fills in `responses.csv`. The columns can also be edited by
hand on the session computer.

| Column | Values |
| --- | --- |
| `completed` | `yes`, `partial` or `no`, judged against `complete_when` in `tasks.csv` |
| `time_seconds` | whole seconds, 1 to 600 |
| `confidence` | 1 to 5 |
| `trust` | 1 to 5 |
| `note` | optional short observation, with no personal details |

Leave all four measure columns blank for a task that was not attempted. Keep
names, visa details, contact details and any other personal information out
of every file.

## Summary

```bash
mm study-summary --study-dir study
```

This checks `responses.csv` against the schedule and writes `summary.csv` and
`summary.md`. Results are descriptive. With fewer than four participants the
summary says so, and the report should not compare the methods beyond
describing what happened. The summary holds only group results, so it can be
committed with the report evidence.
