# Student study

Trello S10-04, GitHub #64. A small comparison of MelbourneMate with searching
official websites. It is impact evidence, not a research question, and it is
reported descriptively.

## Participants

Recruit at least four international students, P01 to P04. Each does all four
tasks, two with MelbourneMate and two with official search, which gives 16 task
records. P05 and P06 are optional. A participant never repeats a task with the
other method, because the second attempt would already know the answer.

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
| `final-template.csv` | Blank sheet for each participant's final would-use rating |

The study page copies the templates to `responses.csv` and `final.csv`. These
files hold participant-level data, so they are listed in `.gitignore` and stay
on the session computer. Only the blank templates are committed.

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

The page shows a short **For participants** guide at the top. It tells the
participant to use only the method shown, write a one or two sentence answer,
rate confidence and trust, and leave out personal details. These researcher
steps are under **For the researcher**.

1. Choose the participant code. Never enter a name.
2. The page lists the tasks in schedule order. For `official-search`, the
   participant follows the official search rule above.
3. Before the first `melbournemate` task, press **Warm up MelbourneMate**. It
   runs one fixed question through the full retrieval and generation pipeline,
   so loading MPNet, the index and the local model is not timed. The timer and
   the question box for MelbourneMate tasks stay locked until this is done.
4. For `melbournemate` tasks, the task text is already in the question box, so
   every participant asks the same question. Press **Ask MelbourneMate**. It
   uses the same pipeline as the MelbourneMate app.
5. Allow up to 5 minutes per task. Press **Start timer** when the task is read
   and **Stop timer** when the participant answers or gives up. The time can be
   corrected by hand.
6. After each task, the participant writes their answer in **Your answer** and
   rates confidence and trust, each from 1 (not at all) to 5 (fully). Judge
   completion, then press **Save task**.
7. After the fourth task, ask the final question and press **Save final
   answer**. *How likely are you to use MelbourneMate instead of searching
   multiple official websites?* 1 means very unlikely and 5 means very likely.

The completion rule for each task gives away the answer, so it is hidden.
Turn on **Show completion rules (researcher only)** only when judging
completion out of the participant's view.

## Recording

The study page fills in `responses.csv`. The columns can also be edited by
hand on the session computer.

| Column | Values |
| --- | --- |
| `completed` | `yes`, `partial` or `no`, judged against `complete_when` in `tasks.csv` |
| `time_seconds` | whole seconds, 1 to 300 |
| `confidence` | 1 to 5 |
| `trust` | 1 to 5 |
| `answer` | the participant's answer in one or two sentences, with no personal details |
| `note` | optional short observation, with no personal details |

Leave all four measure columns blank for a task that was not attempted. Keep
names, visa details, contact details and any other personal information out
of every file.

## Summary

```bash
mm study-summary --study-dir study
```

This checks `responses.csv` against the schedule and writes `summary.csv` and
`summary.md`. The summary table compares task completion, median time, mean
confidence and mean trust for MelbourneMate and official search, and adds the
mean would-use rating. Results are descriptive. With fewer than four participants the
summary says so, and the report should not compare the methods beyond
describing what happened. The summary holds only group results, so it can be
committed with the report evidence.
