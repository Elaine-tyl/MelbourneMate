# Student study

Trello S10-04, GitHub #7. A small comparison of MelbourneMate with searching
official websites. It is impact evidence, not a research question, and it is
reported descriptively.

## Before recruiting

- Confirm with the teaching team whether this study needs course approval or a
  set consent form. Update `information-and-consent.md` if it does.
- Run sessions on a machine that answers in seconds. The chatbot uses the final
  MPNet pipeline and the local `qwen2.5:7b-instruct` model.
- Run one internal pilot with a team member and record it outside this folder.
- After the first participant, do not change the model, prompts, tasks,
  collection or recording fields. If a change is unavoidable, stop, record why
  and restart with new participant codes.

## Materials

| File | Use |
| --- | --- |
| `information-and-consent.md` | Read with the participant before starting |
| `tasks.csv` | The four tasks and the rule for judging completion |
| `schedule.csv` | Task order and method for P01 to P06 |
| `responses.csv` | One row per task, filled in during the session |

The schedule is counterbalanced. Each participant does two tasks with
MelbourneMate and two with official websites. Across P01 to P04, and again
across P01 to P06, every task is done equally often with each method and half
the participants start with each method.

## Session

1. Read the information sheet and record consent. Do not write the
   participant's name anywhere in this folder.
2. Give the tasks in the order listed in `schedule.csv`.
3. For `melbournemate`, start the chatbot with
   `streamlit run src/melbourne_mate/interface/app.py`. For
   `official-search`, the participant may use any search engine and any
   official website, but not MelbourneMate.
4. Allow up to 10 minutes per task. Start timing when the task is read and
   stop when the participant gives an answer or gives up.
5. After each task, ask how confident they are in their answer and how much
   they trust the information, each from 1 (not at all) to 5 (fully).

## Recording

Fill in the blank columns of `responses.csv`.

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
describing what happened.
