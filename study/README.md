# Student study

Trello S10-04, GitHub #7. A small comparison of MelbourneMate with searching
official websites. It is impact evidence, not a research question, and it is
reported descriptively.

## Before recruiting

- Confirm with the teaching team whether this study needs course approval or a
  set consent form. If the course has its own form, use it and update
  `information-and-consent.md` to match. Add the team contact before use.
- Run sessions on Elaine's computer. The chatbot uses the final MPNet pipeline
  and the local `qwen2.5:7b-instruct` model, which is too slow on an 8 GB
  CPU-only laptop for timed tasks.
- Run one internal pilot with a team member and record it outside this folder.
- After the first participant, do not change the model, prompts, tasks,
  collection or recording fields. If a change is unavoidable, stop, record why
  and restart with new participant codes.

## Materials

| File | Use |
| --- | --- |
| `information-and-consent.md` | Information shown on the study page before consent |
| `tasks.csv` | The four tasks and the rule for judging completion |
| `schedule.csv` | Task order and method for P01 to P06 |
| `responses.csv` | One row per task, filled in by the study page |
| `consent.csv` | Created by the study page, one row per consenting code |

The schedule is counterbalanced. Each participant does two tasks with
MelbourneMate and two with official websites. Across P01 to P04, and again
across P01 to P06, every task is done equally often with each method and half
the participants start with each method.

## Session

Start the study page and the chatbot in two terminals.

```bash
streamlit run src/melbourne_mate/interface/study_app.py
streamlit run src/melbourne_mate/interface/app.py --server.port 8502
```

1. On the study page, choose the participant code. Never enter a name.
2. Show the information to the participant. The three consent boxes must all
   be ticked before **Start session** works. The page saves the code and time
   in `consent.csv`.
3. The page lists the tasks in schedule order. For `melbournemate`, the
   participant uses the chatbot tab. For `official-search`, they may use any
   search engine and any official website, but not MelbourneMate.
4. Allow up to 10 minutes per task. Press **Start timer** when the task is
   read and **Stop timer** when the participant answers or gives up. The time
   can be corrected by hand.
5. After each task, ask how confident they are in their answer and how much
   they trust the information, each from 1 (not at all) to 5 (fully). Judge
   completion against the rule shown under the task, then press **Save task**.

## Recording

The study page fills in `responses.csv`. The columns can also be edited by
hand.

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
