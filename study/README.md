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
- On that computer, install the retriever and app extras with
  `pip install -e ".[dense,app]"`, start Ollama and check that
  `qwen2.5:7b-instruct` is listed by `ollama list`.
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

The page tells every participant that taking part is voluntary and they can
stop at any time. Their answers, times and ratings are saved under a
participant code on the session computer only, and they are asked not to write
their name or other personal details. No separate consent form is used for
this classroom activity.

Official search means using a search engine to open official websites, such
as Home Affairs, Study Melbourne or Consumer Affairs Victoria. AI summaries, AI
overviews and chat tools are not allowed. Ask participants to scroll past any
AI overview the search engine shows.

The main area is for the participant. It shows the notice, a short **How it
works** guide and one task at a time with a progress bar. The sidebar is for
the researcher. It holds the participant code, the warm-up and the list of
steps, including **Review** at the end.

1. Choose the participant code in the sidebar. Never enter a name.
2. Press **Prepare MelbourneMate** before the participant starts. It runs one
   fixed question through the full retrieval and generation pipeline, so
   loading MPNet, the index and the local model is not timed. MelbourneMate
   tasks stay locked until this is done, and it lasts 25 minutes.
3. Press **Start timer** when the participant starts reading a task. For a
   MelbourneMate task, the task text is already in the question box, so
   every participant asks the same question. For official search, the
   participant follows the rule above.
4. The participant writes their answer **in their own words** in **Your
   answer**, in one or two sentences. MelbourneMate's answer appears in its
   own box and is not copied in, so both methods are compared the same way.
5. Press **Stop timer** when the participant finishes the answer or gives up.
   Each task has up to 5 minutes, and the time can be corrected by hand. The
   participant then rates confidence and trust from 1 (not at all) to 5
   (fully).
6. Press **Save and continue** in the **Researcher** box. The next task opens
   automatically. Pick a step in the sidebar to correct a saved task.
7. After the fourth task, **Final question** opens. Ask *How likely are you to
   use MelbourneMate instead of searching multiple official websites?* (1 very
   unlikely, 5 very likely) and press **Save final answer**.
8. When the participant has left, open **Review** in the sidebar. It opens only
   after the final answer is saved and lists each saved answer next to its
   completion rule. Choose yes, partial or no for each and press **Save
   judgements**.

The participant and the researcher share one computer. The completion rule
gives away the answer, so it only appears in **Review**, after the session.
Judging from the saved answers also keeps the judgement consistent across
participants.

## Recording

The study page fills in `responses.csv`. The columns can also be edited by
hand on the session computer.

| Column | Values |
| --- | --- |
| `completed` | `yes`, `partial` or `no`, judged in **Review** against `complete_when` in `tasks.csv` |
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

This checks `responses.csv` against the schedule and counts only participants
who finished all four tasks. Anyone who stopped early is listed as excluded,
and the command stops while a counted task still needs a judgement. It writes
`summary.csv` and `summary.md`, which compare completion, median time, mean
confidence and mean trust for each method and add the mean would-use rating.
With fewer than four participants the summary says the methods should only be
described, not compared. Both files hold only group results, so they are committed.
