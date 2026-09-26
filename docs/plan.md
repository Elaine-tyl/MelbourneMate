# Baseline delivery plan — recorded 23 September 2026

This is the original delivery baseline retained as project-management evidence.
Current completion status and remaining submission work are tracked in Trello.

Fixed dates: **oral presentation due Tue 20 Oct, 10:00** · **written report due
Sun 25 Oct, 23:59**. The presentation is five days before the report, so it —
not the report — sets the date by which every number must be final.

**All results frozen: Fri 16 Oct.** Nothing is re-run after that except to fix
a defect, and a re-run after that date is recorded as such.

| Week | Dates | Goal | Done when |
|---|---|---|---|
| 1 | 23–29 Sep | Collection, half of it | 15 topics with passages, questions and judgements; `mm validate` passes; split file committed; repo, CI and board set up |
| 2 | 30 Sep – 6 Oct | Collection finished and frozen | 30 topics, 90 answerable questions and 30 OOKB; Walert-style ground truth converted to `qrels.txt`; new MPNet candidates receive targeted verification; BM25s and Dense validation runs complete |
| 3 | 7–13 Oct | Parameters frozen, test split scored once | Gate thresholds and top-k chosen on validation; `config.py` tagged `config-frozen`; BM25s and Dense test runs + paired comparison; slice table by question form and containment bin |
| 4 | 14–19 Oct | Generation evidence and the video | Frozen generation sample run on BM25, Dense and no-context; refusal, citation and 2×2 tables; manual validation of 20–30 answers by both members; one documented refine → re-evaluate cycle; video recorded |
| 5 | 20–25 Oct | Report | Presentation submitted Mon 19 Oct (a day early); report, evidence appendix, contribution sheet and AI declaration submitted by Fri 23 Oct, leaving the weekend as buffer |

## The one-way doors

1. **Judgements freeze before any system runs** (end of week 2). Judging after
   seeing results invalidates the whole comparison.
2. **Parameters freeze before the test split is scored** (mid week 3). Tune on
   validation, then tag, then score the test split once.
3. **The test split is scored once.** If a genuine defect forces a re-run, the
   report says so and reports both.

## Weekly rhythm

Monday 30 min planning, Thursday 30 min check, Sunday 15 min: board tidy, and
one line each in `docs/contributions.md` — task, evidence link, outcome. Written
weekly, not reconstructed in week 5.

## Git

Commit subjects begin with the matching Trello card id and use plain language,
for example `[S9-06] Compare BM25 and MPNet retrieval`. Conventional prefixes
are not required. Tags: `collection-v1`, `config-frozen`, `baseline-bm25`,
`baseline-dense`. History is never rewritten.

## If something slips

Cut in this order, and say so in the limitations section:

1. the optional reranker — already out of scope;
2. the second paraphrase drops on the easiest 10 topics (never on the hardest);
3. manual validation drops from 30 answers to 20;
4. the report focuses on the primary BM25-versus-Dense comparison.

Never cut: the frozen split, the judgement freeze, the OOKB set, or the
no-context ablation. Those four are what the argument rests on.

## Risks

| Risk | Sign | Response |
|---|---|---|
| Collection takes longer than two weeks | Fewer than 15 topics done by 29 Sep | Reduce topic count transparently; keep three question forms per topic and preserve the held-out split |
| The local model is slow or unavailable mid-run | `mm generate` stops on the failing question | Nothing is lost: the sample is frozen, so restart that arm with a new run id. `mm spike` checks the server before a long run |
| Dense model too slow on a laptop | Indexing takes minutes | Cache the passage matrix to disk; 120 passages is seconds either way |
| Paraphrases come in above 0.55 containment | `mm quality` after the first batch | Rewrite them with the passage closed — an easy benchmark cannot answer RQ2 |
