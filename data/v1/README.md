# Official-source collection v1

This is the formal MelbourneMate collection. It is newly authored for
international students settling in Melbourne and is built from official
government and student-support pages. It does not reuse Walert or Milestone 1
questions, passages or judgements.

## Contents

| Element | Count |
|---|---:|
| Official sources | 31 |
| Passages | 41 (34 marked `volatile`) |
| Topics | 30 (19 known, 11 inferred) |
| Validation / test split | 18 / 12 topics |
| Answerable questions | 90 (canonical plus two paraphrases per topic) |
| Out-of-KB questions | 30 |
| Judged question-passage pairs | 123 |
| Confusable topic pairs | 6 |

Splits are assigned per topic, so a topic's three phrasings always stay on the
same side of the validation/test boundary.

## Sources and access dates

Every source URL was opened and checked against the passages that cite it on
2 October 2026. `sources.csv` records that access date. `source-log.csv`
records, for each source, the page's own "last updated" date where shown, the
passages it supports, the outcome of the source check, and whether the final
recheck found a change since the previous check:

- `verified`: the page supports the cited passages as written;
- `url-updated`: the old URL moved, redirected or no longer returned the page,
  and the current official URL now replaces it;
- `passage-corrected`: passage wording was changed to match the current page.

The final recheck found 28 sources with no substantive change, two current
official URLs to update, and one cross-source pricing difference to record.
Study Melbourne still states a saving over $1,000, while the current Transport
Victoria page advertises a saving over $550. The formal collection remains the
frozen snapshot used by the saved evaluation runs, so the difference is logged
rather than silently changing a tested passage after evaluation.

Passages are concise paraphrased summaries, not verbatim copies. Facts about
rates, dates, eligibility or procedures are marked `volatile` and must be
rechecked before use; the collection is a dated snapshot.

## Relevance labels

`judgements.csv` and `qrels.txt` hold Walert-style topic-to-passage seed labels.
Out-of-KB questions have no labels. New MPNet candidates are checked separately
under S9-05 before final qrels are generated.

- A `known` topic has one passage that fully answers each of its questions
  (grade 2).
- An `inferred` topic has two passages that each answer part of the question
  (grade 1 each). No single passage is a complete answer, so a system needs
  both. `mm quality` reports question-term containment for grade-2 passages
  only, so its containment figures describe the known topics.

`gold.csv` holds one reference answer per answerable question. Each answer
covers every part of its own phrasing and is built from the passage sentences
listed in `support`; for inferred topics it uses both passages. The answers are
an auditable target and do not make the seed labels independent.

## Generation sample

`generation-sample.csv` freezes the 66 questions used by every generation
method: all 36 held-out test questions and all 30 out-of-KB questions. Its risk
category supports the safety results. It does not add relevance labels.

## Check

```bash
mm --data data/v1 validate
mm --data data/v1 quality
```
