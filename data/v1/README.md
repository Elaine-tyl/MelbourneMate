# Official-source collection v1

This is the formal MelbourneMate collection. It is newly authored for
international students settling in Melbourne and is built from official
government and student-support pages. It does not reuse Walert or Milestone 1
questions, passages or judgements.

## Contents

| Element | Count |
|---|---:|
| Official sources | 30 |
| Passages | 41 (34 marked `volatile`) |
| Topics | 30 (20 known, 10 inferred) |
| Validation / test split | 18 / 12 topics |
| Answerable questions | 90 (canonical plus two paraphrases per topic) |
| Out-of-KB questions | 30 |
| Judged question-passage pairs | 123 |
| Confusable topic pairs | 6 |

Splits are assigned per topic, so a topic's three phrasings always stay on the
same side of the validation/test boundary.

## Sources and access dates

Every source URL was opened and checked against the passages that cite it on
27 September 2026. `sources.csv` records that access date. `source-log.csv`
records, for each source, the page's own "last updated" date where shown, the
passages it supports and the outcome of the check:

- `verified`: the page supports the cited passages as written;
- `url-updated`: the old URL moved, redirected or no longer returned the page,
  and the current official URL now replaces it;
- `passage-corrected`: passage wording was changed to match the current page.

Passages are concise paraphrased summaries, not verbatim copies. Facts about
rates, dates, eligibility or procedures are marked `volatile` and must be
rechecked before use; the collection is a dated snapshot.

## Relevance labels

`judgements.csv` and `qrels.txt` hold Walert-style topic-to-passage seed labels:
each answerable question is linked to the passage or passages written for its
topic (grade 2 fully answers, grade 1 partially relevant). Out-of-KB questions
have no labels. New MPNet candidates are checked separately under S9-05 before
final qrels are generated.

`gold.csv` holds reference answers with sentence support spans for every
answerable question. They are an auditable answer target and do not make the
seed labels independent.

## Check

```bash
mm --data data/v1 validate
mm --data data/v1 quality
```
