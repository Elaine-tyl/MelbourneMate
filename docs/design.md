# MelbourneMate — evaluation design

**Protocol:** frozen current configuration · **Frozen:** 23 September 2026 · **Team:** Elaine, Sriporn

## 1. Problem and stakeholders

International students arriving in Melbourne need answers that are correct and
current on subjects where a wrong answer has real consequences: work-hour
conditions on a student visa, health cover, transport, and renting. The
official answers exist, but they are spread across separate government and
university sites and written in institutional English — "subclass 500",
"OSHC", "bond lodgement". A student who does not already know those words has
to guess the search term before they can find the page.

Stakeholders: new international students (primary); university international
student support teams, who answer the same questions repeatedly; and the public
bodies whose pages are the authoritative source.

## 2. What the project tests

The system is a small RAG pipeline over a hand-built knowledge base of official
passages. The claim under test is not "the chatbot works" — it is that
**retrieval design decides whether a student can reach official information
when they do not use official wording**.

That claim is made measurable by a property of the collection itself. For every
question we compute *containment*: the share of its content words that already
appear in the passage that answers it. Canonical questions are expected to have
higher containment than student-style paraphrases. Binning questions by
containment turns the claim into a dose-response curve — as wording moves away
from the source, does the lexical baseline fall away faster than the dense
retriever? — rather than a single two-cell comparison.

### Research questions and pre-specified hypotheses

| | Question | Hypothesis | Primary evidence |
|---|---|---|---|
| **RQ1** | Does dense retrieval beat the lexical baseline overall? | Dense ≥ BM25 on NDCG@5 | Paired comparison, topic-clustered CI |
| **RQ2** | Where does the gap come from? | The BM25-to-dense gap widens monotonically as question-term containment falls | NDCG@5 by `containment_bin` (low / medium / high) and by `question_form` |
| **RQ3** | Does the retrieval layer add value over the LLM alone? | Grounded answers have higher manually judged correctness and a lower unsupported-answer rate than no-context answers | Blind review + no-context ablation on the frozen generation sample |
| **RQ4** | Does the system fail safely? | Correct-refusal rate is high on out-of-knowledge-base questions without a high false-refusal rate on answerable ones | Refusal rates + retrieval × answer 2×2 |

These questions and hypotheses are fixed before the held-out test split is scored.

### Methodological lineage

Walert is the methodological baseline for the test design, not a source-code or
dataset dependency. The mapping, extensions and deliberate differences are in
[`walert-methodology-mapping.md`](walert-methodology-mapping.md).

## 3. System variants

| Arm | Retrieval | Generation | Purpose |
|---|---|---|---|
| `bm25` | BM25 (bm25s), k1=1.5, b=0.75 | Ollama, local | Lexical baseline |
| `dense` | pinned multi-qa-mpnet-base-cos-v1, normalised-vector inner product | Ollama, local | Semantic retrieval for question-to-passage matching |
| `none` | — | same model, **closed-book prompt** | No-context ablation for RQ3 |

One generator across all arms. The project is not a model competition: holding
generation fixed is what makes the retrieval differences attributable.

## 4. Data design

* Retrieval is scored on every answerable question in the split.
* Generation is scored on a **sample frozen before any arm runs**: all 36
  held-out test questions and all 30 out-of-knowledge-base questions
  (`data/v1/generation-sample.csv`). Every generation arm sees the identical
  question ids.
* Splits are assigned **per topic**. A topic's canonical and paraphrased
  questions never straddle the validation/test boundary, and neither does a
  confusable pair — a near-miss split across the boundary can never be
  observed.
* Parameters are chosen on validation only. The test split is scored **once**,
  after `config.py` is frozen and tagged.
* Following Walert, formal qrels are generated from curated topic-passage
  ground truth. The targeted check keeps missing MPNet top-five candidates only
  when the passage is in the same category or a listed confusable topic pair.
  Each candidate has one primary grade; a teammate checks only flagged cases.
  No agreement statistic is claimed for retrieval qrels.

## 5. Metrics

**Retrieval** (graded qrels, 1 = partially relevant, 2 = fully answers):
NDCG@1/3/5 (primary: NDCG@5), Recall@5, MRR, retrieval-failure rate. A question
with no retrieved passages stays in the denominator. NDCG uses **exponential
gain** (2^g − 1), frozen in config and stamped into every run manifest; linear-gain
numbers from elsewhere are not comparable and `mm compare` refuses to mix them.

**Generation**, deterministic and judge-free:

| Metric | Denominator | Reads as |
|---|---|---|
| Correct-refusal rate | OOKB questions | knows when not to answer |
| False-refusal rate | answerable questions **whose context held a relevant passage** | not over-cautious |
| Unsupported-answer rate | OOKB questions | answered something it had no evidence for |
| High-risk unsupported rate | OOKB questions on visa/health/employment/emergency | the failure that matters most |
| Citation validity rate | answered answerable questions | citations resolve to retrieved, relevant passages, no stray URLs, and every figure appears in a cited passage. **Not** a faithfulness measure |
| Truncation rate | all generated answers | responses stopped by the token limit; excluded from every rate above |

Manual validation of a stratified subset (20–30 answers), annotated
independently by both members with agreement reported, is the qualitative layer
on top. Manual judgement is authoritative wherever it disagrees with anything
automated.

## 6. Analysis plan

Written before any result is seen, so that no test is chosen after the numbers
are known. One primary claim per research question; everything else is
diagnostic and is labelled as such in the report.

| RQ | Claim tested | Procedure | Evidence | Reported as positive when |
|---|---|---|---|---|
| RQ1 | Dense ≥ BM25 overall | paired comparison on NDCG@5, topic-clustered bootstrap + randomisation | `mm evaluate-retrieval` and `mm compare` | the 95% CI excludes zero **and** p < 0.05 |
| RQ2 | The gap widens as containment falls | NDCG@5 by `containment_bin`, and the pairwise difference within the `low` bin | retrieval run CSV grouped by containment and question form | the low-bin difference is larger than the high-bin difference, with the low-bin CI excluding zero |
| RQ2 diagnostic | Neighbouring topics are distinguished | displacement rate over six confusable pairs | `pairs.csv` and ranked-output error review | displacement below 10% of pair questions |
| RQ3a | Retrieval reduces unsupported answers | paired answered-rate comparison on OOKB questions | generation runs and deterministic metrics | the grounded arm is lower, the CI excludes zero and at least 5 questions are discordant |
| RQ3b | Grounded answers are actually correct | blind manual validation, both members, κ reported | blind-review CSV and agreement calculation | correctness is higher for a grounded arm, with κ ≥ 0.6 |
| RQ4 | The system fails safely | correct-refusal and false-refusal rates, each with a cluster bootstrap CI | deterministic refusal table | correct refusal high while false refusal stays low — neither read alone |

Citation validity is compared only between grounded arms (BM25 and Dense).
The no-context prompt has no citation requirement, so treating its missing
citations as failures would hand the retrieval arms a win by construction.

The small student study is impact evidence rather than a research question. It
uses four counterbalanced tasks whose answers are in the frozen collection, and
its completion, time, confidence and trust are reported descriptively
([`study/README.md`](../study/README.md)).

**Multiplicity.** NDCG@5 is the only primary retrieval metric; NDCG@1/@3,
Recall@5 and MRR are diagnostic and no claim rests on them alone. The slices
(question form, containment bin, category, confusable pair) are exploratory:
a difference found only in a slice is reported as a hypothesis for future work,
not as a result.

**Thin comparisons.** The generation comparison reports how many questions the
two arms actually disagree on. Under five discordant questions, the report
states the evidence is insufficient and quotes no p-value as if it settled
anything.

**Truncated answers** are excluded from every rate and reported separately. A
truncation rate above 10% invalidates the generation comparison for that arm
and the run is repeated with a larger token budget, recorded as a re-run.

**Manual judgement wins.** Where manual validation and any automated check
disagree, the manual verdict is authoritative and the disagreement is reported
rather than reconciled silently.

### Failure taxonomy

Every error examined in the analysis is labelled with one of these, so that
"we looked at the failures" becomes a table rather than an anecdote:

| Code | Layer | Meaning |
|---|---|---|
| `R-MISS` | retrieval | no relevant passage in the top k |
| `R-NEIGHBOUR` | retrieval | a confusable topic's passage displaced the right one |
| `R-PARTIAL` | retrieval | only a grade-1 passage retrieved for an inferred topic |
| `G-UNGROUNDED` | generation | claim not supported by the cited passage |
| `G-NUMBER` | generation | a figure not present in any cited passage |
| `G-OVERREACH` | generation | answered an out-of-KB question |
| `G-TIMID` | generation | refused although the evidence was present |
| `G-FORMAT` | generation | citation missing or malformed |
| `G-CUT` | generation | truncated by the token limit |

## 7. Statistics

Three questions share a topic, so their scores are not independent. Both
procedures resample **topics**, not questions: a 95 % percentile bootstrap over
topic clusters for the interval, and a paired randomisation test that flips the
sign of whole clusters for the p-value (add-one corrected, so it is never
exactly zero). Seed and resample count are frozen in `config.py` and recorded
in every run manifest.

## 8. Ethics and responsible use

1. **Traceability.** Every passage stores its source URL and access date, so an
   answer can be traced to an official page and a stale one can be found.
2. **Refusal over guessing.** The evidence gate runs before generation, and
   refusal behaviour on visa, health and employment questions is *measured*,
   not assumed.
3. **No personal data leaves the machine.** The generator runs locally through
   Ollama, so questions are never sent to a third party, and study responses
   stay on the session computer. The report still states what a real deployment
   would need beyond a student prototype.
4. **Prompt injection.** Collected page text is delimited and neutralised
   before it enters a prompt; a URL that no citation supports is flagged.

## 9. What this design does not claim

The automated generation checks establish citation validity, not truth. Manual
validation is the only evidence of correctness, and the report keeps the two
separate.

The project makes no production-readiness claim and no demographic fairness
claim. The containment slice is about **wording**, not about people. No claim
goes beyond the sample sizes actually used, and generation-side intervals are
wider than retrieval-side ones.
