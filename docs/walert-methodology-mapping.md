# Relationship to Walert

MelbourneMate is a methodological adaptation of Walert, not an exact
reproduction. Walert evaluates a conversational agent for RMIT computing FAQs.
MelbourneMate applies the same broad evaluation structure to **international
students adjusting to life in Melbourne**, including transport, health and
OSHC, renting and bonds, visa conditions, employment and arrival tasks.

The collection is independent: its official URLs, passages, questions,
judgements and reference answers were newly authored for MelbourneMate.

## Methodology mapping

| Walert evaluation idea | MelbourneMate implementation | Status |
|---|---|---|
| Manually curated domain knowledge | Dated official information for international students adjusting to life in Melbourne | Adapted to a new domain |
| Known questions | Questions answerable from one principal evidence passage | Retained |
| Inferred questions | Questions requiring evidence from two or more passages | Retained |
| Out-of-knowledge-base questions | Thirty questions with no approved answer in the collection | Retained |
| Lexical retrieval baseline | BM25 implemented with `bm25s` | Reimplemented |
| Graded retrieval effectiveness | Graded NDCG@1/3/5, with NDCG@5 pre-specified as primary | Retained and extended |
| Unanswered out-of-KB behaviour | Correct-refusal and unsupported-answer rates, reported together with false refusal on answerable questions | Extended |
| Generated-answer quality | Deterministic citation checks plus blind human scores for correctness, evidence support and fallback appropriateness | Replaced with a more auditable protocol |

## Extensions beyond the baseline

- BM25s and MPNet are tested on the same passages, questions, qrels and top-k.
- Topic-level splitting keeps canonical and paraphrased questions together and
  prevents test leakage.
- Topic-clustered confidence intervals and paired testing report uncertainty.
- BM25s, MPNet and no-context generation use one frozen question sample.
- Saved manifests and fingerprints make reported results traceable.
- Two reviewers independently score correctness, evidence support and fallback
  appropriateness.

## Generation safety metrics

- **Correct-refusal rate:** among OOKB questions, the proportion correctly
  refused. Higher is better.
- **Unsupported-answer rate:** among OOKB questions, the proportion answered
  without approved evidence. Lower is better.
- **False-refusal rate:** among answerable questions where relevant evidence
  was retrieved, the proportion incorrectly refused. Lower is better.

These metrics must be read together. A system that refuses every question can
look safe on OOKB questions while failing to answer supported questions.

## Deliberate differences

- MelbourneMate does not reproduce Walert's full software stack, intent-based
  agent, Alexa deployment or original RMIT FAQ collection.
- BM25s replaces Walert's original retrieval tooling, and the Dense arm is a
  QA-trained MPNet bi-encoder rather than an exact reproduction of Walert's DPR
  implementation.
- Generation uses a pinned local Qwen model through Ollama. This avoids API
  quota and keeps study questions on the local machine.
- ROUGE and BERTScore are not treated as primary evidence. For this safety- and
  source-sensitive domain, refusal behaviour, citation checks and blind human
  judgement answer the project questions more directly.

## References

- Walert project page: <https://www.damianospina.com/project/walert/>
- Walert code and evaluation artefacts: <https://github.com/rmit-ir/walert>
- Pathiyan Cherumanal et al. (2024), *Walert: Putting Conversational Information
  Seeking Knowledge into Action by Building and Evaluating a Large Language
  Model-Powered Chatbot*: <https://doi.org/10.1145/3627508.3638309>
