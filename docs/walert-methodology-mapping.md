# Relationship to Walert

MelbourneMate uses Walert as a methodological baseline, but it is an independent
implementation for a different domain. Walert studies a conversational agent
for RMIT School of Computing Technologies FAQs and reports retrieval and answer
evaluation using known, inferred and out-of-knowledge-base scenarios. Its
published repository highlights NDCG, the percentage of unanswered
out-of-knowledge-base questions, BERTScore and ROUGE-1.

MelbourneMate keeps the parts of that design that answer the WIL project's core
question: can a RAG system retrieve useful evidence, answer when evidence is
available and avoid answering when it is not? It then extends the protocol to
make a smaller two-person project easier to audit and reproduce.

## Methodology mapping

| Walert evaluation idea | MelbourneMate implementation | Status |
|---|---|---|
| Manually curated domain knowledge | A new English-only collection of dated official information for international students settling in Melbourne | Adapted to a new domain |
| Known questions | Questions answerable from one principal evidence passage | Retained |
| Inferred questions | Questions requiring evidence from two or more passages | Retained |
| Out-of-knowledge-base questions | Thirty questions with no approved answer in the collection | Retained |
| Lexical retrieval baseline | BM25 implemented with `bm25s` | Reimplemented |
| Graded retrieval effectiveness | Graded NDCG@1/3/5, with NDCG@5 pre-specified as primary | Retained and extended |
| Unanswered out-of-KB behaviour | Correct-refusal and unsupported-answer rates, reported together with false refusal on answerable questions | Extended |
| Generated-answer quality | Deterministic citation checks plus blind human scores for correctness, evidence support and fallback appropriateness | Replaced with a more auditable protocol |

## Extensions beyond the baseline

1. **Lexical versus semantic retrieval.** BM25s and a pinned QA-trained MPNet
   bi-encoder are evaluated on the same passages, questions, qrels and top-k.
2. **Frozen topic-level hold-out.** Canonical and paraphrased versions of one
   information need remain in the same split. Configuration choices use
   validation only; the held-out test is not used for tuning.
3. **Independent relevance review.** The held-out BM25s/Dense top-five union is
   reviewed independently, with 25% overlap and agreement reported before final
   qrels are produced.
4. **Uncertainty that respects the data structure.** Confidence intervals and
   paired randomisation operate on topic clusters rather than treating three
   phrasings of one topic as independent observations.
5. **No-context ablation.** The same local generator is evaluated with BM25s
   context, Dense context and no retrieved context. This separates retrieval
   value from the quality of the language model alone.
6. **Traceable evidence.** Formal runs are write-once. Their manifests record
   collection, qrels, configuration and model fingerprints so every reported
   number resolves to a specific experiment.
7. **Blind manual assessment.** Two reviewers score an anonymised answer sample
   for correctness, evidence support and fallback appropriateness, with
   agreement reported.

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

## Claim boundary

The project should be described as a **methodological adaptation and extension
of Walert**, not as an exact reproduction. MelbourneMate's scores must not be
compared numerically with Walert's published scores because the corpus,
questions, relevance labels, models and evaluation protocol differ.

## References

- Walert project page: <https://www.damianospina.com/project/walert/>
- Walert code and evaluation artefacts: <https://github.com/rmit-ir/walert>
- Pathiyan Cherumanal et al. (2024), *Walert: Putting Conversational Information
  Seeking Knowledge into Action by Building and Evaluating a Large Language
  Model-Powered Chatbot*: <https://doi.org/10.1145/3627508.3638309>
