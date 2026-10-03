# Team handoff

This records the Sprint 9 evaluation work, its evidence and the frozen results.
Commands to repeat each step are in the [README](../README.md#core-commands).

## Sprint 9 evidence

| Trello task | Work | GitHub issue | Commits | Evidence |
| --- | --- | --- | --- | --- |
| S9-17 | Compared MiniLM and MPNet on validation and selected MPNet | [#51](https://github.com/Elaine-tyl/MelbourneMate/issues/51) | [`398ba97`](https://github.com/Elaine-tyl/MelbourneMate/commit/398ba97) | [Decision](dense-encoder-comparison.md); [MiniLM run](../runs/retrieval/s9-minilm-validation/); [MPNet run](../runs/retrieval/s9-mpnet-validation/) |
| S9-04 | Saved BM25s and MPNet held-out rankings | [#38](https://github.com/Elaine-tyl/MelbourneMate/issues/38) | [`ff31c35`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff31c35) | [BM25s run](../runs/retrieval/s9-formal-bm25-test/); [MPNet run](../runs/retrieval/s9-formal-mpnet-test/) |
| S9-05 | Checked new MPNet candidates and built the final qrels | [#39](https://github.com/Elaine-tyl/MelbourneMate/issues/39) | [`fe3882d`](https://github.com/Elaine-tyl/MelbourneMate/commit/fe3882d), [`ff229dd`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff229dd), [`46ab029`](https://github.com/Elaine-tyl/MelbourneMate/commit/46ab029), [`79b1a6a`](https://github.com/Elaine-tyl/MelbourneMate/commit/79b1a6a), [`af7021c`](https://github.com/Elaine-tyl/MelbourneMate/commit/af7021c) | [Completed review](../review/targeted-qrels/s9-mpnet-candidates.csv); [final qrels](../review/targeted-qrels/qrels-final.txt) |
| S9-06 | Rescored the saved rankings and compared BM25s with MPNet | [#40](https://github.com/Elaine-tyl/MelbourneMate/issues/40) | [`bd4b791`](https://github.com/Elaine-tyl/MelbourneMate/commit/bd4b791), [`265114a`](https://github.com/Elaine-tyl/MelbourneMate/commit/265114a) | [BM25s final run](../runs/retrieval/s9-final-bm25-test/); [MPNet final run](../runs/retrieval/s9-final-mpnet-test/); [paired comparison](../runs/retrieval/s9-final-comparison.csv) |
| S9-07 | Ran BM25s, MPNet and no-context generation | [#41](https://github.com/Elaine-tyl/MelbourneMate/issues/41) | [`a9c0840`](https://github.com/Elaine-tyl/MelbourneMate/commit/a9c0840), [`d9a0aba`](https://github.com/Elaine-tyl/MelbourneMate/commit/d9a0aba), [`562404a`](https://github.com/Elaine-tyl/MelbourneMate/commit/562404a) | [BM25s generation](../runs/generation/s9-qwen25-20260927-bm25/); [MPNet generation](../runs/generation/s9-qwen25-20260927-mpnet/); [no-context generation](../runs/generation/s9-qwen25-20260927-no-context/) |

## Targeted qrels

Elaine graded the 37 original candidate pairs and flagged uncertain rows.
Siriporn checked only the flagged rows, so no agreement statistic is claimed
for this check. During the final audit, a chatbot test showed that one official
Study Melbourne passage about unpaid wages had been missed for two questions.
Those two pairs were added as **partial support** because the passage explains
where to get help but does not cover every part of either question.

The final sheet therefore contains 39 checked pairs. The final qrels contain
71 positive pairs: 57 seed pairs and 14 verified additions. Their SHA-256
fingerprint is
`fbb2b3e9ab3cb8b1882942d241d4e8770bad85006eb344e14964850b795aea69`.
They were built with this command.

```bash
mm --data data/v1 targeted-qrels \
  --base-qrels data/v1/qrels.txt \
  --verification review/targeted-qrels/s9-mpnet-candidates.csv \
  --run runs/retrieval/s9-formal-mpnet-test \
  --out review/targeted-qrels/qrels-final.txt
```

Retrieval was not tuned or rerun after this review. The saved rankings stayed
fixed; only their scores were recalculated with the corrected labels. Saved
model answers also stayed fixed; only their citation labels and summaries were
rechecked.

## Final retrieval comparison

| Retriever | NDCG@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25s | 0.8509 | 0.8843 | 0.9120 |
| MPNet | 0.9710 | 1.0000 | 0.9815 |

MPNet minus BM25s on NDCG@5 was +0.1200, with a topic-clustered 95% bootstrap
interval of [+0.0441, +0.2316] and paired randomisation p = 0.0065. The
comparison covers 36 questions from 12 held-out topics, with 10,000 resamples
and 10,000 permutations.

## Generation evaluation

All three arms used `qwen2.5:7b-instruct` through local Ollama, model digest
`845dbda0ea48ed749ca`, temperature 0.0, seed 20260923 and the same 66
questions, 36 answerable and 30 OOKB.

| Arm | Correct refusal | Unsupported answer | False refusal | Citation validity |
| --- | ---: | ---: | ---: | ---: |
| BM25s | 96.7% | 3.3% | 5.7% | 88.2% |
| MPNet | 86.7% | 13.3% | 8.3% | 84.8% |
| No-context | 0.0% | 100.0% | N/A | N/A |

The correction did not change any refusal or unsupported-answer decision. It
changed whether two existing citations counted as supported and slightly
changed the BM25s false-refusal denominator. The main finding is unchanged:
MPNet ranked passages better, while BM25s refused unsupported questions more
reliably in this experiment.

Both reviewers then scored the same 30 saved answers blind. The agreement and
agreed scores are in [`review/generation-s9/report.md`](../review/generation-s9/report.md).
The error analysis is in
[`runs/analysis/s9-error-analysis/report.md`](../runs/analysis/s9-error-analysis/report.md).

Use a new run ID for any reproduction. Do not overwrite saved evidence.
