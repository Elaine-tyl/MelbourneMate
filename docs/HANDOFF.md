# Team handoff

This records the Sprint 9 evaluation work, its evidence and the frozen results.
Commands to repeat each step are in the [README](../README.md#core-commands).

## Sprint 9 evidence

| Trello task | Work | GitHub issue | Commits | Evidence |
| --- | --- | --- | --- | --- |
| S9-17 | Compared MiniLM and MPNet on validation and selected MPNet | [#51](https://github.com/Elaine-tyl/MelbourneMate/issues/51) | [`398ba97`](https://github.com/Elaine-tyl/MelbourneMate/commit/398ba97) | [Decision](dense-encoder-comparison.md); [MiniLM run](../runs/retrieval/s9-minilm-validation/); [MPNet run](../runs/retrieval/s9-mpnet-validation/) |
| S9-04 | Saved BM25s and MPNet held-out rankings | [#38](https://github.com/Elaine-tyl/MelbourneMate/issues/38) | [`ff31c35`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff31c35) | [BM25s run](../runs/retrieval/s9-formal-bm25-test/); [MPNet run](../runs/retrieval/s9-formal-mpnet-test/) |
| S9-05 | Checked 37 new MPNet candidates and built the final qrels | [#39](https://github.com/Elaine-tyl/MelbourneMate/issues/39) | [`fe3882d`](https://github.com/Elaine-tyl/MelbourneMate/commit/fe3882d), [`ff229dd`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff229dd), [`46ab029`](https://github.com/Elaine-tyl/MelbourneMate/commit/46ab029), [`79b1a6a`](https://github.com/Elaine-tyl/MelbourneMate/commit/79b1a6a), [`af7021c`](https://github.com/Elaine-tyl/MelbourneMate/commit/af7021c) | [Completed review](../review/targeted-qrels/s9-mpnet-candidates.csv); [final qrels](../review/targeted-qrels/qrels-final.txt) |
| S9-06 | Rescored the saved rankings and compared BM25s with MPNet | [#40](https://github.com/Elaine-tyl/MelbourneMate/issues/40) | [`bd4b791`](https://github.com/Elaine-tyl/MelbourneMate/commit/bd4b791), [`265114a`](https://github.com/Elaine-tyl/MelbourneMate/commit/265114a) | [BM25s final run](../runs/retrieval/s9-final-bm25-test/); [MPNet final run](../runs/retrieval/s9-final-mpnet-test/); [paired comparison](../runs/retrieval/s9-final-comparison.csv) |
| S9-07 | Ran BM25s, MPNet and no-context generation | [#41](https://github.com/Elaine-tyl/MelbourneMate/issues/41) | [`a9c0840`](https://github.com/Elaine-tyl/MelbourneMate/commit/a9c0840), [`d9a0aba`](https://github.com/Elaine-tyl/MelbourneMate/commit/d9a0aba), [`562404a`](https://github.com/Elaine-tyl/MelbourneMate/commit/562404a) | [BM25s generation](../runs/generation/s9-qwen25-20260927-bm25/); [MPNet generation](../runs/generation/s9-qwen25-20260927-mpnet/); [no-context generation](../runs/generation/s9-qwen25-20260927-no-context/) |

## Targeted qrels

Elaine gave each of the 37 candidate pairs one grade and flagged uncertain
rows. Siriporn checked only the flagged rows. No kappa is reported for this
check. The final file has 69 positive pairs, 57 seed pairs and 12 verified
additions, with SHA-256 fingerprint
`a3ecafcdf28fb918a663677ecb324c1de1b38fdd3f32a27ac4af331116c233f1`. It was
built with this command.

```bash
mm --data data/v1 targeted-qrels \
  --base-qrels data/v1/qrels.txt \
  --verification review/targeted-qrels/s9-mpnet-candidates.csv \
  --run runs/retrieval/s9-formal-mpnet-test \
  --out review/targeted-qrels/qrels-final.txt
```

Retrieval was not tuned or rerun after this review.

## Final retrieval comparison

| Retriever | NDCG@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25s | 0.8353 | 0.8750 | 0.8843 |
| MPNet | 0.9709 | 1.0000 | 0.9815 |

MPNet minus BM25s on NDCG@5 was +0.1356, with a topic-clustered 95% bootstrap
interval of [+0.0486, +0.2519] and paired randomisation p = 0.0065. The
comparison covers 36 questions from 12 held-out topics, with 10,000 resamples
and 10,000 permutations.

## Generation evaluation

All three arms used `qwen2.5:7b-instruct` through local Ollama, model digest
`845dbda0ea48ed749ca`, temperature 0.0, seed 20260923 and the same 66
questions, 36 answerable and 30 OOKB.

| Arm | Correct refusal | Unsupported answer |
| --- | ---: | ---: |
| BM25s | 96.7% | 3.3% |
| MPNet | 86.7% | 13.3% |
| No-context | 0.0% | 100.0% |

Both reviewers then scored the same 30 saved answers blind. The agreement and
agreed scores are in [`review/generation-s9/report.md`](../review/generation-s9/report.md).
The error analysis is in
[`runs/analysis/s9-error-analysis/report.md`](../runs/analysis/s9-error-analysis/report.md).

Use a new run ID for any reproduction. Do not overwrite saved evidence.
