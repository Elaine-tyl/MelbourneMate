# Team handoff

The current base includes the checked `data/v1` collection from the latest
`main` branch.

## Elaine's completed work

- S9-17: compared pinned MiniLM and MPNet on validation and selected MPNet.
- S9-04: saved BM25s and MPNet held-out rankings, metrics and manifests.
- S9-05: completed the 37-pair targeted qrels check and generated final qrels.
- S9-06: rescored the saved rankings and compared BM25s with MPNet.

The model decision is recorded in `docs/dense-encoder-comparison.md`. Saved
runs are under `runs/retrieval/`.

## Sprint 9 evidence links

| Trello task | GitHub issue | Commits | Evidence |
| --- | --- | --- | --- |
| S9-17 | [#51](https://github.com/Elaine-tyl/MelbourneMate/issues/51) | [`398ba97`](https://github.com/Elaine-tyl/MelbourneMate/commit/398ba97) | [Decision](dense-encoder-comparison.md); [MiniLM run](../runs/retrieval/s9-minilm-validation/); [MPNet run](../runs/retrieval/s9-mpnet-validation/) |
| S9-04 | [#38](https://github.com/Elaine-tyl/MelbourneMate/issues/38) | [`ff31c35`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff31c35) | [BM25s run](../runs/retrieval/s9-formal-bm25-test/); [MPNet run](../runs/retrieval/s9-formal-mpnet-test/) |
| S9-05 | [#39](https://github.com/Elaine-tyl/MelbourneMate/issues/39) | [`fe3882d`](https://github.com/Elaine-tyl/MelbourneMate/commit/fe3882d), [`ff229dd`](https://github.com/Elaine-tyl/MelbourneMate/commit/ff229dd), [`46ab029`](https://github.com/Elaine-tyl/MelbourneMate/commit/46ab029), [`79b1a6a`](https://github.com/Elaine-tyl/MelbourneMate/commit/79b1a6a), [`af7021c`](https://github.com/Elaine-tyl/MelbourneMate/commit/af7021c) | [Completed review](../review/targeted-qrels/s9-mpnet-candidates.csv); [final qrels](../review/targeted-qrels/qrels-final.txt) |
| S9-06 | [#40](https://github.com/Elaine-tyl/MelbourneMate/issues/40) | [`bd4b791`](https://github.com/Elaine-tyl/MelbourneMate/commit/bd4b791), [`265114a`](https://github.com/Elaine-tyl/MelbourneMate/commit/265114a) | [BM25s final run](../runs/retrieval/s9-final-bm25-test/); [MPNet final run](../runs/retrieval/s9-final-mpnet-test/); [paired comparison](../runs/retrieval/s9-final-comparison.csv) |

## Targeted qrels review

Use `review/targeted-qrels/s9-mpnet-candidates.csv`.

1. Elaine assigns one grade to each pair: 0, 1 or 2.
2. Elaine writes `Elaine` in `primary_reviewer`.
3. Elaine marks uncertain rows in `needs_second_review` and adds a short note.
4. Siriporn checks only marked rows and records the agreed decision in `note`.
5. Keep one final grade per pair. Do not calculate Cohen's kappa for this check.

After the sheet is complete, create the final test qrels from the saved run:

```bash
mm --data data/v1 targeted-qrels \
  --base-qrels data/v1/qrels.txt \
  --verification review/targeted-qrels/s9-mpnet-candidates.csv \
  --run runs/retrieval/s9-formal-mpnet-test \
  --out review/targeted-qrels/qrels-final.txt
```

The completed file contains 69 positive pairs: 57 seed pairs and 12 verified
additions. Its SHA-256 fingerprint is
`a3ecafcdf28fb918a663677ecb324c1de1b38fdd3f32a27ac4af331116c233f1`.
Use it to rescore the saved BM25s and MPNet rankings. Retrieval must not be
tuned or rerun after this review.

## Final retrieval comparison

The saved rankings were rescored against the final qrels. BM25s achieved
NDCG@5 `0.8353`, Recall@5 `0.8750` and MRR `0.8843`. MPNet achieved NDCG@5
`0.9709`, Recall@5 `1.0000` and MRR `0.9815`.

MPNet minus BM25s on NDCG@5 was `+0.1356`. The topic-clustered 95% bootstrap
interval was `[+0.0486, +0.2519]`, with paired randomisation `p = 0.0065`.
The comparison covers 36 questions from 12 held-out topics and uses 10,000
bootstrap resamples and 10,000 randomisation permutations.

## Checks

```bash
python -m pytest -q
ruff check src tests
mm --data data/v1 validate
mm --data data/v1 quality
```

Use a new run ID for any reproduction. Do not overwrite saved evidence.
