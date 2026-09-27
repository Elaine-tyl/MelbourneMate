# Team handoff

The current base includes the checked `data/v1` collection from the latest
`main` branch.

## Elaine's completed work

- S9-17: compared pinned MiniLM and MPNet on validation and selected MPNet.
- S9-04: saved BM25s and MPNet held-out rankings, metrics and manifests.
- S9-05: generated the 37-pair targeted qrels review sheet.

The model decision is recorded in `docs/dense-encoder-comparison.md`. Saved
runs are under `runs/retrieval/`.

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

Return the completed CSV to Elaine. Elaine will check the row count and qrels
fingerprint, then rescore the saved BM25s and MPNet rankings. Retrieval must not
be tuned or rerun after this review.

## Checks

```bash
python -m pytest -q
ruff check src tests
mm --data data/v1 validate
mm --data data/v1 quality
```

Use a new run ID for any reproduction. Do not overwrite saved evidence.
