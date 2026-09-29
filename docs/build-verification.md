# Build verification

Trello S10-08, GitHub #45. Checked by Siriporn on 29 September 2026 against
`main` at `ffb819c`, after the Sprint 9 pull requests were merged.

## Environment

| Item | Version |
| --- | --- |
| Machine | macOS 15.8, Intel x86_64, 8 GB RAM, CPU only |
| Python | 3.14.0 |
| bm25s | 0.3.11 |
| PyStemmer | 3.1.0 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| streamlit | 1.64.0 |
| pytest | 9.1.1 |
| ruff | 0.16.9 |
| Ollama model | `qwen2.5:7b-instruct`, ID `845dbda0ea48` |

The formal runs were produced on a different machine with Python 3.12. The
Ollama model ID matches the digest recorded in the generation run manifests.

## Checks

| Check | Command | Result |
| --- | --- | --- |
| Tests | `python -m pytest -q` | 63 passed |
| Lint | `ruff check src tests` | passed |
| Collection | `mm --data data/v1 validate` | VALID, fingerprint `1f04bcc7ed2cd814` |
| Collection quality | `mm --data data/v1 quality` | 31 sources, 41 passages, 30 topics, 120 questions |
| Final qrels | `mm --data data/v1 targeted-qrels ...` | identical to `review/targeted-qrels/qrels-final.txt` |
| BM25s test run | `mm retrieve` then `mm rescore-retrieval` | metrics identical to `s9-final-bm25-test` |
| Retrieval comparison | `mm compare` on the two final runs | +0.1356, 95% CI [+0.0486, +0.2519], p = 0.0065 |
| Local model | `ollama list` | `qwen2.5:7b-instruct` present |

The rerun outputs were written to a scratch folder, so no saved run was
replaced.

## Tests added

`tests/test_retrievers.py` adds direct tests for both retrievers.

- BM25s ranks the matching passage first, uses the stemmer, drops passages
  with no shared terms, limits `k` and rejects bad input.
- Dense ranks by similarity, breaks equal scores by passage ID, works with the
  offline hashing encoder and rejects a wrong number of passage vectors.

## Cleanup

- No caches, compiled files or system files are tracked in Git.
- Local `__pycache__`, `.pytest_cache` and `.ruff_cache` folders were removed
  and are covered by `.gitignore`.
- `src/melbourne_mate/evaluation/proportions.py` is not imported or tested. It
  was left in place for the team to decide whether rate intervals are needed
  in the report.

## Limits

- MPNet and generation were not rerun here. The Dense extra is not installed
  and the local model is too slow on this CPU-only machine. Their saved
  manifests, fingerprints and metrics were checked instead.
- BM25s does not break equal scores by passage ID. In the test split this only
  changes the order of two non-relevant passages at rank 5 for Q18P1, so no
  metric changes.
