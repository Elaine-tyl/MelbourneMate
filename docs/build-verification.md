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
| build | 1.6.1 |
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

## Package build

The package was built from this branch, and the wheel was installed into a
new virtual environment. The installed `mm` command then validated the
collection and reproduced the retrieval comparison.

| Output | Size |
| --- | --- |
| `melbourne_mate-0.1.0-py3-none-any.whl` | 55,444 bytes |
| `melbourne_mate-0.1.0.tar.gz` | 55,978 bytes |

The wheel contains all 33 modules and the `mm` entry point. Run these commands
from the repository root to repeat the check.

```bash
python3 -m venv /tmp/mm-build
/tmp/mm-build/bin/pip install build
/tmp/mm-build/bin/python -m build --outdir /tmp/mm-dist .
python3 -m venv /tmp/mm-wheel
/tmp/mm-wheel/bin/pip install /tmp/mm-dist/melbourne_mate-0.1.0-py3-none-any.whl
/tmp/mm-wheel/bin/mm --data data/v1 validate
/tmp/mm-wheel/bin/mm --data data/v1 compare \
  --left runs/retrieval/s9-final-bm25-test \
  --right runs/retrieval/s9-final-mpnet-test
```

Expected output is `VALID` with fingerprint `1f04bcc7ed2cd814`, then a mean
difference of +0.1356 with p = 0.0065.

The development checks are listed in the [README](../README.md#development-setup).

## Tests added

`tests/test_retrievers.py` adds direct tests for both retrievers.

- BM25s ranks the matching passage first, uses the stemmer, drops passages
  with no shared terms, returns all three matches when `k` is larger than the
  collection and rejects bad input.
- Dense ranks by similarity, breaks equal scores by passage ID, works with the
  offline hashing encoder and rejects a wrong number of passage vectors.

## Cleanup

- No caches, compiled files or system files are tracked in Git.
- Local `__pycache__`, `.pytest_cache` and `.ruff_cache` folders were removed
  and are covered by `.gitignore`.
- `src/melbourne_mate/evaluation/proportions.py` was removed. No code, test
  or document used it. Comparisons and rate intervals use `evaluation/stats.py`.

## Limits

- MPNet and generation were not rerun here. The Dense extra is not installed
  and the local model is too slow on this CPU-only machine. Their saved
  manifests, fingerprints and metrics were checked instead.
- BM25s does not break equal scores by passage ID. In the test split this only
  changes the order of two non-relevant passages at rank 5 for Q18P1, so no
  metric changes.
