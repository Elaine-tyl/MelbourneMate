# Team handoff and dependency gates

This file separates Elaine's work from Sriporn's work and prevents dependent
results from being committed before their inputs exist in Git.

## Branch 1: foundation and methodology

Owner: Elaine

May be committed now because it does not depend on final data, scoring sheets
or test results.

- repository and package foundation;
- Walert-to-MelbourneMate methodology mapping;
- system architecture and evaluation design;
- contribution boundaries and branch order.

## Branch 2: retrieval and evaluation code

Owner: Elaine

This branch may be committed after Branch 1. It contains Elaine's base
implementation and intentionally excludes Sriporn's later S9-14 fixes. It adds:

- BM25s and MPNet retrieval;
- frozen configuration and model revisions;
- immutable run and manifest code;
- retrieval metrics, statistics and review tooling;
- grounded generation evaluation code.

It must not include Sriporn's dataset, tests, scoring sheets or no-context run.
After this branch is available, Sriporn can apply the S9-14 encoder-name and
BM25 dependency changes in her own reviewed pull request.

## Branch 3: final evidence and interface

Owner: Elaine for integration; shared files retain joint credit.

Do not finalise this branch until Sriporn's pull requests provide:

- the official-source `data/v1` collection and collection documentation;
- the test suite and build-verification record;
- any MPNet qrels candidates she was asked to check and her generation answer
  review sheet;
- the no-context generation run.

After those dependencies are merged, Elaine may add the Walert-style final
qrels, the targeted verification record, rescored rankings, statistical
comparison, error analysis, shared Streamlit work and the final evidence index.

## Verification after integration

Sriporn should rerun these commands after Branch 2 is combined with her files:

```bash
python -m pytest -q
python -m ruff check src tests tools
python -m melbourne_mate.cli --data data/v1 validate
python -m melbourne_mate.cli --data data/v1 quality
```

For the local model and interface:

```bash
ollama list
MM_DATA_DIR=data/v1 MM_ENCODER=multi-qa-mpnet \
  streamlit run src/melbourne_mate/interface/app.py
```

Record the command output in Sriporn's build-verification evidence. If any
result changes, create a new run ID; never overwrite an existing formal run.

## Pull-request order

Merge the three branches in order. Use a merge commit rather than squash so
the Trello-linked commits remain visible. Each pull request requires one review
from the other member.
