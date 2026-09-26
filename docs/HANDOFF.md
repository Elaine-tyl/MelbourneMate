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

## Branch 2: RAG pipeline core

Owner: Elaine

Branch: `elaine/s9-rag-pipeline`

This branch may be committed after Branch 1. It contains Elaine's base
implementation and intentionally excludes Sriporn's later S9-14 fixes. It adds:

- BM25s and MPNet retrieval;
- frozen configuration and model revisions;
- immutable run and manifest code;
- retrieval metrics, topic-clustered statistics and targeted qrels tooling;
- grounded generation evaluation code.

The qrels workflow follows Walert's topic-to-passage ground truth. It adds only
a small targeted check for candidates unique to the frozen MPNet top-five
pool. Full retrieval double review, Cohen's kappa and complex adjudication are
outside this branch.

It must not include Sriporn's dataset, tests, scoring sheets or no-context run.
After this branch is available, Sriporn can apply the S9-14 encoder-name and
BM25 dependency changes in her own reviewed pull request.

## Sriporn's work package

Complete these items on a separate branch:

1. Add the checked official-source collection under `data/v1/` and update its
   source dates.
2. Add the full test suite and record the commands and results in
   `docs/build-verification.md`.
3. Check the flagged MPNet question-passage pairs and return the completed
   verification CSV.
4. Complete the assigned generated-answer review sheet.
5. Apply the S9-14 encoder-name and BM25 dependency fixes.
6. Run the no-context arm on the agreed frozen question list. Do not edit the
   BM25 or MPNet run folders.

Put the frozen question IDs in `data/v1/generation-sample.txt`, one ID per
line, then run:

```bash
python -m melbourne_mate.cli --data data/v1 generate \
  --arm none \
  --split test \
  --run-id s9-no-context-test \
  --sample generation-sample.txt \
  --model qwen2.5:7b-instruct \
  --runs-dir runs/generation
```

Return these items to Elaine:

- `data/v1/` and its source log;
- the test files and `docs/build-verification.md`;
- the checked MPNet pairs and generated-answer review CSV;
- the new no-context run folder;
- the S9-14 code changes.

Elaine will then integrate the final qrels, rerun the saved rankings, compare
the systems and prepare the final evidence.

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
