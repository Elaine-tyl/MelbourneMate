# MelbourneMate

MelbourneMate is an independent Test-Driven RAG project for international
students who need current official information about settling in Melbourne.
The project compares a BM25s lexical baseline with a QA-trained MPNet dense
retriever and evaluates whether retrieved evidence improves answer quality and
safe refusal behaviour.

## From Walert to MelbourneMate

Walert is the methodological baseline for MelbourneMate. The project adapts
Walert's known, inferred and out-of-knowledge-base question scenarios, graded
retrieval evaluation and explicit measurement of unanswered questions.

MelbourneMate extends that baseline with:

- a topic-level validation and held-out test split;
- BM25s versus pinned MPNet retrieval;
- Walert-style qrels with targeted checks for new MPNet candidates;
- topic-clustered confidence intervals and paired testing;
- grounded and no-context generation arms;
- deterministic citation checks and blind generated-answer review;
- immutable run manifests and data/configuration fingerprints;
- a local Ollama generator and Streamlit interface.

The projects use different domains, collections, questions, models and
protocols, so their numerical scores are not directly comparable. See
[`docs/walert-methodology-mapping.md`](docs/walert-methodology-mapping.md).

### Walert method versus MelbourneMate tooling

Walert does not use the `mm` command. Its published evaluation runs separate
Python programs for data preparation, retrieval and evaluation, plus a shell
command for BM25 indexing. MelbourneMate follows the same broad experimental
sequence but exposes it through one project-local command:

```text
validate data -> run BM25s -> run Dense -> calculate metrics -> compare runs
```

The `mm` command changes only how the steps are started. It does not introduce
a different retrieval method or make the results directly comparable with
Walert. The automation reduces manual command errors and gives both team
members one reproducible entry point.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dense,app,dev]"
```

Do not commit virtual environments, model weights, `.env` files, caches or
personal participant data.

## Core commands

`mm` is the short command installed by MelbourneMate. It runs
`melbourne_mate.cli:main`; it is not a Walert command or an external tool.

The same commands work with the sample fixture now and with `data/v1` after the
collection PR is merged:

```bash
mm --data data/sample validate
mm --data data/sample quality
mm --data data/sample retrieve --arm bm25 --split validation \
  --run-id bm25-validation
mm --data data/sample retrieve --arm dense --encoder multi-qa-mpnet \
  --split validation --run-id mpnet-validation
mm --data data/sample compare \
  --left runs/retrieval/bm25-validation \
  --right runs/retrieval/mpnet-validation
```

Run the complete validation retrieval workflow with one command:

```bash
mm --data data/v1 evaluate-retrieval \
  --split validation --run-prefix s9-validation
```

Use `--split test` only for the frozen held-out run. Every run ID is
write-once, so use a new prefix instead of replacing evidence.

Team responsibilities and the reviewed branch order are recorded in
[`docs/HANDOFF.md`](docs/HANDOFF.md).
