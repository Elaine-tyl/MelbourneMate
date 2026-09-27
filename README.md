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

### Walert method and local tooling

MelbourneMate follows Walert's broad experimental sequence:

```text
validate data -> run BM25s -> run Dense -> calculate metrics -> compare runs
```

Walert uses separate scripts. MelbourneMate uses the project-local `mm` command
to automate the same sequence and reduce command errors. This does not make the
two projects' scores directly comparable.

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

`mm` is the short command installed by MelbourneMate. Validate the formal
collection before running an experiment:

```bash
mm --data data/v1 validate
mm --data data/v1 quality
```

The Dense encoder decision used validation only:

```bash
mm --data data/v1 retrieve --arm dense --split validation \
  --encoder all-minilm --run-id s9-minilm-validation
mm --data data/v1 retrieve --arm dense --split validation \
  --encoder multi-qa-mpnet --run-id s9-mpnet-validation
mm --data data/v1 compare \
  --left runs/retrieval/s9-minilm-validation \
  --right runs/retrieval/s9-mpnet-validation --metric ndcg@5
```

After MPNet was frozen, BM25s and MPNet were run once on the held-out test:

```bash
mm --data data/v1 evaluate-retrieval --split test \
  --encoder multi-qa-mpnet --run-prefix recheck-yourname
```

The submitted rankings are in `runs/retrieval/s9-formal-bm25-test/` and
`runs/retrieval/s9-formal-mpnet-test/`. They use the current Walert-style qrels.
After the small targeted qrels check, rescore these rankings without tuning or
rerunning either retriever. Each run ID is write-once, so a reproduction needs
a new prefix.

Run the checks before handing work to another team member:

```bash
python -m pytest -q
ruff check src tests
```

Team responsibilities and the reviewed branch order are recorded in
[`docs/HANDOFF.md`](docs/HANDOFF.md).
