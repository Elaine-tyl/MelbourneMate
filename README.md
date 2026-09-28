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
`runs/retrieval/s9-formal-mpnet-test/`. After the targeted qrels check, the
same rankings were rescored without tuning or rerunning either retriever:

```bash
mm --data data/v1 rescore-retrieval \
  --source-run runs/retrieval/s9-formal-bm25-test \
  --qrels review/targeted-qrels/qrels-final.txt \
  --run-id recheck-yourname-bm25-test
mm --data data/v1 rescore-retrieval \
  --source-run runs/retrieval/s9-formal-mpnet-test \
  --qrels review/targeted-qrels/qrels-final.txt \
  --run-id recheck-yourname-mpnet-test
mm --data data/v1 compare \
  --left runs/retrieval/recheck-yourname-bm25-test \
  --right runs/retrieval/recheck-yourname-mpnet-test \
  --out runs/retrieval/recheck-yourname-comparison.csv
```

The final submitted results are in `runs/retrieval/s9-final-bm25-test/`,
`runs/retrieval/s9-final-mpnet-test/` and
`runs/retrieval/s9-final-comparison.csv`. Each run ID is write-once.

## Generation evaluation

Install the local model, then run all three methods with one command:

```bash
ollama pull qwen2.5:7b-instruct
mm --data data/v1 validate
mm --data data/v1 evaluate-generation \
  --sample generation-sample.csv \
  --qrels review/targeted-qrels/qrels-final.txt \
  --run-prefix recheck-yourname \
  --model qwen2.5:7b-instruct
```

This creates BM25s, MPNet, and no-context runs with the same 66 questions and
settings. Use a new prefix when repeating the test.

## Error analysis

Build the final error report from saved runs. This does not rerun retrieval or
Ollama.

```bash
mm --data data/v1 analyse-errors \
  --qrels review/targeted-qrels/qrels-final.txt \
  --bm25-retrieval runs/retrieval/s9-final-bm25-test \
  --mpnet-retrieval runs/retrieval/s9-final-mpnet-test \
  --bm25-generation runs/generation/s9-qwen25-20260927-bm25 \
  --mpnet-generation runs/generation/s9-qwen25-20260927-mpnet \
  --no-context-generation runs/generation/s9-qwen25-20260927-no-context \
  --out runs/analysis/recheck-yourname
```

The report uses the shared high-risk categories in
`src/melbourne_mate/evaluation/risk.py`.

## Streamlit demo

Start Ollama, then launch the final MPNet chatbot:

```bash
ollama serve
streamlit run src/melbourne_mate/interface/app.py
```

The four checked tasks and one citation warning are recorded in
`review/chatbot-test-s9/results.csv`. The app uses the same pipeline, final
qrels, frozen encoder and local model as the evaluation runs.

Create the 37-pair review sheet from the saved MPNet top-five rankings:

```bash
mm --data data/v1 qrels-pool \
  --run runs/retrieval/s9-formal-mpnet-test \
  --out review/targeted-qrels/recheck-yourname.csv
```

The submitted sheet is `review/targeted-qrels/s9-mpnet-candidates.csv`. It keeps
only candidates absent from the current qrels whose passage is in the same
category as the question or in a listed confusable topic pair.

Run the checks before handing work to another team member:

```bash
python -m pytest -q
ruff check src tests
```

Team responsibilities and the reviewed branch order are recorded in
[`docs/HANDOFF.md`](docs/HANDOFF.md).
