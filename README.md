# MelbourneMate

MelbourneMate is an independent Test-Driven RAG project for international
students who need current official information about settling in Melbourne.
The project compares a BM25s lexical baseline with a QA-trained MPNet dense
retriever and evaluates whether retrieved evidence improves answer quality and
safe refusal behaviour.

## From Walert to MelbourneMate

Walert is the methodological baseline, not a source-code or dataset dependency.
MelbourneMate independently adapts Walert's known, inferred and
out-of-knowledge-base question scenarios, graded retrieval evaluation and
explicit measurement of unanswered questions.

MelbourneMate extends that baseline with:

- a topic-level validation and held-out test split;
- BM25s versus pinned MPNet retrieval;
- independently reviewed relevance judgements;
- topic-clustered confidence intervals and paired testing;
- grounded and no-context generation arms;
- deterministic citation checks and blind human review;
- immutable run manifests and data/configuration fingerprints;
- a local Ollama generator and Streamlit interface.

The projects use different domains, collections, questions, models and
protocols, so their numerical scores are not directly comparable. See
[`docs/walert-methodology-mapping.md`](docs/walert-methodology-mapping.md).

## Elaine's Sprint 9 scope

Elaine owns the following technical and evaluation work recorded in Trello:

- S9-03: BM25s and MPNet retrieval;
- S9-04: frozen settings, manifests and reproducible run records;
- S9-06: BM25s/MPNet comparison and statistical testing;
- S9-09: retrieval and generation error analysis;
- S9-15: shared high-risk definitions and refreshed analysis;
- S9-16: system architecture and evidence flow;
- S9-17: all-MiniLM/MPNet validation comparison and final encoder decision.

S9-01, S9-05, S9-07, S9-08 and S9-10 are shared tasks. Their final evidence is
added only after both members' inputs are present. Files owned solely by
Sriporn are not included in Elaine's branch.

## Repository delivery order

The project is delivered through three reviewed branches:

1. `elaine/s9-foundation-methodology`
2. `elaine/s9-retrieval-evaluation-code`
3. `elaine/s9-final-evidence-interface`

Each branch is reviewed through a pull request before it reaches `main`.
Branch 2 contains Elaine's base implementation and excludes Sriporn's later
fixes. Branch 3 must not be finalised before the dependencies listed in
[`docs/HANDOFF.md`](docs/HANDOFF.md) are available.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dense,app,dev]"
```

Do not commit virtual environments, model weights, `.env` files, caches or
personal participant data.

## Current branch boundary

The first branch contains project structure, a small non-reportable fixture,
the architecture and the Walert methodology mapping. It intentionally does not
claim that the final dataset, tests, manual reviews or reported results are
already present. Those items enter later branches after their owners' pull
requests and reviews.
