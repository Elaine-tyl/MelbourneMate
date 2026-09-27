# Dense encoder comparison

Trello: S9-17  
GitHub: #51  
Date: 27 September 2026

## Method

Both encoders retrieved the top five passages for the same validation questions
in `data/v1`. The collection fingerprint was `1f04bcc7ed2cd814`. No test
questions were used for model selection.

| Encoder | Revision |
| --- | --- |
| all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| multi-qa-mpnet-base-cos-v1 | `d51b22a1dfa8184e9258074e56e2875e50612dca` |

## Validation results

| Encoder | NDCG@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| MiniLM | 0.8990 | 0.9815 | 0.8910 |
| MPNet | 0.9214 | 0.9815 | 0.9136 |

For NDCG@5, MPNet minus MiniLM was +0.0224. The topic-clustered 95% bootstrap
interval was [-0.0193, +0.0681], with paired randomisation p = 0.3530.

## Decision

Use MPNet for the frozen test run. It had the higher validation NDCG@5 and MRR
without reducing Recall@5. The difference was not statistically significant, so
the report must describe this as a validation-based model choice, not proof that
MPNet is generally better.

The saved run folders are:

- `runs/retrieval/s9-minilm-validation/`
- `runs/retrieval/s9-mpnet-validation/`
