# Sprint 9 error analysis

This report uses saved rankings and answers. No model was rerun.

## Retrieval

- BM25s has higher NDCG@5 on 2 questions.
- MPNet has higher NDCG@5 on 10 paraphrased questions.
- Both retrievers miss all relevant top-five passages on 0 questions.

## Generation

| Arm | Hit but refused | Miss but answered | Invalid citation | OOKB answered | High-risk failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 2 | 1 | 4 | 1 | 5 |
| mpnet | 3 | 0 | 5 | 4 | 9 |
| no-context | 0 | 0 | 0 | 30 | 15 |

## Example question IDs

- BM25s better: Q04P1 (bm25, employment), Q28P1 (bm25, visa)
- MPNet better paraphrases: Q08P2 (mpnet, health), Q10P1 (mpnet, emergency), Q10P2 (mpnet, emergency)
- Both retrieval methods fail: None
- Retrieval hit but refused: Q20P2 (bm25, housing), Q23P1 (bm25, employment), Q20P2 (mpnet, housing)
- Retrieval miss but answered: Q25P2 (bm25, general)
- Invalid citation: Q10P1 (bm25, emergency), Q10P2 (bm25, emergency), Q25P1 (bm25, general)
- OOKB answered: QX14 (bm25, housing), QX11 (mpnet, housing), QX14 (mpnet, housing)

The CSV files contain every case. These counts are descriptive and do not replace human answer review.
