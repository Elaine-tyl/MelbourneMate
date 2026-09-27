# Sprint 9 error analysis

This report uses saved rankings and answers. No model was rerun.

## Retrieval

- BM25s has higher NDCG@5 on 1 question.
- MPNet has higher NDCG@5 on 11 paraphrased questions.
- Both retrievers miss all relevant top-five passages on 0 questions.

## Generation

| Arm | Hit but refused | Miss but answered | Invalid citation | OOKB answered | High-risk failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 2 | 2 | 6 | 1 | 7 |
| mpnet | 3 | 0 | 7 | 4 | 11 |
| no-context | 0 | 0 | 0 | 30 | 15 |

## Example question IDs

- BM25s better: Q28P1 (bm25, visa)
- MPNet better paraphrases: Q04P1 (mpnet, employment), Q08P2 (mpnet, health), Q10P1 (mpnet, emergency)
- Both retrieval methods fail: None
- Retrieval hit but refused: Q20P2 (bm25, housing), Q23P1 (bm25, employment), Q20P2 (mpnet, housing)
- Retrieval miss but answered: Q04C (bm25, employment), Q25P2 (bm25, general)
- Invalid citation: Q04C (bm25, employment), Q04P1 (bm25, employment), Q10P1 (bm25, emergency)
- OOKB answered: QX14 (bm25, housing), QX11 (mpnet, housing), QX14 (mpnet, housing)

The CSV files contain every case. These counts are descriptive and do not replace human answer review.
