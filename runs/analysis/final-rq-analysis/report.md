# Final RQ2, RQ3a and RQ4 analyses

Computed from saved runs. No model was rerun. Intervals are 95% cluster
bootstrap intervals and p-values come from paired cluster randomisation.

## RQ2. MPNet minus BM25s on NDCG@5

| Slice | Questions | Topics | BM25s | MPNet | Difference | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| all | 36 | 12 | 0.835 | 0.971 | +0.136 | [+0.049, +0.252] | 0.0065 |
| low | 27 | 11 | 0.811 | 0.966 | +0.154 | [+0.051, +0.292] | 0.0155 |
| medium | 6 | 4 | 0.886 | 0.989 | +0.102 | [+0.000, +0.287] | 0.5013 |
| high | 3 | 3 | 0.949 | 0.984 | +0.035 | [+0.000, +0.104] | 1.0000 |
| canonical | 12 | 12 | 0.845 | 0.990 | +0.146 | [+0.017, +0.336] | 0.1273 |
| paraphrased | 24 | 12 | 0.831 | 0.961 | +0.131 | [+0.040, +0.263] | 0.0104 |

Slices are exploratory except the low and high containment comparison.

## RQ3a. Answers to out-of-KB questions

| Comparison | Rate before | Rate after | Difference | 95% CI | p | Discordant | Evidence |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| bm25 | | 3.3% | | [0.0%, 10.0%] | | | |
| mpnet | | 13.3% | | [3.3%, 26.7%] | | | |
| no-context | | 100.0% | | [100.0%, 100.0%] | | | |
| bm25 minus no-context | 100.0% | 3.3% | -0.967 | [-1.000, -0.900] | 0.0001 | 29 | sufficient |
| mpnet minus no-context | 100.0% | 13.3% | -0.867 | [-0.967, -0.733] | 0.0001 | 26 | sufficient |
| mpnet minus bm25 | 3.3% | 13.3% | +0.100 | [+0.000, +0.233] | 0.2541 | 3 | insufficient |

Fewer than 5 discordant questions is reported as insufficient evidence.

## RQ4. Refusal behaviour

| Arm | Metric | Questions | Rate | 95% CI |
| --- | --- | ---: | ---: | --- |
| bm25 | correct refusal | 30 | 96.7% | [90.0%, 100.0%] |
| bm25 | false refusal | 34 | 5.9% | [0.0%, 14.3%] |
| mpnet | correct refusal | 30 | 86.7% | [73.3%, 96.7%] |
| mpnet | false refusal | 36 | 8.3% | [0.0%, 16.7%] |
| no-context | correct refusal | 30 | 0.0% | [0.0%, 0.0%] |
| no-context | false refusal | 0 | n/a | not applicable without retrieval |

Correct refusal and false refusal are read together. A system that refuses
everything would score full correct refusal and full false refusal.
False refusal counts only answerable questions whose evidence was retrieved.
