**Table: Matched-budget replay (appendix).** Matched-budget replay on the development pools (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254). For each policy, the questions on which it made its first advisor call are rescored with the forced-call result of that advisor on the same question, and the other questions are scored with the direct answer; this gives the replay accuracy (%). Calls (E/R/V): the number of first calls the policy made to the Extractor, Reasoner and Verifier. The same number of calls per advisor is then reassigned to uniformly random questions 2,000 times; Random 95% is the 2.5–97.5 percentile interval of the resulting accuracies, p the fraction of the 2,000 random assignments whose accuracy is at least the policy's replay accuracy, and Above whether the replay accuracy exceeds the upper end of the interval. Bold: replay accuracy above the interval. R3/R5: round 3 or 5 of the chain; no GRPO: the v3 chain trained with SFT only; +GRPO: the main v3+GRPO chain. †: taken from the results page because the per-question files of these runs were not archived, so their per-advisor call counts are unavailable (—), and for the AQuA-RAT round-5 policies only p and the above/below outcome were recorded.

| Benchmark | Policy | n | Calls (E/R/V) | Replay acc. | Random 95% | p | Above |
|:--|:--|--:|:--:|--:|:--:|--:|:--:|
| MedQA | S_1 | 200 | 2 / 40 / 32 | **79.5** | [73.0, 78.5] | 0.012 | yes |
| MedQA | RSI (dynamic), no GRPO, R3 | 200 | 2 / 26 / 0 | **78.5** | [73.5, 77.0] | 0.002 | yes |
| MedQA | RSI (static), no GRPO, R3 | 200 | 3 / 22 / 26 | **78.5** | [73.0, 78.0] | 0.013 | yes |
| MedQA | RSI (dynamic), no GRPO, R5 † | 200 | — | 76.0 | [73.5, 77.5] | 0.370 | no |
| MedQA | RSI (static), no GRPO, R5 † | 200 | — | **79.0** | [72.0, 78.5] | 0.015 | yes |
| MMLU-Pro | S_1 | 200 | 2 / 7 / 52 | 58.5 | [55.5, 62.0] | 0.587 | no |
| MMLU-Pro | RSI (dynamic), no GRPO, R3 | 200 | 0 / 3 / 41 | 58.0 | [53.0, 58.5] | 0.096 | no |
| MMLU-Pro | RSI (static), no GRPO, R3 | 200 | 8 / 4 / 41 | 56.5 | [53.5, 59.5] | 0.599 | no |
| MMLU-Pro | RSI (dynamic), +GRPO, R3 † | 200 | — | **64.5** | [54.5, 61.5] | <0.001 | yes |
| MMLU-Pro | RSI (dynamic), no GRPO, R5 † | 200 | — | **60.5** | [53.0, 59.0] | 0.003 | yes |
| GPQA | S_1 | 100 | 0 / 16 / 46 | 47.0 | [42.0, 51.0] | 0.552 | no |
| GPQA | RSI (dynamic), no GRPO, R3 | 100 | 0 / 26 / 35 | 46.0 | [41.0, 51.0] | 0.547 | no |
| GPQA | RSI (static), no GRPO, R3 | 100 | 0 / 13 / 34 | 47.0 | [43.0, 52.0] | 0.682 | no |
| AQuA-RAT | S_1 | 254 | 0 / 0 / 198 | 70.1 | [66.1, 71.7] | 0.238 | no |
| AQuA-RAT | RSI (dynamic), no GRPO, R3 | 254 | 0 / 0 / 208 | 71.7 | [66.9, 72.0] | 0.080 | no |
| AQuA-RAT | RSI (static), no GRPO, R3 | 254 | 0 / 0 / 180 | **70.1** | [63.4, 69.3] | 0.011 | yes |
| AQuA-RAT | RSI (dynamic), no GRPO, R5 † | 254 | — | — | — | 0.150 | no |
| AQuA-RAT | RSI (static), no GRPO, R5 † | 254 | — | — | — | 0.002 | yes |
