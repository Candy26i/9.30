**Table: Matched-budget replay (appendix).** Matched-budget replay on the development pools: replay accuracy scores each first call with that advisor's cached forced-call result; the same calls per advisor (E/R/V) are reassigned to random questions 2,000 times; p is the share of draws at least as accurate; bold = above the random 95% interval. †: from the results page (per-question files not archived).

| Benchmark | Policy | n | Calls (E/R/V) | Replay acc. | Random 95% | p | Above |
|:--|:--|--:|:--:|--:|:--:|--:|:--:|
| MedQA | S_1 | 200 | 2 / 40 / 32 | **79.5** | [73.0, 78.5] | 0.012 | yes |
| MedQA | Dynamic R3 (SFT) | 200 | 2 / 26 / 0 | **78.5** | [73.5, 77.0] | 0.002 | yes |
| MedQA | Static R3 (SFT) | 200 | 3 / 22 / 26 | **78.5** | [73.0, 78.0] | 0.014 | yes |
| MedQA | Dynamic R5 (SFT) † | 200 | — | 76.0 | [73.5, 77.5] | 0.370 | no |
| MedQA | Static R5 (SFT) † | 200 | — | **79.0** | [72.0, 78.5] | 0.015 | yes |
| MMLU-Pro | S_1 | 200 | 2 / 7 / 52 | 58.5 | [55.5, 62.0] | 0.587 | no |
| MMLU-Pro | Dynamic R3 (SFT) | 200 | 0 / 3 / 41 | 58.0 | [53.0, 58.5] | 0.096 | no |
| MMLU-Pro | Static R3 (SFT) | 200 | 8 / 4 / 41 | 56.5 | [53.5, 59.5] | 0.599 | no |
| MMLU-Pro | Dynamic R3 (+GRPO) † | 200 | — | **64.5** | [54.5, 61.5] | <0.001 | yes |
| MMLU-Pro | Dynamic R5 (SFT) † | 200 | — | **60.5** | [53.0, 59.0] | 0.003 | yes |
| GPQA | S_1 | 100 | 0 / 16 / 46 | 47.0 | [42.0, 51.0] | 0.552 | no |
| GPQA | Dynamic R3 (SFT) | 100 | 0 / 26 / 35 | 46.0 | [41.0, 51.0] | 0.547 | no |
| GPQA | Static R3 (SFT) | 100 | 0 / 13 / 34 | 47.0 | [43.0, 52.0] | 0.683 | no |
| AQuA-RAT | S_1 | 254 | 0 / 0 / 198 | 70.1 | [66.1, 71.7] | 0.239 | no |
| AQuA-RAT | Dynamic R3 (SFT) | 254 | 0 / 0 / 208 | 71.7 | [66.9, 72.0] | 0.080 | no |
| AQuA-RAT | Static R3 (SFT) | 254 | 0 / 0 / 180 | **70.1** | [63.4, 69.3] | 0.011 | yes |
| AQuA-RAT | Dynamic R5 (SFT) † | 254 | — | — | — | 0.150 | no |
| AQuA-RAT | Static R5 (SFT) † | 254 | — | — | — | 0.002 | yes |
