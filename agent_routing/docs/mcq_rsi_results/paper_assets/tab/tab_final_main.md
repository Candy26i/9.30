| Policy | Round | MedQA Acc. | MedQA Calls | MMLU-Pro Acc. | MMLU-Pro Calls | GPQA Acc. | GPQA Calls | AQuA-RAT Acc. | AQuA-RAT Calls |
|:--|:-:|--:|--:|--:|--:|--:|--:|--:|--:|
| S₁ | 1 | 78.8 | 0.37 | 63.6 | 0.35 | 54.0 | 0.52 | 77.2 | 0.78 |
| *SFT + GRPO* | | | | | | | | | |
| Static | 3 | 79.8 | 0.28 | 61.4 | 0.26 | 50.0 | 0.47 | 73.6 | 0.72 |
| Dynamic | 3 | 78.6 | 0.17 | 68.6 | 0.36 | 55.0 | 0.60 | 77.6 | 0.77 |
| Success | 3 | 80.2 | 0.99 | **71.8** | 1.00 | **58.0** | 1.00 | **82.7** | 1.00 |
| *SFT only* | | | | | | | | | |
| Static | 3 | 78.6 | 0.27 | 58.0 | 0.26 | 55.0 | 0.49 | 74.8 | 0.73 |
| Dynamic | 3 | 78.6 | 0.15 | 64.4 | 0.27 | 54.0 | 0.58 | 78.0 | 0.79 |
| Static | 5 | **81.0** | 0.37 | 58.0 | 0.25 | — | — | 77.2 | 0.76 |
| Dynamic | 5 | 77.4 | 0.17 | 61.4 | 0.30 | — | — | 71.7 | 0.72 |

Locked-test accuracy (%) and advisor calls per question, single runs (MedQA 500, MMLU-Pro 500 test, GPQA 100, AQuA-RAT 254; – = not run); bold = best per benchmark. A GRPO update is kept only if it passes the development gate; MedQA kept 0 of 7.
