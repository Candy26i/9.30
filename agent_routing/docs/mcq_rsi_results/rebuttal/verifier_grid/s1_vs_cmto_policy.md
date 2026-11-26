### Learned calling of two managers on the same questions: S_1 against commit-only, pool `test`

Each manager's first learned call followed by the forced-commit revision (the equal-information grid). Accuracy (%) / advisor calls per question.

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| S_1: learned calling | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| commit-only: learned calling | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| S_1: first call to Extractor / Reasoner / Verifier | 2 / 109 / 72 | 10 / 12 / 144 | 1 / 9 / 42 | 0 / 0 / 196 |
| commit-only: first call to Extractor / Reasoner / Verifier | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| S_1: draft accuracy (%) | 73.4 | 50.8 | 41.0 | 36.2 |
| commit-only: draft accuracy (%) | 73.6 | 53.2 | 42.0 | 38.6 |
| S_1: accuracy gained per call (pt) | 14.8 | 37.3 | 25.0 | 53.1 |
| commit-only: accuracy gained per call (pt) | – | – | – | – |
| commit-only: always Verifier with the candidate (%) | 83.2 | 74.6 | 57.0 | 84.3 |
| S_1 − commit-only, learned calling: pt [95% CI], exact McNemar p | +5.2 [+2.0, +8.4], .002 | +10.0 [+6.2, +13.8], <.001 | +12.0 [+5.0, +19.0], .002 | +38.6 [+31.5, +45.7], <.001 |

Pooled over the 1354 questions: S_1 − commit-only = +13.7 [+11.4, +16.2], <.001.
