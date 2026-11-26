### Learned calling of two managers on the same questions: S_1 against success distillation, pool `test`

Each manager's first learned call followed by the forced-commit revision (the equal-information grid). Accuracy (%) / advisor calls per question.

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| S_1: learned calling | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| success distillation: learned calling | 81.6 / 0.98 | 73.2 / 1.00 | 50.0 / 0.99 | 83.1 / 0.96 |
| S_1: first call to Extractor / Reasoner / Verifier | 2 / 109 / 72 | 10 / 12 / 144 | 1 / 9 / 42 | 0 / 0 / 196 |
| success distillation: first call to Extractor / Reasoner / Verifier | 66 / 191 / 231 | 73 / 59 / 367 | 54 / 16 / 29 | 0 / 0 / 244 |
| S_1: draft accuracy (%) | 73.4 | 50.8 | 41.0 | 36.2 |
| success distillation: draft accuracy (%) | 71.8 | 50.0 | 40.0 | 37.8 |
| S_1: accuracy gained per call (pt) | 14.8 | 37.3 | 25.0 | 53.1 |
| success distillation: accuracy gained per call (pt) | 10.0 | 23.2 | 10.1 | 47.1 |
| success distillation: always Verifier with the candidate (%) | 83.6 | 73.6 | 57.0 | 83.9 |
| S_1 − success distillation, learned calling: pt [95% CI], exact McNemar p | -2.8 [-5.4, -0.2], .044 | -10.0 [-13.2, -6.8], <.001 | +4.0 [-4.0, +12.0], .454 | -5.9 [-9.8, -2.0], .006 |

Pooled over the 1354 questions: S_1 − success distillation = -5.5 [-7.3, -3.8], <.001.
