### Deployed evaluation of the manager across training seeds, locked tests

Paper protocol: the manager routes freely and may call more than once. `s1` is the original checkpoint's run on the same pools (the S_1 row of the RSI table); `seedK` are re-trainings of the same label file with the recipe printed in the paper-era logs and training seed K. Each cell lists the checkpoints in that order, then mean ± sample SD.

**MedQA** (n = 500); checkpoints: s1, seed1

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Accuracy (%) | 78.8 | 78.0 | 78.4 ± 0.6 |
| Advisor calls per question | 0.366 | 0.348 | 0.357 ± 0.013 |
| Draft accuracy (%) | 73.4 | 74.4 | 73.9 ± 0.7 |
| Call rate when the draft is wrong | 0.632 | 0.594 | 0.613 ± 0.027 |
| Corrected (% of questions) | 9.2 | 7.8 | 8.5 ± 1.0 |
| Corrupted (% of questions) | 3.8 | 4.2 | 4.0 ± 0.3 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 ± 0.0 |

**MMLU-Pro** (n = 500); checkpoints: s1, seed1

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Accuracy (%) | 63.6 | 65.4 | 64.5 ± 1.3 |
| Advisor calls per question | 0.352 | 0.374 | 0.363 ± 0.016 |
| Draft accuracy (%) | 51.0 | 51.4 | 51.2 ± 0.3 |
| Call rate when the draft is wrong | 0.490 | 0.510 | 0.500 ± 0.014 |
| Corrected (% of questions) | 14.2 | 15.2 | 14.7 ± 0.7 |
| Corrupted (% of questions) | 1.6 | 1.2 | 1.4 ± 0.3 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 ± 0.0 |

**GPQA** (n = 100); checkpoints: s1

| | s1 | mean ± SD |
|:--|--:|--:|
| Accuracy (%) | 54.0 | 54.0 |
| Advisor calls per question | 0.520 | 0.520 |
| Draft accuracy (%) | 41.0 | 41.0 |
| Call rate when the draft is wrong | 0.508 | 0.508 |
| Corrected (% of questions) | 13.0 | 13.0 |
| Corrupted (% of questions) | 0.0 | 0.0 |
| Valid answers (%) | 100.0 | 100.0 |

**AQuA** (n = 254); checkpoints: s1, seed1

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Accuracy (%) | 77.2 | 76.0 | 76.6 ± 0.8 |
| Advisor calls per question | 0.776 | 0.740 | 0.758 ± 0.025 |
| Draft accuracy (%) | 36.2 | 36.2 | 36.2 ± 0.0 |
| Call rate when the draft is wrong | 0.846 | 0.821 | 0.833 ± 0.017 |
| Corrected (% of questions) | 43.7 | 42.5 | 43.1 ± 0.8 |
| Corrupted (% of questions) | 2.8 | 2.8 | 2.8 ± 0.0 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 ± 0.0 |
