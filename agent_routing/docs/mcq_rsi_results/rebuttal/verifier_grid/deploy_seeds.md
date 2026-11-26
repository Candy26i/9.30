### Deployed evaluation of the manager across training seeds, locked tests

Paper protocol: the manager routes freely and may call more than once. `s1` is the original checkpoint's run on the same pools (the S_1 row of the RSI table); `seedK` are re-trainings of the same label file with the recipe printed in the paper-era logs and training seed K. Each cell lists the checkpoints in that order, then mean ± sample SD.

**MedQA** (n = 500); checkpoints: s1, seed1, seed2

| | s1 | seed1 | seed2 | mean ± SD |
|:--|--:|--:|--:|--:|
| Accuracy (%) | 78.8 | 78.0 | 78.6 | 78.5 ± 0.4 |
| Advisor calls per question | 0.366 | 0.348 | 0.386 | 0.367 ± 0.019 |
| Draft accuracy (%) | 73.4 | 74.4 | 74.6 | 74.1 ± 0.6 |
| Call rate when the draft is wrong | 0.632 | 0.594 | 0.614 | 0.613 ± 0.019 |
| Corrected (% of questions) | 9.2 | 7.8 | 7.8 | 8.3 ± 0.8 |
| Corrupted (% of questions) | 3.8 | 4.2 | 3.8 | 3.9 ± 0.2 |
| Valid answers (%) | 100.0 | 100.0 | 99.8 | 99.9 ± 0.1 |

**MMLU-Pro** (n = 500); checkpoints: s1, seed1, seed2

| | s1 | seed1 | seed2 | mean ± SD |
|:--|--:|--:|--:|--:|
| Accuracy (%) | 63.6 | 65.4 | 65.6 | 64.9 ± 1.1 |
| Advisor calls per question | 0.352 | 0.374 | 0.404 | 0.377 ± 0.026 |
| Draft accuracy (%) | 51.0 | 51.4 | 51.4 | 51.3 ± 0.2 |
| Call rate when the draft is wrong | 0.490 | 0.510 | 0.531 | 0.510 ± 0.021 |
| Corrected (% of questions) | 14.2 | 15.2 | 15.6 | 15.0 ± 0.7 |
| Corrupted (% of questions) | 1.6 | 1.2 | 1.4 | 1.4 ± 0.2 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 |

**GPQA** (n = 100); checkpoints: s1, seed1

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Accuracy (%) | 54.0 | 50.0 | 52.0 ± 2.8 |
| Advisor calls per question | 0.520 | 0.550 | 0.535 ± 0.021 |
| Draft accuracy (%) | 41.0 | 40.0 | 40.5 ± 0.7 |
| Call rate when the draft is wrong | 0.508 | 0.550 | 0.529 ± 0.029 |
| Corrected (% of questions) | 13.0 | 12.0 | 12.5 ± 0.7 |
| Corrupted (% of questions) | 0.0 | 2.0 | 1.0 ± 1.4 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 ± 0.0 |

**AQuA** (n = 254); checkpoints: s1, seed1, seed2

| | s1 | seed1 | seed2 | mean ± SD |
|:--|--:|--:|--:|--:|
| Accuracy (%) | 77.2 | 76.0 | 76.0 | 76.4 ± 0.7 |
| Advisor calls per question | 0.776 | 0.740 | 0.748 | 0.755 ± 0.019 |
| Draft accuracy (%) | 36.2 | 36.2 | 35.8 | 36.1 ± 0.2 |
| Call rate when the draft is wrong | 0.846 | 0.821 | 0.828 | 0.832 ± 0.013 |
| Corrected (% of questions) | 43.7 | 42.5 | 42.9 | 43.0 ± 0.6 |
| Corrupted (% of questions) | 2.8 | 2.8 | 2.8 | 2.8 ± 0.0 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 |
