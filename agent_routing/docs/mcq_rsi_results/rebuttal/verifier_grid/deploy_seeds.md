### Deployed evaluation of the manager across training seeds, locked tests

Paper protocol: the manager routes freely and may call more than once. `s1` is the original checkpoint's run on the same pools (the S_1 row of the RSI table); `seedK` are re-trainings of the same label file with the recipe printed in the paper-era logs and training seed K. Each cell lists the checkpoints in that order, then mean ± sample SD.

**MedQA** (n = 500); checkpoints: s1, seed1, seed2, seed3

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Accuracy (%) | 78.8 | 78.0 | 78.6 | 79.0 | 78.6 ± 0.4 |
| Advisor calls per question | 0.366 | 0.348 | 0.386 | 0.362 | 0.365 ± 0.016 |
| Draft accuracy (%) | 73.4 | 74.4 | 74.6 | 73.2 | 73.9 ± 0.7 |
| Call rate when the draft is wrong | 0.632 | 0.594 | 0.614 | 0.612 | 0.613 ± 0.015 |
| Corrected (% of questions) | 9.2 | 7.8 | 7.8 | 9.6 | 8.6 ± 0.9 |
| Corrupted (% of questions) | 3.8 | 4.2 | 3.8 | 3.8 | 3.9 ± 0.2 |
| Valid answers (%) | 100.0 | 100.0 | 99.8 | 99.8 | 99.9 ± 0.1 |

**MMLU-Pro** (n = 500); checkpoints: s1, seed1, seed2, seed3

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Accuracy (%) | 63.6 | 65.4 | 65.6 | 65.2 | 65.0 ± 0.9 |
| Advisor calls per question | 0.352 | 0.374 | 0.404 | 0.402 | 0.383 ± 0.025 |
| Draft accuracy (%) | 51.0 | 51.4 | 51.4 | 50.8 | 51.1 ± 0.3 |
| Call rate when the draft is wrong | 0.490 | 0.510 | 0.531 | 0.528 | 0.515 ± 0.019 |
| Corrected (% of questions) | 14.2 | 15.2 | 15.6 | 15.6 | 15.1 ± 0.7 |
| Corrupted (% of questions) | 1.6 | 1.2 | 1.4 | 1.2 | 1.4 ± 0.2 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 |

**GPQA** (n = 100); checkpoints: s1, seed1, seed2, seed3

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Accuracy (%) | 54.0 | 50.0 | 51.0 | 52.0 | 51.8 ± 1.7 |
| Advisor calls per question | 0.520 | 0.550 | 0.440 | 0.730 | 0.560 ± 0.122 |
| Draft accuracy (%) | 41.0 | 40.0 | 43.0 | 41.0 | 41.2 ± 1.3 |
| Call rate when the draft is wrong | 0.508 | 0.550 | 0.439 | 0.695 | 0.548 ± 0.108 |
| Corrected (% of questions) | 13.0 | 12.0 | 9.0 | 14.0 | 12.0 ± 2.2 |
| Corrupted (% of questions) | 0.0 | 2.0 | 1.0 | 3.0 | 1.5 ± 1.3 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 |

**AQuA** (n = 254); checkpoints: s1, seed1, seed2, seed3

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Accuracy (%) | 77.2 | 76.0 | 76.0 | 75.6 | 76.2 ± 0.7 |
| Advisor calls per question | 0.776 | 0.740 | 0.748 | 0.724 | 0.747 ± 0.022 |
| Draft accuracy (%) | 36.2 | 36.2 | 35.8 | 37.4 | 36.4 ± 0.7 |
| Call rate when the draft is wrong | 0.846 | 0.821 | 0.828 | 0.811 | 0.827 ± 0.015 |
| Corrected (% of questions) | 43.7 | 42.5 | 42.9 | 40.9 | 42.5 ± 1.2 |
| Corrupted (% of questions) | 2.8 | 2.8 | 2.8 | 2.8 | 2.8 ± 0.0 |
| Valid answers (%) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 |
