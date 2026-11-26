### Deployed evaluation of the success-distillation control, locked tests

Paper protocol: the manager routes freely and may call more than once. `s1` is the original checkpoint's run on the same pools (the S_1 row of the RSI table); `seedK` are re-trainings of the same label file with the recipe printed in the paper-era logs and training seed K. Each cell lists the checkpoints in that order, then mean ± sample SD.

**MedQA** (n = 500); checkpoints: sucd

| | sucd | mean ± SD |
|:--|--:|--:|
| Accuracy (%) | 81.6 | 81.6 |
| Advisor calls per question | 0.976 | 0.976 |
| Draft accuracy (%) | 71.8 | 71.8 |
| Call rate when the draft is wrong | 0.979 | 0.979 |
| Corrected (% of questions) | 13.6 | 13.6 |
| Corrupted (% of questions) | 3.8 | 3.8 |
| Valid answers (%) | 100.0 | 100.0 |

**MMLU-Pro** (n = 500); checkpoints: sucd

| | sucd | mean ± SD |
|:--|--:|--:|
| Accuracy (%) | 73.2 | 73.2 |
| Advisor calls per question | 0.998 | 0.998 |
| Draft accuracy (%) | 50.0 | 50.0 |
| Call rate when the draft is wrong | 1.000 | 1.000 |
| Corrected (% of questions) | 25.2 | 25.2 |
| Corrupted (% of questions) | 2.0 | 2.0 |
| Valid answers (%) | 100.0 | 100.0 |

**GPQA** (n = 100); checkpoints: sucd

| | sucd | mean ± SD |
|:--|--:|--:|
| Accuracy (%) | 50.0 | 50.0 |
| Advisor calls per question | 0.990 | 0.990 |
| Draft accuracy (%) | 40.0 | 40.0 |
| Call rate when the draft is wrong | 0.983 | 0.983 |
| Corrected (% of questions) | 12.0 | 12.0 |
| Corrupted (% of questions) | 2.0 | 2.0 |
| Valid answers (%) | 100.0 | 100.0 |

**AQuA** (n = 254); checkpoints: sucd

| | sucd | mean ± SD |
|:--|--:|--:|
| Accuracy (%) | 83.1 | 83.1 |
| Advisor calls per question | 0.961 | 0.961 |
| Draft accuracy (%) | 37.8 | 37.8 |
| Call rate when the draft is wrong | 0.987 | 0.987 |
| Corrected (% of questions) | 47.6 | 47.6 |
| Corrupted (% of questions) | 2.4 | 2.4 |
| Valid answers (%) | 100.0 | 100.0 |
