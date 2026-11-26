### Commit cap of the manager's training data, pool `test`

Managers trained on the same round-1 collection with the S_1 recipe; only the commit cap rho differs (every rescue record is kept, the commit records are cut to int(rho x rescue records)). The row marked S_1 is the paper's manager. Accuracy in %, calls per question; every reference in a row is computed from that manager's own equal-information grid (its own draft and confidences). The confidence gate sends the Verifier to the manager's least confident candidates, as many as the learned policy calls, with the cut placed inside this pool; the learned policy's calls can go to another advisor. Differences: pt [95% CI], exact McNemar p.

The second column is computed from the round-1 collection counts with the cap rule, not counted from each manager's label file. Recorded label files or training logs agree with it for S_1 on all four benchmarks, for MedQA at rho 0 / 1 / 2 and for the no-cap managers of MedQA, MMLU-Pro and GPQA; for MMLU-Pro rho 1, GPQA rho 1 and AQuA rho 0 / 0.5 / no cap no label file or training log was read (the GPQA collection counts are derived, see `configuration.md`).

**MedQA** (n = 500; round-1 collection: 62 rescue and 304 commit records)

| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling | Random questions, same calls per advisor | Verifier on its own least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 69.6 | 1.00 | 82.2 | 82.7 | 84.6 | -2.4 [-4.8, +0.0], .073 | 84.6 | -2.4 [-4.8, +0.0], .073 |
| 1 | 62 | 71.4 | 0.60 | 80.0 | 78.2 | 82.8 | -2.8 [-5.4, -0.2], .049 | 84.0 | -4.0 [-6.6, -1.4], .005 |
| 2 | 124 | 72.4 | 0.50 | 78.8 | 77.4 | 82.2 | -3.4 [-6.0, -1.0], .014 | 83.8 | -5.0 [-7.6, -2.4], <.001 |
| 3 (S_1) | 186 | 73.4 | 0.37 | 78.8 | 76.6 | 79.6 | -0.8 [-3.6, +2.0], .672 | 83.4 | -4.6 [-7.4, -1.8], .001 |
| no cap | 304 | 73.0 | 0.23 | 77.2 | 75.2 | 78.2 | -1.0 [-3.6, +1.6], .542 | 83.2 | -6.0 [-8.8, -3.2], <.001 |

**MMLU-Pro** (n = 500; round-1 collection: 81 rescue and 213 commit records)

| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling | Random questions, same calls per advisor | Verifier on its own least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 81 | 48.8 | 0.33 | 62.0 | 56.1 | 60.6 | +1.4 [-2.0, +4.8], .489 | 72.8 | -10.8 [-13.8, -7.8], <.001 |
| 2 (S_1) | 162 | 50.8 | 0.33 | 63.2 | 57.2 | 61.6 | +1.6 [-1.6, +5.0], .416 | 72.4 | -9.2 [-12.2, -6.2], <.001 |
| no cap | 213 | 51.8 | 0.21 | 59.2 | 55.7 | 59.0 | +0.2 [-3.0, +3.4], 1.000 | 73.2 | -14.0 [-17.6, -10.4], <.001 |

**GPQA** (n = 100; round-1 collection: 37 rescue and 86 commit records)

| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling | Random questions, same calls per advisor | Verifier on its own least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 37 | 37.0 | 0.82 | 51.0 | 50.5 | 52.0 | -1.0 [-5.0, +3.0], 1.000 | 54.0 | -3.0 [-8.0, +2.0], .453 |
| 2 (S_1) | 74 | 41.0 | 0.52 | 54.0 | 47.7 | 52.0 | +2.0 [-4.0, +8.0], .754 | 56.0 | -2.0 [-9.0, +5.0], .774 |
| no cap | 86 | 40.0 | 0.68 | 54.0 | 48.8 | 53.0 | +1.0 [-5.0, +7.0], 1.000 | 55.0 | -1.0 [-7.0, +5.0], 1.000 |

**AQuA** (n = 254; round-1 collection: 132 rescue and 217 commit records)

| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling | Random questions, same calls per advisor | Verifier on its own least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 30.3 | 1.00 | 85.0 | 84.4 | 85.4 | -0.4 [-1.2, +0.0], 1.000 | 85.4 | -0.4 [-1.2, +0.0], 1.000 |
| 0.5 | 66 | 34.3 | 0.88 | 81.9 | 78.7 | 80.7 | +1.2 [-2.0, +4.3], .629 | 84.6 | -2.8 [-5.1, -0.8], .039 |
| 1 (S_1) | 132 | 36.2 | 0.77 | 77.2 | 73.9 | 76.8 | +0.4 [-4.3, +5.1], 1.000 | 85.0 | -7.9 [-11.4, -4.7], <.001 |
| no cap | 217 | 37.0 | 0.63 | 69.7 | 67.0 | 68.9 | +0.8 [-5.1, +7.1], .899 | 84.6 | -15.0 [-19.7, -10.6], <.001 |
