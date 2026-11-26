### Equal-information comparison: manager `cmto`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Learned calling (first call, then commit) | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Learned selection, Verifier only | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Random subset, Verifier only (same budget) | 73.6 / 0.00 [73.6, 73.6] | 53.2 / 0.00 [53.2, 53.2] | 42.0 / 0.00 [42.0, 42.0] | 38.6 / 0.00 [38.6, 38.6] |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Always Verifier, with the candidate | 83.2 / 1.00 | 74.6 / 1.00 | 57.0 / 1.00 | 84.3 / 1.00 |
| Always Verifier, candidate removed | 84.0 / 1.00 | 67.2 / 1.00 | 58.0 / 1.00 | 78.3 / 1.00 |
| Always Reasoner | 80.0 / 1.00 | 56.4 / 1.00 | 50.0 / 1.00 | 40.2 / 1.00 |
| Always Extractor | 76.6 / 1.00 | 53.6 / 1.00 | 42.0 / 1.00 | 38.6 / 1.00 |
| Revision turn on an empty Verifier output | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Oracle (best of candidate / one call) | 88.4 / 0.15 | 78.0 / 0.25 | 61.0 / 0.19 | 86.6 / 0.48 |

First call of the learned policy: MedQA 0/500 questions (E 0, R 0, V 0); MMLU-Pro 0/500 questions (E 0, R 0, V 0); GPQA 0/100 questions (E 0, R 0, V 0); AQuA 0/254 questions (E 0, R 0, V 0).

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -9.6 [-12.8, -6.6], <.001 | -21.4 [-25.4, -17.4], <.001 | -15.0 [-23.0, -7.0], .001 | -45.7 [-52.4, -39.0], <.001 | -21.1 [-23.6, -18.8], <.001 |
| Learned − learned selection sent to the Verifier | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 |
| Learned − confidence gate (same number of calls) | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 |
| Always Verifier: with − without the candidate | -0.8 [-3.4, +1.8], .644 | +7.4 [+4.0, +11.0], <.001 | -1.0 [-8.0, +5.0], 1.000 | +5.9 [+1.2, +10.6], .020 | +3.5 [+1.6, +5.4], <.001 |
| Always Verifier − empty-output revision | +9.6 [+6.6, +12.8], <.001 | +21.4 [+17.4, +25.4], <.001 | +15.0 [+7.0, +23.0], .001 | +45.7 [+39.0, +52.4], <.001 | +21.1 [+18.8, +23.6], <.001 |
| Empty-output revision − candidate | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 |
