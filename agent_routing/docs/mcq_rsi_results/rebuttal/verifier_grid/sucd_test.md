### Equal-information comparison: manager `sucd`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 71.8 / 0.00 | 50.0 / 0.00 | 40.0 / 0.00 | 37.8 / 0.00 |
| Learned calling (first call, then commit) | 81.6 / 0.98 | 73.2 / 1.00 | 50.0 / 0.99 | 83.1 / 0.96 |
| Learned selection, Verifier only | 83.2 / 0.98 | 73.6 / 1.00 | 57.0 / 0.99 | 83.1 / 0.96 |
| Random questions, same calls per advisor | 81.7 / 0.98 [80.0, 83.4] | 67.8 / 1.00 [66.0, 69.6] | 45.7 / 0.99 [42.0, 50.0] | 82.1 / 0.96 [80.7, 83.5] |
| Random subset, Verifier only (same budget) | 83.3 / 0.98 [82.8, 83.8] | 73.6 / 1.00 [73.4, 73.6] | 56.8 / 0.99 [56.0, 58.0] | 82.1 / 0.96 [80.7, 83.5] |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 83.6 / 0.98 | 73.6 / 1.00 | 57.0 / 0.99 | 82.3 / 0.96 |
| Always Verifier, with the candidate | 83.6 / 1.00 | 73.6 / 1.00 | 57.0 / 1.00 | 83.9 / 1.00 |
| Always Verifier, candidate removed | 84.4 / 1.00 | 67.6 / 1.00 | 57.0 / 1.00 | 78.7 / 1.00 |
| Always Reasoner | 81.4 / 1.00 | 53.0 / 1.00 | 48.0 / 1.00 | 39.0 / 1.00 |
| Always Extractor | 77.6 / 1.00 | 51.2 / 1.00 | 39.0 / 1.00 | 38.2 / 1.00 |
| Revision turn on an empty Verifier output | 71.4 / 0.00 | 50.0 / 0.00 | 40.0 / 0.00 | 37.8 / 0.00 |
| Oracle (best of candidate / one call) | 90.6 / 0.19 | 77.6 / 0.28 | 61.0 / 0.21 | 86.2 / 0.48 |

First call of the learned policy: MedQA 488/500 questions (E 66, R 191, V 231), commit forced after 0; MMLU-Pro 499/500 questions (E 73, R 59, V 367), commit forced after 0; GPQA 99/100 questions (E 54, R 16, V 29), commit forced after 0; AQuA 244/254 questions (E 0, R 0, V 244), commit forced after 0.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .580; MMLU-Pro <.001; GPQA .041; AQuA .111. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA .786; MMLU-Pro .756; GPQA .807; AQuA .111.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -2.0 [-4.2, +0.2], .121 | -0.4 [-2.2, +1.4], .824 | -7.0 [-14.0, -1.0], .065 | -0.8 [-2.0, +0.0], .500 | -1.6 [-2.7, -0.4], .014 |
| Learned − learned selection sent to the Verifier | -1.6 [-3.8, +0.6], .215 | -0.4 [-2.2, +1.4], .824 | -7.0 [-14.0, -1.0], .065 | +0.0 [+0.0, +0.0], 1.000 | -1.3 [-2.4, -0.1], .043 |
| Learned − confidence gate (same number of calls) | -2.0 [-4.2, +0.2], .121 | -0.4 [-2.2, +1.4], .824 | -7.0 [-14.0, -1.0], .065 | +0.8 [-1.2, +2.8], .688 | -1.3 [-2.5, -0.1], .057 |
| Always Verifier: with − without the candidate | -0.8 [-3.6, +2.0], .665 | +6.0 [+2.8, +9.4], <.001 | +0.0 [-7.0, +7.0], 1.000 | +5.1 [+0.8, +9.8], .041 | +2.9 [+1.0, +4.8], .003 |
| Always Verifier − empty-output revision | +12.2 [+8.8, +15.8], <.001 | +23.6 [+19.6, +27.8], <.001 | +17.0 [+8.0, +26.0], <.001 | +46.1 [+39.4, +52.8], <.001 | +23.1 [+20.6, +25.7], <.001 |
| Empty-output revision − candidate | -0.4 [-1.4, +0.6], .688 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -0.1 [-0.5, +0.2], .688 |
