### Equal-information comparison: manager `seed3`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 73.2 / 0.00 | 50.8 / 0.00 | 41.0 / 0.00 | 37.4 / 0.00 |
| Learned calling (first call, then commit) | 79.0 / 0.36 | 64.8 / 0.37 | 52.0 / 0.73 | 75.6 / 0.72 |
| Learned selection, Verifier only | 80.6 / 0.36 | 65.4 / 0.37 | 52.0 / 0.73 | 75.6 / 0.72 |
| Random questions, same calls per advisor | 76.5 / 0.36 [74.8, 78.2] | 58.2 / 0.37 [56.4, 60.2] | 49.5 / 0.73 [45.0, 54.0] | 71.7 / 0.72 [68.5, 74.8] |
| Random subset, Verifier only (same budget) | 77.4 / 0.36 [75.8, 79.0] | 59.3 / 0.37 [57.4, 61.2] | 51.2 / 0.73 [48.0, 55.0] | 71.7 / 0.72 [68.5, 74.8] |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 81.0 / 0.36 | 64.6 / 0.37 | 53.0 / 0.73 | 74.0 / 0.72 |
| Always Verifier, with the candidate | 84.8 / 1.00 | 73.6 / 1.00 | 55.0 / 1.00 | 85.0 / 1.00 |
| Always Verifier, candidate removed | 84.4 / 1.00 | 68.4 / 1.00 | 59.0 / 1.00 | 78.7 / 1.00 |
| Always Reasoner | 81.8 / 1.00 | 56.6 / 1.00 | 47.0 / 1.00 | 35.4 / 1.00 |
| Always Extractor | 79.4 / 1.00 | 51.8 / 1.00 | 38.0 / 1.00 | 29.9 / 1.00 |
| Revision turn on an empty Verifier output | 72.6 / 0.00 | 51.0 / 0.00 | 41.0 / 0.00 | 35.8 / 0.00 |
| Oracle (best of candidate / one call) | 91.2 / 0.18 | 79.8 / 0.29 | 62.0 / 0.21 | 88.6 / 0.51 |

First call of the learned policy: MedQA 179/500 questions (E 7, R 134, V 38), commit forced after 0; MMLU-Pro 185/500 questions (E 14, R 12, V 159), commit forced after 9; GPQA 73/100 questions (E 3, R 14, V 56), commit forced after 0; AQuA 183/254 questions (E 0, R 0, V 183), commit forced after 1.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .005; MMLU-Pro <.001; GPQA .196; AQuA .009. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA <.001; MMLU-Pro <.001; GPQA .452; AQuA .009.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -5.8 [-8.8, -2.8], <.001 | -8.8 [-11.6, -6.0], <.001 | -3.0 [-8.0, +1.0], .375 | -9.4 [-13.4, -5.9], <.001 | -7.4 [-9.2, -5.7], <.001 |
| Learned − learned selection sent to the Verifier | -1.6 [-3.8, +0.6], .215 | -0.6 [-1.6, +0.2], .375 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -0.8 [-1.7, +0.1], .099 |
| Learned − confidence gate (same number of calls) | -2.0 [-5.0, +0.8], .229 | +0.2 [-3.0, +3.6], 1.000 | -1.0 [-6.0, +3.0], 1.000 | +1.6 [-3.9, +6.7], .665 | -0.4 [-2.4, +1.6], .711 |
| Always Verifier: with − without the candidate | +0.4 [-2.2, +3.0], .883 | +5.2 [+1.8, +8.6], .004 | -4.0 [-11.0, +3.0], .424 | +6.3 [+1.6, +11.0], .014 | +3.0 [+1.0, +4.9], .003 |
| Always Verifier − empty-output revision | +12.2 [+8.8, +15.6], <.001 | +22.6 [+18.4, +26.8], <.001 | +14.0 [+5.0, +23.0], .004 | +49.2 [+42.5, +55.9], <.001 | +23.1 [+20.6, +25.7], <.001 |
| Empty-output revision − candidate | -0.6 [-2.0, +0.8], .581 | +0.2 [-0.8, +1.2], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -1.6 [-4.3, +1.2], .388 | -0.4 [-1.3, +0.4], .377 |
