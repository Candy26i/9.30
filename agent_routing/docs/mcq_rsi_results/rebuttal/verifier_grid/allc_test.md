### Equal-information comparison: manager `allc`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 73.0 / 0.00 | 51.8 / 0.00 | 40.0 / 0.00 | 37.0 / 0.00 |
| Learned calling (first call, then commit) | 77.2 / 0.23 | 59.2 / 0.21 | 54.0 / 0.68 | 69.7 / 0.63 |
| Learned selection, Verifier only | 76.8 / 0.23 | 59.0 / 0.21 | 53.0 / 0.68 | 69.7 / 0.63 |
| Random questions, same calls per advisor | 75.2 / 0.23 [73.8, 76.6] | 55.7 / 0.21 [54.0, 57.4] | 48.8 / 0.68 [44.0, 54.0] | 67.0 / 0.63 [63.8, 70.1] |
| Random subset, Verifier only (same budget) | 75.4 / 0.23 [74.0, 76.8] | 56.3 / 0.21 [54.6, 58.0] | 50.2 / 0.68 [46.0, 55.0] | 67.0 / 0.63 [63.8, 70.1] |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 78.2 / 0.23 | 59.0 / 0.21 | 53.0 / 0.68 | 68.9 / 0.63 |
| Always Verifier, with the candidate | 83.2 / 1.00 | 73.2 / 1.00 | 55.0 / 1.00 | 84.6 / 1.00 |
| Always Verifier, candidate removed | 84.6 / 1.00 | 69.6 / 1.00 | 61.0 / 1.00 | 79.1 / 1.00 |
| Always Reasoner | 81.6 / 1.00 | 57.0 / 1.00 | 47.0 / 1.00 | 39.4 / 1.00 |
| Always Extractor | 79.2 / 1.00 | 52.0 / 1.00 | 39.0 / 1.00 | 31.5 / 1.00 |
| Revision turn on an empty Verifier output | 71.4 / 0.00 | 51.4 / 0.00 | 41.0 / 0.00 | 34.6 / 0.00 |
| Oracle (best of candidate / one call) | 90.6 / 0.18 | 80.4 / 0.29 | 65.0 / 0.25 | 89.0 / 0.52 |

First call of the learned policy: MedQA 117/500 questions (E 3, R 56, V 58), commit forced after 0; MMLU-Pro 106/500 questions (E 9, R 10, V 87), commit forced after 8; GPQA 68/100 questions (E 1, R 16, V 51), commit forced after 0; AQuA 160/254 questions (E 0, R 0, V 160), commit forced after 4.

First turn equal to unconstrained greedy decoding: MedQA 499/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .005; MMLU-Pro <.001; GPQA .035; AQuA .057. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA .035; MMLU-Pro <.001; GPQA .152; AQuA .057.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -6.0 [-8.8, -3.2], <.001 | -14.0 [-17.6, -10.4], <.001 | -1.0 [-7.0, +5.0], 1.000 | -15.0 [-19.7, -10.6], <.001 | -10.3 [-12.3, -8.3], <.001 |
| Learned − learned selection sent to the Verifier | +0.4 [-1.2, +2.0], .815 | +0.2 [-0.4, +1.0], 1.000 | +1.0 [-2.0, +5.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.3 [-0.4, +1.0], .541 |
| Learned − confidence gate (same number of calls) | -1.0 [-3.6, +1.6], .542 | +0.2 [-3.0, +3.4], 1.000 | +1.0 [-5.0, +7.0], 1.000 | +0.8 [-5.1, +7.1], .899 | -0.1 [-2.0, +1.8], 1.000 |
| Always Verifier: with − without the candidate | -1.4 [-4.2, +1.2], .382 | +3.6 [+0.2, +7.0], .050 | -6.0 [-14.0, +2.0], .238 | +5.5 [+0.8, +10.6], .038 | +1.4 [-0.5, +3.3], .181 |
| Always Verifier − empty-output revision | +11.8 [+8.4, +15.2], <.001 | +21.8 [+17.8, +26.0], <.001 | +14.0 [+5.0, +23.0], .007 | +50.0 [+43.3, +56.7], <.001 | +22.8 [+20.2, +25.4], <.001 |
| Empty-output revision − candidate | -1.6 [-3.4, +0.0], .115 | -0.4 [-2.0, +1.2], .815 | +1.0 [+0.0, +3.0], 1.000 | -2.4 [-5.5, +0.8], .210 | -1.1 [-2.1, +0.0], .058 |
