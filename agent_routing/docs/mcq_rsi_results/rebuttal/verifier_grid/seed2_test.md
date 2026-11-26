### Equal-information comparison: manager `seed2`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 74.6 / 0.00 | 51.4 / 0.00 | 43.0 / 0.00 | 35.8 / 0.00 |
| Learned calling (first call, then commit) | 78.6 / 0.38 | 65.4 / 0.37 | 51.0 / 0.44 | 76.0 / 0.75 |
| Learned selection, Verifier only | 80.0 / 0.38 | 65.8 / 0.37 | 52.0 / 0.44 | 76.0 / 0.75 |
| Random questions, same calls per advisor | 77.4 / 0.38 [75.6, 79.2] | 58.5 / 0.37 [56.6, 60.6] | 47.9 / 0.44 [43.0, 53.0] | 71.7 / 0.75 [68.9, 74.8] |
| Random subset, Verifier only (same budget) | 77.9 / 0.38 [76.2, 79.6] | 59.4 / 0.37 [57.4, 61.4] | 48.7 / 0.44 [44.0, 53.0] | 71.7 / 0.75 [68.9, 74.8] |
| Confidence-gated Verifier (dev threshold, matched call rate) | 80.2 / 0.39 | 62.6 / 0.34 | 50.0 / 0.60 | 76.0 / 0.79 |
| Confidence-gated Verifier (dev threshold, best dev accuracy) | 82.6 / 0.65 | 73.2 / 0.79 | 55.0 / 0.86 | 80.7 / 0.93 |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 80.2 / 0.38 | 63.2 / 0.37 | 50.0 / 0.44 | 73.6 / 0.75 |
| Always Verifier, with the candidate | 83.2 / 1.00 | 73.2 / 1.00 | 56.0 / 1.00 | 83.9 / 1.00 |
| Always Verifier, candidate removed | 84.2 / 1.00 | 68.8 / 1.00 | 59.0 / 1.00 | 79.5 / 1.00 |
| Always Reasoner | 81.6 / 1.00 | 56.6 / 1.00 | 48.0 / 1.00 | 36.2 / 1.00 |
| Always Extractor | 79.0 / 1.00 | 52.2 / 1.00 | 40.0 / 1.00 | 31.9 / 1.00 |
| Revision turn on an empty Verifier output | 73.0 / 0.00 | 51.8 / 0.00 | 43.0 / 0.00 | 35.8 / 0.00 |
| Oracle (best of candidate / one call) | 91.0 / 0.16 | 80.0 / 0.29 | 64.0 / 0.21 | 89.4 / 0.54 |

First call of the learned policy: MedQA 191/500 questions (E 9, R 122, V 60), commit forced after 1; MMLU-Pro 183/500 questions (E 13, R 9, V 161), commit forced after 10; GPQA 44/100 questions (E 1, R 9, V 34), commit forced after 0; AQuA 190/254 questions (E 0, R 0, V 190), commit forced after 0.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .139; MMLU-Pro <.001; GPQA .141; AQuA .005. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA .011; MMLU-Pro <.001; GPQA .115; AQuA .005.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -4.6 [-7.6, -1.6], .004 | -7.8 [-10.6, -5.0], <.001 | -5.0 [-12.0, +2.0], .267 | -7.9 [-11.4, -4.3], <.001 | -6.4 [-8.2, -4.7], <.001 |
| Learned − learned selection sent to the Verifier | -1.4 [-3.6, +0.8], .296 | -0.4 [-1.0, +0.0], .500 | -1.0 [-3.0, +0.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -0.7 [-1.6, +0.1], .132 |
| Learned − confidence gate (dev threshold, matched) | -1.6 [-4.6, +1.4], .350 | +2.8 [-0.4, +6.0], .120 | +1.0 [-6.0, +8.0], 1.000 | +0.0 [-5.1, +5.1], 1.000 | +0.5 [-1.4, +2.4], .656 |
| Learned − confidence gate (dev threshold, best) | -4.0 [-7.0, -1.2], .009 | -7.8 [-10.6, -5.0], <.001 | -4.0 [-11.0, +3.0], .388 | -4.7 [-9.1, -0.4], .043 | -5.5 [-7.3, -3.8], <.001 |
| Learned − confidence gate (same number of calls) | -1.6 [-4.6, +1.4], .350 | +2.2 [-1.0, +5.4], .228 | +1.0 [-6.0, +8.0], 1.000 | +2.4 [-2.8, +7.5], .461 | +0.7 [-1.2, +2.7], .507 |
| Always Verifier: with − without the candidate | -1.0 [-3.8, +1.8], .576 | +4.4 [+1.0, +7.8], .017 | -3.0 [-10.0, +5.0], .607 | +4.3 [-0.8, +9.1], .117 | +1.8 [-0.1, +3.8], .077 |
| Always Verifier − empty-output revision | +10.2 [+6.8, +13.6], <.001 | +21.4 [+17.2, +25.6], <.001 | +13.0 [+4.0, +22.0], .011 | +48.0 [+40.9, +54.7], <.001 | +21.6 [+19.1, +24.3], <.001 |
| Empty-output revision − candidate | -1.6 [-3.0, -0.4], .021 | +0.4 [-0.8, +1.6], .754 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [-2.0, +2.0], 1.000 | -0.4 [-1.2, +0.3], .327 |
