### Equal-information comparison: manager `seed1`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 74.4 / 0.00 | 51.4 / 0.00 | 40.0 / 0.00 | 36.2 / 0.00 |
| Learned calling (first call, then commit) | 78.0 / 0.35 | 65.2 / 0.35 | 50.0 / 0.55 | 76.0 / 0.74 |
| Learned selection, Verifier only | 79.8 / 0.35 | 65.8 / 0.35 | 49.0 / 0.55 | 76.0 / 0.74 |
| Random questions, same calls per advisor | 77.0 / 0.35 [75.2, 78.8] | 58.4 / 0.35 [56.6, 60.4] | 47.6 / 0.55 [43.0, 53.0] | 72.1 / 0.74 [68.9, 75.2] |
| Random subset, Verifier only (same budget) | 77.3 / 0.35 [75.8, 79.0] | 59.2 / 0.35 [57.2, 61.2] | 48.8 / 0.55 [44.0, 53.0] | 72.1 / 0.74 [68.9, 75.2] |
| Confidence-gated Verifier (dev threshold, matched call rate) | 79.2 / 0.36 | 63.6 / 0.33 | 53.0 / 0.69 | 75.2 / 0.77 |
| Confidence-gated Verifier (dev threshold, best dev accuracy) | 82.4 / 0.68 | 73.2 / 0.76 | 55.0 / 0.86 | 81.5 / 0.93 |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 79.0 / 0.35 | 63.8 / 0.35 | 50.0 / 0.55 | 73.2 / 0.74 |
| Always Verifier, with the candidate | 82.8 / 1.00 | 73.6 / 1.00 | 56.0 / 1.00 | 84.6 / 1.00 |
| Always Verifier, candidate removed | 84.4 / 1.00 | 67.4 / 1.00 | 59.0 / 1.00 | 79.9 / 1.00 |
| Always Reasoner | 81.4 / 1.00 | 56.4 / 1.00 | 48.0 / 1.00 | 37.0 / 1.00 |
| Always Extractor | 79.0 / 1.00 | 52.0 / 1.00 | 38.0 / 1.00 | 33.1 / 1.00 |
| Revision turn on an empty Verifier output | 73.2 / 0.00 | 51.6 / 0.00 | 40.0 / 0.00 | 36.2 / 0.00 |
| Oracle (best of candidate / one call) | 90.8 / 0.16 | 79.8 / 0.28 | 65.0 / 0.25 | 89.4 / 0.53 |

First call of the learned policy: MedQA 174/500 questions (E 3, R 118, V 53), commit forced after 0; MMLU-Pro 176/500 questions (E 13, R 7, V 156), commit forced after 6; GPQA 55/100 questions (E 1, R 13, V 41), commit forced after 0; AQuA 188/254 questions (E 0, R 0, V 188), commit forced after 0.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .160; MMLU-Pro <.001; GPQA .233; AQuA .007. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA .002; MMLU-Pro <.001; GPQA .555; AQuA .007.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -4.8 [-7.8, -2.0], .002 | -8.4 [-11.2, -5.6], <.001 | -6.0 [-13.0, +1.0], .180 | -8.7 [-12.6, -5.1], <.001 | -6.9 [-8.7, -5.2], <.001 |
| Learned − learned selection sent to the Verifier | -1.8 [-4.0, +0.4], .150 | -0.6 [-1.4, +0.0], .250 | +1.0 [-2.0, +5.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -0.8 [-1.7, +0.1], .099 |
| Learned − confidence gate (dev threshold, matched) | -1.2 [-4.0, +1.6], .480 | +1.6 [-1.6, +4.8], .396 | -3.0 [-10.0, +4.0], .581 | +0.8 [-4.7, +6.3], .885 | +0.1 [-1.8, +2.0], 1.000 |
| Learned − confidence gate (dev threshold, best) | -4.4 [-7.2, -1.6], .003 | -8.0 [-10.8, -5.2], <.001 | -5.0 [-12.0, +2.0], .267 | -5.5 [-9.8, -1.2], .020 | -6.0 [-7.8, -4.2], <.001 |
| Learned − confidence gate (same number of calls) | -1.0 [-3.8, +1.8], .576 | +1.4 [-1.8, +4.6], .470 | +0.0 [-7.0, +7.0], 1.000 | +2.8 [-2.8, +8.3], .401 | +0.7 [-1.3, +2.6], .554 |
| Always Verifier: with − without the candidate | -1.6 [-4.4, +1.2], .322 | +6.2 [+2.8, +9.6], <.001 | -3.0 [-10.0, +5.0], .607 | +4.7 [+0.0, +9.4], .073 | +2.4 [+0.4, +4.3], .021 |
| Always Verifier − empty-output revision | +9.6 [+6.4, +13.0], <.001 | +22.0 [+17.8, +26.2], <.001 | +16.0 [+7.0, +25.0], .002 | +48.4 [+41.3, +55.1], <.001 | +21.9 [+19.4, +24.5], <.001 |
| Empty-output revision − candidate | -1.2 [-2.6, +0.0], .146 | +0.2 [-0.8, +1.2], 1.000 | +0.0 [+0.0, +0.0], 1.000 | +0.0 [-1.6, +1.6], 1.000 | -0.4 [-1.1, +0.3], .405 |
