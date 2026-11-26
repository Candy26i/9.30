### Equal-information comparison: manager `base`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 70.6 / 0.00 | 44.2 / 0.00 | 32.0 / 0.00 | 32.7 / 0.00 |
| Learned calling (first call, then commit) | 73.6 / 0.99 | 45.4 / 0.91 | 31.0 / 1.00 | 29.5 / 0.91 |
| Learned selection, Verifier only | 83.4 / 0.99 | 66.6 / 0.91 | 52.0 / 1.00 | 78.7 / 0.91 |
| Random questions, same calls per advisor | 73.6 / 0.99 [73.4, 73.8] | 47.1 / 0.91 [46.4, 48.0] | 31.0 / 1.00 [31.0, 31.0] | 29.8 / 0.91 [28.7, 31.1] |
| Random subset, Verifier only (same budget) | 83.5 / 0.99 [83.2, 83.8] | 68.1 / 0.91 [67.0, 69.2] | 52.0 / 1.00 [52.0, 52.0] | 77.6 / 0.91 [75.6, 79.5] |
| Confidence-gated Verifier (dev threshold, matched call rate) | 83.6 / 0.95 | – | – | – |
| Confidence-gated Verifier (dev threshold, best dev accuracy) | 83.4 / 0.84 | – | – | – |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 83.6 / 0.99 | 70.4 / 0.91 | 52.0 / 1.00 | 78.7 / 0.91 |
| Always Verifier, with the candidate | 83.6 / 1.00 | 70.4 / 1.00 | 52.0 / 1.00 | 81.9 / 1.00 |
| Always Verifier, candidate removed | 84.0 / 1.00 | 67.6 / 1.00 | 55.0 / 1.00 | 77.6 / 1.00 |
| Always Reasoner | 80.0 / 1.00 | 52.6 / 1.00 | 48.0 / 1.00 | 33.1 / 1.00 |
| Always Extractor | 73.6 / 1.00 | 46.8 / 1.00 | 31.0 / 1.00 | 29.5 / 1.00 |
| Revision turn on an empty Verifier output | 70.8 / 0.00 | 45.0 / 0.00 | 32.0 / 0.00 | 33.1 / 0.00 |
| Oracle (best of candidate / one call) | 89.8 / 0.19 | 76.2 / 0.32 | 60.0 / 0.28 | 84.6 / 0.52 |

First call of the learned policy: MedQA 497/500 questions (E 497, R 0, V 0), commit forced after 497; MMLU-Pro 456/500 questions (E 444, R 0, V 12), commit forced after 456; GPQA 100/100 questions (E 100, R 0, V 0), commit forced after 100; AQuA 232/254 questions (E 232, R 0, V 0), commit forced after 232.

First turn equal to unconstrained greedy decoding: MedQA 0/500; MMLU-Pro 2/500; GPQA 0/100; AQuA 27/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .871; MMLU-Pro 1.000; GPQA 1.000; AQuA .758. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA .933; MMLU-Pro .995; GPQA 1.000; AQuA .170.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -10.0 [-13.6, -6.6], <.001 | -25.0 [-29.4, -20.6], <.001 | -21.0 [-30.0, -11.0], <.001 | -52.4 [-58.7, -45.7], <.001 | -24.3 [-27.0, -21.6], <.001 |
| Learned − learned selection sent to the Verifier | -9.8 [-13.4, -6.4], <.001 | -21.2 [-25.4, -17.2], <.001 | -21.0 [-30.0, -11.0], <.001 | -49.2 [-55.5, -42.9], <.001 | -22.2 [-24.8, -19.6], <.001 |
| Learned − confidence gate (dev threshold, matched) | -10.0 [-13.6, -6.6], <.001 | – | – | – | – |
| Learned − confidence gate (dev threshold, best) | -9.8 [-13.4, -6.4], <.001 | – | – | – | – |
| Learned − confidence gate (same number of calls) | -10.0 [-13.6, -6.6], <.001 | -25.0 [-29.4, -20.6], <.001 | -21.0 [-30.0, -11.0], <.001 | -49.2 [-55.9, -42.5], <.001 | -23.7 [-26.4, -21.0], <.001 |
| Always Verifier: with − without the candidate | -0.4 [-3.0, +2.2], .883 | +2.8 [-0.8, +6.6], .165 | -3.0 [-11.0, +4.0], .607 | +4.3 [-0.8, +9.4], .126 | +1.5 [-0.5, +3.5], .170 |
| Always Verifier − empty-output revision | +12.8 [+9.2, +16.6], <.001 | +25.4 [+21.2, +29.6], <.001 | +20.0 [+11.0, +29.0], <.001 | +48.8 [+42.1, +55.5], <.001 | +24.7 [+22.2, +27.4], <.001 |
| Empty-output revision − candidate | +0.2 [-0.4, +1.0], 1.000 | +0.8 [+0.2, +1.6], .125 | +0.0 [+0.0, +0.0], 1.000 | +0.4 [-0.8, +2.0], 1.000 | +0.4 [+0.0, +1.0], .109 |
