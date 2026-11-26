### Equal-information comparison: manager `s1`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 73.4 / 0.00 | 50.8 / 0.00 | 41.0 / 0.00 | 36.2 / 0.00 |
| Learned calling (first call, then commit) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| Learned selection, Verifier only | 80.2 / 0.37 | 63.4 / 0.33 | 53.0 / 0.52 | 77.2 / 0.77 |
| Random questions, same calls per advisor | 76.6 / 0.37 [74.8, 78.4] | 57.2 / 0.33 [55.2, 59.2] | 47.7 / 0.52 [43.0, 52.0] | 73.9 / 0.77 [70.9, 76.8] |
| Random subset, Verifier only (same budget) | 77.1 / 0.37 [75.4, 78.8] | 58.0 / 0.33 [56.0, 60.0] | 48.8 / 0.52 [44.0, 53.0] | 73.9 / 0.77 [70.9, 76.8] |
| Confidence-gated Verifier (dev threshold, matched call rate) | 79.2 / 0.31 | 58.0 / 0.24 | 53.0 / 0.71 | 76.0 / 0.75 |
| Confidence-gated Verifier (dev threshold, best dev accuracy) | 83.0 / 0.80 | 71.4 / 0.68 | 55.0 / 0.84 | 81.9 / 0.93 |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 79.6 / 0.37 | 61.6 / 0.33 | 52.0 / 0.52 | 76.8 / 0.77 |
| Always Verifier, with the candidate | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 | 85.0 / 1.00 |
| Always Verifier, candidate removed | 84.2 / 1.00 | 69.2 / 1.00 | 61.0 / 1.00 | 78.3 / 1.00 |
| Always Reasoner | 81.2 / 1.00 | 55.6 / 1.00 | 45.0 / 1.00 | 37.0 / 1.00 |
| Always Extractor | 79.0 / 1.00 | 52.0 / 1.00 | 39.0 / 1.00 | 29.5 / 1.00 |
| Revision turn on an empty Verifier output | 72.8 / 0.00 | 51.4 / 0.00 | 41.0 / 0.00 | 32.7 / 0.00 |
| Oracle (best of candidate / one call) | 90.4 / 0.17 | 79.6 / 0.29 | 63.0 / 0.22 | 88.6 / 0.52 |

First call of the learned policy: MedQA 183/500 questions (E 2, R 109, V 72), commit forced after 0; MMLU-Pro 166/500 questions (E 10, R 12, V 144), commit forced after 8; GPQA 52/100 questions (E 1, R 9, V 42), commit forced after 0; AQuA 196/254 questions (E 0, R 0, V 196), commit forced after 1.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 499/500; GPQA 100/100; AQuA 254/254.

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA .008; MMLU-Pro <.001; GPQA .005; AQuA .017. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA <.001; MMLU-Pro <.001; GPQA .052; AQuA .017.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -4.6 [-7.4, -1.8], .001 | -9.2 [-12.2, -6.2], <.001 | -2.0 [-9.0, +5.0], .774 | -7.9 [-11.4, -4.7], <.001 | -6.7 [-8.4, -5.0], <.001 |
| Learned − learned selection sent to the Verifier | -1.4 [-3.4, +0.6], .248 | -0.2 [-1.2, +0.6], 1.000 | +1.0 [-2.0, +4.0], 1.000 | +0.0 [+0.0, +0.0], 1.000 | -0.5 [-1.4, +0.4], .311 |
| Learned − confidence gate (dev threshold, matched) | -0.4 [-3.2, +2.4], .888 | +5.2 [+1.8, +8.6], .004 | +1.0 [-5.0, +7.0], 1.000 | +1.2 [-3.5, +5.9], .755 | +2.1 [+0.1, +4.0], .043 |
| Learned − confidence gate (dev threshold, best) | -4.2 [-7.0, -1.6], .002 | -8.2 [-11.4, -5.2], <.001 | -1.0 [-8.0, +5.0], 1.000 | -4.7 [-8.7, -0.8], .029 | -5.5 [-7.3, -3.8], <.001 |
| Learned − confidence gate (same number of calls) | -0.8 [-3.6, +2.0], .672 | +1.6 [-1.6, +5.0], .416 | +2.0 [-4.0, +8.0], .754 | +0.4 [-4.3, +5.1], 1.000 | +0.5 [-1.3, +2.4], .648 |
| Always Verifier: with − without the candidate | -0.8 [-3.6, +2.0], .672 | +3.2 [-0.2, +6.6], .089 | -5.0 [-13.0, +3.0], .332 | +6.7 [+2.0, +11.4], .012 | +1.8 [-0.2, +3.8], .091 |
| Always Verifier − empty-output revision | +10.6 [+7.2, +14.2], <.001 | +21.0 [+17.0, +25.2], <.001 | +15.0 [+6.0, +24.0], .003 | +52.4 [+45.7, +59.1], <.001 | +22.6 [+20.1, +25.3], <.001 |
| Empty-output revision − candidate | -0.6 [-2.0, +0.6], .549 | +0.6 [+0.0, +1.4], .250 | +0.0 [+0.0, +0.0], 1.000 | -3.5 [-6.3, -0.8], .022 | -0.7 [-1.4, +0.1], .122 |
