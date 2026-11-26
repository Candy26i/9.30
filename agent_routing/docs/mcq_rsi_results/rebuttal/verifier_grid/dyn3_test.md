### Equal-information comparison: manager `dyn3`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 74.8 / 0.00 | 51.0 / 0.00 | 41.0 / 0.00 | 35.4 / 0.00 |
| Learned calling (first call, then commit) | 78.6 / 0.15 | 64.2 / 0.27 | 53.0 / 0.58 | 77.6 / 0.80 |
| Learned selection, Verifier only | 78.4 / 0.15 | 63.8 / 0.27 | 55.0 / 0.58 | 77.6 / 0.80 |
| Random subset, Verifier only (same budget) | 76.0 / 0.15 [75.0, 77.2] | 57.0 / 0.27 [55.0, 58.8] | 50.3 / 0.58 [45.0, 55.0] | 74.6 / 0.80 [71.7, 77.6] |
| Confidence-gated Verifier (dev threshold, matched call rate) | 77.6 / 0.11 | 59.0 / 0.21 | 52.0 / 0.68 | 76.0 / 0.78 |
| Confidence-gated Verifier (dev threshold, best dev accuracy) | 82.8 / 0.72 | 72.8 / 0.67 | 57.0 / 0.92 | 82.3 / 0.94 |
| Confidence-gated Verifier (same number of calls, in-pool ranking) | 77.4 / 0.15 | 61.0 / 0.27 | 50.0 / 0.58 | 76.8 / 0.80 |
| Always Verifier, with the candidate | 83.2 / 1.00 | 73.4 / 1.00 | 57.0 / 1.00 | 84.6 / 1.00 |
| Always Verifier, candidate removed | 84.8 / 1.00 | 68.6 / 1.00 | 58.0 / 1.00 | 79.1 / 1.00 |
| Always Reasoner | 81.4 / 1.00 | 55.6 / 1.00 | 47.0 / 1.00 | 36.6 / 1.00 |
| Always Extractor | 79.2 / 1.00 | 52.0 / 1.00 | 41.0 / 1.00 | 30.3 / 1.00 |
| Revision turn on an empty Verifier output | 74.4 / 0.00 | 51.4 / 0.00 | 41.0 / 0.00 | 32.7 / 0.00 |
| Oracle (best of candidate / one call) | 89.8 / 0.15 | 79.6 / 0.29 | 65.0 / 0.24 | 89.0 / 0.54 |

First call of the learned policy: MedQA 75/500 questions (E 0, R 74, V 1), commit forced after 0; MMLU-Pro 134/500 questions (E 0, R 5, V 129), commit forced after 0; GPQA 58/100 questions (E 0, R 21, V 37), commit forced after 0; AQuA 202/254 questions (E 0, R 0, V 202), commit forced after 0.

First turn equal to unconstrained greedy decoding: MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254.

| Paired difference in accuracy, pt [95% CI], bootstrap p / McNemar p | MedQA | MMLU-Pro | GPQA | AQuA |
|:--|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -4.6 [-7.4, -1.8], .002 / .003 | -9.2 [-12.4, -6.2], <.001 / <.001 | -4.0 [-10.0, +2.0], .259 / .344 | -7.1 [-10.6, -3.9], <.001 / <.001 |
| Learned − learned selection sent to the Verifier | +0.2 [-1.4, +2.0], .906 / 1.000 | +0.4 [+0.0, +1.0], .283 / .500 | -2.0 [-6.0, +2.0], .450 / .625 | +0.0 [+0.0, +0.0], 1.000 / 1.000 |
| Learned − confidence gate (dev threshold, matched) | +1.0 [-1.2, +3.4], .447 / .500 | +5.2 [+2.0, +8.2], <.001 / .002 | +1.0 [-6.0, +8.0], .887 / 1.000 | +1.6 [-2.4, +5.9], .504 / .572 |
| Learned − confidence gate (dev threshold, best) | -4.2 [-7.0, -1.6], .004 / .005 | -8.6 [-11.6, -5.6], <.001 / <.001 | -4.0 [-10.0, +2.0], .259 / .344 | -4.7 [-8.3, -1.6], .005 / .012 |
| Learned − confidence gate (same number of calls) | +1.2 [-1.2, +3.6], .355 / .405 | +3.2 [+0.0, +6.2], .052 / .060 | +3.0 [-5.0, +11.0], .510 / .607 | +0.8 [-3.1, +4.7], .768 / .845 |
| Always Verifier: with − without the candidate | -1.6 [-4.2, +1.0], .266 / .302 | +4.8 [+1.4, +8.2], .005 / .008 | -1.0 [-8.0, +7.0], .909 / 1.000 | +5.5 [+0.8, +10.2], .022 / .029 |
| Always Verifier − empty-output revision | +8.8 [+5.6, +12.2], <.001 / <.001 | +22.0 [+17.8, +26.2], <.001 / <.001 | +16.0 [+6.0, +25.0], .001 / .002 | +52.0 [+44.9, +58.7], <.001 / <.001 |
| Empty-output revision − candidate | -0.4 [-1.2, +0.4], .460 / .625 | +0.4 [-0.8, +1.6], .592 / .727 | +0.0 [+0.0, +0.0], 1.000 / 1.000 | -2.8 [-4.7, -0.8], .002 / .016 |
