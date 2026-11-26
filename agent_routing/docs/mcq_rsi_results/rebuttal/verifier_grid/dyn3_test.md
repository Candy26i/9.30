### Equal-information comparison: manager `dyn3`, pool `test`

Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses the same checkpoint, decoding and forced-commit revision, and makes at most one call.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Candidate only (no call) | 74.8 / 0.00 | 51.0 / 0.00 | 41.0 / 0.00 | 35.4 / 0.00 |
| Learned calling (first call, then commit) | 78.6 / 0.15 | 64.2 / 0.27 | 53.0 / 0.58 | 77.6 / 0.80 |
| Learned selection, Verifier only | 78.4 / 0.15 | 63.8 / 0.27 | 55.0 / 0.58 | 77.6 / 0.80 |
| Random questions, same calls per advisor | 75.8 / 0.15 [74.6, 77.0] | 56.8 / 0.27 [55.0, 58.6] | 48.1 / 0.58 [43.0, 53.0] | 74.6 / 0.80 [71.7, 77.6] |
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

Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor MedQA <.001; MMLU-Pro <.001; GPQA .042; AQuA .025. Verifier-only random subsets against the learned selection sent to the Verifier: MedQA <.001; MMLU-Pro <.001; GPQA .038; AQuA .025.

| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | MedQA | MMLU-Pro | GPQA | AQuA | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned − always Verifier (with candidate) | -4.6 [-7.4, -1.8], .003 | -9.2 [-12.4, -6.2], <.001 | -4.0 [-10.0, +2.0], .344 | -7.1 [-10.6, -3.9], <.001 | -6.7 [-8.4, -5.0], <.001 |
| Learned − learned selection sent to the Verifier | +0.2 [-1.4, +2.0], 1.000 | +0.4 [+0.0, +1.0], .500 | -2.0 [-6.0, +2.0], .625 | +0.0 [+0.0, +0.0], 1.000 | +0.1 [-0.7, +0.8], 1.000 |
| Learned − confidence gate (dev threshold, matched) | +1.0 [-1.2, +3.4], .500 | +5.2 [+2.0, +8.2], .002 | +1.0 [-6.0, +8.0], 1.000 | +1.6 [-2.4, +5.9], .572 | +2.7 [+1.0, +4.4], .003 |
| Learned − confidence gate (dev threshold, best) | -4.2 [-7.0, -1.6], .005 | -8.6 [-11.6, -5.6], <.001 | -4.0 [-10.0, +2.0], .344 | -4.7 [-8.3, -1.6], .012 | -5.9 [-7.7, -4.2], <.001 |
| Learned − confidence gate (same number of calls) | +1.2 [-1.2, +3.6], .405 | +3.2 [+0.0, +6.2], .060 | +3.0 [-5.0, +11.0], .607 | +0.8 [-3.1, +4.7], .845 | +2.0 [+0.3, +3.7], .028 |
| Always Verifier: with − without the candidate | -1.6 [-4.2, +1.0], .302 | +4.8 [+1.4, +8.2], .008 | -1.0 [-8.0, +7.0], 1.000 | +5.5 [+0.8, +10.2], .029 | +2.1 [+0.2, +4.0], .033 |
| Always Verifier − empty-output revision | +8.8 [+5.6, +12.2], <.001 | +22.0 [+17.8, +26.2], <.001 | +16.0 [+6.0, +25.0], .002 | +52.0 [+44.9, +58.7], <.001 | +22.3 [+19.7, +25.0], <.001 |
| Empty-output revision − candidate | -0.4 [-1.2, +0.4], .625 | +0.4 [-0.8, +1.6], .727 | +0.0 [+0.0, +0.0], 1.000 | -2.8 [-4.7, -0.8], .016 | -0.5 [-1.2, +0.1], .167 |
