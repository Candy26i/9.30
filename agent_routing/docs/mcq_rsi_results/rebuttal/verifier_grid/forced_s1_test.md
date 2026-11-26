### Original fixed-role baselines and equal-information rules: manager `s1`, pool `test`

Accuracy (%) / advisor calls per question; valid-answer rate in parentheses when below 100%. The original baselines are the submission's protocol (`evaluate --forced`): the manager does not write a draft before the forced call and the Verifier receives no candidate. The equal-information rows start from the manager's own draft and are scored by the forced-commit revision.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Original baseline: forced Verifier (no draft, no candidate) | 84.4 / 1.00 | 68.2 / 1.00 | 60.0 / 1.00 | 77.6 / 1.00 |
| Original baseline: forced Reasoner | 81.6 / 1.00 | 55.2 / 1.00 | 48.0 / 1.00 | 33.9 / 1.00 (93.3% valid) |
| Original baseline: forced Extractor | 76.8 / 1.00 (98.2% valid) | 51.0 / 1.00 (99.6% valid) | 41.0 / 1.00 | 24.4 / 1.00 (63.4% valid) |
| Original baseline: forced Extractor, Reasoner, Verifier | 84.0 / 3.00 | 70.0 / 3.00 | 56.0 / 3.00 | 79.1 / 3.00 |
| Equal information: candidate only | 73.4 / 0.00 | 50.8 / 0.00 | 41.0 / 0.00 | 36.2 / 0.00 |
| Equal information: learned calling (first call) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| Equal information: always Verifier, with the candidate | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 | 85.0 / 1.00 |
| Equal information: always Verifier, candidate removed | 84.2 / 1.00 | 69.2 / 1.00 | 61.0 / 1.00 | 78.3 / 1.00 |
| Equal information: always Reasoner | 81.2 / 1.00 | 55.6 / 1.00 | 45.0 / 1.00 | 37.0 / 1.00 |
| Equal information: always Extractor | 79.0 / 1.00 | 52.0 / 1.00 | 39.0 / 1.00 | 29.5 / 1.00 |

**Question by question** (the same questions, joined on the example id). First minus second, accuracy in pt [95% CI], bootstrap p / exact McNemar p; in parentheses the share of questions (%) with the same outcome under the two rules.

| Contrast | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) | Pooled (n=1354) |
|:--|--:|--:|--:|--:|--:|
| Learned calling − original forced Verifier | -5.6 [-9.0, -2.4], <.001 / .001 (85.6) | -5.0 [-9.2, -0.8], .020 / .026 (76.6) | -6.0 [-14.0, +2.0], .199 / .238 (82.0) | -0.4 [-6.3, +5.1], .957 / 1.000 (78.3) | -4.4 [-6.7, -2.1], <.001 / <.001 |
| Learned calling − original forced Extractor, Reasoner, Verifier (3 calls) | -5.2 [-8.4, -2.0], <.001 / .003 (86.0) | -6.8 [-11.0, -2.6], .001 / .002 (76.8) | -2.0 [-10.0, +6.0], .727 / .804 (84.0) | -2.0 [-7.5, +3.5], .532 / .583 (79.1) | -4.9 [-7.2, -2.7], <.001 / <.001 |
| Always Verifier with the candidate − original forced Verifier | -1.0 [-3.8, +1.8], .523 / .568 (90.2) | +4.2 [+0.8, +7.6], .016 / .020 (85.0) | -4.0 [-12.0, +4.0], .424 / .481 (82.0) | +7.5 [+2.8, +12.2], .001 / .003 (84.6) | +2.3 [+0.4, +4.2], .025 / .025 |
| Original forced Verifier − always Verifier, candidate removed | +0.2 [-0.8, +1.2], .843 / 1.000 (98.6) | -1.0 [-2.4, +0.4], .201 / .267 (97.4) | -1.0 [-6.0, +4.0], .834 / 1.000 (93.0) | -0.8 [-3.1, +1.2], .598 / .727 (96.9) | -0.5 [-1.4, +0.4], .262 / .311 |
| Original forced Reasoner − always Reasoner | +0.4 [-1.2, +2.0], .700 / .804 (96.8) | -0.4 [-2.8, +2.0], .823 / .871 (92.4) | +3.0 [-2.0, +8.0], .341 / .453 (93.0) | -3.1 [-7.9, +1.2], .196 / .229 (86.6) | -0.4 [-1.8, +1.0], .662 / .682 |
| Original forced Extractor − always Extractor | -2.2 [-4.4, +0.0], .058 / .071 (93.8) | -1.0 [-3.2, +1.0], .407 / .458 (94.2) | +2.0 [-4.0, +8.0], .644 / .754 (90.0) | -5.1 [-9.8, -0.4], .044 / .053 (84.6) | -2.0 [-3.5, -0.5], .010 / .012 |
