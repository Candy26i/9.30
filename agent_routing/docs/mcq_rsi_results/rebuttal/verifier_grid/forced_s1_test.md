### Original fixed-role baselines and equal-information rules: manager `s1`, pool `test`

Accuracy (%) / advisor calls per question; valid-answer rate in parentheses when below 100%. The original baselines are the submission's protocol (`evaluate --forced`): the manager does not write a draft before the forced call and the Verifier receives no candidate. The equal-information rows start from the manager's own draft and are scored by the forced-commit revision.

| Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Original baseline: forced Verifier (no draft, no candidate) | 84.4 / 1.00 | 68.2 / 1.00 | 60.0 / 1.00 | 77.6 / 1.00 |
| Original baseline: forced Reasoner | 81.6 / 1.00 | 55.2 / 1.00 | 48.0 / 1.00 | 33.9 / 1.00 (93% valid) |
| Original baseline: forced Extractor | 76.8 / 1.00 (98% valid) | 51.0 / 1.00 (100% valid) | 41.0 / 1.00 | 24.4 / 1.00 (63% valid) |
| Original baseline: forced Extractor, Reasoner, Verifier | 84.0 / 3.00 | 70.0 / 3.00 | 56.0 / 3.00 | 79.1 / 3.00 |
| Equal information: candidate only | 73.4 / 0.00 | 50.8 / 0.00 | 41.0 / 0.00 | 36.2 / 0.00 |
| Equal information: learned calling (first call) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| Equal information: always Verifier, with the candidate | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 | 85.0 / 1.00 |
| Equal information: always Verifier, candidate removed | 84.2 / 1.00 | 69.2 / 1.00 | 61.0 / 1.00 | 78.3 / 1.00 |
| Equal information: always Reasoner | 81.2 / 1.00 | 55.6 / 1.00 | 45.0 / 1.00 | 37.0 / 1.00 |
| Equal information: always Extractor | 79.0 / 1.00 | 52.0 / 1.00 | 39.0 / 1.00 | 29.5 / 1.00 |
