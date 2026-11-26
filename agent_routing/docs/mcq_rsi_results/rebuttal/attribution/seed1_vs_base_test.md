### What training changed in the manager: `seed1` against the untrained `base`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The untrained manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 0/500; MMLU-Pro 2/500; GPQA 0/100; AQuA 27/254 questions: on its own it does not follow the draft / commit / call format, so its rows describe the base model under the same output grammar as the trained manager.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, accuracy (%) | 70.6 | 44.2 | 32.0 | 32.7 |
| Trained manager, accuracy (%) | 74.4 | 51.4 | 40.0 | 36.2 |
| Trained − untrained | +3.8 [+1.0, +6.6], .011 / .013 | +7.2 [+3.4, +11.0], <.001 / <.001 | +8.0 [+0.0, +16.0], .071 / .096 | +3.5 [-2.4, +9.4], .261 / .298 |
| Same candidate (%) | 86.2 | 61.4 | 67.0 | 59.4 |
| Reference: untrained model, plain prompt, letter only | 74.6 | 53.0 | 45.0 | 41.7 |
| Reference: trained manager's weights, plain prompt, letter only | 74.4 | 53.8 | 44.0 | 42.1 |

The untrained draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1354 questions: +5.3 [+3.2, +7.5], <.001 / <.001.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: untrained → trained; then trained − untrained.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Verifier, with the candidate | 85.4 → 85.4; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 78.5 → 78.8; +0.3 [-0.7, +1.6], .791 / 1.000 | 58.2 → 56.7; -1.5 [-4.5, +0.0], .739 / 1.000 | 82.8 → 84.1; +1.3 [-2.0, +4.6], .530 / .688 |
| Verifier, candidate removed | 86.3 → 86.3; +0.0 [-1.2, +1.2], 1.000 / 1.000 | 73.3 → 72.0; -1.3 [-2.6, -0.3], .034 / .125 | 62.7 → 62.7; +0.0 [-4.5, +4.5], 1.000 / 1.000 | 80.1 → 81.5; +1.3 [-1.3, +4.0], .445 / .625 |
| Reasoner | 84.0 → 85.2; +1.2 [-0.2, +2.8], .169 / .227 | 64.2 → 64.8; +0.7 [-2.0, +3.3], .688 / .804 | 47.8 → 47.8; +0.0 [-4.5, +4.5], 1.000 / 1.000 | 39.1 → 41.1; +2.0 [-3.3, +7.3], .546 / .629 |
| Extractor | 80.5 → 81.7; +1.2 [-0.9, +3.2], .327 / .383 | 62.5 → 62.5; +0.0 [-2.0, +2.0], 1.000 / 1.000 | 35.8 → 40.3; +4.5 [+0.0, +10.4], .103 / .250 | 36.4 → 33.8; -2.6 [-6.6, +1.3], .258 / .344 |
| Empty-output revision turn | 78.0 → 77.0; -0.9 [-1.9, -0.2], .034 / .125 | 62.5 → 62.2; -0.3 [-1.3, +0.7], .804 / 1.000 | 40.3 → 40.3; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 38.4 → 37.7; -0.7 [-2.0, +0.0], .744 / 1.000 |
| Verifier: wrong candidates fixed (%), untrained / trained | 47.4 / 47.4 | 47.0 / 47.8 | 35.0 / 35.0 | 76.3 / 80.6 |
| Verifier: right candidates broken (%), untrained / trained | 3.9 / 3.9 | 2.6 / 2.6 | 7.4 / 11.1 | 6.9 / 10.3 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 78.0 / 52.2; MMLU-Pro 62.5 / 33.7; GPQA 40.3 / 39.4; AQuA 38.4 / 33.0). Pooled over the 956 questions: +0.2 [-0.4, +0.8], .632 / .754.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Calls per question | 0.35 | 0.35 | 0.55 | 0.74 |
| The learned policy | 79.8 | 65.8 | 49.0 | 76.0 |
| The trained manager's confidence | 79.0 | 63.8 | 50.0 | 73.2 |
| The untrained manager's confidence in its own draft | 79.8 | 63.4 | 52.0 | 74.0 |
| Chance (95% of random draws) | 77.3 [75.8, 79.0] | 59.2 [57.2, 61.2] | 48.8 [44.0, 53.0] | 72.1 [68.9, 75.2] |
| Learned − base confidence | +0.0 [-2.0, +2.0], 1.000 / 1.000 | +2.4 [-0.8, +5.6], .143 / .169 | -3.0 [-8.0, +2.0], .339 / .453 | +2.0 [-2.8, +6.7], .480 / .522 |
| Learned − trained confidence | +0.8 [-1.2, +2.8], .463 / .541 | +2.0 [-1.2, +5.2], .247 / .275 | -1.0 [-7.0, +6.0], .899 / 1.000 | +2.8 [-2.8, +8.3], .364 / .401 |
| Trained confidence − base confidence | -0.8 [-2.4, +0.8], .403 / .481 | +0.4 [-2.6, +3.2], .844 / .892 | -2.0 [-9.0, +5.0], .659 / .774 | -0.8 [-5.1, +3.5], .782 / .856 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 77.0 [75.2, 78.8]; 0.160 | 58.4 [56.6, 60.4]; 0.000 | 47.6 [43.0, 53.0]; 0.233 | 72.1 [68.9, 75.2]; 0.007 |
| AUROC for a wrong trained draft: trained confidence | 0.728 | 0.771 | 0.614 | 0.590 |
| AUROC for a wrong trained draft: untrained confidence (in its own draft) | 0.732 | 0.726 | 0.510 | 0.600 |
| AUROC for a wrong trained draft: the learned call decision | 0.665 | 0.654 | 0.500 | 0.612 |
| Learned call rate: wrong / right candidates (%) | 59.4 / 26.3 | 51.0 / 20.2 | 55.0 / 55.0 | 82.1 / 59.8 |

Pooled:  learned − untrained confidence, over the 1354 questions: +1.0 [-0.6, +2.7], .248 / .265.  learned − trained confidence, over the 1354 questions: +1.5 [-0.4, +3.2], .120 / .125.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, candidate only | 70.6 / 0.00 | 44.2 / 0.00 | 32.0 / 0.00 | 32.7 / 0.00 |
| Untrained manager + Verifier on its least confident candidates | 80.4 / 0.35 | 58.4 / 0.35 | 48.0 / 0.55 | 70.9 / 0.74 |
| Untrained manager + Verifier below a confidence threshold fixed on dev | 80.4 / 0.31 | 58.6 / 0.35 | 49.0 / 0.70 | 72.8 / 0.78 |
| Trained manager, learned calling (first call) | 78.0 / 0.35 | 65.2 / 0.35 | 50.0 / 0.55 | 76.0 / 0.74 |
| Untrained manager + always Verifier | 83.6 / 1.00 | 70.4 / 1.00 | 52.0 / 1.00 | 81.9 / 1.00 |
| Trained manager + always Verifier | 82.8 / 1.00 | 73.6 / 1.00 | 56.0 / 1.00 | 84.6 / 1.00 |
| Learned calling − untrained manager with a confidence gate | -2.4 [-5.2, +0.4], .094 / .119 | +6.8 [+2.8, +11.0], <.001 / .001 | +2.0 [-5.0, +9.0], .657 / .774 | +5.1 [-0.8, +11.0], .096 / .111 |
| Learned calling − untrained manager with the dev-threshold gate | -2.4 [-5.2, +0.4], .111 / .134 | +6.6 [+2.6, +10.8], .002 / .002 | +1.0 [-7.0, +9.0], .887 / 1.000 | +3.1 [-2.4, +8.7], .304 / .341 |
| Always Verifier: trained − untrained | -0.8 [-2.4, +0.8], .383 / .454 | +3.2 [+0.4, +6.2], .031 / .040 | +4.0 [+0.0, +9.0], .129 / .219 | +2.8 [-0.4, +5.9], .104 / .143 |
|   of which: draft | +3.8 | +7.2 | +8.0 | +3.5 |
|   of which: what the same confidence-gated Verifier calls add (trained − untrained) | -5.2 | -1.8 | -6.0 | -1.2 |
|   of which: learned calling − the trained manager's own confidence gate | -1.0 | +1.4 | +0.0 | +2.8 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 10.3 / 8.4 | 39.2 / 22.2 | 18.2 / 16.0 | 53.7 / 48.4 |
| Reference: untrained manager deployed without the grammar (up to 3 calls) | 82.8 / 2.98 (valid answers 95%) | 69.0 / 2.51 (valid answers 85%) | 39.0 / 2.53 (valid answers 52%) | 78.3 / 1.62 (valid answers 84%) |
| Reference: trained manager deployed (Table 1; a second call is allowed) | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 |

The three "of which" rows add up exactly to the gap between learned calling and the untrained manager with a confidence gate. Pooled over the 1354 questions: +2.7 [+0.5, +4.9], .017 / .017. Always Verifier, trained − untrained, over the 1354 questions: +1.7 [+0.3, +3.1], .017 / .022.

Dev-threshold gate: the Verifier is called when the untrained manager's confidence in its draft is below a threshold fixed on the dev pool, the cut between two distinct dev confidences of that manager whose dev call count is closest to the trained policy's (MedQA 36.0% of dev questions called (learned policy on dev: 36.0%); MMLU-Pro 38.5% of dev questions called (learned policy on dev: 38.5%); GPQA 61.0% of dev questions called (learned policy on dev: 61.0%); AQuA 76.8% of dev questions called (learned policy on dev: 76.8%)). The gate's calls on this pool are whatever that threshold gives. Learned calling − the dev-threshold gate, over the 1354 questions: +2.2 [-0.1, +4.4], .057 / .057.
