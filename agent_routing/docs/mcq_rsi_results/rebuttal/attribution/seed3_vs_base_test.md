### What training changed in the manager: `seed3` against the untrained `base`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The untrained manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 0/500; MMLU-Pro 2/500; GPQA 0/100; AQuA 27/254 questions: on its own it does not follow the draft / commit / call format, so its rows describe the base model under the same output grammar as the trained manager.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, accuracy (%) | 70.6 | 44.2 | 32.0 | 32.7 |
| Trained manager, accuracy (%) | 73.2 | 50.8 | 41.0 | 37.4 |
| Trained − untrained | +2.6 [-0.6, +5.8], .116 / .136 | +6.6 [+2.8, +10.4], .001 / <.001 | +9.0 [+1.0, +17.0], .047 / .064 | +4.7 [-1.2, +10.6], .130 / .148 |
| Same candidate (%) | 83.8 | 62.4 | 66.0 | 61.0 |
| Reference: untrained model, plain prompt, letter only | 74.6 | 53.0 | 45.0 | 41.7 |
| Reference: trained manager's weights, plain prompt, letter only | 74.4 | 53.8 | 44.0 | 42.1 |

The untrained draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1354 questions: +4.9 [+2.7, +7.2], <.001 / <.001.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: untrained → trained; then trained − untrained.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Verifier, with the candidate | 86.2 → 87.1; +1.0 [+0.2, +1.9], .033 / .125 | 77.9 → 78.2; +0.3 [-0.6, +1.6], .771 / 1.000 | 57.6 → 56.1; -1.5 [-4.5, +0.0], .744 / 1.000 | 84.5 → 85.8; +1.3 [-1.9, +4.5], .538 / .688 |
| Verifier, candidate removed | 86.6 → 86.6; +0.0 [-1.4, +1.4], 1.000 / 1.000 | 72.8 → 72.4; -0.3 [-1.9, +1.3], .859 / 1.000 | 62.1 → 62.1; +0.0 [-4.5, +4.5], 1.000 / 1.000 | 80.0 → 80.6; +0.6 [-1.3, +3.2], .797 / 1.000 |
| Reasoner | 83.8 → 84.5; +0.7 [-0.7, +2.1], .401 / .508 | 63.1 → 64.7; +1.6 [-1.0, +4.5], .286 / .359 | 48.5 → 47.0; -1.5 [-4.5, +0.0], .729 / 1.000 | 38.7 → 39.4; +0.6 [-3.9, +5.8], .883 / 1.000 |
| Extractor | 80.7 → 82.8; +2.1 [+0.0, +4.3], .055 / .078 | 61.5 → 61.5; +0.0 [-2.2, +2.2], 1.000 / 1.000 | 36.4 → 39.4; +3.0 [+0.0, +7.6], .260 / .500 | 36.1 → 29.0; -7.1 [-11.6, -3.2], <.001 / .003 |
| Empty-output revision turn | 77.8 → 77.1; -0.7 [-1.7, +0.0], .099 / .250 | 61.2 → 61.2; +0.0 [-1.3, +1.3], 1.000 / 1.000 | 40.9 → 40.9; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 38.7 → 36.8; -1.9 [-5.2, +0.6], .245 / .375 |
| Verifier: wrong candidates fixed (%), untrained / trained | 48.9 / 53.3 | 47.1 / 47.9 | 33.3 / 33.3 | 77.9 / 82.1 |
| Verifier: right candidates broken (%), untrained / trained | 3.4 / 3.4 | 2.6 / 2.6 | 7.4 / 11.1 | 5.0 / 8.3 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 78.0 / 48.1; MMLU-Pro 61.2 / 33.5; GPQA 40.9 / 41.2; AQuA 38.7 / 35.4). Pooled over the 952 questions: +0.6 [-0.1, +1.4], .132 / .180.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Calls per question | 0.36 | 0.37 | 0.73 | 0.72 |
| The learned policy | 80.6 | 65.4 | 52.0 | 75.6 |
| The trained manager's confidence | 81.0 | 64.6 | 53.0 | 74.0 |
| The untrained manager's confidence in its own draft | 80.2 | 64.4 | 51.0 | 74.0 |
| Chance (95% of random draws) | 77.4 [75.8, 79.0] | 59.3 [57.4, 61.2] | 51.2 [48.0, 55.0] | 71.7 [68.5, 74.8] |
| Learned − base confidence | +0.4 [-2.0, +2.8], .806 / .864 | +1.0 [-2.2, +4.2], .573 / .620 | +1.0 [-3.0, +5.0], .830 / 1.000 | +1.6 [-3.5, +6.7], .590 / .636 |
| Learned − trained confidence | -0.4 [-2.4, +1.6], .786 / .851 | +0.8 [-2.4, +4.0], .671 / .724 | -1.0 [-6.0, +3.0], .827 / 1.000 | +1.6 [-3.9, +6.7], .613 / .665 |
| Trained confidence − base confidence | +0.8 [-1.4, +2.8], .522 / .572 | +0.2 [-2.6, +3.0], .958 / 1.000 | +2.0 [-2.0, +6.0], .455 / .625 | +0.0 [-3.9, +3.9], 1.000 / 1.000 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 76.5 [74.8, 78.2]; 0.005 | 58.2 [56.4, 60.2]; 0.000 | 49.5 [45.0, 54.0]; 0.196 | 71.7 [68.5, 74.8]; 0.009 |
| AUROC for a wrong trained draft: trained confidence | 0.760 | 0.777 | 0.604 | 0.603 |
| AUROC for a wrong trained draft: untrained confidence (in its own draft) | 0.724 | 0.726 | 0.484 | 0.595 |
| AUROC for a wrong trained draft: the learned call decision | 0.673 | 0.656 | 0.457 | 0.621 |
| Learned call rate: wrong / right candidates (%) | 61.2 / 26.5 | 52.8 / 21.7 | 69.5 / 78.0 | 81.1 / 56.8 |

Pooled:  learned − untrained confidence, over the 1354 questions: +0.9 [-0.8, +2.7], .330 / .359.  learned − trained confidence, over the 1354 questions: +0.4 [-1.4, +2.1], .720 / .747.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, candidate only | 70.6 / 0.00 | 44.2 / 0.00 | 32.0 / 0.00 | 32.7 / 0.00 |
| Untrained manager + Verifier on its least confident candidates | 80.6 / 0.36 | 58.8 / 0.37 | 48.0 / 0.73 | 70.5 / 0.72 |
| Trained manager, learned calling (first call) | 79.0 / 0.36 | 64.8 / 0.37 | 52.0 / 0.73 | 75.6 / 0.72 |
| Untrained manager + always Verifier | 83.6 / 1.00 | 70.4 / 1.00 | 52.0 / 1.00 | 81.9 / 1.00 |
| Trained manager + always Verifier | 84.8 / 1.00 | 73.6 / 1.00 | 55.0 / 1.00 | 85.0 / 1.00 |
| Learned calling − untrained manager with a confidence gate | -1.6 [-4.4, +1.2], .292 / .332 | +6.0 [+2.0, +10.2], .008 / .006 | +4.0 [-2.0, +10.0], .248 / .344 | +5.1 [-0.8, +11.0], .097 / .111 |
| Always Verifier: trained − untrained | +1.2 [-0.6, +3.2], .244 / .286 | +3.2 [+0.4, +6.2], .033 / .040 | +3.0 [-1.0, +8.0], .241 / .375 | +3.1 [+0.0, +6.3], .065 / .096 |
|   of which: draft | +2.6 | +6.6 | +9.0 | +4.7 |
|   of which: what the same confidence-gated Verifier calls add (trained − untrained) | -2.2 | -0.8 | -4.0 | -1.2 |
|   of which: learned calling − the trained manager's own confidence gate | -2.0 | +0.2 | -1.0 | +1.6 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 16.2 / 11.6 | 37.8 / 22.8 | 15.1 / 14.0 | 53.0 / 47.6 |
| Reference: untrained manager deployed without the grammar (up to 3 calls) | 82.8 / 2.98 (valid answers 95%) | 69.0 / 2.51 (valid answers 85%) | 39.0 / 2.53 (valid answers 52%) | 78.3 / 1.62 (valid answers 84%) |
| Reference: trained manager deployed (Table 1; a second call is allowed) | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 |

The three "of which" rows add up exactly to the gap between learned calling and the untrained manager with a confidence gate. Pooled over the 1354 questions: +2.9 [+0.7, +5.1], .011 / .013. Always Verifier, trained − untrained, over the 1354 questions: +2.4 [+1.0, +3.9], .001 / .001.
