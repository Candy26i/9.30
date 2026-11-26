### What training changed in the manager: `seed2` against the untrained `base`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The untrained manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 0/500; MMLU-Pro 2/500; GPQA 0/100; AQuA 27/254 questions: on its own it does not follow the draft / commit / call format, so its rows describe the base model under the same output grammar as the trained manager.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, accuracy (%) | 70.6 | 44.2 | 32.0 | 32.7 |
| Trained manager, accuracy (%) | 74.6 | 51.4 | 43.0 | 35.8 |
| Trained − untrained | +4.0 [+1.2, +6.8], .008 / .007 | +7.2 [+3.4, +11.2], <.001 / <.001 | +11.0 [+2.0, +20.0], .018 / .027 | +3.1 [-2.8, +9.1], .316 / .358 |
| Same candidate (%) | 86.6 | 61.2 | 66.0 | 59.4 |
| Reference: untrained model, plain prompt, letter only | 74.6 | 53.0 | 45.0 | 41.7 |
| Reference: trained manager's weights, plain prompt, letter only | 74.4 | 53.8 | 44.0 | 42.1 |

The untrained draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1354 questions: +5.5 [+3.4, +7.7], <.001 / <.001.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: untrained → trained; then trained − untrained.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Verifier, with the candidate | 85.9 → 86.1; +0.2 [+0.0, +0.7], .741 / 1.000 | 77.8 → 78.1; +0.3 [-0.7, +1.6], .800 / 1.000 | 57.6 → 56.1; -1.5 [-4.5, +0.0], .744 / 1.000 | 82.8 → 84.1; +1.3 [-2.0, +4.6], .533 / .688 |
| Verifier, candidate removed | 86.4 → 86.1; -0.2 [-1.4, +0.9], .861 / 1.000 | 72.9 → 72.9; +0.0 [-2.0, +2.0], 1.000 / 1.000 | 62.1 → 62.1; +0.0 [-4.5, +4.5], 1.000 / 1.000 | 80.8 → 82.1; +1.3 [-1.3, +4.0], .445 / .625 |
| Reasoner | 84.1 → 85.0; +0.9 [-0.7, +2.5], .313 / .388 | 63.7 → 64.7; +1.0 [-1.6, +3.6], .543 / .629 | 48.5 → 47.0; -1.5 [-4.5, +0.0], .706 / 1.000 | 39.1 → 41.1; +2.0 [-3.3, +7.3], .555 / .629 |
| Extractor | 80.1 → 81.5; +1.4 [-0.7, +3.5], .221 / .263 | 62.4 → 62.7; +0.3 [-1.6, +2.3], .871 / 1.000 | 36.4 → 39.4; +3.0 [+0.0, +7.6], .259 / .500 | 36.4 → 32.5; -4.0 [-7.3, -1.3], .003 / .031 |
| Empty-output revision turn | 77.8 → 76.9; -0.9 [-1.8, -0.2], .036 / .125 | 62.1 → 62.1; +0.0 [-1.3, +1.3], 1.000 / 1.000 | 40.9 → 40.9; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 38.4 → 37.7; -0.7 [-2.0, +0.0], .725 / 1.000 |
| Verifier: wrong candidates fixed (%), untrained / trained | 49.5 / 50.5 | 45.7 / 46.6 | 33.3 / 33.3 | 76.3 / 80.6 |
| Verifier: right candidates broken (%), untrained / trained | 3.8 / 3.8 | 2.6 / 2.6 | 7.4 / 11.1 | 6.9 / 10.3 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 78.1 / 52.2; MMLU-Pro 62.1 / 34.5; GPQA 40.9 / 47.1; AQuA 38.4 / 32.0). Pooled over the 956 questions: +0.3 [-0.3, +1.0], .436 / .549.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Calls per question | 0.38 | 0.37 | 0.44 | 0.75 |
| The learned policy | 80.0 | 65.8 | 52.0 | 76.0 |
| The trained manager's confidence | 80.2 | 63.2 | 50.0 | 73.6 |
| The untrained manager's confidence in its own draft | 80.4 | 64.0 | 49.0 | 73.6 |
| Chance (95% of random draws) | 77.9 [76.2, 79.6] | 59.4 [57.4, 61.4] | 48.7 [44.0, 53.0] | 71.7 [68.9, 74.8] |
| Learned − base confidence | -0.4 [-2.4, +1.8], .763 / .851 | +1.8 [-1.4, +5.0], .281 / .321 | +3.0 [-2.0, +8.0], .338 / .453 | +2.4 [-2.4, +7.1], .381 / .418 |
| Learned − trained confidence | -0.2 [-2.2, +1.8], .919 / 1.000 | +2.6 [-0.6, +5.8], .124 / .148 | +2.0 [-5.0, +9.0], .666 / .774 | +2.4 [-2.8, +7.5], .418 / .461 |
| Trained confidence − base confidence | -0.2 [-2.0, +1.6], .913 / 1.000 | -0.8 [-3.4, +1.8], .608 / .665 | +1.0 [-4.0, +6.0], .859 / 1.000 | +0.0 [-3.9, +3.9], 1.000 / 1.000 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 77.4 [75.6, 79.2]; 0.139 | 58.5 [56.6, 60.6]; 0.000 | 47.9 [43.0, 53.0]; 0.141 | 71.7 [68.9, 74.8]; 0.005 |
| AUROC for a wrong trained draft: trained confidence | 0.722 | 0.766 | 0.565 | 0.604 |
| AUROC for a wrong trained draft: untrained confidence (in its own draft) | 0.734 | 0.727 | 0.490 | 0.608 |
| AUROC for a wrong trained draft: the learned call decision | 0.656 | 0.660 | 0.498 | 0.612 |
| Learned call rate: wrong / right candidates (%) | 61.4 / 30.3 | 53.1 / 21.0 | 43.9 / 44.2 | 82.8 / 60.4 |

Pooled:  learned − untrained confidence, over the 1354 questions: +1.2 [-0.5, +2.9], .191 / .201.  learned − trained confidence, over the 1354 questions: +1.5 [-0.3, +3.2], .117 / .123.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, candidate only | 70.6 / 0.00 | 44.2 / 0.00 | 32.0 / 0.00 | 32.7 / 0.00 |
| Untrained manager + Verifier on its least confident candidates | 80.6 / 0.38 | 58.6 / 0.37 | 44.0 / 0.44 | 71.3 / 0.75 |
| Untrained manager + Verifier below a confidence threshold fixed on dev | 80.4 / 0.35 | 58.8 / 0.37 | 49.0 / 0.64 | 74.0 / 0.80 |
| Trained manager, learned calling (first call) | 78.6 / 0.38 | 65.4 / 0.37 | 51.0 / 0.44 | 76.0 / 0.75 |
| Untrained manager + always Verifier | 83.6 / 1.00 | 70.4 / 1.00 | 52.0 / 1.00 | 81.9 / 1.00 |
| Trained manager + always Verifier | 83.2 / 1.00 | 73.2 / 1.00 | 56.0 / 1.00 | 83.9 / 1.00 |
| Learned calling − untrained manager with a confidence gate | -2.0 [-4.8, +0.8], .172 / .212 | +6.8 [+2.6, +11.0], .002 / .002 | +7.0 [-1.0, +15.0], .093 / .143 | +4.7 [-1.2, +10.6], .122 / .141 |
| Learned calling − untrained manager with the dev-threshold gate | -1.8 [-4.6, +1.0], .222 / .262 | +6.6 [+2.4, +10.8], .003 / .003 | +2.0 [-6.0, +10.0], .694 / .804 | +2.0 [-3.5, +7.5], .521 / .568 |
| Always Verifier: trained − untrained | -0.4 [-2.0, +1.2], .732 / .815 | +2.8 [+0.0, +5.6], .058 / .076 | +4.0 [+0.0, +9.0], .126 / .219 | +2.0 [-1.2, +5.1], .265 / .332 |
|   of which: draft | +4.0 | +7.2 | +11.0 | +3.1 |
|   of which: what the same confidence-gated Verifier calls add (trained − untrained) | -4.4 | -2.6 | -5.0 | -0.8 |
|   of which: learned calling − the trained manager's own confidence gate | -1.6 | +2.2 | +1.0 | +2.4 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 10.5 / 8.6 | 38.3 / 21.8 | 18.2 / 13.0 | 53.7 / 48.0 |
| Reference: untrained manager deployed without the grammar (up to 3 calls) | 82.8 / 2.98 (valid answers 95%) | 69.0 / 2.51 (valid answers 85%) | 39.0 / 2.53 (valid answers 52%) | 78.3 / 1.62 (valid answers 84%) |
| Reference: trained manager deployed (Table 1; a second call is allowed) | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 |

The three "of which" rows add up exactly to the gap between learned calling and the untrained manager with a confidence gate. Pooled over the 1354 questions: +3.2 [+0.9, +5.4], .006 / .007. Always Verifier, trained − untrained, over the 1354 questions: +1.6 [+0.1, +3.0], .033 / .040.

Dev-threshold gate: the Verifier is called when the untrained manager's confidence in its draft is below a threshold fixed on the dev pool, the cut between two distinct dev confidences of that manager whose dev call count is closest to the trained policy's (MedQA 40.0% of dev questions called (learned policy on dev: 40.0%); MMLU-Pro 41.0% of dev questions called (learned policy on dev: 41.0%); GPQA 56.0% of dev questions called (learned policy on dev: 56.0%); AQuA 79.1% of dev questions called (learned policy on dev: 79.1%)). The gate's calls on this pool are whatever that threshold gives. Learned calling − the dev-threshold gate, over the 1354 questions: +2.3 [+0.1, +4.5], .045 / .048.
