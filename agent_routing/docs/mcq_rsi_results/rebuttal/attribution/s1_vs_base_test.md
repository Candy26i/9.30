### What training changed in the manager: `s1` against the untrained `base`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The untrained manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 0/500; MMLU-Pro 2/500; GPQA 0/100; AQuA 27/254 questions: on its own it does not follow the draft / commit / call format, so its rows describe the base model under the same output grammar as the trained manager.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, accuracy (%) | 70.6 | 44.2 | 32.0 | 32.7 |
| Trained manager, accuracy (%) | 73.4 | 50.8 | 41.0 | 36.2 |
| Trained − untrained | +2.8 [+0.2, +5.4], .049 / .059 | +6.6 [+2.8, +10.4], .001 / <.001 | +9.0 [+1.0, +18.0], .047 / .064 | +3.5 [-2.4, +9.4], .285 / .314 |
| Same candidate (%) | 86.8 | 62.6 | 65.0 | 59.1 |
| Reference: untrained model, plain prompt, letter only | 74.6 | 53.0 | 45.0 | 41.7 |
| Reference: trained manager's weights, plain prompt, letter only | 74.4 | 53.8 | 44.0 | 42.1 |

The untrained draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1354 questions: +4.8 [+2.7, +6.9], <.001 / <.001.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: untrained → trained; then trained − untrained.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Verifier, with the candidate | 85.7 → 85.7; +0.0 [-0.7, +0.7], 1.000 / 1.000 | 77.6 → 77.6; +0.0 [-1.3, +1.3], 1.000 / 1.000 | 58.5 → 56.9; -1.5 [-4.6, +0.0], .738 / 1.000 | 84.7 → 86.0; +1.3 [-2.0, +4.7], .534 / .688 |
| Verifier, candidate removed | 85.9 → 85.9; +0.0 [-1.4, +1.4], 1.000 / 1.000 | 72.5 → 73.2; +0.6 [-1.0, +2.6], .592 / .727 | 63.1 → 64.6; +1.5 [-3.1, +7.7], .774 / 1.000 | 79.3 → 79.3; +0.0 [-2.7, +2.7], 1.000 / 1.000 |
| Reasoner | 84.1 → 84.6; +0.5 [-1.2, +2.1], .659 / .774 | 62.6 → 64.2; +1.6 [-1.3, +4.5], .315 / .383 | 47.7 → 44.6; -3.1 [-7.7, +0.0], .267 / .500 | 37.3 → 41.3; +4.0 [-1.3, +9.3], .165 / .210 |
| Extractor | 80.0 → 82.5; +2.5 [+0.5, +4.8], .023 / .035 | 61.0 → 61.0; +0.0 [-2.6, +2.6], 1.000 / 1.000 | 36.9 → 41.5; +4.6 [+0.0, +10.8], .098 / .250 | 35.3 → 31.3; -4.0 [-9.3, +1.3], .182 / .238 |
| Empty-output revision turn | 77.4 → 77.0; -0.5 [-1.4, +0.5], .453 / .625 | 61.0 → 61.3; +0.3 [+0.0, +1.0], .739 / 1.000 | 41.5 → 41.5; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 37.3 → 34.7; -2.7 [-6.0, +0.7], .134 / .219 |
| Verifier: wrong candidates fixed (%), untrained / trained | 50.0 / 50.0 | 46.7 / 47.5 | 34.2 / 34.2 | 77.7 / 81.9 |
| Verifier: right candidates broken (%), untrained / trained | 3.9 / 3.9 | 2.6 / 3.1 | 7.4 / 11.1 | 3.6 / 7.1 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 77.4 / 47.0; MMLU-Pro 61.0 / 33.7; GPQA 41.5 / 40.0; AQuA 37.3 / 34.6). Pooled over the 962 questions: +0.1 [-0.6, +0.8], .886 / 1.000.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Calls per question | 0.37 | 0.33 | 0.52 | 0.77 |
| The learned policy | 80.2 | 63.4 | 53.0 | 77.2 |
| The trained manager's confidence | 79.6 | 61.6 | 52.0 | 76.8 |
| The untrained manager's confidence in its own draft | 79.8 | 62.0 | 52.0 | 75.6 |
| Chance (95% of random draws) | 77.1 [75.4, 78.8] | 58.0 [56.0, 60.0] | 48.8 [44.0, 53.0] | 73.9 [70.9, 76.8] |
| Learned − base confidence | +0.4 [-1.6, +2.4], .790 / .851 | +1.4 [-1.8, +4.6], .434 / .464 | +1.0 [-3.0, +5.0], .843 / 1.000 | +1.6 [-3.1, +6.3], .551 / .618 |
| Learned − trained confidence | +0.6 [-1.4, +2.6], .617 / .690 | +1.8 [-1.4, +5.0], .310 / .342 | +1.0 [-5.0, +7.0], .876 / 1.000 | +0.4 [-4.3, +5.1], .942 / 1.000 |
| Trained confidence − base confidence | -0.2 [-1.8, +1.4], .889 / 1.000 | -0.4 [-3.4, +2.4], .845 / .892 | +0.0 [-6.0, +6.0], 1.000 / 1.000 | +1.2 [-2.4, +4.7], .593 / .678 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 76.6 [74.8, 78.4]; 0.008 | 57.2 [55.2, 59.2]; 0.000 | 47.7 [43.0, 52.0]; 0.005 | 73.9 [70.9, 76.8]; 0.017 |
| AUROC for a wrong trained draft: trained confidence | 0.751 | 0.774 | 0.597 | 0.632 |
| AUROC for a wrong trained draft: untrained confidence (in its own draft) | 0.741 | 0.720 | 0.474 | 0.616 |
| AUROC for a wrong trained draft: the learned call decision | 0.681 | 0.653 | 0.486 | 0.602 |
| Learned call rate: wrong / right candidates (%) | 63.2 / 27.0 | 48.8 / 18.1 | 50.8 / 53.7 | 84.6 / 64.1 |

Pooled:  learned − untrained confidence, over the 1354 questions: +1.0 [-0.7, +2.7], .253 / .265.  learned − trained confidence, over the 1354 questions: +1.0 [-0.7, +2.8], .259 / .279.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Untrained manager, candidate only | 70.6 / 0.00 | 44.2 / 0.00 | 32.0 / 0.00 | 32.7 / 0.00 |
| Untrained manager + Verifier on its least confident candidates | 80.6 / 0.37 | 57.6 / 0.33 | 47.0 / 0.52 | 71.7 / 0.77 |
| Untrained manager + Verifier below a confidence threshold fixed on dev | 80.6 / 0.32 | 56.0 / 0.28 | 49.0 / 0.71 | 73.6 / 0.79 |
| Trained manager, learned calling (first call) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| Untrained manager + always Verifier | 83.6 / 1.00 | 70.4 / 1.00 | 52.0 / 1.00 | 81.9 / 1.00 |
| Trained manager + always Verifier | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 | 85.0 / 1.00 |
| Learned calling − untrained manager with a confidence gate | -1.8 [-4.4, +0.8], .199 / .243 | +5.6 [+1.4, +9.8], .012 / .012 | +7.0 [+0.0, +14.0], .081 / .118 | +5.5 [-0.4, +11.0], .065 / .081 |
| Learned calling − untrained manager with the dev-threshold gate | -1.8 [-4.6, +0.8], .216 / .253 | +7.2 [+3.0, +11.6], .001 / .001 | +5.0 [-1.0, +12.0], .154 / .227 | +3.5 [-2.0, +9.1], .243 / .272 |
| Always Verifier: trained − untrained | -0.2 [-1.8, +1.6], .901 / 1.000 | +2.0 [-0.8, +5.0], .184 / .220 | +4.0 [-1.0, +10.0], .192 / .289 | +3.1 [+0.0, +6.3], .068 / .096 |
|   of which: draft | +2.8 | +6.6 | +9.0 | +3.5 |
|   of which: what the same confidence-gated Verifier calls add (trained − untrained) | -3.8 | -2.6 | -4.0 | +1.6 |
|   of which: learned calling − the trained manager's own confidence gate | -0.8 | +1.6 | +2.0 | +0.4 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 14.8 / 10.0 | 37.3 / 21.6 | 25.0 / 15.0 | 53.1 / 48.8 |
| Reference: untrained manager deployed without the grammar (up to 3 calls) | 82.8 / 2.98 (valid answers 95%) | 69.0 / 2.51 (valid answers 85%) | 39.0 / 2.53 (valid answers 52%) | 78.3 / 1.62 (valid answers 84%) |
| Reference: trained manager deployed (Table 1; a second call is allowed) | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 |

The three "of which" rows add up exactly to the gap between learned calling and the untrained manager with a confidence gate. Pooled over the 1354 questions: +3.0 [+0.7, +5.2], .011 / .011. Always Verifier, trained − untrained, over the 1354 questions: +1.6 [+0.1, +3.0], .038 / .044.

Dev-threshold gate: the untrained manager calls the Verifier when its confidence is below a threshold fixed on the dev pool, the cut between two distinct dev confidences whose dev call count is closest to the trained policy's (MedQA 37.0% of dev questions called (learned policy on dev: 37.0%); MMLU-Pro 30.5% of dev questions called (learned policy on dev: 30.5%); GPQA 62.0% of dev questions called (learned policy on dev: 62.0%); AQuA 78.0% of dev questions called (learned policy on dev: 78.0%)). Its calls on this pool are whatever that threshold gives. Learned calling − the dev-threshold gate, over the 1354 questions: +3.0 [+0.9, +5.2], .008 / .009.
