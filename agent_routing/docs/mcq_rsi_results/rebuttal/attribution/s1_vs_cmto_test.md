### What training changed in the manager: `s1` against the commit-only `cmto`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The commit-only manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 500/500; MMLU-Pro 500/500; GPQA 100/100; AQuA 254/254 questions.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Commit-only manager, accuracy (%) | 73.6 | 53.2 | 42.0 | 38.6 |
| Trained manager, accuracy (%) | 73.4 | 50.8 | 41.0 | 36.2 |
| Trained − commit-only | -0.2 [-2.4, +2.0], .940 / 1.000 | -2.4 [-5.0, +0.2], .071 / .088 | -1.0 [-5.0, +3.0], .841 / 1.000 | -2.4 [-6.3, +1.6], .291 / .327 |
| Same candidate (%) | 92.4 | 82.8 | 89.0 | 80.3 |

The commit-only draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1354 questions: -1.5 [-3.0, +0.0], .056 / .062.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: commit-only → trained; then trained − commit-only.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Verifier, with the candidate | 84.4 → 84.8; +0.4 [-0.6, +1.5], .540 / .688 | 78.0 → 76.3; -1.7 [-3.1, -0.5], .017 / .039 | 59.6 → 58.4; -1.1 [-5.6, +2.2], .771 / 1.000 | 85.3 → 84.8; -0.5 [-2.5, +1.5], .829 / 1.000 |
| Verifier, candidate removed | 85.3 → 85.5; +0.2 [-1.1, +1.7], .877 / 1.000 | 70.5 → 71.7; +1.2 [-0.5, +3.1], .235 / .302 | 60.7 → 62.9; +2.2 [-2.2, +6.7], .453 / .625 | 78.4 → 79.9; +1.5 [+0.0, +3.4], .098 / .250 |
| Reasoner | 82.9 → 83.5; +0.6 [-1.3, +2.8], .577 / .690 | 60.1 → 60.6; +0.5 [-2.2, +3.1], .807 / .860 | 50.6 → 47.2; -3.4 [-9.0, +1.1], .251 / .375 | 41.2 → 40.2; -1.0 [-5.4, +3.9], .774 / .839 |
| Extractor | 79.4 → 80.5; +1.1 [-0.9, +3.0], .327 / .383 | 58.0 → 57.7; -0.2 [-2.2, +1.7], .907 / 1.000 | 42.7 → 40.4; -2.2 [-5.6, +0.0], .265 / .500 | 40.2 → 32.4; -7.8 [-12.7, -2.9], .001 / .004 |
| Empty-output revision turn | 76.2 → 75.5; -0.6 [-1.7, +0.2], .234 / .375 | 57.7 → 58.0; +0.2 [+0.0, +0.7], .731 / 1.000 | 43.8 → 43.8; +0.0 [+0.0, +0.0], 1.000 / 1.000 | 40.2 → 36.8; -3.4 [-6.4, -1.0], .019 / .039 |
| Verifier: wrong candidates fixed (%), commit-only / trained | 44.5 / 47.3 | 51.4 / 50.9 | 32.0 / 34.0 | 78.7 / 80.3 |
| Verifier: right candidates broken (%), commit-only / trained | 3.1 / 3.4 | 2.5 / 5.0 | 5.1 / 10.3 | 4.9 / 8.5 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 76.2 / 39.5; MMLU-Pro 57.7 / 17.4; GPQA 43.8 / 18.2; AQuA 40.2 / 20.0). Pooled over the 1169 questions: -0.6 [-1.5, +0.2], .168 / .210.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Calls per question | 0.37 | 0.33 | 0.52 | 0.77 |
| The learned policy | 80.2 | 63.4 | 53.0 | 77.2 |
| The trained manager's confidence | 79.6 | 61.6 | 52.0 | 76.8 |
| The commit-only manager's confidence in its own draft | 80.6 | 62.8 | 49.0 | 75.6 |
| Chance (95% of random draws) | 77.1 [75.4, 78.8] | 58.0 [56.0, 60.0] | 48.8 [44.0, 53.0] | 73.9 [70.9, 76.8] |
| Learned − base confidence | -0.4 [-2.4, +1.6], .756 / .839 | +0.6 [-2.4, +3.6], .755 / .795 | +4.0 [-2.0, +10.0], .255 / .344 | +1.6 [-3.1, +5.9], .550 / .608 |
| Learned − trained confidence | +0.6 [-1.4, +2.6], .617 / .690 | +1.8 [-1.4, +5.0], .310 / .342 | +1.0 [-5.0, +7.0], .876 / 1.000 | +0.4 [-4.3, +5.1], .942 / 1.000 |
| Trained confidence − base confidence | -1.0 [-2.2, +0.2], .121 / .180 | -1.2 [-3.6, +1.2], .354 / .405 | +3.0 [-1.0, +7.0], .235 / .375 | +1.2 [-1.6, +4.3], .517 / .607 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 76.6 [74.8, 78.4]; 0.008 | 57.2 [55.2, 59.2]; 0.000 | 47.7 [43.0, 52.0]; 0.005 | 73.9 [70.9, 76.8]; 0.017 |
| AUROC for a wrong trained draft: trained confidence | 0.751 | 0.774 | 0.597 | 0.632 |
| AUROC for a wrong trained draft: commit-only confidence (in its own draft) | 0.775 | 0.781 | 0.599 | 0.660 |
| AUROC for a wrong trained draft: the learned call decision | 0.681 | 0.653 | 0.486 | 0.602 |
| Learned call rate: wrong / right candidates (%) | 63.2 / 27.0 | 48.8 / 18.1 | 50.8 / 53.7 | 84.6 / 64.1 |

Pooled:  learned − commit-only confidence, over the 1354 questions: +0.7 [-1.0, +2.3], .451 / .478.  learned − trained confidence, over the 1354 questions: +1.0 [-0.7, +2.8], .259 / .279.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| Commit-only manager, candidate only | 73.6 / 0.00 | 53.2 / 0.00 | 42.0 / 0.00 | 38.6 / 0.00 |
| Commit-only manager + Verifier on its least confident candidates | 80.2 / 0.37 | 66.0 / 0.33 | 49.0 / 0.52 | 74.8 / 0.77 |
| Commit-only manager + Verifier below a confidence threshold fixed on dev | 80.6 / 0.39 | 64.4 / 0.30 | 54.0 / 0.68 | 74.0 / 0.76 |
| Trained manager, learned calling (first call) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 | 77.2 / 0.77 |
| Commit-only manager + always Verifier | 83.2 / 1.00 | 74.6 / 1.00 | 57.0 / 1.00 | 84.3 / 1.00 |
| Trained manager + always Verifier | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 | 85.0 / 1.00 |
| Learned calling − commit-only manager with a confidence gate | -1.4 [-4.4, +1.4], .376 / .410 | -2.8 [-6.4, +0.8], .131 / .156 | +5.0 [-1.0, +12.0], .156 / .227 | +2.4 [-3.1, +7.5], .416 / .461 |
| Learned calling − commit-only manager with the dev-threshold gate | -1.8 [-4.8, +1.0], .247 / .272 | -1.2 [-4.8, +2.4], .539 / .590 | +0.0 [-5.0, +5.0], 1.000 / 1.000 | +3.1 [-2.4, +8.3], .275 / .312 |
| Always Verifier: trained − commit-only | +0.2 [-1.6, +2.0], .894 / 1.000 | -2.2 [-4.4, +0.0], .056 / .071 | -1.0 [-5.0, +3.0], .835 / 1.000 | +0.8 [-2.0, +3.5], .662 / .774 |
|   of which: draft | -0.2 | -2.4 | -1.0 | -2.4 |
|   of which: what the same confidence-gated Verifier calls add (trained − commit-only) | -0.4 | -2.0 | +4.0 | +4.3 |
|   of which: learned calling − the trained manager's own confidence gate | -0.8 | +1.6 | +2.0 | +0.4 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 14.8 / 10.0 | 37.3 / 21.6 | 25.0 / 15.0 | 53.1 / 48.8 |

The three "of which" rows add up exactly to the gap between learned calling and the commit-only manager with a confidence gate. Pooled over the 1354 questions: -0.7 [-2.8, +1.3], .498 / .518. Always Verifier, trained − commit-only, over the 1354 questions: -0.7 [-1.8, +0.5], .292 / .328.

Dev-threshold gate: the Verifier is called when the commit-only manager's confidence in its draft is below a threshold fixed on the dev pool, the cut between two distinct dev confidences of that manager whose dev call count is closest to the trained policy's (MedQA 37.0% of dev questions called (learned policy on dev: 37.0%); MMLU-Pro 30.5% of dev questions called (learned policy on dev: 30.5%); GPQA 62.0% of dev questions called (learned policy on dev: 62.0%); AQuA 78.0% of dev questions called (learned policy on dev: 78.0%)). The gate's calls on this pool are whatever that threshold gives. Learned calling − the dev-threshold gate, over the 1354 questions: -0.5 [-2.5, +1.5], .650 / .668.
