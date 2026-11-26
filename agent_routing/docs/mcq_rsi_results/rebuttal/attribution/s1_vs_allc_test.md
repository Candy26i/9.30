### What training changed in the manager: `s1` against the all-commit `allc`, pool `test`

Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt [95% CI], bootstrap p / McNemar p.

The all-commit manager's grammar-constrained first turn equals its unconstrained greedy turn on MedQA 499/500; MMLU-Pro 500/500; GPQA 100/100 questions.

**1. Draft (candidate before any call)**

| | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) |
|:--|--:|--:|--:|
| All-commit manager, accuracy (%) | 73.0 | 51.8 | 40.0 |
| Trained manager, accuracy (%) | 73.4 | 50.8 | 41.0 |
| Trained − all-commit | +0.4 [-1.2, +2.0], .720 / .804 | -1.0 [-3.0, +1.0], .377 / .424 | +1.0 [-3.0, +6.0], .823 / 1.000 |
| Same candidate (%) | 96.8 | 90.8 | 89.0 |

The all-commit draft is written under the manager prompt and output grammar; the letter-only rows are the same weights answering a plain multiple-choice prompt. Pooled over the 1100 questions: -0.2 [-1.4, +1.0], .821 / .883.

**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). Accuracy (%) after the forced call: all-commit → trained; then trained − all-commit.

| Forced call | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) |
|:--|--:|--:|--:|
| Verifier, with the candidate | 83.3 → 83.9; +0.6 [-0.2, +1.7], .248 / .375 | 74.0 → 73.3; -0.7 [-1.5, +0.0], .098 / .250 | 57.3 → 56.2; -1.1 [-3.4, +0.0], .745 / 1.000 |
| Verifier, candidate removed | 85.1 → 84.7; -0.4 [-1.0, +0.0], .279 / .500 | 70.7 → 70.0; -0.7 [-1.5, +0.0], .096 / .250 | 62.9 → 62.9; +0.0 [-3.4, +3.4], 1.000 / 1.000 |
| Reasoner | 82.2 → 81.6; -0.6 [-1.7, +0.2], .256 / .375 | 58.6 → 57.5; -1.1 [-2.6, +0.2], .174 / .227 | 46.1 → 46.1; +0.0 [+0.0, +0.0], 1.000 / 1.000 |
| Extractor | 80.4 → 79.5; -0.8 [-1.9, +0.0], .129 / .219 | 54.2 → 54.0; -0.2 [-1.3, +0.9], .865 / 1.000 | 39.3 → 40.4; +1.1 [-2.2, +5.6], .806 / 1.000 |
| Empty-output revision turn | 72.7 → 73.1; +0.4 [-0.6, +1.7], .590 / .727 | 53.7 → 54.0; +0.2 [-1.5, +2.0], .898 / 1.000 | 42.7 → 42.7; +0.0 [+0.0, +0.0], 1.000 / 1.000 |
| Verifier: wrong candidates fixed (%), all-commit / trained | 46.8 / 49.2 | 49.5 / 48.1 | 33.3 / 31.4 |
| Verifier: right candidates broken (%), all-commit / trained | 3.9 / 3.9 | 4.9 / 4.9 | 10.5 / 10.5 |

Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: MedQA 74.0 / 56.2; MMLU-Pro 53.7 / 21.7; GPQA 42.7 / 27.3). Pooled over the 1027 questions: -0.1 [-0.7, +0.5], .853 / 1.000.

**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number of calls; only the choice of questions differs). Accuracy (%).

| Questions chosen by | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) |
|:--|--:|--:|--:|
| Calls per question | 0.37 | 0.33 | 0.52 |
| The learned policy | 80.2 | 63.4 | 53.0 |
| The trained manager's confidence | 79.6 | 61.6 | 52.0 |
| The all-commit manager's confidence in its own draft | 80.6 | 63.0 | 51.0 |
| Chance (95% of random draws) | 77.1 [75.4, 78.8] | 58.0 [56.0, 60.0] | 48.8 [44.0, 53.0] |
| Learned − base confidence | -0.4 [-2.4, +1.6], .757 / .839 | +0.4 [-3.0, +3.8], .859 / .906 | +2.0 [-5.0, +9.0], .662 / .774 |
| Learned − trained confidence | +0.6 [-1.4, +2.6], .617 / .690 | +1.8 [-1.4, +5.0], .310 / .342 | +1.0 [-5.0, +7.0], .876 / 1.000 |
| Trained confidence − base confidence | -1.0 [-2.2, +0.0], .067 / .125 | -1.4 [-3.4, +0.6], .205 / .248 | +1.0 [-2.0, +4.0], .793 / 1.000 |
| Random questions, same calls per advisor; share of draws ≥ learned calling | 76.6 [74.8, 78.4]; 0.008 | 57.2 [55.2, 59.2]; 0.000 | 47.7 [43.0, 52.0]; 0.005 |
| AUROC for a wrong trained draft: trained confidence | 0.751 | 0.774 | 0.597 |
| AUROC for a wrong trained draft: all-commit confidence (in its own draft) | 0.765 | 0.789 | 0.566 |
| AUROC for a wrong trained draft: the learned call decision | 0.681 | 0.653 | 0.486 |
| Learned call rate: wrong / right candidates (%) | 63.2 / 27.0 | 48.8 / 18.1 | 50.8 / 53.7 |

Pooled:  learned − all-commit confidence, over the 1100 questions: +0.2 [-1.6, +2.0], .889 / .923.  learned − trained confidence, over the 1100 questions: +1.2 [-0.5, +3.0], .212 / .241.

**4. Whole systems.** Accuracy (%) / advisor calls per question.

| System | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) |
|:--|--:|--:|--:|
| All-commit manager, candidate only | 73.0 / 0.00 | 51.8 / 0.00 | 40.0 / 0.00 |
| All-commit manager + Verifier on its least confident candidates | 80.4 / 0.37 | 64.2 / 0.33 | 50.0 / 0.52 |
| Trained manager, learned calling (first call) | 78.8 / 0.37 | 63.2 / 0.33 | 54.0 / 0.52 |
| All-commit manager + always Verifier | 83.2 / 1.00 | 73.2 / 1.00 | 55.0 / 1.00 |
| Trained manager + always Verifier | 83.4 / 1.00 | 72.4 / 1.00 | 56.0 / 1.00 |
| Learned calling − all-commit manager with a confidence gate | -1.6 [-4.4, +1.2], .277 / .312 | -1.0 [-4.6, +2.6], .626 / .672 | +4.0 [-4.0, +12.0], .400 / .481 |
| Always Verifier: trained − all-commit | +0.2 [-1.0, +1.4], .872 / 1.000 | -0.8 [-2.4, +0.8], .384 / .454 | +1.0 [-3.0, +6.0], .803 / 1.000 |
|   of which: draft | +0.4 | -1.0 | +1.0 |
|   of which: what the same confidence-gated Verifier calls add (trained − all-commit) | -1.2 | -1.6 | +1.0 |
|   of which: learned calling − the trained manager's own confidence gate | -0.8 | +1.6 | +2.0 |
| Accuracy gained per advisor call (pt): learned calling / always Verifier (trained) | 14.8 / 10.0 | 37.3 / 21.6 | 25.0 / 15.0 |

The three "of which" rows add up exactly to the gap between learned calling and the all-commit manager with a confidence gate. Pooled over the 1100 questions: -0.8 [-3.0, +1.4], .490 / .521. Always Verifier, trained − all-commit, over the 1100 questions: -0.2 [-1.2, +0.7], .772 / .856.
