### base_mmlu_pro_dev

n = 200; the grammar-constrained first action is a call on 175 questions (extractor 173, reasoner 0, verifier 2). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 3/200 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 42.5 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.88 | 45.5 | after 175 of 175 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.88 | 66.0 |  |
| Random questions, same calls per advisor | 0.88 | 44.5 | 95% of draws in [43.5, 45.5]; 0.105 of draws ≥ learned |
| Random questions, Verifier only | 0.88 | 66.6 | 95% of draws in [64.5, 68.5]; 1.000 of draws ≥ learned; 0.765 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.88 | 70.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.73 | 70.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 70.0 |  |
| Always Verifier, candidate removed | 1.00 | 67.0 |  |
| Always Reasoner | 1.00 | 51.5 |  |
| Always Extractor | 1.00 | 44.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 43.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.30 | 72.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -24.5 | [-30.5, -18.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -20.5 | [-26.5, -14.5] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | +3.0 | [-1.5, +7.5] | 0.232 | 0.286 |
| evidence value: always_verifier - always_null | +27.0 | [+20.5, +33.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.5 | [+0.0, +1.5] | 0.743 | 1.000 |
| learned - confidence_matched | -24.5 | [-30.5, -18.0] | <0.001 | <0.001 |
