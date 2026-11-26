### base_mmlu_pro_test

n = 500; the grammar-constrained first action is a call on 456 questions (extractor 444, reasoner 0, verifier 12). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 2/500 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 44.2 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.91 | 45.4 | after 456 of 456 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.91 | 66.6 |  |
| Random questions, same calls per advisor | 0.91 | 47.1 | 95% of draws in [46.4, 48.0]; 1.000 of draws ≥ learned |
| Random questions, Verifier only | 0.91 | 68.1 | 95% of draws in [67.0, 69.2]; 1.000 of draws ≥ learned; 0.995 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.91 | 70.4 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.85 | 69.8 | dev: 70.0 at 0.88 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.71 | 68.6 | dev: 70.0 at 0.73 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.87 | 70.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 70.4 |  |
| Always Verifier, candidate removed | 1.00 | 67.6 |  |
| Always Reasoner | 1.00 | 52.6 |  |
| Always Extractor | 1.00 | 46.8 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 45.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.32 | 76.2 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -25.0 | [-29.4, -20.6] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -21.2 | [-25.4, -17.2] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | +2.8 | [-0.8, +6.6] | 0.143 | 0.165 |
| evidence value: always_verifier - always_null | +25.4 | [+21.2, +29.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.8 | [+0.2, +1.6] | 0.038 | 0.125 |
| learned - confidence_matched | -25.0 | [-29.4, -20.6] | <0.001 | <0.001 |
| learned - confidence_dev_matched | -24.4 | [-28.8, -20.0] | <0.001 | <0.001 |
| learned - confidence_dev_best | -23.2 | [-27.4, -19.0] | <0.001 | <0.001 |
