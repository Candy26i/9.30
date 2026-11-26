### base_medqa_test

n = 500; the grammar-constrained first action is a call on 497 questions (extractor 497, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 0/500 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 70.6 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.99 | 73.6 | after 497 of 497 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.99 | 83.4 |  |
| Random questions, same calls per advisor | 0.99 | 73.6 | 95% of draws in [73.4, 73.8]; 0.871 of draws ≥ learned |
| Random questions, Verifier only | 0.99 | 83.5 | 95% of draws in [83.2, 83.8]; 1.000 of draws ≥ learned; 0.933 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.99 | 83.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.95 | 83.6 | dev: 85.0 at 0.97 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.84 | 83.4 | dev: 85.0 at 0.88 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.87 | 83.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.6 |  |
| Always Verifier, candidate removed | 1.00 | 84.0 |  |
| Always Reasoner | 1.00 | 80.0 |  |
| Always Extractor | 1.00 | 73.6 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 70.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.19 | 89.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -10.0 | [-13.6, -6.6] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -9.8 | [-13.4, -6.4] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | -0.4 | [-3.0, +2.2] | 0.840 | 0.883 |
| evidence value: always_verifier - always_null | +12.8 | [+9.2, +16.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.2 | [-0.4, +1.0] | 0.780 | 1.000 |
| learned - confidence_matched | -10.0 | [-13.6, -6.6] | <0.001 | <0.001 |
| learned - confidence_dev_matched | -10.0 | [-13.6, -6.6] | <0.001 | <0.001 |
| learned - confidence_dev_best | -9.8 | [-13.4, -6.4] | <0.001 | <0.001 |
