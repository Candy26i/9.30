### base_aqua_test

n = 254; the grammar-constrained first action is a call on 232 questions (extractor 232, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 27/254 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 32.7 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.91 | 29.5 | after 232 of 232 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.91 | 78.7 |  |
| Random questions, same calls per advisor | 0.91 | 29.8 | 95% of draws in [28.7, 31.1]; 0.758 of draws ≥ learned |
| Random questions, Verifier only | 0.91 | 77.6 | 95% of draws in [75.6, 79.5]; 1.000 of draws ≥ learned; 0.170 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.91 | 78.7 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.90 | 78.0 | dev: 80.3 at 0.91 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.96 | 80.7 | dev: 81.9 at 0.97 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.98 | 81.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.9 |  |
| Always Verifier, candidate removed | 1.00 | 77.6 |  |
| Always Reasoner | 1.00 | 33.1 |  |
| Always Extractor | 1.00 | 29.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 33.1 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.52 | 84.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -52.4 | [-58.7, -45.7] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -49.2 | [-55.5, -42.9] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | +4.3 | [-0.8, +9.4] | 0.103 | 0.126 |
| evidence value: always_verifier - always_null | +48.8 | [+42.1, +55.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.4 | [-0.8, +2.0] | 0.795 | 1.000 |
| learned - confidence_matched | -49.2 | [-55.9, -42.5] | <0.001 | <0.001 |
| learned - confidence_dev_matched | -48.4 | [-55.1, -41.7] | <0.001 | <0.001 |
| learned - confidence_dev_best | -51.2 | [-57.9, -44.5] | <0.001 | <0.001 |
