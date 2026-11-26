### base_aqua_dev

n = 254; the grammar-constrained first action is a call on 232 questions (extractor 232, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 27/254 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 34.6 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.91 | 33.9 | after 232 of 232 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.91 | 78.7 |  |
| Random questions, same calls per advisor | 0.91 | 32.5 | 95% of draws in [31.5, 33.5]; 0.018 of draws ≥ learned |
| Random questions, Verifier only | 0.91 | 77.8 | 95% of draws in [76.0, 79.5]; 1.000 of draws ≥ learned; 0.234 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.91 | 80.3 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.97 | 81.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.9 |  |
| Always Verifier, candidate removed | 1.00 | 73.6 |  |
| Always Reasoner | 1.00 | 38.6 |  |
| Always Extractor | 1.00 | 32.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 34.3 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.52 | 86.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -48.0 | [-54.3, -40.9] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -44.9 | [-51.2, -38.2] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | +8.3 | [+3.1, +13.4] | 0.002 | 0.002 |
| evidence value: always_verifier - always_null | +47.6 | [+40.6, +54.3] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-1.2, +0.0] | 0.733 | 1.000 |
| learned - confidence_matched | -46.5 | [-53.1, -39.8] | <0.001 | <0.001 |
