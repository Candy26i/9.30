### base_gpqa_dev

n = 100; the grammar-constrained first action is a call on 100 questions (extractor 100, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 0/100 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 43.0 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 1.00 | 43.0 | after 100 of 100 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 50.0 |  |
| Random questions, same calls per advisor | 1.00 | 43.0 | 95% of draws in [43.0, 43.0]; 1.000 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 50.0 | 95% of draws in [50.0, 50.0]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 50.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.66 | 55.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 50.0 |  |
| Always Verifier, candidate removed | 1.00 | 51.0 |  |
| Always Reasoner | 1.00 | 45.0 |  |
| Always Extractor | 1.00 | 43.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 44.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.24 | 67.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.0 | [-17.0, +3.0] | 0.206 | 0.248 |
| learned - learned_selection_verifier | -7.0 | [-17.0, +3.0] | 0.206 | 0.248 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-10.0, +8.0] | 0.918 | 1.000 |
| evidence value: always_verifier - always_null | +6.0 | [-4.0, +16.0] | 0.267 | 0.327 |
| revision without evidence: always_null - candidate | +1.0 | [+0.0, +3.0] | 0.730 | 1.000 |
| learned - confidence_matched | -7.0 | [-17.0, +3.0] | 0.206 | 0.248 |
