### base_medqa_dev

n = 200; the grammar-constrained first action is a call on 195 questions (extractor 195, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 0/200 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 68.5 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 0.97 | 71.5 | after 195 of 195 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.97 | 83.5 |  |
| Random questions, same calls per advisor | 0.97 | 72.9 | 95% of draws in [72.5, 73.5]; 1.000 of draws ≥ learned |
| Random questions, Verifier only | 0.97 | 84.6 | 95% of draws in [83.5, 85.5]; 1.000 of draws ≥ learned; 0.998 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.97 | 85.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.88 | 85.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.0 |  |
| Always Verifier, candidate removed | 1.00 | 83.5 |  |
| Always Reasoner | 1.00 | 82.5 |  |
| Always Extractor | 1.00 | 73.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 69.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.23 | 91.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -13.5 | [-19.5, -7.5] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -12.0 | [-18.0, -6.5] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | +1.5 | [-3.0, +6.0] | 0.571 | 0.664 |
| evidence value: always_verifier - always_null | +16.0 | [+10.5, +22.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.5 | [+0.0, +1.5] | 0.729 | 1.000 |
| learned - confidence_matched | -13.5 | [-19.5, -7.5] | <0.001 | <0.001 |
