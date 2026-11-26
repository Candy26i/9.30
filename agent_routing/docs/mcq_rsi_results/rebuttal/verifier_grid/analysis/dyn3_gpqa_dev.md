### dyn3_gpqa_dev

n = 100; the learned policy's first action is a call on 61 questions (extractor 0, reasoner 24, verifier 37). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 42.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.61 | 52.0 | after 0 of 61 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.61 | 55.0 |  |
| Random questions, same calls per advisor | 0.61 | 48.5 | 95% of draws in [43.0, 54.0]; 0.150 of draws ≥ learned |
| Random questions, Verifier only | 0.61 | 51.1 | 95% of draws in [46.0, 56.0]; 0.431 of draws ≥ learned; 0.095 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.61 | 55.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.77 | 59.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 54.0 |  |
| Always Reasoner | 1.00 | 46.0 |  |
| Always Extractor | 1.00 | 44.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 44.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.28 | 70.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.0 | [-12.0, +2.0] | 0.206 | 0.267 |
| learned - learned_selection_verifier | -3.0 | [-8.0, +2.0] | 0.345 | 0.453 |
| candidate value: always_verifier - always_verifier_nocand | +3.0 | [-7.0, +13.0] | 0.631 | 0.701 |
| evidence value: always_verifier - always_null | +13.0 | [+3.0, +23.0] | 0.019 | 0.029 |
| revision without evidence: always_null - candidate | +2.0 | [+0.0, +5.0] | 0.275 | 0.500 |
| learned - confidence_matched | -3.0 | [-10.0, +4.0] | 0.477 | 0.581 |
