### allc_mmlu_pro_test

n = 500; the learned policy's first action is a call on 106 questions (extractor 9, reasoner 10, verifier 87). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.21 | 59.2 | after 8 of 106 calls (7.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.21 | 59.0 |  |
| Random questions, same calls per advisor | 0.21 | 55.7 | 95% of draws in [54.0, 57.4]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.21 | 56.3 | 95% of draws in [54.6, 58.0]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.21 | 59.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.89 | 73.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.2 |  |
| Always Verifier, candidate removed | 1.00 | 69.6 |  |
| Always Reasoner | 1.00 | 57.0 |  |
| Always Extractor | 1.00 | 52.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 80.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -14.0 | [-17.6, -10.4] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.2 | [-0.4, +1.0] | 0.789 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +3.6 | [+0.2, +7.0] | 0.041 | 0.050 |
| evidence value: always_verifier - always_null | +21.8 | [+17.8, +26.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-2.0, +1.2] | 0.729 | 0.815 |
| learned - confidence_matched | +0.2 | [-3.0, +3.4] | 0.939 | 1.000 |
