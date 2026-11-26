### allc_medqa_test

n = 500; the learned policy's first action is a call on 117 questions (extractor 3, reasoner 56, verifier 58). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 499/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 73.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.23 | 77.2 | after 0 of 117 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.23 | 76.8 |  |
| Random questions, same calls per advisor | 0.23 | 75.2 | 95% of draws in [73.8, 76.6]; 0.005 of draws ≥ learned |
| Random questions, Verifier only | 0.23 | 75.4 | 95% of draws in [74.0, 76.8]; 0.012 of draws ≥ learned; 0.035 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.23 | 78.2 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.91 | 83.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.2 |  |
| Always Verifier, candidate removed | 1.00 | 84.6 |  |
| Always Reasoner | 1.00 | 81.6 |  |
| Always Extractor | 1.00 | 79.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.18 | 90.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -6.0 | [-8.8, -3.2] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.4 | [-1.2, +2.0] | 0.729 | 0.815 |
| candidate value: always_verifier - always_verifier_nocand | -1.4 | [-4.2, +1.2] | 0.351 | 0.382 |
| evidence value: always_verifier - always_null | +11.8 | [+8.4, +15.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -1.6 | [-3.4, +0.0] | 0.087 | 0.115 |
| learned - confidence_matched | -1.0 | [-3.6, +1.6] | 0.490 | 0.542 |
