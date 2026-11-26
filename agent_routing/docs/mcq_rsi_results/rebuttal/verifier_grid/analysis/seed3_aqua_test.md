### seed3_aqua_test

n = 254; the learned policy's first action is a call on 183 questions (extractor 0, reasoner 0, verifier 183). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.72 | 75.6 | after 1 of 183 calls (0.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.72 | 75.6 |  |
| Random questions, same calls per advisor | 0.72 | 71.7 | 95% of draws in [68.5, 74.8]; 0.009 of draws ≥ learned |
| Random questions, Verifier only | 0.72 | 71.7 | 95% of draws in [68.5, 74.8]; 0.009 of draws ≥ learned; 0.009 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.72 | 74.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 85.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.0 |  |
| Always Verifier, candidate removed | 1.00 | 78.7 |  |
| Always Reasoner | 1.00 | 35.4 |  |
| Always Extractor | 1.00 | 29.9 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 35.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.51 | 88.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.4 | [-13.4, -5.9] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +6.3 | [+1.6, +11.0] | 0.008 | 0.014 |
| evidence value: always_verifier - always_null | +49.2 | [+42.5, +55.9] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -1.6 | [-4.3, +1.2] | 0.314 | 0.388 |
| learned - confidence_matched | +1.6 | [-3.9, +6.7] | 0.613 | 0.665 |
