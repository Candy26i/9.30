### s1_aqua_test

n = 254; the learned policy's first action is a call on 196 questions (extractor 0, reasoner 0, verifier 196). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 36.2 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.77 | 77.2 | after 1 of 196 calls (0.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.77 | 77.2 |  |
| Random questions, same calls per advisor | 0.77 | 73.9 | 95% of draws in [70.9, 76.8]; 0.017 of draws ≥ learned |
| Random questions, Verifier only | 0.77 | 73.9 | 95% of draws in [70.9, 76.8]; 0.017 of draws ≥ learned; 0.017 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.77 | 76.8 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 85.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.0 |  |
| Always Verifier, candidate removed | 1.00 | 78.3 |  |
| Always Reasoner | 1.00 | 37.0 |  |
| Always Extractor | 1.00 | 29.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 32.7 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.52 | 88.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.9 | [-11.4, -4.7] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +6.7 | [+2.0, +11.4] | 0.006 | 0.012 |
| evidence value: always_verifier - always_null | +52.4 | [+45.7, +59.1] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -3.5 | [-6.3, -0.8] | 0.012 | 0.022 |
| learned - confidence_matched | +0.4 | [-4.3, +5.1] | 0.942 | 1.000 |
