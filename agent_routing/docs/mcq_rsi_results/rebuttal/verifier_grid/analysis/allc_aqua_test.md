### allc_aqua_test

n = 254; the learned policy's first action is a call on 160 questions (extractor 0, reasoner 0, verifier 160). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.63 | 69.7 | after 4 of 160 calls (2.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.63 | 69.7 |  |
| Random questions, same calls per advisor | 0.63 | 67.0 | 95% of draws in [63.8, 70.1]; 0.057 of draws ≥ learned |
| Random questions, Verifier only | 0.63 | 67.0 | 95% of draws in [63.8, 70.1]; 0.057 of draws ≥ learned; 0.057 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.63 | 68.9 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 84.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.6 |  |
| Always Verifier, candidate removed | 1.00 | 79.1 |  |
| Always Reasoner | 1.00 | 39.4 |  |
| Always Extractor | 1.00 | 31.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 34.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.52 | 89.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -15.0 | [-19.7, -10.6] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.5 | [+0.8, +10.6] | 0.032 | 0.038 |
| evidence value: always_verifier - always_null | +50.0 | [+43.3, +56.7] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -2.4 | [-5.5, +0.8] | 0.160 | 0.210 |
| learned - confidence_matched | +0.8 | [-5.1, +7.1] | 0.867 | 0.899 |
