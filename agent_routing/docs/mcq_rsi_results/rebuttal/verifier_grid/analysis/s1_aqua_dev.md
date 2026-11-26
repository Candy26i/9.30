### s1_aqua_dev

n = 254; the learned policy's first action is a call on 198 questions (extractor 0, reasoner 0, verifier 198). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 39.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.78 | 73.2 | after 2 of 198 calls (1.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.78 | 73.2 |  |
| Random questions, same calls per advisor | 0.78 | 71.8 | 95% of draws in [69.3, 74.4]; 0.197 of draws ≥ learned |
| Random questions, Verifier only | 0.78 | 71.8 | 95% of draws in [69.3, 74.4]; 0.197 of draws ≥ learned; 0.197 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.78 | 76.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.97 | 81.1 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.1 |  |
| Always Verifier, candidate removed | 1.00 | 75.6 |  |
| Always Reasoner | 1.00 | 38.6 |  |
| Always Extractor | 1.00 | 34.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 38.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.48 | 86.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.9 | [-11.4, -4.7] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.5 | [+0.8, +10.6] | 0.031 | 0.044 |
| evidence value: always_verifier - always_null | +42.5 | [+36.2, +49.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-2.4, +1.6] | 0.860 | 1.000 |
| learned - confidence_matched | -2.8 | [-7.1, +1.6] | 0.248 | 0.296 |
