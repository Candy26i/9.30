### rho0_aqua_test

n = 254; the learned policy's first action is a call on 254 questions (extractor 1, reasoner 4, verifier 249). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 30.3 |  |
| Learned calling, first call only (policy's advisor, then commit) | 1.00 | 85.0 | after 2 of 254 calls (0.8%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 85.4 |  |
| Random questions, same calls per advisor | 1.00 | 84.4 | 95% of draws in [83.5, 85.4]; 0.192 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 85.4 | 95% of draws in [85.4, 85.4]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 85.4 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 85.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.4 |  |
| Always Verifier, candidate removed | 1.00 | 78.0 |  |
| Always Reasoner | 1.00 | 35.0 |  |
| Always Extractor | 1.00 | 28.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 29.9 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.59 | 89.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -0.4 | [-1.2, +0.0] | 0.743 | 1.000 |
| learned - learned_selection_verifier | -0.4 | [-1.2, +0.0] | 0.743 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +7.5 | [+2.8, +12.2] | <0.001 | 0.003 |
| evidence value: always_verifier - always_null | +55.5 | [+48.8, +62.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-2.0, +1.2] | 0.827 | 1.000 |
| learned - confidence_matched | -0.4 | [-1.2, +0.0] | 0.743 | 1.000 |
