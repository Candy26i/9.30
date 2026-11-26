### rho2_medqa_test

n = 500; the learned policy's first action is a call on 251 questions (extractor 1, reasoner 110, verifier 140). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 72.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.50 | 78.8 | after 0 of 251 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.50 | 81.2 |  |
| Random questions, same calls per advisor | 0.50 | 77.4 | 95% of draws in [75.4, 79.4]; 0.096 of draws ≥ learned |
| Random questions, Verifier only | 0.50 | 78.1 | 95% of draws in [76.4, 80.0]; 0.254 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.50 | 82.2 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.90 | 83.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.8 |  |
| Always Verifier, candidate removed | 1.00 | 84.4 |  |
| Always Reasoner | 1.00 | 80.8 |  |
| Always Extractor | 1.00 | 78.4 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.18 | 90.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.0 | [-7.6, -2.4] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -2.4 | [-4.6, -0.2] | 0.033 | 0.043 |
| candidate value: always_verifier - always_verifier_nocand | -0.6 | [-3.2, +2.0] | 0.708 | 0.761 |
| evidence value: always_verifier - always_null | +12.4 | [+9.0, +16.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -1.0 | [-2.6, +0.4] | 0.244 | 0.302 |
| learned - confidence_matched | -3.4 | [-6.0, -1.0] | 0.011 | 0.014 |
