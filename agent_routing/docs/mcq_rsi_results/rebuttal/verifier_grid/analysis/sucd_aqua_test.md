### sucd_aqua_test

n = 254; the learned policy's first action is a call on 244 questions (extractor 0, reasoner 0, verifier 244). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.96 | 83.1 | after 0 of 244 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.96 | 83.1 |  |
| Random questions, same calls per advisor | 0.96 | 82.1 | 95% of draws in [80.7, 83.5]; 0.111 of draws ≥ learned |
| Random questions, Verifier only | 0.96 | 82.1 | 95% of draws in [80.7, 83.5]; 0.111 of draws ≥ learned; 0.111 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.96 | 82.3 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 83.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.9 |  |
| Always Verifier, candidate removed | 1.00 | 78.7 |  |
| Always Reasoner | 1.00 | 39.0 |  |
| Always Extractor | 1.00 | 38.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 37.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.48 | 86.2 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -0.8 | [-2.0, +0.0] | 0.271 | 0.500 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.1 | [+0.8, +9.8] | 0.030 | 0.041 |
| evidence value: always_verifier - always_null | +46.1 | [+39.4, +52.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.8 | [-1.2, +2.8] | 0.539 | 0.688 |
