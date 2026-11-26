### rho05_aqua_test

n = 254; the learned policy's first action is a call on 224 questions (extractor 0, reasoner 0, verifier 224). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 34.3 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.88 | 81.9 | after 1 of 224 calls (0.4%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.88 | 81.9 |  |
| Random questions, same calls per advisor | 0.88 | 78.7 | 95% of draws in [76.4, 81.1]; 0.005 of draws ≥ learned |
| Random questions, Verifier only | 0.88 | 78.7 | 95% of draws in [76.4, 81.1]; 0.005 of draws ≥ learned; 0.005 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.88 | 80.7 |  |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 84.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.6 |  |
| Always Verifier, candidate removed | 1.00 | 78.7 |  |
| Always Reasoner | 1.00 | 36.6 |  |
| Always Extractor | 1.00 | 29.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 35.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.56 | 89.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.8 | [-5.1, -0.8] | 0.015 | 0.039 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.9 | [+1.2, +10.6] | 0.018 | 0.024 |
| evidence value: always_verifier - always_null | +49.6 | [+42.5, +56.7] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.8 | [-1.2, +3.1] | 0.584 | 0.727 |
| learned - confidence_matched | +1.2 | [-2.0, +4.3] | 0.554 | 0.629 |
