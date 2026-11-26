### rho1_gpqa_test

n = 100; the learned policy's first action is a call on 82 questions (extractor 0, reasoner 9, verifier 73). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.82 | 51.0 | after 0 of 82 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.82 | 51.0 |  |
| Random questions, same calls per advisor | 0.82 | 50.5 | 95% of draws in [46.0, 55.0]; 0.492 of draws ≥ learned |
| Random questions, Verifier only | 0.82 | 51.0 | 95% of draws in [47.0, 54.0]; 0.606 of draws ≥ learned; 0.606 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.82 | 52.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.93 | 54.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 54.0 |  |
| Always Verifier, candidate removed | 1.00 | 57.0 |  |
| Always Reasoner | 1.00 | 49.0 |  |
| Always Extractor | 1.00 | 37.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 37.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.27 | 64.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -3.0 | [-8.0, +2.0] | 0.334 | 0.453 |
| learned - learned_selection_verifier | +0.0 | [-3.0, +3.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -3.0 | [-11.0, +5.0] | 0.555 | 0.629 |
| evidence value: always_verifier - always_null | +17.0 | [+8.0, +26.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -1.0 | [-5.0, +3.0] | 0.830 | 1.000 |
