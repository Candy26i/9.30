### sucd_gpqa_test

n = 100; the learned policy's first action is a call on 99 questions (extractor 54, reasoner 16, verifier 29). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 40.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.99 | 50.0 | after 0 of 99 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.99 | 57.0 |  |
| Random questions, same calls per advisor | 0.99 | 45.7 | 95% of draws in [42.0, 50.0]; 0.041 of draws ≥ learned |
| Random questions, Verifier only | 0.99 | 56.8 | 95% of draws in [56.0, 58.0]; 1.000 of draws ≥ learned; 0.807 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.99 | 57.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.93 | 57.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 57.0 |  |
| Always Reasoner | 1.00 | 48.0 |  |
| Always Extractor | 1.00 | 39.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 40.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.21 | 61.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.0 | [-14.0, -1.0] | 0.035 | 0.065 |
| learned - learned_selection_verifier | -7.0 | [-14.0, -1.0] | 0.035 | 0.065 |
| candidate value: always_verifier - always_verifier_nocand | +0.0 | [-7.0, +7.0] | 1.000 | 1.000 |
| evidence value: always_verifier - always_null | +17.0 | [+8.0, +26.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -7.0 | [-14.0, -1.0] | 0.035 | 0.065 |
