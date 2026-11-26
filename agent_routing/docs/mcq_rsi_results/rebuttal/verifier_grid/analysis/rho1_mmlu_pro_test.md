### rho1_mmlu_pro_test

n = 500; the learned policy's first action is a call on 166 questions (extractor 10, reasoner 6, verifier 150). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 499/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 48.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.33 | 62.0 | after 8 of 166 calls (4.8%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.33 | 62.6 |  |
| Random questions, same calls per advisor | 0.33 | 56.1 | 95% of draws in [54.2, 58.0]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.33 | 56.8 | 95% of draws in [54.8, 58.8]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.33 | 60.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.80 | 73.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 72.8 |  |
| Always Verifier, candidate removed | 1.00 | 68.0 |  |
| Always Reasoner | 1.00 | 55.0 |  |
| Always Extractor | 1.00 | 49.4 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 49.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.30 | 78.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -10.8 | [-13.8, -7.8] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.6 | [-1.4, +0.2] | 0.246 | 0.375 |
| candidate value: always_verifier - always_verifier_nocand | +4.8 | [+1.6, +8.2] | 0.008 | 0.007 |
| evidence value: always_verifier - always_null | +23.8 | [+19.6, +28.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.2 | [-1.2, +1.6] | 0.890 | 1.000 |
| learned - confidence_matched | +1.4 | [-2.0, +4.8] | 0.446 | 0.489 |
