### seed1_aqua_test

n = 254; the learned policy's first action is a call on 188 questions (extractor 0, reasoner 0, verifier 188). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 36.2 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.74 | 76.0 | after 0 of 188 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.74 | 76.0 |  |
| Random questions, same calls per advisor | 0.74 | 72.1 | 95% of draws in [68.9, 75.2]; 0.007 of draws ≥ learned |
| Random questions, Verifier only | 0.74 | 72.1 | 95% of draws in [68.9, 75.2]; 0.007 of draws ≥ learned; 0.007 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.74 | 73.2 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.77 | 75.2 | dev: 75.2 at 0.77 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.93 | 81.5 | dev: 81.5 at 0.95 calls |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 84.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.6 |  |
| Always Verifier, candidate removed | 1.00 | 79.9 |  |
| Always Reasoner | 1.00 | 37.0 |  |
| Always Extractor | 1.00 | 33.1 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 36.2 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.53 | 89.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -8.7 | [-12.6, -5.1] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +4.7 | [+0.0, +9.4] | 0.064 | 0.073 |
| evidence value: always_verifier - always_null | +48.4 | [+41.3, +55.1] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [-1.6, +1.6] | 1.000 | 1.000 |
| learned - confidence_matched | +2.8 | [-2.8, +8.3] | 0.364 | 0.401 |
| learned - confidence_dev_matched | +0.8 | [-4.7, +6.3] | 0.835 | 0.885 |
| learned - confidence_dev_best | -5.5 | [-9.8, -1.2] | 0.016 | 0.020 |
