### seed2_aqua_test

n = 254; the learned policy's first action is a call on 190 questions (extractor 0, reasoner 0, verifier 190). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 35.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.75 | 76.0 | after 0 of 190 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.75 | 76.0 |  |
| Random questions, same calls per advisor | 0.75 | 71.7 | 95% of draws in [68.9, 74.8]; 0.005 of draws ≥ learned |
| Random questions, Verifier only | 0.75 | 71.7 | 95% of draws in [68.9, 74.8]; 0.005 of draws ≥ learned; 0.005 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.75 | 73.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.79 | 76.0 | dev: 76.4 at 0.79 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.93 | 80.7 | dev: 81.9 at 0.96 calls |
| Confidence-gated Verifier, best threshold on this pool | 1.00 | 83.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.9 |  |
| Always Verifier, candidate removed | 1.00 | 79.5 |  |
| Always Reasoner | 1.00 | 36.2 |  |
| Always Extractor | 1.00 | 31.9 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 35.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.54 | 89.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.9 | [-11.4, -4.3] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +4.3 | [-0.8, +9.1] | 0.102 | 0.117 |
| evidence value: always_verifier - always_null | +48.0 | [+40.9, +54.7] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [-2.0, +2.0] | 1.000 | 1.000 |
| learned - confidence_matched | +2.4 | [-2.8, +7.5] | 0.418 | 0.461 |
| learned - confidence_dev_matched | +0.0 | [-5.1, +5.1] | 1.000 | 1.000 |
| learned - confidence_dev_best | -4.7 | [-9.1, -0.4] | 0.032 | 0.043 |
