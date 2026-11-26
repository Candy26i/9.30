### seed3_medqa_test

n = 500; the learned policy's first action is a call on 179 questions (extractor 7, reasoner 134, verifier 38). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 73.2 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.36 | 79.0 | after 0 of 179 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.36 | 80.6 |  |
| Random questions, same calls per advisor | 0.36 | 76.5 | 95% of draws in [74.8, 78.2]; 0.005 of draws ≥ learned |
| Random questions, Verifier only | 0.36 | 77.4 | 95% of draws in [75.8, 79.0]; 0.041 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.36 | 81.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.34 | 81.0 | dev: 79.5 at 0.34 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.59 | 84.0 | dev: 85.5 at 0.65 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.81 | 84.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.8 |  |
| Always Verifier, candidate removed | 1.00 | 84.4 |  |
| Always Reasoner | 1.00 | 81.8 |  |
| Always Extractor | 1.00 | 79.4 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 72.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.18 | 91.2 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.8 | [-8.8, -2.8] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -1.6 | [-3.8, +0.6] | 0.186 | 0.215 |
| candidate value: always_verifier - always_verifier_nocand | +0.4 | [-2.2, +3.0] | 0.814 | 0.883 |
| evidence value: always_verifier - always_null | +12.2 | [+8.8, +15.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.6 | [-2.0, +0.8] | 0.473 | 0.581 |
| learned - confidence_matched | -2.0 | [-5.0, +0.8] | 0.196 | 0.229 |
| learned - confidence_dev_matched | -2.0 | [-5.0, +1.0] | 0.206 | 0.237 |
| learned - confidence_dev_best | -5.0 | [-8.0, -2.2] | <0.001 | <0.001 |
