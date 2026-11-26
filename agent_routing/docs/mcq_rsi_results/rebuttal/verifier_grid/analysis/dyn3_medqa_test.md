### dyn3_medqa_test

n = 500; the learned policy's first action is a call on 75 questions (extractor 0, reasoner 74, verifier 1). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 74.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.15 | 78.6 | after 0 of 75 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.15 | 78.4 |  |
| Random questions, same calls per advisor | 0.15 | 75.8 | 95% of draws in [74.6, 77.0]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.15 | 76.0 | 95% of draws in [75.0, 77.2]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.15 | 77.4 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.11 | 77.6 | dev: 77.0 at 0.14 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.72 | 82.8 | dev: 85.0 at 0.75 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.84 | 83.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.2 |  |
| Always Verifier, candidate removed | 1.00 | 84.8 |  |
| Always Reasoner | 1.00 | 81.4 |  |
| Always Extractor | 1.00 | 79.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 74.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.15 | 89.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.6 | [-7.4, -1.8] | 0.002 | 0.003 |
| learned - learned_selection_verifier | +0.2 | [-1.4, +2.0] | 0.906 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -1.6 | [-4.2, +1.0] | 0.266 | 0.302 |
| evidence value: always_verifier - always_null | +8.8 | [+5.6, +12.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-1.2, +0.4] | 0.460 | 0.625 |
| learned - confidence_matched | +1.2 | [-1.2, +3.6] | 0.355 | 0.405 |
| learned - confidence_dev_matched | +1.0 | [-1.2, +3.4] | 0.447 | 0.500 |
| learned - confidence_dev_best | -4.2 | [-7.0, -1.6] | 0.004 | 0.005 |
