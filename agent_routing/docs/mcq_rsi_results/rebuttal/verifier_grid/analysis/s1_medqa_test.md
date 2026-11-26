### s1_medqa_test

n = 500; the learned policy's first action is a call on 183 questions (extractor 2, reasoner 109, verifier 72). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 73.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.37 | 78.8 | after 0 of 183 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.37 | 80.2 |  |
| Random questions, same calls per advisor | 0.37 | 76.6 | 95% of draws in [74.8, 78.4]; 0.008 of draws ≥ learned |
| Random questions, Verifier only | 0.37 | 77.1 | 95% of draws in [75.4, 78.8]; 0.026 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.37 | 79.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.31 | 79.2 | dev: 80.5 at 0.37 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.80 | 83.0 | dev: 84.5 at 0.89 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.98 | 83.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.4 |  |
| Always Verifier, candidate removed | 1.00 | 84.2 |  |
| Always Reasoner | 1.00 | 81.2 |  |
| Always Extractor | 1.00 | 79.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 72.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.17 | 90.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.6 | [-7.4, -1.8] | <0.001 | 0.001 |
| learned - learned_selection_verifier | -1.4 | [-3.4, +0.6] | 0.216 | 0.248 |
| candidate value: always_verifier - always_verifier_nocand | -0.8 | [-3.6, +2.0] | 0.624 | 0.672 |
| evidence value: always_verifier - always_null | +10.6 | [+7.2, +14.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.6 | [-2.0, +0.6] | 0.454 | 0.549 |
| learned - confidence_matched | -0.8 | [-3.6, +2.0] | 0.626 | 0.672 |
| learned - confidence_dev_matched | -0.4 | [-3.2, +2.4] | 0.827 | 0.888 |
| learned - confidence_dev_best | -4.2 | [-7.0, -1.6] | 0.001 | 0.002 |
