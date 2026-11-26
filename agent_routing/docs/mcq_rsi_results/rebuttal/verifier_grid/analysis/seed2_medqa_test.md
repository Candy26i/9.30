### seed2_medqa_test

n = 500; the learned policy's first action is a call on 191 questions (extractor 9, reasoner 122, verifier 60). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 74.6 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.38 | 78.6 | after 1 of 191 calls (0.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.38 | 80.0 |  |
| Random questions, same calls per advisor | 0.38 | 77.4 | 95% of draws in [75.6, 79.2]; 0.139 of draws ≥ learned |
| Random questions, Verifier only | 0.38 | 77.9 | 95% of draws in [76.2, 79.6]; 0.241 of draws ≥ learned; 0.011 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.38 | 80.2 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.39 | 80.2 | dev: 78.0 at 0.40 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.65 | 82.6 | dev: 84.0 at 0.69 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.80 | 83.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.2 |  |
| Always Verifier, candidate removed | 1.00 | 84.2 |  |
| Always Reasoner | 1.00 | 81.6 |  |
| Always Extractor | 1.00 | 79.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 73.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.16 | 91.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.6 | [-7.6, -1.6] | 0.003 | 0.004 |
| learned - learned_selection_verifier | -1.4 | [-3.6, +0.8] | 0.251 | 0.296 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-3.8, +1.8] | 0.535 | 0.576 |
| evidence value: always_verifier - always_null | +10.2 | [+6.8, +13.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -1.6 | [-3.0, -0.4] | 0.008 | 0.021 |
| learned - confidence_matched | -1.6 | [-4.6, +1.4] | 0.310 | 0.350 |
| learned - confidence_dev_matched | -1.6 | [-4.6, +1.4] | 0.310 | 0.350 |
| learned - confidence_dev_best | -4.0 | [-7.0, -1.2] | 0.009 | 0.009 |
