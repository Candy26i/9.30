### seed1_medqa_test

n = 500; the learned policy's first action is a call on 174 questions (extractor 3, reasoner 118, verifier 53). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 74.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.35 | 78.0 | after 0 of 174 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.35 | 79.8 |  |
| Random questions, same calls per advisor | 0.35 | 77.0 | 95% of draws in [75.2, 78.8]; 0.160 of draws ≥ learned |
| Random questions, Verifier only | 0.35 | 77.3 | 95% of draws in [75.8, 79.0]; 0.242 of draws ≥ learned; 0.002 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.35 | 79.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.36 | 79.2 | dev: 77.5 at 0.36 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.68 | 82.4 | dev: 83.5 at 0.72 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.80 | 82.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 82.8 |  |
| Always Verifier, candidate removed | 1.00 | 84.4 |  |
| Always Reasoner | 1.00 | 81.4 |  |
| Always Extractor | 1.00 | 79.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 73.2 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.16 | 90.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.8 | [-7.8, -2.0] | 0.001 | 0.002 |
| learned - learned_selection_verifier | -1.8 | [-4.0, +0.4] | 0.118 | 0.150 |
| candidate value: always_verifier - always_verifier_nocand | -1.6 | [-4.4, +1.2] | 0.293 | 0.322 |
| evidence value: always_verifier - always_null | +9.6 | [+6.4, +13.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -1.2 | [-2.6, +0.0] | 0.095 | 0.146 |
| learned - confidence_matched | -1.0 | [-3.8, +1.8] | 0.529 | 0.576 |
| learned - confidence_dev_matched | -1.2 | [-4.0, +1.6] | 0.434 | 0.480 |
| learned - confidence_dev_best | -4.4 | [-7.2, -1.6] | 0.002 | 0.003 |
