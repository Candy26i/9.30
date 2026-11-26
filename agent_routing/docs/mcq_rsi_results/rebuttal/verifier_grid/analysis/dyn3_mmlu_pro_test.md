### dyn3_mmlu_pro_test

n = 500; the learned policy's first action is a call on 134 questions (extractor 0, reasoner 5, verifier 129). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.27 | 64.2 | after 0 of 134 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.27 | 63.8 |  |
| Random questions, same calls per advisor | 0.27 | 56.8 | 95% of draws in [55.0, 58.6]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.27 | 57.0 | 95% of draws in [55.0, 58.8]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.27 | 61.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.81 | 73.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.4 |  |
| Always Verifier, candidate removed | 1.00 | 68.6 |  |
| Always Reasoner | 1.00 | 55.6 |  |
| Always Extractor | 1.00 | 52.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 79.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.2 | [-12.4, -6.2] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.4 | [+0.0, +1.0] | 0.283 | 0.500 |
| candidate value: always_verifier - always_verifier_nocand | +4.8 | [+1.4, +8.2] | 0.005 | 0.008 |
| evidence value: always_verifier - always_null | +22.0 | [+17.8, +26.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.4 | [-0.8, +1.6] | 0.592 | 0.727 |
| learned - confidence_matched | +3.2 | [+0.0, +6.2] | 0.052 | 0.060 |
