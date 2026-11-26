### seed2_mmlu_pro_test

n = 500; the learned policy's first action is a call on 183 questions (extractor 13, reasoner 9, verifier 161). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.37 | 65.4 | after 10 of 183 calls (5.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.37 | 65.8 |  |
| Random questions, same calls per advisor | 0.37 | 58.5 | 95% of draws in [56.6, 60.6]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.37 | 59.4 | 95% of draws in [57.4, 61.4]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.37 | 63.2 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.82 | 73.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.2 |  |
| Always Verifier, candidate removed | 1.00 | 68.8 |  |
| Always Reasoner | 1.00 | 56.6 |  |
| Always Extractor | 1.00 | 52.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 80.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.8 | [-10.6, -5.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.4 | [-1.0, +0.0] | 0.270 | 0.500 |
| candidate value: always_verifier - always_verifier_nocand | +4.4 | [+1.0, +7.8] | 0.011 | 0.017 |
| evidence value: always_verifier - always_null | +21.4 | [+17.2, +25.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.4 | [-0.8, +1.6] | 0.627 | 0.754 |
| learned - confidence_matched | +2.2 | [-1.0, +5.4] | 0.198 | 0.228 |
