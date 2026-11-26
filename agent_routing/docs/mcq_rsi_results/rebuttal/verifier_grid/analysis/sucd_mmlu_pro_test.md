### sucd_mmlu_pro_test

n = 500; the learned policy's first action is a call on 499 questions (extractor 73, reasoner 59, verifier 367). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 50.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 1.00 | 73.2 | after 0 of 499 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 73.6 |  |
| Random questions, same calls per advisor | 1.00 | 67.8 | 95% of draws in [66.0, 69.6]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 73.6 | 95% of draws in [73.4, 73.6]; 1.000 of draws ≥ learned; 0.756 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 73.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.80 | 73.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.6 |  |
| Always Verifier, candidate removed | 1.00 | 67.6 |  |
| Always Reasoner | 1.00 | 53.0 |  |
| Always Extractor | 1.00 | 51.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 50.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.28 | 77.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -0.4 | [-2.2, +1.4] | 0.738 | 0.824 |
| learned - learned_selection_verifier | -0.4 | [-2.2, +1.4] | 0.738 | 0.824 |
| candidate value: always_verifier - always_verifier_nocand | +6.0 | [+2.8, +9.4] | <0.001 | <0.001 |
| evidence value: always_verifier - always_null | +23.6 | [+19.6, +27.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -0.4 | [-2.2, +1.4] | 0.738 | 0.824 |
