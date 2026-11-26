### dyn3_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 44 questions (extractor 0, reasoner 3, verifier 41). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 52.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.22 | 62.5 | after 0 of 44 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.22 | 63.5 |  |
| Random questions, same calls per advisor | 0.22 | 56.6 | 95% of draws in [54.0, 59.5]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.22 | 56.8 | 95% of draws in [54.0, 59.5]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.22 | 60.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.71 | 72.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 72.0 |  |
| Always Verifier, candidate removed | 1.00 | 68.5 |  |
| Always Reasoner | 1.00 | 57.5 |  |
| Always Extractor | 1.00 | 52.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 53.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.25 | 77.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.5 | [-15.0, -4.5] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -1.0 | [-2.5, +0.0] | 0.263 | 0.500 |
| candidate value: always_verifier - always_verifier_nocand | +3.5 | [-1.5, +8.5] | 0.187 | 0.230 |
| evidence value: always_verifier - always_null | +18.5 | [+12.0, +25.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.0 | [-1.0, +3.0] | 0.443 | 0.625 |
| learned - confidence_matched | +2.0 | [-2.5, +6.5] | 0.457 | 0.523 |
