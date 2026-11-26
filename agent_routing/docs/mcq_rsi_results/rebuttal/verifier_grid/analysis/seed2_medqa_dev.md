### seed2_medqa_dev

n = 200; the learned policy's first action is a call on 80 questions (extractor 5, reasoner 39, verifier 36). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 70.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.40 | 81.0 | after 0 of 80 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.40 | 81.0 |  |
| Random questions, same calls per advisor | 0.40 | 75.8 | 95% of draws in [73.0, 79.0]; 0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.40 | 75.6 | 95% of draws in [73.0, 78.5]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.40 | 78.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.69 | 84.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.0 |  |
| Always Verifier, candidate removed | 1.00 | 85.0 |  |
| Always Reasoner | 1.00 | 85.5 |  |
| Always Extractor | 1.00 | 78.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 70.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.23 | 92.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -3.0 | [-7.0, +1.0] | 0.178 | 0.238 |
| learned - learned_selection_verifier | +0.0 | [-2.5, +3.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-5.5, +3.5] | 0.734 | 0.824 |
| evidence value: always_verifier - always_null | +14.0 | [+8.5, +19.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [-2.0, +2.0] | 1.000 | 1.000 |
| learned - confidence_matched | +3.0 | [-1.5, +7.5] | 0.222 | 0.263 |
