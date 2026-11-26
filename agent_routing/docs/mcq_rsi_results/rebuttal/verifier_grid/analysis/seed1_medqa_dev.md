### seed1_medqa_dev

n = 200; the learned policy's first action is a call on 72 questions (extractor 3, reasoner 39, verifier 30). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 69.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.36 | 81.0 | after 0 of 72 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.36 | 80.5 |  |
| Random questions, same calls per advisor | 0.36 | 74.6 | 95% of draws in [71.5, 77.5]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.36 | 74.2 | 95% of draws in [71.5, 77.0]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.36 | 77.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.72 | 83.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.5 |  |
| Always Verifier, candidate removed | 1.00 | 85.0 |  |
| Always Reasoner | 1.00 | 85.5 |  |
| Always Extractor | 1.00 | 78.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 69.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.23 | 92.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.5 | [-7.0, +1.5] | 0.289 | 0.359 |
| learned - learned_selection_verifier | +0.5 | [-2.5, +3.5] | 0.887 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -1.5 | [-6.0, +3.0] | 0.592 | 0.664 |
| evidence value: always_verifier - always_null | +14.5 | [+9.0, +20.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [-2.5, +2.5] | 1.000 | 1.000 |
| learned - confidence_matched | +3.5 | [-0.5, +7.5] | 0.131 | 0.167 |
