### seed3_medqa_dev

n = 200; the learned policy's first action is a call on 69 questions (extractor 6, reasoner 41, verifier 22). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 72.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.34 | 82.0 | after 0 of 69 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.34 | 82.5 |  |
| Random questions, same calls per advisor | 0.34 | 76.5 | 95% of draws in [73.5, 79.5]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.34 | 76.7 | 95% of draws in [74.0, 79.5]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.34 | 79.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.65 | 85.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.5 |  |
| Always Verifier, candidate removed | 1.00 | 85.5 |  |
| Always Reasoner | 1.00 | 85.5 |  |
| Always Extractor | 1.00 | 79.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 72.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.20 | 92.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -3.5 | [-8.0, +1.0] | 0.147 | 0.189 |
| learned - learned_selection_verifier | -0.5 | [-3.5, +2.5] | 0.875 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +0.0 | [-4.0, +4.0] | 1.000 | 1.000 |
| evidence value: always_verifier - always_null | +13.0 | [+8.0, +18.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.5 | [-1.5, +3.0] | 0.828 | 1.000 |
| learned - confidence_matched | +2.5 | [-2.0, +7.0] | 0.324 | 0.383 |
