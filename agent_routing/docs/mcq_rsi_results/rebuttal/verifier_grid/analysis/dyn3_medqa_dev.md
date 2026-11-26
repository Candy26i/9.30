### dyn3_medqa_dev

n = 200; the learned policy's first action is a call on 28 questions (extractor 2, reasoner 26, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 74.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.14 | 79.0 | after 0 of 28 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.14 | 79.0 |  |
| Random questions, same calls per advisor | 0.14 | 75.5 | 95% of draws in [74.0, 77.5]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.14 | 75.6 | 95% of draws in [74.0, 77.5]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.14 | 77.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.75 | 85.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 85.0 |  |
| Always Verifier, candidate removed | 1.00 | 85.0 |  |
| Always Reasoner | 1.00 | 85.0 |  |
| Always Extractor | 1.00 | 78.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 73.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.18 | 92.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -6.0 | [-11.0, -1.5] | 0.014 | 0.023 |
| learned - learned_selection_verifier | +0.0 | [-2.5, +2.5] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +0.0 | [-4.0, +4.0] | 1.000 | 1.000 |
| evidence value: always_verifier - always_null | +11.5 | [+6.5, +16.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.5 | [-1.5, +0.0] | 0.720 | 1.000 |
| learned - confidence_matched | +2.0 | [-2.0, +6.0] | 0.390 | 0.454 |
