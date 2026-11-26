### seed2_aqua_dev

n = 254; the learned policy's first action is a call on 201 questions (extractor 1, reasoner 1, verifier 199). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.79 | 74.8 | after 1 of 201 calls (0.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.79 | 74.8 |  |
| Random questions, same calls per advisor | 0.79 | 72.3 | 95% of draws in [69.7, 74.8]; 0.043 of draws ≥ learned |
| Random questions, Verifier only | 0.79 | 72.6 | 95% of draws in [70.1, 75.2]; 0.073 of draws ≥ learned; 0.073 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.79 | 76.4 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.96 | 81.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.9 |  |
| Always Verifier, candidate removed | 1.00 | 75.6 |  |
| Always Reasoner | 1.00 | 41.3 |  |
| Always Extractor | 1.00 | 34.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 37.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.48 | 85.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.1 | [-10.6, -4.3] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +6.3 | [+1.2, +11.4] | 0.016 | 0.023 |
| evidence value: always_verifier - always_null | +44.9 | [+38.6, +51.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-1.2, +0.0] | 0.721 | 1.000 |
| learned - confidence_matched | -1.6 | [-5.9, +2.8] | 0.555 | 0.597 |
