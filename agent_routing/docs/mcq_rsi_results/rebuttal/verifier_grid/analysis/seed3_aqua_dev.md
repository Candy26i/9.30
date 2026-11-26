### seed3_aqua_dev

n = 254; the learned policy's first action is a call on 191 questions (extractor 1, reasoner 0, verifier 190). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.75 | 72.0 | after 1 of 191 calls (0.5%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.75 | 72.0 |  |
| Random questions, same calls per advisor | 0.75 | 70.2 | 95% of draws in [67.3, 73.2]; 0.129 of draws ≥ learned |
| Random questions, Verifier only | 0.75 | 70.4 | 95% of draws in [67.7, 73.2]; 0.160 of draws ≥ learned; 0.160 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.75 | 74.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.95 | 81.1 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.1 |  |
| Always Verifier, candidate removed | 1.00 | 75.6 |  |
| Always Reasoner | 1.00 | 39.0 |  |
| Always Extractor | 1.00 | 34.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 39.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.47 | 85.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.1 | [-12.6, -5.9] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.5 | [+0.8, +10.6] | 0.031 | 0.044 |
| evidence value: always_verifier - always_null | +41.7 | [+35.0, +48.4] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.6 | [-0.4, +3.9] | 0.202 | 0.289 |
| learned - confidence_matched | -2.0 | [-6.7, +2.8] | 0.462 | 0.511 |
