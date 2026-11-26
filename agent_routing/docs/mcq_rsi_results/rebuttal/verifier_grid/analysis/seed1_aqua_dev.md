### seed1_aqua_dev

n = 254; the learned policy's first action is a call on 195 questions (extractor 0, reasoner 0, verifier 195). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 37.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.77 | 74.0 | after 0 of 195 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.77 | 74.0 |  |
| Random questions, same calls per advisor | 0.77 | 71.4 | 95% of draws in [68.5, 74.4]; 0.045 of draws ≥ learned |
| Random questions, Verifier only | 0.77 | 71.4 | 95% of draws in [68.5, 74.4]; 0.045 of draws ≥ learned; 0.045 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.77 | 75.2 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.95 | 81.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.5 |  |
| Always Verifier, candidate removed | 1.00 | 75.6 |  |
| Always Reasoner | 1.00 | 40.2 |  |
| Always Extractor | 1.00 | 35.4 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 37.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.47 | 85.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.5 | [-11.0, -4.3] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.9 | [+0.8, +11.0] | 0.024 | 0.036 |
| evidence value: always_verifier - always_null | +43.7 | [+37.4, +50.4] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -1.2 | [-5.5, +3.1] | 0.684 | 0.728 |
