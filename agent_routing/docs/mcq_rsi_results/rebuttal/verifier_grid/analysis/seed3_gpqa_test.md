### seed3_gpqa_test

n = 100; the learned policy's first action is a call on 73 questions (extractor 3, reasoner 14, verifier 56). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 41.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.73 | 52.0 | after 0 of 73 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.73 | 52.0 |  |
| Random questions, same calls per advisor | 0.73 | 49.5 | 95% of draws in [45.0, 54.0]; 0.196 of draws ≥ learned |
| Random questions, Verifier only | 0.73 | 51.2 | 95% of draws in [48.0, 55.0]; 0.452 of draws ≥ learned; 0.452 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.73 | 53.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.91 | 55.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 55.0 |  |
| Always Verifier, candidate removed | 1.00 | 59.0 |  |
| Always Reasoner | 1.00 | 47.0 |  |
| Always Extractor | 1.00 | 38.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 41.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.21 | 62.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -3.0 | [-8.0, +1.0] | 0.235 | 0.375 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -4.0 | [-11.0, +3.0] | 0.357 | 0.424 |
| evidence value: always_verifier - always_null | +14.0 | [+5.0, +23.0] | 0.003 | 0.004 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -1.0 | [-6.0, +3.0] | 0.827 | 1.000 |
