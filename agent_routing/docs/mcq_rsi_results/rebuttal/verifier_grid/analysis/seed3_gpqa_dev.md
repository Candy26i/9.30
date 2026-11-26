### seed3_gpqa_dev

n = 100; the learned policy's first action is a call on 74 questions (extractor 1, reasoner 17, verifier 56). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 39.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.74 | 55.0 | after 0 of 74 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.74 | 55.0 |  |
| Random questions, same calls per advisor | 0.74 | 50.3 | 95% of draws in [45.0, 56.0]; 0.056 of draws ≥ learned |
| Random questions, Verifier only | 0.74 | 52.3 | 95% of draws in [48.0, 57.0]; 0.151 of draws ≥ learned; 0.151 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.74 | 58.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.79 | 59.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 50.0 |  |
| Always Reasoner | 1.00 | 46.0 |  |
| Always Extractor | 1.00 | 41.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 40.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.27 | 66.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.0 | [-8.0, +3.0] | 0.566 | 0.727 |
| learned - learned_selection_verifier | +0.0 | [-4.0, +4.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +7.0 | [-2.0, +16.0] | 0.169 | 0.210 |
| evidence value: always_verifier - always_null | +17.0 | [+7.0, +27.0] | 0.001 | 0.002 |
| revision without evidence: always_null - candidate | +1.0 | [+0.0, +3.0] | 0.740 | 1.000 |
| learned - confidence_matched | -3.0 | [-9.0, +3.0] | 0.389 | 0.508 |
