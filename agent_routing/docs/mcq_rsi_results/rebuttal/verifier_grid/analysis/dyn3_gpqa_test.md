### dyn3_gpqa_test

n = 100; the learned policy's first action is a call on 58 questions (extractor 0, reasoner 21, verifier 37). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 41.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.58 | 53.0 | after 0 of 58 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.58 | 55.0 |  |
| Random questions, same calls per advisor | 0.58 | 48.1 | 95% of draws in [43.0, 53.0]; 0.042 of draws ≥ learned |
| Random questions, Verifier only | 0.58 | 50.3 | 95% of draws in [45.0, 55.0]; 0.179 of draws ≥ learned; 0.038 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.58 | 50.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.92 | 57.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 58.0 |  |
| Always Reasoner | 1.00 | 47.0 |  |
| Always Extractor | 1.00 | 41.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 41.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.24 | 65.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.0 | [-10.0, +2.0] | 0.259 | 0.344 |
| learned - learned_selection_verifier | -2.0 | [-6.0, +2.0] | 0.450 | 0.625 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-8.0, +7.0] | 0.909 | 1.000 |
| evidence value: always_verifier - always_null | +16.0 | [+6.0, +25.0] | 0.001 | 0.002 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +3.0 | [-5.0, +11.0] | 0.510 | 0.607 |
