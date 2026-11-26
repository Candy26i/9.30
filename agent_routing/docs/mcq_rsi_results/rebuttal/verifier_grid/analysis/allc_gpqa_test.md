### allc_gpqa_test

n = 100; the learned policy's first action is a call on 68 questions (extractor 1, reasoner 16, verifier 51). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 40.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.68 | 54.0 | after 0 of 68 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.68 | 53.0 |  |
| Random questions, same calls per advisor | 0.68 | 48.8 | 95% of draws in [44.0, 54.0]; 0.035 of draws ≥ learned |
| Random questions, Verifier only | 0.68 | 50.2 | 95% of draws in [46.0, 55.0]; 0.069 of draws ≥ learned; 0.152 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.68 | 53.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.91 | 55.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 55.0 |  |
| Always Verifier, candidate removed | 1.00 | 61.0 |  |
| Always Reasoner | 1.00 | 47.0 |  |
| Always Extractor | 1.00 | 39.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 41.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.25 | 65.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -1.0 | [-7.0, +5.0] | 0.862 | 1.000 |
| learned - learned_selection_verifier | +1.0 | [-2.0, +5.0] | 0.809 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -6.0 | [-14.0, +2.0] | 0.191 | 0.238 |
| evidence value: always_verifier - always_null | +14.0 | [+5.0, +23.0] | 0.002 | 0.007 |
| revision without evidence: always_null - candidate | +1.0 | [+0.0, +3.0] | 0.723 | 1.000 |
| learned - confidence_matched | +1.0 | [-5.0, +7.0] | 0.886 | 1.000 |
