### s1_gpqa_dev

n = 100; the learned policy's first action is a call on 62 questions (extractor 0, reasoner 15, verifier 47). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 41.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.62 | 53.0 | after 0 of 62 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.62 | 55.0 |  |
| Random questions, same calls per advisor | 0.62 | 49.1 | 95% of draws in [44.0, 54.0]; 0.111 of draws ≥ learned |
| Random questions, Verifier only | 0.62 | 50.8 | 95% of draws in [46.0, 56.0]; 0.255 of draws ≥ learned; 0.075 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.62 | 57.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.69 | 59.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 52.0 |  |
| Always Reasoner | 1.00 | 46.0 |  |
| Always Extractor | 1.00 | 44.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 43.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 70.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.0 | [-11.0, +3.0] | 0.295 | 0.388 |
| learned - learned_selection_verifier | -2.0 | [-6.0, +2.0] | 0.436 | 0.625 |
| candidate value: always_verifier - always_verifier_nocand | +5.0 | [-5.0, +15.0] | 0.376 | 0.424 |
| evidence value: always_verifier - always_null | +14.0 | [+3.0, +25.0] | 0.014 | 0.020 |
| revision without evidence: always_null - candidate | +2.0 | [+0.0, +5.0] | 0.274 | 0.500 |
| learned - confidence_matched | -4.0 | [-11.0, +3.0] | 0.298 | 0.388 |
