### seed1_gpqa_dev

n = 100; the learned policy's first action is a call on 61 questions (extractor 2, reasoner 14, verifier 45). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 40.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.61 | 53.0 | after 0 of 61 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.61 | 54.0 |  |
| Random questions, same calls per advisor | 0.61 | 48.4 | 95% of draws in [43.0, 53.0]; 0.054 of draws ≥ learned |
| Random questions, Verifier only | 0.61 | 50.3 | 95% of draws in [46.0, 55.0]; 0.196 of draws ≥ learned; 0.096 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.61 | 54.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.72 | 59.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 50.0 |  |
| Always Reasoner | 1.00 | 45.0 |  |
| Always Extractor | 1.00 | 42.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 41.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.27 | 67.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.0 | [-11.0, +3.0] | 0.297 | 0.388 |
| learned - learned_selection_verifier | -1.0 | [-5.0, +2.0] | 0.787 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +7.0 | [-2.0, +16.0] | 0.169 | 0.210 |
| evidence value: always_verifier - always_null | +16.0 | [+6.0, +26.0] | 0.003 | 0.005 |
| revision without evidence: always_null - candidate | +1.0 | [+0.0, +3.0] | 0.740 | 1.000 |
| learned - confidence_matched | -1.0 | [-9.0, +6.0] | 0.902 | 1.000 |
