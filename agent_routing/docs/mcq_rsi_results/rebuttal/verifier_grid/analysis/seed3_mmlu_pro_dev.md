### seed3_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 79 questions (extractor 3, reasoner 11, verifier 65). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 52.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.40 | 60.5 | after 5 of 79 calls (6.3%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.40 | 63.5 |  |
| Random questions, same calls per advisor | 0.40 | 58.8 | 95% of draws in [55.5, 62.0]; 0.178 of draws ≥ learned |
| Random questions, Verifier only | 0.40 | 60.0 | 95% of draws in [57.0, 63.0]; 0.446 of draws ≥ learned; 0.018 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.40 | 66.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.79 | 72.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 72.0 |  |
| Always Verifier, candidate removed | 1.00 | 68.5 |  |
| Always Reasoner | 1.00 | 56.0 |  |
| Always Extractor | 1.00 | 52.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 53.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.26 | 77.5 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -11.5 | [-17.0, -6.5] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -3.0 | [-6.0, -0.5] | 0.037 | 0.070 |
| candidate value: always_verifier - always_verifier_nocand | +3.5 | [-1.5, +8.5] | 0.207 | 0.248 |
| evidence value: always_verifier - always_null | +19.0 | [+13.0, +25.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.0 | [-1.0, +3.0] | 0.436 | 0.625 |
| learned - confidence_matched | -5.5 | [-11.0, +0.0] | 0.055 | 0.071 |
