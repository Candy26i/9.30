### seed1_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 77 questions (extractor 3, reasoner 5, verifier 69). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 53.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.39 | 63.0 | after 3 of 77 calls (3.9%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.39 | 64.0 |  |
| Random questions, same calls per advisor | 0.39 | 60.2 | 95% of draws in [57.0, 63.0]; 0.048 of draws ≥ learned |
| Random questions, Verifier only | 0.39 | 60.9 | 95% of draws in [58.0, 63.5]; 0.105 of draws ≥ learned; 0.021 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.39 | 66.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.78 | 73.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 72.5 |  |
| Always Verifier, candidate removed | 1.00 | 69.0 |  |
| Always Reasoner | 1.00 | 58.0 |  |
| Always Extractor | 1.00 | 54.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 55.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.24 | 78.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.5 | [-14.5, -5.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -1.0 | [-3.0, +1.0] | 0.451 | 0.625 |
| candidate value: always_verifier - always_verifier_nocand | +3.5 | [-1.5, +8.5] | 0.201 | 0.248 |
| evidence value: always_verifier - always_null | +17.5 | [+11.5, +23.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.5 | [-0.5, +4.0] | 0.241 | 0.375 |
| learned - confidence_matched | -3.5 | [-8.5, +1.5] | 0.201 | 0.230 |
