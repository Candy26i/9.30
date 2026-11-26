### seed2_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 82 questions (extractor 4, reasoner 8, verifier 70). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.41 | 59.5 | after 3 of 82 calls (3.7%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.41 | 61.0 |  |
| Random questions, same calls per advisor | 0.41 | 58.3 | 95% of draws in [55.0, 61.5]; 0.266 of draws ≥ learned |
| Random questions, Verifier only | 0.41 | 59.3 | 95% of draws in [56.0, 62.5]; 0.507 of draws ≥ learned; 0.161 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.41 | 65.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.80 | 71.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 71.0 |  |
| Always Verifier, candidate removed | 1.00 | 68.0 |  |
| Always Reasoner | 1.00 | 57.5 |  |
| Always Extractor | 1.00 | 51.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 52.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.25 | 76.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -11.5 | [-16.5, -7.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -1.5 | [-3.5, +0.0] | 0.100 | 0.250 |
| candidate value: always_verifier - always_verifier_nocand | +3.0 | [-2.0, +8.0] | 0.271 | 0.327 |
| evidence value: always_verifier - always_null | +18.5 | [+12.5, +24.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.5 | [+0.0, +3.5] | 0.095 | 0.250 |
| learned - confidence_matched | -6.0 | [-11.5, -1.0] | 0.025 | 0.036 |
