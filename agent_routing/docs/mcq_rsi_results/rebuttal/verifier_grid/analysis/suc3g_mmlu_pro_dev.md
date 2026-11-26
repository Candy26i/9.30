### suc3g_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 200 questions (extractor 19, reasoner 79, verifier 102). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 52.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 1.00 | 70.0 | after 0 of 200 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 73.5 |  |
| Random questions, same calls per advisor | 1.00 | 65.4 | 95% of draws in [62.0, 68.5]; 0.003 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 73.5 | 95% of draws in [73.5, 73.5]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 73.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.76 | 73.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.5 |  |
| Always Verifier, candidate removed | 1.00 | 67.5 |  |
| Always Reasoner | 1.00 | 58.0 |  |
| Always Extractor | 1.00 | 53.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 53.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.26 | 78.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -3.5 | [-6.5, -1.0] | 0.002 | 0.016 |
| learned - learned_selection_verifier | -3.5 | [-6.5, -1.0] | 0.002 | 0.016 |
| candidate value: always_verifier - always_verifier_nocand | +6.0 | [+0.5, +11.5] | 0.042 | 0.050 |
| evidence value: always_verifier - always_null | +20.5 | [+14.0, +26.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.5 | [+0.0, +1.5] | 0.729 | 1.000 |
| learned - confidence_matched | -3.5 | [-6.5, -1.0] | 0.002 | 0.016 |
