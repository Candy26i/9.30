### seed1_mmlu_pro_test

n = 500; the learned policy's first action is a call on 176 questions (extractor 13, reasoner 7, verifier 156). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.35 | 65.2 | after 6 of 176 calls (3.4%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.35 | 65.8 |  |
| Random questions, same calls per advisor | 0.35 | 58.4 | 95% of draws in [56.6, 60.4]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.35 | 59.2 | 95% of draws in [57.2, 61.2]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.35 | 63.8 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.82 | 73.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.6 |  |
| Always Verifier, candidate removed | 1.00 | 67.4 |  |
| Always Reasoner | 1.00 | 56.4 |  |
| Always Extractor | 1.00 | 52.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.28 | 79.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -8.4 | [-11.2, -5.6] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.6 | [-1.4, +0.0] | 0.097 | 0.250 |
| candidate value: always_verifier - always_verifier_nocand | +6.2 | [+2.8, +9.6] | <0.001 | <0.001 |
| evidence value: always_verifier - always_null | +22.0 | [+17.8, +26.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.2 | [-0.8, +1.2] | 0.846 | 1.000 |
| learned - confidence_matched | +1.4 | [-1.8, +4.6] | 0.436 | 0.470 |
