### dyn3g_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 76 questions (extractor 0, reasoner 10, verifier 66). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 52.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.38 | 68.5 | after 2 of 76 calls (2.6%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.38 | 68.5 |  |
| Random questions, same calls per advisor | 0.38 | 59.8 | 95% of draws in [56.5, 63.0]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.38 | 60.5 | 95% of draws in [57.5, 64.0]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.38 | 65.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.72 | 74.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.5 |  |
| Always Verifier, candidate removed | 1.00 | 70.0 |  |
| Always Reasoner | 1.00 | 58.0 |  |
| Always Extractor | 1.00 | 53.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 53.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.26 | 78.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.0 | [-8.5, -1.5] | 0.005 | 0.013 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +3.5 | [-1.5, +8.5] | 0.200 | 0.248 |
| evidence value: always_verifier - always_null | +20.0 | [+13.5, +26.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.0 | [+0.0, +2.5] | 0.260 | 0.500 |
| learned - confidence_matched | +3.0 | [+0.0, +6.5] | 0.088 | 0.146 |
