### s1_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 61 questions (extractor 2, reasoner 6, verifier 53). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 198/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 55.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.30 | 62.0 | after 3 of 61 calls (4.9%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.30 | 62.5 |  |
| Random questions, same calls per advisor | 0.30 | 60.4 | 95% of draws in [57.5, 63.5]; 0.185 of draws ≥ learned |
| Random questions, Verifier only | 0.30 | 61.0 | 95% of draws in [58.0, 64.0]; 0.313 of draws ≥ learned; 0.203 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.30 | 65.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.71 | 74.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.5 |  |
| Always Verifier, candidate removed | 1.00 | 69.0 |  |
| Always Reasoner | 1.00 | 58.5 |  |
| Always Extractor | 1.00 | 55.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 57.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.23 | 79.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -11.5 | [-17.0, -6.5] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.5 | [-2.5, +1.0] | 0.784 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +4.5 | [-0.5, +9.5] | 0.093 | 0.122 |
| evidence value: always_verifier - always_null | +16.5 | [+10.5, +22.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.5 | [+0.0, +3.5] | 0.095 | 0.250 |
| learned - confidence_matched | -3.0 | [-8.0, +2.0] | 0.298 | 0.345 |
