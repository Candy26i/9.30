### s1_medqa_dev

n = 200; the learned policy's first action is a call on 74 questions (extractor 2, reasoner 41, verifier 31). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 70.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.37 | 79.0 | after 0 of 74 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.37 | 80.0 |  |
| Random questions, same calls per advisor | 0.37 | 75.7 | 95% of draws in [73.0, 78.5]; 0.024 of draws ≥ learned |
| Random questions, Verifier only | 0.37 | 75.7 | 95% of draws in [73.0, 78.5]; 0.013 of draws ≥ learned; 0.003 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.37 | 80.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.89 | 84.5 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.5 |  |
| Always Verifier, candidate removed | 1.00 | 85.5 |  |
| Always Reasoner | 1.00 | 85.0 |  |
| Always Extractor | 1.00 | 78.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.23 | 93.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.5 | [-10.0, -1.0] | 0.016 | 0.027 |
| learned - learned_selection_verifier | -1.0 | [-4.0, +2.0] | 0.620 | 0.754 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-5.5, +3.5] | 0.741 | 0.824 |
| evidence value: always_verifier - always_null | +13.5 | [+8.0, +19.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.5 | [-1.5, +3.0] | 0.836 | 1.000 |
| learned - confidence_matched | -1.5 | [-6.0, +3.0] | 0.573 | 0.664 |
