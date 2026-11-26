### cmto_medqa_test

n = 500; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 73.6 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 73.6 |  |
| Learned selection, Verifier only | 0.00 | 73.6 |  |
| Random questions, Verifier only | 0.00 | 73.6 | 95% of draws in [73.6, 73.6]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 73.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.84 | 83.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.2 |  |
| Always Verifier, candidate removed | 1.00 | 84.0 |  |
| Always Reasoner | 1.00 | 80.0 |  |
| Always Extractor | 1.00 | 76.6 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 73.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.15 | 88.4 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.6 | [-12.8, -6.6] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -0.8 | [-3.4, +1.8] | 0.585 | 0.644 |
| evidence value: always_verifier - always_null | +9.6 | [+6.6, +12.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
