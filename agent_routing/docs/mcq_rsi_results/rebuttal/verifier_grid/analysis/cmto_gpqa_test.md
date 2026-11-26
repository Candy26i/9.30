### cmto_gpqa_test

n = 100; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 42.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 42.0 |  |
| Learned selection, Verifier only | 0.00 | 42.0 |  |
| Random questions, Verifier only | 0.00 | 42.0 | 95% of draws in [42.0, 42.0]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 42.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.85 | 57.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 57.0 |  |
| Always Verifier, candidate removed | 1.00 | 58.0 |  |
| Always Reasoner | 1.00 | 50.0 |  |
| Always Extractor | 1.00 | 42.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 42.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.19 | 61.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -15.0 | [-23.0, -7.0] | <0.001 | 0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -1.0 | [-8.0, +5.0] | 0.892 | 1.000 |
| evidence value: always_verifier - always_null | +15.0 | [+7.0, +23.0] | <0.001 | 0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
