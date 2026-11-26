### sucd_medqa_test

n = 500; the learned policy's first action is a call on 488 questions (extractor 66, reasoner 191, verifier 231). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 71.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.98 | 81.6 | after 0 of 488 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.98 | 83.2 |  |
| Random questions, same calls per advisor | 0.98 | 81.7 | 95% of draws in [80.0, 83.4]; 0.580 of draws ≥ learned |
| Random questions, Verifier only | 0.98 | 83.3 | 95% of draws in [82.8, 83.8]; 1.000 of draws ≥ learned; 0.786 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.98 | 83.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.84 | 83.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 83.6 |  |
| Always Verifier, candidate removed | 1.00 | 84.4 |  |
| Always Reasoner | 1.00 | 81.4 |  |
| Always Extractor | 1.00 | 77.6 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.19 | 90.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.0 | [-4.2, +0.2] | 0.093 | 0.121 |
| learned - learned_selection_verifier | -1.6 | [-3.8, +0.6] | 0.175 | 0.215 |
| candidate value: always_verifier - always_verifier_nocand | -0.8 | [-3.6, +2.0] | 0.611 | 0.665 |
| evidence value: always_verifier - always_null | +12.2 | [+8.8, +15.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-1.4, +0.6] | 0.521 | 0.688 |
| learned - confidence_matched | -2.0 | [-4.2, +0.2] | 0.093 | 0.121 |
