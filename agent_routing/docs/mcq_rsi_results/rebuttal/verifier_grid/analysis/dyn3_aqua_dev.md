### dyn3_aqua_dev

n = 254; the learned policy's first action is a call on 206 questions (extractor 0, reasoner 0, verifier 206). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 35.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.81 | 76.8 | after 0 of 206 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.81 | 76.8 |  |
| Random questions, same calls per advisor | 0.81 | 73.0 | 95% of draws in [70.5, 75.6]; 0.002 of draws ≥ learned |
| Random questions, Verifier only | 0.81 | 73.0 | 95% of draws in [70.5, 75.6]; 0.002 of draws ≥ learned; 0.002 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.81 | 77.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.96 | 81.9 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 81.9 |  |
| Always Verifier, candidate removed | 1.00 | 74.8 |  |
| Always Reasoner | 1.00 | 39.4 |  |
| Always Extractor | 1.00 | 32.7 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 35.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.52 | 86.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.1 | [-8.7, -2.0] | <0.001 | 0.004 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +7.1 | [+2.4, +12.2] | 0.006 | 0.010 |
| evidence value: always_verifier - always_null | +46.1 | [+39.4, +52.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.8 | [-1.2, +2.8] | 0.549 | 0.688 |
| learned - confidence_matched | -0.8 | [-4.7, +3.1] | 0.789 | 0.851 |
