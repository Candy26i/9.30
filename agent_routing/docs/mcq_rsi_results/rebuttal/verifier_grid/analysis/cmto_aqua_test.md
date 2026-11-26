### cmto_aqua_test

n = 254; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 38.6 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 38.6 |  |
| Learned selection, Verifier only | 0.00 | 38.6 |  |
| Random questions, Verifier only | 0.00 | 38.6 | 95% of draws in [38.6, 38.6]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 38.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.00 | 38.6 | dev: 39.4 at 0.00 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.92 | 82.3 | dev: 81.1 at 0.96 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.97 | 84.3 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.3 |  |
| Always Verifier, candidate removed | 1.00 | 78.3 |  |
| Always Reasoner | 1.00 | 40.2 |  |
| Always Extractor | 1.00 | 38.6 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 38.6 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.48 | 86.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -45.7 | [-52.4, -39.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.9 | [+1.2, +10.6] | 0.015 | 0.020 |
| evidence value: always_verifier - always_null | +45.7 | [+39.0, +52.4] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_dev_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_dev_best | -43.7 | [-50.4, -37.0] | <0.001 | <0.001 |
