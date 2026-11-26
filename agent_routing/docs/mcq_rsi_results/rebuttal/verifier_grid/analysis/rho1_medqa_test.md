### rho1_medqa_test

n = 500; the learned policy's first action is a call on 301 questions (extractor 2, reasoner 148, verifier 151). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 71.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.60 | 80.0 | after 0 of 301 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.60 | 82.0 |  |
| Random questions, same calls per advisor | 0.60 | 78.2 | 95% of draws in [76.2, 80.2]; 0.051 of draws ≥ learned |
| Random questions, Verifier only | 0.60 | 79.0 | 95% of draws in [77.2, 80.8]; 0.150 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.60 | 82.8 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.98 | 84.2 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.0 |  |
| Always Verifier, candidate removed | 1.00 | 84.4 |  |
| Always Reasoner | 1.00 | 81.4 |  |
| Always Extractor | 1.00 | 78.8 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.19 | 90.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -4.0 | [-6.6, -1.4] | 0.002 | 0.005 |
| learned - learned_selection_verifier | -2.0 | [-4.4, +0.2] | 0.092 | 0.121 |
| candidate value: always_verifier - always_verifier_nocand | -0.4 | [-3.0, +2.2] | 0.839 | 0.880 |
| evidence value: always_verifier - always_null | +13.0 | [+9.6, +16.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.4 | [-2.0, +1.2] | 0.707 | 0.804 |
| learned - confidence_matched | -2.8 | [-5.4, -0.2] | 0.039 | 0.049 |
