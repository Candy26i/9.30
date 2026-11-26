### rho0_medqa_test

n = 500; the learned policy's first action is a call on 500 questions (extractor 2, reasoner 294, verifier 204). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 69.6 |  |
| Learned calling, first call only (policy's advisor, then commit) | 1.00 | 82.2 | after 0 of 500 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 84.6 |  |
| Random questions, same calls per advisor | 1.00 | 82.7 | 95% of draws in [81.2, 84.2]; 0.785 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 84.6 | 95% of draws in [84.6, 84.6]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 84.6 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.84 | 84.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.6 |  |
| Always Verifier, candidate removed | 1.00 | 84.2 |  |
| Always Reasoner | 1.00 | 81.4 |  |
| Always Extractor | 1.00 | 78.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 71.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.21 | 90.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.4 | [-4.8, +0.0] | 0.058 | 0.073 |
| learned - learned_selection_verifier | -2.4 | [-4.8, +0.0] | 0.058 | 0.073 |
| candidate value: always_verifier - always_verifier_nocand | +0.4 | [-2.2, +3.0] | 0.820 | 0.883 |
| evidence value: always_verifier - always_null | +13.2 | [+9.8, +16.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +1.8 | [-0.8, +4.4] | 0.205 | 0.233 |
| learned - confidence_matched | -2.4 | [-4.8, +0.0] | 0.058 | 0.073 |
