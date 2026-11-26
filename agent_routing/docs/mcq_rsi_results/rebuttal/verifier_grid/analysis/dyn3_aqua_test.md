### dyn3_aqua_test

n = 254; the learned policy's first action is a call on 202 questions (extractor 0, reasoner 0, verifier 202). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 254/254 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 35.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.80 | 77.6 | after 0 of 202 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.80 | 77.6 |  |
| Random questions, same calls per advisor | 0.80 | 74.6 | 95% of draws in [71.7, 77.6]; 0.025 of draws ≥ learned |
| Random questions, Verifier only | 0.80 | 74.6 | 95% of draws in [71.7, 77.6]; 0.025 of draws ≥ learned; 0.025 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.80 | 76.8 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.78 | 76.0 | dev: 77.6 at 0.81 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.94 | 82.3 | dev: 81.9 at 0.96 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.99 | 84.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 84.6 |  |
| Always Verifier, candidate removed | 1.00 | 79.1 |  |
| Always Reasoner | 1.00 | 36.6 |  |
| Always Extractor | 1.00 | 30.3 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 32.7 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.54 | 89.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -7.1 | [-10.6, -3.9] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.5 | [+0.8, +10.2] | 0.022 | 0.029 |
| evidence value: always_verifier - always_null | +52.0 | [+44.9, +58.7] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -2.8 | [-4.7, -0.8] | 0.002 | 0.016 |
| learned - confidence_matched | +0.8 | [-3.1, +4.7] | 0.768 | 0.845 |
| learned - confidence_dev_matched | +1.6 | [-2.4, +5.9] | 0.504 | 0.572 |
| learned - confidence_dev_best | -4.7 | [-8.3, -1.6] | 0.005 | 0.012 |
