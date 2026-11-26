### s1_mmlu_pro_test

n = 500; the learned policy's first action is a call on 166 questions (extractor 10, reasoner 12, verifier 144). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 499/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 50.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.33 | 63.2 | after 8 of 166 calls (4.8%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.33 | 63.4 |  |
| Random questions, same calls per advisor | 0.33 | 57.2 | 95% of draws in [55.2, 59.2]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.33 | 58.0 | 95% of draws in [56.0, 60.0]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.33 | 61.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.24 | 58.0 | dev: 65.0 at 0.30 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.68 | 71.4 | dev: 74.0 at 0.71 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.75 | 72.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 72.4 |  |
| Always Verifier, candidate removed | 1.00 | 69.2 |  |
| Always Reasoner | 1.00 | 55.6 |  |
| Always Extractor | 1.00 | 52.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 79.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -9.2 | [-12.2, -6.2] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.2 | [-1.2, +0.6] | 0.830 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +3.2 | [-0.2, +6.6] | 0.072 | 0.089 |
| evidence value: always_verifier - always_null | +21.0 | [+17.0, +25.2] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.6 | [+0.0, +1.4] | 0.101 | 0.250 |
| learned - confidence_matched | +1.6 | [-1.6, +5.0] | 0.380 | 0.416 |
| learned - confidence_dev_matched | +5.2 | [+1.8, +8.6] | 0.002 | 0.004 |
| learned - confidence_dev_best | -8.2 | [-11.4, -5.2] | <0.001 | <0.001 |
