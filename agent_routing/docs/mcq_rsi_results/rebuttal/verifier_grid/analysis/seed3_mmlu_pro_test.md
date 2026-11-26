### seed3_mmlu_pro_test

n = 500; the learned policy's first action is a call on 185 questions (extractor 14, reasoner 12, verifier 159). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 50.8 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.37 | 64.8 | after 9 of 185 calls (4.9%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.37 | 65.4 |  |
| Random questions, same calls per advisor | 0.37 | 58.2 | 95% of draws in [56.4, 60.2]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.37 | 59.3 | 95% of draws in [57.4, 61.2]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.37 | 64.6 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.35 | 63.8 | dev: 66.0 at 0.40 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.77 | 73.4 | dev: 72.5 at 0.79 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.81 | 73.8 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.6 |  |
| Always Verifier, candidate removed | 1.00 | 68.4 |  |
| Always Reasoner | 1.00 | 56.6 |  |
| Always Extractor | 1.00 | 51.8 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 79.8 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -8.8 | [-11.6, -6.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.6 | [-1.6, +0.2] | 0.251 | 0.375 |
| candidate value: always_verifier - always_verifier_nocand | +5.2 | [+1.8, +8.6] | 0.003 | 0.004 |
| evidence value: always_verifier - always_null | +22.6 | [+18.4, +26.8] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.2 | [-0.8, +1.2] | 0.846 | 1.000 |
| learned - confidence_matched | +0.2 | [-3.0, +3.6] | 0.939 | 1.000 |
| learned - confidence_dev_matched | +1.0 | [-2.4, +4.4] | 0.601 | 0.644 |
| learned - confidence_dev_best | -8.6 | [-11.4, -5.8] | <0.001 | <0.001 |
