### seed2_gpqa_test

n = 100; the learned policy's first action is a call on 44 questions (extractor 1, reasoner 9, verifier 34). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 43.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.44 | 51.0 | after 0 of 44 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.44 | 52.0 |  |
| Random questions, same calls per advisor | 0.44 | 47.9 | 95% of draws in [43.0, 53.0]; 0.141 of draws ≥ learned |
| Random questions, Verifier only | 0.44 | 48.7 | 95% of draws in [44.0, 53.0]; 0.221 of draws ≥ learned; 0.115 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.44 | 50.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.60 | 50.0 | dev: 53.0 at 0.56 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.86 | 55.0 | dev: 59.0 at 0.72 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.93 | 56.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 56.0 |  |
| Always Verifier, candidate removed | 1.00 | 59.0 |  |
| Always Reasoner | 1.00 | 48.0 |  |
| Always Extractor | 1.00 | 40.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 43.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.21 | 64.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.0 | [-12.0, +2.0] | 0.208 | 0.267 |
| learned - learned_selection_verifier | -1.0 | [-3.0, +0.0] | 0.742 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -3.0 | [-10.0, +5.0] | 0.537 | 0.607 |
| evidence value: always_verifier - always_null | +13.0 | [+4.0, +22.0] | 0.007 | 0.011 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +1.0 | [-6.0, +8.0] | 0.889 | 1.000 |
| learned - confidence_dev_matched | +1.0 | [-6.0, +8.0] | 0.890 | 1.000 |
| learned - confidence_dev_best | -4.0 | [-11.0, +3.0] | 0.312 | 0.388 |
