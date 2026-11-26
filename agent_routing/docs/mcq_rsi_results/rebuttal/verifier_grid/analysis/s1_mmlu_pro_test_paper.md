### s1_mmlu_pro_test_paper

n = 500; the learned policy's first action is a call on 166 questions (extractor 6, reasoner 20, verifier 140). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 498/500 questions.

200 of the 500 questions are also in the dev grid; the two dev-threshold rows and their tests use the other 300.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 49.2 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.33 | 59.2 | after 6 of 166 calls (3.6%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.33 | 59.8 |  |
| Random questions, same calls per advisor | 0.33 | 55.4 | 95% of draws in [53.4, 57.4]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.33 | 56.2 | 95% of draws in [54.4, 58.2]; 0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.33 | 62.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.26 | 56.3 | dev: 65.0 at 0.30 calls; n = 300 (200 dev questions excluded; learned on these: 57.3 at 0.35 calls) |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.70 | 66.7 | dev: 74.0 at 0.71 calls; n = 300 (200 dev questions excluded; learned on these: 57.3 at 0.35 calls) |
| Confidence-gated Verifier, best threshold on this pool | 0.98 | 70.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 70.4 |  |
| Always Verifier, candidate removed | 1.00 | 66.0 |  |
| Always Reasoner | 1.00 | 55.0 |  |
| Always Extractor | 1.00 | 51.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 50.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.30 | 79.2 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -11.2 | [-14.6, -7.8] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -0.6 | [-2.0, +0.6] | 0.436 | 0.549 |
| candidate value: always_verifier - always_verifier_nocand | +4.4 | [+1.0, +7.8] | 0.013 | 0.018 |
| evidence value: always_verifier - always_null | +20.4 | [+16.2, +24.6] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.8 | [+0.0, +1.8] | 0.130 | 0.219 |
| learned - confidence_matched | -2.8 | [-5.8, +0.2] | 0.072 | 0.087 |
| learned - confidence_dev_matched | +1.0 | [-3.0, +5.0] | 0.697 | 0.749 |
| learned - confidence_dev_best | -9.3 | [-14.0, -5.3] | <0.001 | <0.001 |
