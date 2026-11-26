### s1_gpqa_test

n = 100; the learned policy's first action is a call on 52 questions (extractor 1, reasoner 9, verifier 42). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 41.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.52 | 54.0 | after 0 of 52 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.52 | 53.0 |  |
| Random questions, same calls per advisor | 0.52 | 47.7 | 95% of draws in [43.0, 52.0]; 0.005 of draws ≥ learned |
| Random questions, Verifier only | 0.52 | 48.8 | 95% of draws in [44.0, 53.0]; 0.017 of draws ≥ learned; 0.052 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.52 | 52.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.71 | 53.0 | dev: 57.0 at 0.62 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.84 | 55.0 | dev: 59.0 at 0.69 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.91 | 56.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 56.0 |  |
| Always Verifier, candidate removed | 1.00 | 61.0 |  |
| Always Reasoner | 1.00 | 45.0 |  |
| Always Extractor | 1.00 | 39.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 41.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.22 | 63.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -2.0 | [-9.0, +5.0] | 0.653 | 0.774 |
| learned - learned_selection_verifier | +1.0 | [-2.0, +4.0] | 0.784 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | -5.0 | [-13.0, +3.0] | 0.279 | 0.332 |
| evidence value: always_verifier - always_null | +15.0 | [+6.0, +24.0] | <0.001 | 0.003 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +2.0 | [-4.0, +8.0] | 0.638 | 0.754 |
| learned - confidence_dev_matched | +1.0 | [-5.0, +7.0] | 0.881 | 1.000 |
| learned - confidence_dev_best | -1.0 | [-8.0, +5.0] | 0.878 | 1.000 |
