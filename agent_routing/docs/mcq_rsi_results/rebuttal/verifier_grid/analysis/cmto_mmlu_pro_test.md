### cmto_mmlu_pro_test

n = 500; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 53.2 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 53.2 |  |
| Learned selection, Verifier only | 0.00 | 53.2 |  |
| Random questions, Verifier only | 0.00 | 53.2 | 95% of draws in [53.2, 53.2]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 53.2 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.00 | 53.2 | dev: 57.5 at 0.00 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.64 | 73.4 | dev: 74.0 at 0.65 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.81 | 74.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 74.6 |  |
| Always Verifier, candidate removed | 1.00 | 67.2 |  |
| Always Reasoner | 1.00 | 56.4 |  |
| Always Extractor | 1.00 | 53.6 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 53.2 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.25 | 78.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -21.4 | [-25.4, -17.4] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +7.4 | [+4.0, +11.0] | <0.001 | <0.001 |
| evidence value: always_verifier - always_null | +21.4 | [+17.4, +25.4] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_dev_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_dev_best | -20.2 | [-24.2, -16.4] | <0.001 | <0.001 |
