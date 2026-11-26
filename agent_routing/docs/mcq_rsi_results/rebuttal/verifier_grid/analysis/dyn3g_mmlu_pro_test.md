### dyn3g_mmlu_pro_test

n = 500; the learned policy's first action is a call on 176 questions (extractor 0, reasoner 20, verifier 156). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 51.4 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.35 | 68.6 | after 1 of 176 calls (0.6%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 0.35 | 68.6 |  |
| Random questions, same calls per advisor | 0.35 | 58.6 | 95% of draws in [56.6, 60.6]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 0.35 | 59.2 | 95% of draws in [57.2, 61.2]; <0.001 of draws ≥ learned; <0.001 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.35 | 64.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 0.35 | 64.2 | dev: 65.5 at 0.38 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.73 | 73.2 | dev: 74.0 at 0.72 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.88 | 73.6 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.6 |  |
| Always Verifier, candidate removed | 1.00 | 68.2 |  |
| Always Reasoner | 1.00 | 56.4 |  |
| Always Extractor | 1.00 | 52.2 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 51.4 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.29 | 80.2 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -5.0 | [-7.8, -2.4] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [-1.2, +1.2] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +5.4 | [+2.2, +8.6] | 0.002 | 0.002 |
| evidence value: always_verifier - always_null | +22.2 | [+18.0, +26.4] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [-1.0, +1.0] | 1.000 | 1.000 |
| learned - confidence_matched | +4.6 | [+1.8, +7.6] | 0.002 | 0.003 |
| learned - confidence_dev_matched | +4.4 | [+1.6, +7.4] | 0.002 | 0.004 |
| learned - confidence_dev_best | -4.6 | [-7.2, -2.0] | <0.001 | <0.001 |
