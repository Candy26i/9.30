### suc3g_mmlu_pro_test

n = 500; the learned policy's first action is a call on 498 questions (extractor 44, reasoner 200, verifier 254). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 500/500 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 53.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 1.00 | 71.8 | after 0 of 498 calls (0.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 73.4 |  |
| Random questions, same calls per advisor | 1.00 | 65.0 | 95% of draws in [63.0, 67.2]; <0.001 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 73.3 | 95% of draws in [73.0, 73.6]; 1.000 of draws ≥ learned; 0.585 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 73.4 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 1.00 | 73.4 | dev: 73.5 at 1.00 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.77 | 72.8 | dev: 73.5 at 0.76 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.84 | 73.4 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.4 |  |
| Always Verifier, candidate removed | 1.00 | 69.2 |  |
| Always Reasoner | 1.00 | 57.2 |  |
| Always Extractor | 1.00 | 53.4 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 52.8 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.28 | 80.6 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -1.6 | [-4.0, +0.6] | 0.201 | 0.229 |
| learned - learned_selection_verifier | -1.6 | [-4.0, +0.6] | 0.201 | 0.229 |
| candidate value: always_verifier - always_verifier_nocand | +4.2 | [+0.8, +7.6] | 0.016 | 0.022 |
| evidence value: always_verifier - always_null | +20.6 | [+16.4, +25.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | -0.2 | [-1.0, +0.4] | 0.798 | 1.000 |
| learned - confidence_matched | -1.6 | [-4.0, +0.6] | 0.201 | 0.229 |
| learned - confidence_dev_matched | -1.6 | [-4.0, +0.6] | 0.201 | 0.229 |
| learned - confidence_dev_best | -1.0 | [-3.4, +1.2] | 0.456 | 0.500 |
