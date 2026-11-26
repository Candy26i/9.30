### cmto_mmlu_pro_dev

n = 200; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 200/200 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 57.5 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 57.5 |  |
| Learned selection, Verifier only | 0.00 | 57.5 |  |
| Random questions, Verifier only | 0.00 | 57.5 | 95% of draws in [57.5, 57.5]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 57.5 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.65 | 74.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 73.5 |  |
| Always Verifier, candidate removed | 1.00 | 67.5 |  |
| Always Reasoner | 1.00 | 60.5 |  |
| Always Extractor | 1.00 | 57.5 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 57.5 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.18 | 76.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -16.0 | [-21.5, -10.5] | <0.001 | <0.001 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +6.0 | [+1.5, +11.0] | 0.017 | 0.023 |
| evidence value: always_verifier - always_null | +16.0 | [+10.5, +21.5] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
