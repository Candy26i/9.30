### cmto_gpqa_dev

n = 100; the learned policy's first action is a call on 0 questions (extractor 0, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 100/100 questions.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 40.0 |  |
| Learned calling, first call only (policy's advisor, then commit) | 0.00 | 40.0 |  |
| Learned selection, Verifier only | 0.00 | 40.0 |  |
| Random questions, Verifier only | 0.00 | 40.0 | 95% of draws in [40.0, 40.0]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 0.00 | 40.0 |  |
| Confidence-gated Verifier, best threshold on this pool | 0.77 | 53.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 51.0 |  |
| Always Verifier, candidate removed | 1.00 | 48.0 |  |
| Always Reasoner | 1.00 | 44.0 |  |
| Always Extractor | 1.00 | 40.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 40.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.21 | 61.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -11.0 | [-19.0, -3.0] | 0.017 | 0.027 |
| learned - learned_selection_verifier | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| candidate value: always_verifier - always_verifier_nocand | +3.0 | [-6.0, +12.0] | 0.585 | 0.664 |
| evidence value: always_verifier - always_null | +11.0 | [+3.0, +19.0] | 0.017 | 0.027 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
