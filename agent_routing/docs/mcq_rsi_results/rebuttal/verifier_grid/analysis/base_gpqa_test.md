### base_gpqa_test

n = 100; the grammar-constrained first action is a call on 100 questions (extractor 100, reasoner 0, verifier 0). Every rule makes at most one call and is scored by the same forced-commit revision.

The grid's first turn equals unconstrained greedy decoding on 0/100 questions. This manager does not follow the DRAFT / commit / call grammar on its own: the rows below describe the grammar-forced model, not its deployed behaviour, and the budget-matched rows use that forced call budget.

| Rule | Calls per question | Accuracy (%) | Note |
|:--|--:|--:|:--|
| Candidate only (never call) | 0.00 | 32.0 |  |
| Grammar-constrained first action, then commit (not the deployed policy) | 1.00 | 31.0 | after 100 of 100 calls (100.0%) the commit was forced (the model's own next step was not the commit) |
| Learned selection, Verifier only | 1.00 | 52.0 |  |
| Random questions, same calls per advisor | 1.00 | 31.0 | 95% of draws in [31.0, 31.0]; 1.000 of draws ≥ learned |
| Random questions, Verifier only | 1.00 | 52.0 | 95% of draws in [52.0, 52.0]; 1.000 of draws ≥ learned; 1.000 ≥ learned selection (Verifier only) |
| Confidence-gated Verifier, same number of calls | 1.00 | 52.0 |  |
| Confidence-gated Verifier, dev threshold (matched call rate) | 1.00 | 52.0 | dev: 50.0 at 1.00 calls |
| Confidence-gated Verifier, dev threshold (best dev accuracy) | 0.72 | 49.0 | dev: 55.0 at 0.66 calls |
| Confidence-gated Verifier, best threshold on this pool | 0.96 | 52.0 | threshold chosen on this pool (upper bound) |
| Always Verifier, with the candidate | 1.00 | 52.0 |  |
| Always Verifier, candidate removed | 1.00 | 55.0 |  |
| Always Reasoner | 1.00 | 48.0 |  |
| Always Extractor | 1.00 | 31.0 |  |
| Revision turn on an empty Verifier output (no advisor call) | 0.00 | 32.0 | one extra manager revision turn on an empty Verifier output; no advisor call |
| Oracle (best of candidate / one call) | 0.28 | 60.0 |  |

| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |
|:--|--:|:-:|--:|--:|
| learned - always_verifier | -21.0 | [-30.0, -11.0] | <0.001 | <0.001 |
| learned - learned_selection_verifier | -21.0 | [-30.0, -11.0] | <0.001 | <0.001 |
| candidate value: always_verifier - always_verifier_nocand | -3.0 | [-11.0, +4.0] | 0.520 | 0.607 |
| evidence value: always_verifier - always_null | +20.0 | [+11.0, +29.0] | <0.001 | <0.001 |
| revision without evidence: always_null - candidate | +0.0 | [+0.0, +0.0] | 1.000 | 1.000 |
| learned - confidence_matched | -21.0 | [-30.0, -11.0] | <0.001 | <0.001 |
| learned - confidence_dev_matched | -21.0 | [-30.0, -11.0] | <0.001 | <0.001 |
| learned - confidence_dev_best | -18.0 | [-27.0, -9.0] | <0.001 | <0.001 |
