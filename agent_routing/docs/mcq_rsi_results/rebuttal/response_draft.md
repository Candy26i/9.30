# MARGENT author response: Verifier / calling policy (A) and what the trained manager contributes (B)

Draft for the author of topics A and B. GRPO details, RSI results and offline cost belong to the co-authors and are not answered here.

Conventions. Accuracy is in %. A difference is "first minus second" in accuracy points, with its 95% paired-bootstrap interval in brackets and the exact McNemar p. "S_1" is the registered single-pass manager of each benchmark (for MedQA a checkpoint trained with the paper's configuration, because the paper's locked checkpoint was not archived). "Seed 1", "seed 2" and "seed 3" are re-trainings of the S_1 label file with that training seed. "Four seeds" means S_1 and the three re-trainings; "mean ± SD" is the mean and the sample standard deviation over those four. Evidence files are under `agent_routing/docs/mcq_rsi_results/rebuttal/` (below: `REB/`).

---

## Common block (post once; the reviewer sections point to it)

1. **Advisors.** The paper's description of the advisors as "role-specific LoRA specialists" is wrong for every reported number: the adapters' parameter names matched no module of the served model, so each advisor was the base model (Qwen3.5-9B) under its role prompt. Replays confirm this for the Verifier (20 of 20 outputs identical to the base model); the Extractor matched on 5 of 20 and the Reasoner was not checked. No reported number changes.

2. **Evidence base.** New results use test pools sharing no question with the training or development pools; MMLU-Pro has a new pool because the paper's test set, as reconstructed, shared 200 of its 500 questions with the development pool and 196 with training pools (68 label-collection roots, 128 GRPO questions), and GPQA is Diamond-100, which Table 4 also used. Each benchmark has four managers: S_1 and three re-trainings of its label file (training seeds 1 to 3) with the logged recipe (assumed for AQuA, which has no training log), which may differ from S_1 in unlogged details (seed, maximum sequence length, attention kernels). The MedQA S_1 is a same-configuration substitute for the paper's unarchived checkpoint. Intervals cover questions, SDs cover seeds; each benchmark has one test pool, also used by our earlier runs.

3. **Check.** S_1's learned-calling row scores each question (right or wrong) as our re-evaluation of the deployed S_1 does on 500/500, 498/500, 100/100 and 254/254 questions.

Sources for the common block (not part of the post): `REB/configuration.md` sections 1 to 3. The four counts of item 3 are printed in no result table: they are recomputed from `agent_routing/outputs/verifier_grid/s1/<bench>/test/grid_records.jsonl` against the S_1 test outputs in the two result archives (author note 3b).

---

## Table R1. Equal-information comparison on the test pools, four training seeds

Every rule starts from the same manager-written candidate and uses the same checkpoint, greedy decoding and forced-commit revision step, with at most one advisor call. The grid was run for S_1 and for the re-trainings with training seeds 1, 2 and 3 (logged recipe, assumed for AQuA, which has no training log; they may differ from S_1 in the unlogged seed and maximum sequence length and in the attention kernels).

**R1a. Rules.** Each cell: S_1; mean ± SD over the four seeds. Accuracy (%) unless the label says otherwise. Rows e and m list S_1 / seed 1 / seed 2 / seed 3 and the number of seeds below .05.

| | Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|:--|--:|--:|--:|--:|
| a | Candidate only, no call | 73.4; 73.9 ± 0.7 | 50.8; 51.1 ± 0.3 | 41.0; 41.2 ± 1.3 | 36.2; 36.4 ± 0.7 |
| b | Revision turn on an empty Verifier output, no call | 72.8; 72.9 ± 0.3 | 51.4; 51.5 ± 0.3 | 41.0; 41.2 ± 1.3 | 32.7; 35.1 ± 1.7 |
| c | **Learned calling** (first call, then commit) | 78.8; 78.6 ± 0.4 | 63.2; 64.7 ± 1.0 | 54.0; 51.8 ± 1.7 | 77.2; 76.2 ± 0.7 |
| | Calls per question of row c | 0.37; 0.36 ± 0.01 | 0.33; 0.35 ± 0.02 | 0.52; 0.56 ± 0.12 | 0.77; 0.75 ± 0.02 |
| d | Random questions, same calls per advisor (2,000 draws) | 76.6; 76.9 ± 0.4 | 57.2; 58.1 ± 0.6 | 47.7; 48.2 ± 0.9 | 73.9; 72.3 ± 1.1 |
| e | Share of those draws at least as accurate as row c (one-sided) | .008 / .160 / .139 / .005; 2 of 4 | <.001 / <.001 / <.001 / <.001; 4 of 4 | .005 / .233 / .141 / .196; 1 of 4 | .017 / .007 / .005 / .009; 4 of 4 |
| f | Confidence-gated Verifier, same number of calls as row c | 79.6; 80.0 ± 0.9 | 61.6; 63.3 ± 1.3 | 52.0; 51.2 ± 1.5 | 76.8; 74.4 ± 1.6 |
| g | Always Verifier, with the candidate (1.00 call) | 83.4; 83.5 ± 0.9 | 72.4; 73.2 ± 0.6 | 56.0; 55.8 ± 0.5 | 85.0; 84.6 ± 0.6 |
| h | Always Verifier, candidate removed: the paper's baseline design, rerun under this protocol (1.00 call) | 84.2; 84.3 ± 0.1 | 69.2; 68.5 ± 0.8 | 61.0; 59.5 ± 1.0 | 78.3; 79.1 ± 0.7 |
| i | Always Reasoner (1.00 call) | 81.2; 81.5 ± 0.3 | 55.6; 56.3 ± 0.5 | 45.0; 47.0 ± 1.4 | 37.0; 36.4 ± 0.8 |
| j | Always Extractor (1.00 call) | 79.0; 79.1 ± 0.2 | 52.0; 52.0 ± 0.2 | 39.0; 38.8 ± 1.0 | 29.5; 31.1 ± 1.7 |
| k | Learned selection, all sent to the Verifier (same calls as row c) | 80.2; 80.2 ± 0.3 | 63.4; 65.1 ± 1.1 | 53.0; 51.5 ± 1.7 | 77.2; 76.2 ± 0.7 |
| l | Random subset, Verifier only, same number of calls (2,000 draws) | 77.1; 77.4 ± 0.4 | 58.0; 59.0 ± 0.7 | 48.8; 49.4 ± 1.2 | 73.9; 72.3 ± 1.1 |
| m | Share of those subsets at least as accurate as row k (one-sided) | <.001 / .002 / .011 / <.001; 4 of 4 | <.001 / <.001 / <.001 / <.001; 4 of 4 | .052 / .555 / .115 / .452; 0 of 4 | .017 / .007 / .005 / .009; 4 of 4 |

**R1b. Paired contrasts** (first minus second, accuracy points). "Seeds with p < .05" counts the four exact McNemar p-values of the row.

| Contrast | Pool | S_1: difference [95% CI] | Exact McNemar p: S_1 / seed 1 / seed 2 / seed 3 | Four seeds: mean ± SD | Seeds with p < .05 |
|:--|:--|--:|:--|--:|:--|
| **C1** (c − g): learned − always Verifier with the candidate | MedQA | −4.6 [−7.4, −1.8] | .001 / .002 / .004 / <.001 | −5.0 ± 0.6 | 4 of 4 |
| | MMLU-Pro | −9.2 [−12.2, −6.2] | <.001 / <.001 / <.001 / <.001 | −8.5 ± 0.6 | 4 of 4 |
| | GPQA | −2.0 [−9.0, +5.0] | .774 / .180 / .267 / .375 | −4.0 ± 1.8 | 0 of 4 |
| | AQuA | −7.9 [−11.4, −4.7] | <.001 / <.001 / <.001 / <.001 | −8.5 ± 0.8 | 4 of 4 |
| | Pooled (n=1354) | −6.7 [−8.4, −5.0] | <.001 / <.001 / <.001 / <.001 | −6.9 ± 0.4 | 4 of 4 |
| **C2** (c − f): learned − confidence gate at the same calls | MedQA | −0.8 [−3.6, +2.0] | .672 / .576 / .350 / .229 | −1.4 ± 0.6 | 0 of 4 |
| | MMLU-Pro | +1.6 [−1.6, +5.0] | .416 / .470 / .228 / 1.000 | +1.4 ± 0.8 | 0 of 4 |
| | GPQA | +2.0 [−4.0, +8.0] | .754 / 1.000 / 1.000 / 1.000 | +0.5 ± 1.3 | 0 of 4 |
| | AQuA | +0.4 [−4.3, +5.1] | 1.000 / .401 / .461 / .665 | +1.8 ± 1.0 | 0 of 4 |
| | Pooled | +0.5 [−1.3, +2.4] | .648 / .554 / .507 / .711 | +0.4 ± 0.5 | 0 of 4 |
| **C3** (g − h): always Verifier, with − without the candidate | MedQA | −0.8 [−3.6, +2.0] | .672 / .322 / .576 / .883 | −0.8 ± 0.8 | 0 of 4 |
| | MMLU-Pro | +3.2 [−0.2, +6.6] | .089 / <.001 / .017 / .004 | +4.8 ± 1.3 | 3 of 4 |
| | GPQA | −5.0 [−13.0, +3.0] | .332 / .607 / .607 / .424 | −3.8 ± 1.0 | 0 of 4 |
| | AQuA | +6.7 [+2.0, +11.4] | .012 / .073 / .117 / .014 | +5.5 ± 1.2 | 2 of 4 |
| | Pooled | +1.8 [−0.2, +3.8] | .091 / .021 / .077 / .003 | +2.2 ± 0.5 | 2 of 4 |
| **C4** (g − b): Verifier output − empty Verifier output | MedQA | +10.6 [+7.2, +14.2] | <.001 / <.001 / <.001 / <.001 | +10.6 ± 1.1 | 4 of 4 |
| | MMLU-Pro | +21.0 [+17.0, +25.2] | <.001 / <.001 / <.001 / <.001 | +21.8 ± 0.7 | 4 of 4 |
| | GPQA | +15.0 [+6.0, +24.0] | .003 / .002 / .011 / .004 | +14.5 ± 1.3 | 4 of 4 |
| | AQuA | +52.4 [+45.7, +59.1] | <.001 / <.001 / <.001 / <.001 | +49.5 ± 2.0 | 4 of 4 |
| | Pooled | +22.6 [+20.1, +25.3] | <.001 / <.001 / <.001 / <.001 | +22.3 ± 0.7 | 4 of 4 |
| **C5** (b − a): empty-output revision − candidate | MedQA | −0.6 [−2.0, +0.6] | .549 / .146 / .021 / .581 | −1.0 ± 0.5 | 1 of 4 (a drop) |
| | MMLU-Pro | +0.6 [+0.0, +1.4] | .250 / 1.000 / .754 / 1.000 | +0.4 ± 0.2 | 0 of 4 |
| | GPQA | +0.0 [+0.0, +0.0] | 1.000 / 1.000 / 1.000 / 1.000 | +0.0 ± 0.0 | 0 of 4 |
| | AQuA | −3.5 [−6.3, −0.8] | .022 / 1.000 / 1.000 / .388 | −1.3 ± 1.7 | 1 of 4 (a drop) |
| | Pooled | −0.7 [−1.4, +0.1] | .122 / .405 / .327 / .377 | −0.5 ± 0.1 | 0 of 4 |

Sources: `REB/verifier_grid/seeds_test.md` (rows a to h and C1 to C4 per benchmark, with their means and SDs), `REB/verifier_grid/{s1,seed1,seed2,seed3}_test.md` (rows i to m, C5, pooled rows, intervals). The means and SDs of rows i to l, of C5 and of the pooled rows are not printed in those files; they are computed for this response from the full-precision per-seed values in `REB/verifier_grid/analysis/<seed>_<bench>_test.json`. Those files hold no pooled test: a seed's pooled difference is the question-weighted mean (500 / 500 / 100 / 254 of 1354) of its four per-benchmark differences. Recomputing from the rounded values printed in the `*_test.md` files can move a last digit (for example C3 pooled: ± 0.6 instead of ± 0.5; row l on MedQA: ± 0.3 instead of ± 0.4).

Notes to Table R1. Everything below is S_1 unless a seed is named; the development pools, the development-threshold gates and the independent GPQA pool were run for S_1 only.

- **Rows d and l, 95% of draws (S_1).** Row d: [74.8, 78.4], [55.2, 59.2], [43.0, 52.0], [70.9, 76.8]. Row l: [75.4, 78.8], [56.0, 60.0], [44.0, 53.0], [70.9, 76.8]. On AQuA every learned call goes to the Verifier, so rows d and l coincide there.
- **Choice of advisor.** First calls by role (Extractor / Reasoner / Verifier): 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196. Learned calling minus the same questions all sent to the Verifier (row c − row k), pooled: −0.5 [−1.4, +0.4], p = .311; seeds 1 to 3: −0.8 [−1.7, +0.1], p = .099; −0.7 [−1.6, +0.1], p = .132; −0.8 [−1.7, +0.1], p = .099.
- **Confidence gate.** Row f ranks the questions inside the test pool, so it is not a deployable gate. With a threshold fixed on the development pool: (i) threshold chosen to match the learned call rate on development: 79.2 / 58.0 / 53.0 / 76.0 at 0.31 / 0.24 / 0.71 / 0.75 calls; learned minus gate is +2.1 [+0.1, +4.0], p = .043 pooled, but the calls differ from row c, so this is not an equal-budget result. (ii) Threshold chosen for best development accuracy: 83.0 / 71.4 / 55.0 / 81.9 at 0.80 / 0.68 / 0.84 / 0.93 calls; learned minus gate is −5.5 [−7.3, −3.8], p < .001 pooled.
- **Development pools** (n = 200 / 200 / 100 / 254). Share of random draws at least as accurate as learned calling: .024 / .185 / .111 / .197. C1: −5.5 (.027), −11.5 (<.001), −4.0 (.388), −7.9 (<.001). C2: −1.5 (.664), −3.0 (.345), −4.0 (.388), −2.8 (.296). Share of Verifier-only random subsets at least as accurate as the learned selection sent to the Verifier: .003 / .203 / .075 / .197. On the MedQA development pool always-Reasoner is 85.0 against 84.5 for always-Verifier with the candidate.
- **GPQA on an independent pool.** The GPQA development pool has 100 questions that are disjoint from Diamond-100 and from training. There learned calling scores 53.0 at 0.62 calls and always-Verifier with the candidate 57.0 (−4.0 [−11.0, +3.0], p = .388); with − without the candidate is +5.0 [−5.0, +15.0], p = .424. Learned calling is not above random there (.111 of random draws are at least as accurate) and is 4.0 points below the confidence gate at the same calls (p = .388).

Sources for the notes: `REB/verifier_grid/s1_test.md`, `REB/verifier_grid/seed{1,2,3}_test.md`, `REB/verifier_grid/analysis/s1_{medqa,mmlu_pro,gpqa,aqua}_dev.md`, `REB/configuration.md` section 3.

## Table R2. What training changed: the trained manager against the untrained base model as manager

Same questions, advisors and advisor outputs for both managers; the untrained model is held to the same output grammar as the trained manager. Upper block: S_1; cells are untrained → trained, accuracy in %, and the last column is trained − untrained, pooled. Lower block: the pooled estimate of the same rows in each of the four seeds.

| | Quantity (S_1) | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) | Pooled, pt [95% CI], p |
|:--|:--|--:|--:|--:|--:|:--|
| 1 | Questions whose first turn follows the draft / commit / call format without the grammar | 0 → 500 of 500 | 2 → 499 of 500 | 0 → 100 of 100 | 27 → 254 of 254 | untrained: 29 of 1354 |
| 2 | Reference: letter only under a plain prompt (untrained weights; S_1 weights) | 74.6; 74.4 | 53.0; 53.8 | 45.0; 44.0 | 41.7; 42.1 | – |
| 3 | Draft under the manager format | 70.6 → 73.4 | 44.2 → 50.8 | 32.0 → 41.0 | 32.7 → 36.2 | +4.8 [+2.7, +6.9], <.001 |
| 4 | Use of evidence: accuracy after a forced Verifier call on same-candidate questions (n = 434 / 313 / 65 / 150) | 85.7 → 85.7 | 77.6 → 77.6 | 58.5 → 56.9 | 84.7 → 86.0 | +0.1 [−0.6, +0.8], 1.000 (n = 962) |
| 5 | Selection signal: the trained manager's Verifier outcomes on the questions chosen by the untrained model's confidence → by the learned policy, same number of calls | 79.8 → 80.2 | 62.0 → 63.4 | 52.0 → 53.0 | 75.6 → 77.2 | +1.0 [−0.7, +2.7], .265 |
| 6 | Whole system at equal calls (0.37 / 0.33 / 0.52 / 0.77): untrained manager + Verifier on its least confident candidates → learned calling | 80.6 → 78.8 | 57.6 → 63.2 | 47.0 → 54.0 | 71.7 → 77.2 | +3.0 [+0.7, +5.2], .011 |
| | Row 6, difference per benchmark | −1.8 [−4.4, +0.8], .243 | +5.6 [+1.4, +9.8], .012 | +7.0 [+0.0, +14.0], .118 | +5.5 [−0.4, +11.0], .081 | – |
| 7 | Whole system at 1.00 call: always Verifier with the candidate | 83.6 → 83.4 | 70.4 → 72.4 | 52.0 → 56.0 | 81.9 → 85.0 | +1.6 [+0.1, +3.0], .044 |
| 8 | Reference, as deployed: untrained manager without the grammar, up to 3 calls: accuracy / calls (valid answers) | 82.8 / 2.98 (95%) | 69.0 / 2.51 (85%) | 39.0 / 2.53 (52%) | 78.3 / 1.62 (84%) | – |
| 9 | Reference, as deployed: S_1 (a second call is allowed): accuracy / calls; other seeds in Table R3 | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 | – |

**Pooled estimate in each of the four seeds** (trained − untrained over the 1354 questions, pt [95% CI], exact McNemar p).

| Row | Contrast | S_1 | Seed 1 | Seed 2 | Seed 3 |
|:--|:--|:--|:--|:--|:--|
| 3 | Draft | +4.8 [+2.7, +6.9], <.001 | +5.3 [+3.2, +7.5], <.001 | +5.5 [+3.4, +7.7], <.001 | +4.9 [+2.7, +7.2], <.001 |
| 4 | Use of evidence (same-candidate questions) | +0.1 [−0.6, +0.8], 1.000 (n = 962) | +0.2 [−0.4, +0.8], .754 (n = 956) | +0.3 [−0.3, +1.0], .549 (n = 956) | +0.6 [−0.1, +1.4], .180 (n = 952) |
| 5 | Selection signal | +1.0 [−0.7, +2.7], .265 | +1.0 [−0.6, +2.7], .265 | +1.2 [−0.5, +2.9], .201 | +0.9 [−0.8, +2.7], .359 |
| 6 | Whole system at equal calls | +3.0 [+0.7, +5.2], .011 | +2.7 [+0.5, +4.9], .017 | +3.2 [+0.9, +5.4], .007 | +2.9 [+0.7, +5.1], .013 |
| 7 | Whole system at 1.00 call | +1.6 [+0.1, +3.0], .044 | +1.7 [+0.3, +3.1], .022 | +1.6 [+0.1, +3.0], .040 | +2.4 [+1.0, +3.9], .001 |

Sources: `REB/attribution/s1_vs_base_test.md` (and `.json`), `REB/attribution/seed{1,2,3}_vs_base_test.md` (pooled lines at the end of each section).

Notes to Table R2 (S_1 unless a seed is named).

- **Row 4 caveat.** Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: 77.4 / 47.0, 61.0 / 33.7, 41.5 / 40.0, 37.3 / 34.6); questions where the drafts differ cannot be compared this way. With another advisor forced on the same questions the pooled difference is +1.1 [−0.3, +2.6], p = .161 (Reasoner) and +0.8 [−0.7, +2.4], p = .366 (Extractor); per benchmark the Extractor on MedQA is +2.5 [+0.5, +4.8], p = .035. In two re-trainings the Extractor row is significant on AQuA with the trained manager lower: seed 2, −4.0 [−7.3, −1.3], p = .031; seed 3, −7.1 [−11.6, −3.2], p = .003. These three are the only per-benchmark cells of the forced-call rows (Verifier with and without the candidate, Reasoner, Extractor, empty output) that are significant by the exact McNemar test in any seed; for the Verifier's output (row 4) none is, per benchmark or pooled.
- **Right candidates broken by a Verifier call on same-candidate questions (%), trained / untrained.** S_1: 3.9 / 3.9, 3.1 / 2.6, 11.1 / 7.4, 7.1 / 3.6. Seed 1: 3.9 / 3.9, 2.6 / 2.6, 11.1 / 7.4, 10.3 / 6.9. Seed 2: 3.8 / 3.8, 2.6 / 2.6, 11.1 / 7.4, 10.3 / 6.9. Seed 3: 3.4 / 3.4, 2.6 / 2.6, 11.1 / 7.4, 8.3 / 5.0.
- **Learned call rate on wrong / right candidates (%).** S_1: 63.2 / 27.0, 48.8 / 18.1, 50.8 / 53.7, 84.6 / 64.1. Seed 1: 59.4 / 26.3, 51.0 / 20.2, 55.0 / 55.0, 82.1 / 59.8. Seed 2: 61.4 / 30.3, 53.1 / 21.0, 43.9 / 44.2, 82.8 / 60.4. Seed 3: 61.2 / 26.5, 52.8 / 21.7, 69.5 / 78.0, 81.1 / 56.8.
- **Row 6, exact decomposition (S_1)** = draft + what the same confidence-gated Verifier calls add (trained − untrained) + learned calling − the trained manager's own confidence gate. MedQA: −1.8 = +2.8 − 3.8 − 0.8. MMLU-Pro: +5.6 = +6.6 − 2.6 + 1.6. GPQA: +7.0 = +9.0 − 4.0 + 2.0. AQuA: +5.5 = +3.5 + 1.6 + 0.4.
- **Row 6 per benchmark in seeds 1 / 2 / 3** (difference, exact McNemar p). MedQA: −2.4 (.119), −2.0 (.212), −1.6 (.332). MMLU-Pro: +6.8 (.001), +6.8 (.002), +6.0 (.006). GPQA: +2.0 (.774), +7.0 (.143), +4.0 (.344). AQuA: +5.1 (.111), +4.7 (.141), +5.1 (.111). Per benchmark the equal-call difference is significant on MMLU-Pro only, in all four seeds.
- **Rows 8 and 9.** Both are runs on the test pools of this response. Row 9 is our re-evaluation of the deployed S_1, not the submitted Table 1 (79.8 / 60.2 / 76.8 on MedQA / MMLU-Pro / AQuA, as quoted by eCWp): the MedQA checkpoint is a substitute, the MMLU-Pro pool is new and disjoint, and AQuA is the official test split evaluated again. The untrained system of row 8 returns a valid answer on 95 / 85 / 52 / 84% of questions, so its GPQA accuracy (39.0) reflects invalid outputs as well as wrong answers.
- **Sibling of S_1 trained with no commit cap** (same collection and recipe, every commit row kept; one training run; it is not a commit-only or manager-only control). S_1 − sibling, pooled: draft −0.3 [−1.4, +0.8], p = .699; evidence −0.1 [−0.6, +0.4], p = 1.000 (n = 1253); selection +0.1 [−1.6, +1.8], p = 1.000; equal calls −0.7 [−2.7, +1.3], p = .563; always Verifier −0.1 [−0.9, +0.7], p = 1.000. The sibling's learned calling scores 77.2 / 59.2 / 54.0 / 69.7 at 0.23 / 0.21 / 0.68 / 0.63 calls. Sources: `REB/attribution/s1_vs_allc_test.md`, `REB/verifier_grid/allc_test.md`.
- **Matched-data success distillation** (one training run; the same round-1 records and the same commit cap; each kept record trains one successful option drawn at random instead of the shortest one; same SFT recipe; the default selection of the same records reproduces the S_1 label file row for row). Its learned calling scores 81.6 / 73.2 / 50.0 / 83.1 at 0.98 / 1.00 / 0.99 / 0.96 calls, against 78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 for S_1. S_1 − success distillation: MedQA −2.8 [−5.4, −0.2], p = .044; MMLU-Pro −10.0 [−13.2, −6.8], p < .001; GPQA +4.0 [−4.0, +12.0], p = .454; AQuA −5.9 [−9.8, −2.0], p = .006; pooled −5.5 [−7.3, −3.8], p < .001. Accuracy gained per call: 14.8 / 37.3 / 25.0 / 53.1 points for S_1, 10.0 / 23.2 / 10.1 / 47.1 for success distillation. First calls to Extractor / Reasoner / Verifier: S_1 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196; success distillation 66 / 191 / 231, 73 / 59 / 367, 54 / 16 / 29, 0 / 0 / 244. Its draft accuracy is 71.8 / 50.0 / 40.0 / 37.8 (S_1: 73.4 / 50.8 / 41.0 / 36.2). Against always calling the Verifier with the candidate from its own draft it is not ahead (pooled −1.6 [−2.7, −0.4], p = .014). Sources: `REB/verifier_grid/s1_vs_sucd_policy.md`, `REB/verifier_grid/sucd_test.md`.

## Table R3. Deployed evaluation (paper protocol) across training seeds

The manager routes freely and may call more than once. Cells: accuracy (%) / advisor calls per question on the test pools.

| Checkpoint | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| S_1 | 78.8 / 0.366 | 63.6 / 0.352 | 54.0 / 0.520 | 77.2 / 0.776 |
| Seed 1 | 78.0 / 0.348 | 65.4 / 0.374 | 50.0 / 0.550 | 76.0 / 0.740 |
| Seed 2 | 78.6 / 0.386 | 65.6 / 0.404 | [deployed evaluation of seed 2 on GPQA pending] | 76.0 / 0.748 |
| Seed 3 | [deployed evaluation of seed 3 pending] | [deployed evaluation of seed 3 pending] | [deployed evaluation of seed 3 pending] | [deployed evaluation of seed 3 pending] |
| Mean ± SD, checkpoints present | 78.5 ± 0.4 / 0.367 ± 0.019 | 64.9 ± 1.1 / 0.377 ± 0.026 | 52.0 ± 2.8 / 0.535 ± 0.021 (two checkpoints) | 76.4 ± 0.7 / 0.755 ± 0.019 |

Note. Corrupted answers (% of questions; mean ± SD over the checkpoints present): 3.9 ± 0.2 / 1.4 ± 0.2 / 1.0 ± 1.4 / 2.8 ± 0.0. Valid answers: 99.8% for seed 2 on MedQA, 100.0% in every other cell.

Source: `REB/verifier_grid/deploy_seeds.md`.

## Figure R1

File: `agent_routing/docs/mcq_rsi_results/paper_assets/fig/fig_rebuttal_budget_s1.pdf`.

Caption: "Accuracy against advisor calls per question for S_1 on the four test pools. Circle: candidate only. Diamond: learned calling (first call). Curve: Verifier on the least confident candidates. Dashed line: Verifier on a random subset, with the 95% interval at the learned number of calls. Squares: always Verifier with the candidate (filled) and with the candidate removed (open)."

---

## Reviewer cPCG: weakness 1 (the always-Verifier baseline)

Thank you. Weakness 1 is correct, and the rerun changes our conclusion.

**What each call receives.** The policy's Verifier call carries the question, options and context plus two lines naming the manager's draft letter as the candidate to audit; the paper's always-Verifier baseline omitted those lines. The Extractor and Reasoner never receive the candidate.

**Equal-information rerun (Table R1), four training seeds.** All rules share the candidate, checkpoint, greedy decoding and revision step and make at most one call. Always-Verifier with the candidate (row g) is more accurate than learned calling (row c) in every seed on MedQA, MMLU-Pro and AQuA, by 5.0 to 8.5 points on average (contrast C1 lists each exact p). On GPQA (n = 100) it is not significantly ahead in any seed. Without the candidate (row h) it is still numerically above learned calling on every pool in every seed.

**Does the candidate matter (C3)?** It helps on MMLU-Pro (significant in three of four seeds) and AQuA (two of four), has no detectable effect on MedQA, and is unresolved on GPQA, where no seed is significant.

**Other advisors.** Always-Reasoner and always-Extractor (rows i, j) are numerically below always-Verifier on every test pool in every seed; no paired test was run.

**What survives.** Learned calling is a lower-call operating point, not a more accurate one. It is significantly above random questions at the same calls per advisor (row e: share of random draws below .05) in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA. It is not significantly different from a confidence-gated Verifier at the same calls, ranked within the test pool, in any seed on any benchmark (C2).

We withdraw the claim that learned calling matches or exceeds always-Verifier; Table R1 and Figure R1 will replace the baseline.

---

## Reviewer sCpX: weaknesses 1 to 4

Thank you. Over four training seeds, your controls do not support the always-Verifier claim.

**W1.** Agreed.
- Empty-output revision turn: no significant gain in any seed (Table R1, C5).
- Evidence: the Verifier's output adds a mean 10.6 to 49.5 points over that turn, significant in every seed and benchmark (C4).
- SFT against the untrained model, same grammar (Table R2): draft +4.8 to +5.5 points pooled; use of the Verifier's output and selection not significantly different in any seed.
- Equally trained manager-only control: [commit-only control pending].
- Matched-data success distillation (Table R2 notes) calls on almost every question and is 5.5 points more accurate pooled (p < .001; one run): the shortest path adds stopping, not accuracy.

**W2.**
- Fixed roles on the test pools (rows g to j): always-Verifier is numerically highest.
- Random questions, same calls per advisor (row e: share of draws below .05): learned calling is significantly above in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA.
- Confidence-gated Verifier, same calls, ranked within the test pool: not significantly different in any seed or benchmark (C2); a development-threshold gate at 0.68 to 0.93 calls is 5.5 points more accurate pooled (S_1; Table R1 notes).

**W3.** Correct. With identical candidates, checkpoint, decoding and revision, always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1; Figure R1). Random-subset Verifier (rows k to m): significantly below the learned selection in every seed on those three, in none on GPQA.

**W4.** GPQA: conceded; Table 4 used Diamond-100, the evaluation set. On a disjoint pool, S_1's learned calling scores 53.0 against 57.0 for always-Verifier (p = .388; Table R1 notes). AQuA: official validation and test splits, no shared question. The training code selects no checkpoint among epochs; how ρ was selected is not recorded. Seeds: common block, item 2.

W5: co-authors; advisors: common block.

---

## Reviewer A2jt: call selection and roles; attribution and baselines

Thank you. Both weaknesses hold; four training seeds sharpen them.

**Uneven call selection and harmful roles.** Agreed.
- Random questions, same calls per advisor (Table R1, row e: share of draws below .05): learned calling is significantly above in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA; on the development pools (S_1 only), only on MedQA.
- Confidence-gated Verifier, same calls, ranked within the test pool: not significantly different in any seed on any benchmark (C2).
- Call gap: none on GPQA in any seed; 20.5 to 24.3 points on AQuA (Table R2 notes).
- Roles: always-Extractor is numerically below the candidate on AQuA and GPQA in every seed; always-Reasoner is within 2 points of it on AQuA (rows a, i, j; no paired test). The choice of advisor adds nothing measurable (Table R1 notes).
- Corruption: training does not reduce it (Table R2 notes).

**Confounded attribution and asymmetric baselines.** Correct. With the candidate, always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1), at one call per question against 0.33 to 0.77.

Against the untrained model under the same output grammar (Table R2), in all four seeds:
- the draft rises (pooled +4.8 to +5.5) without exceeding that model's letter-only accuracy;
- the use of the Verifier's output on same-candidate questions does not differ significantly;
- the learned choice of questions is not significantly better than that model's own confidence.

A gain from routing alone is not established: training provides protocol-following output and a lower call rate. Matched-data success distillation agrees: trained on a random successful option, the manager calls on almost every question and is 5.5 points more accurate pooled (p < .001; one run).

Forced-all with other role orders and an external learned router were not run; both remain limitations.

---

## Reviewer eCWp: weaknesses 1 and 2

Thank you. Both points are correct.

**W1.** Agreed. In no seed does the trained manager's draft exceed the untrained model's letter-only accuracy on any benchmark (Table R1, row a; Table R2, row 2). The drop comes from the manager format, not the weights (letter-only, S_1 is within one point of the untrained model); under it, training recovers 4.8 to 5.5 points pooled (row 3). The two managers' use of the Verifier's output is not significantly different in any seed (row 4): training taught the protocol.

Your untrained-system comparison stands: the deployed managers ([deployed evaluation of seed 3 pending]) make 0.35 to 0.78 calls per question against at least 1.62, with lower accuracy on MedQA, MMLU-Pro and AQuA (Table R2, rows 8 and 9; Table R3).

Our numbers differ from your Table 1 values: substitute MedQA checkpoint and new MMLU-Pro pool (common block, item 2); AQuA is 77.2 on re-evaluation against 76.8, and we hold no record of the paper run.

**W2.** Agreed. The 0 to 4.5 point advantage of Figure 3 is not supported under equal information and we withdraw it: always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1).

The gain is the Verifier's output (C4: significant over an empty-output revision turn in every seed and benchmark), not that turn alone (C5).

Learned calling is significantly above random questions at the same calls per advisor (row e: share of random draws below .05) in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA; on the development pools (S_1 only), only on MedQA, as you read Table 10. At the same calls it is not significantly different from ranking the test pool by the manager's or the untrained model's confidence in any seed (C2; Table R2, row 5).

We will restate the contribution as protocol-following output at fewer calls, not accuracy.

---

## Changes we will make to the paper

Revised claim sentences.

1. **Abstract, headline claim.** "On four test pools the trained manager makes 0.35 to 0.78 advisor calls per question (the training seeds evaluated as deployed so far; [deployed evaluation of seed 3 pending]; [deployed evaluation of seed 2 on GPQA pending]), against 1.62 to 2.98 for the untrained manager with the same advisors, which returns a valid answer on 95 / 85 / 52 / 84% of questions; accuracy of the original manager is 78.8 / 63.6 / 54.0 / 77.2 against 82.8 / 69.0 / 39.0 / 78.3 (MedQA / MMLU-Pro / GPQA / AQuA-RAT). Against the untrained model held to the same output grammar with a confidence-gated Verifier at the same number of calls, the trained manager is ahead by 2.7 to 3.2 points pooled across four training seeds (original manager: 3.0 [+0.7, +5.2]); per benchmark the difference is significant on MMLU-Pro only (in all four seeds), and on MedQA it is negative in every seed and not significant. Training buys output that follows the protocol and fewer calls, not higher accuracy."
2. **Abstract and Section 4.1, always-Verifier (replaces the Figure 3 claim; the Figure 3 configuration itself was not rerun, limitation 17).** "Under equal information (same candidate, checkpoint, decoding and revision step), calling the Verifier on every question reaches 83.5 ± 0.9 / 73.2 ± 0.6 / 55.8 ± 0.5 / 84.6 ± 0.6 (mean ± SD over four training seeds) at 1.00 call, and learned calling 78.6 ± 0.4 / 64.7 ± 1.0 / 51.8 ± 1.7 / 76.2 ± 0.7 at 0.36 / 0.35 / 0.56 / 0.75 calls. The difference is significant in every seed on MedQA (exact McNemar p = .001, .002, .004, < .001), MMLU-Pro (< .001, < .001, < .001, < .001) and AQuA-RAT (< .001, < .001, < .001, < .001), and in no seed on GPQA (n = 100; p = .774, .180, .267, .375)."
3. **Section 4, the +17.1-point sentence.** "The gain over the manager's pre-call draft is produced by the Verifier's output: a revision turn on an empty Verifier output does not improve the draft in any of four training seeds (pooled −0.5 ± 0.1 points; original manager −0.7 [−1.4, +0.1], p = .122), and the Verifier's output adds 10.6 / 21.8 / 14.5 / 49.5 points over that turn (four-seed means), significant in every seed on every benchmark." Source: Table R1b, C5 and C4; the pooled −0.5 ± 0.1 is printed in no result file and is computed from the four per-seed pooled differences (−0.7 / −0.4 / −0.4 / −0.4 as printed).
4. **Section 4.4, candidate conditioning.** "Passing the candidate to the Verifier raises accuracy on MMLU-Pro (significant in three of four training seeds; +4.8 ± 1.3 points; exact McNemar p = .089, < .001, .017, .004) and on AQuA-RAT (significant in two of four; +5.5 ± 1.2; p = .012, .073, .117, .014). It has no detectable effect on MedQA (−0.8 ± 0.8; p = .672, .322, .576, .883), and on GPQA its sign is unresolved (−3.8 ± 1.0; p = .332, .607, .607, .424; n = 100)."
5. **Selection (discussion of Table 10).** "On the test pools learned calling is more accurate than calling on random questions with the same calls per advisor in four of four training seeds on MMLU-Pro and AQuA-RAT, in two of four on MedQA and in one of four on GPQA (one-sided share of 2,000 random draws below .05); on the development pools, evaluated for the original manager, only on MedQA. Against random subsets sent to the Verifier only, the learned selection is above random in four of four seeds on MedQA, MMLU-Pro and AQuA-RAT and in none on GPQA. It is not significantly different from ranking questions by the manager's draft confidence at the same number of calls in any seed on any benchmark (pooled +0.4 ± 0.5 points). For the original manager, a confidence gate whose threshold is fixed on the development pool for best development accuracy makes 0.80 / 0.68 / 0.84 / 0.93 calls and is 5.5 points more accurate [+3.8, +7.3]." Source: Table R1, rows e and m, C2 and the notes; the pooled +0.4 ± 0.5 is printed in no result file and is computed from the four per-seed pooled differences (+0.5 / +0.7 / +0.7 / −0.4 as printed); the development-pool result comes from `REB/verifier_grid/analysis/s1_{medqa,mmlu_pro,gpqa,aqua}_dev.md`.
6. **New attribution paragraph.** "Relative to the untrained model under the same output grammar, training raises the draft by 4.8 to 5.5 points pooled across four training seeds (original manager: [+2.7, +6.9]) without exceeding that model's letter-only accuracy. The use of the Verifier's output on same-candidate questions is not significantly different (pooled +0.1 to +0.6; 952 to 962 of 1354 questions), and the learned policy does not select questions significantly better than the untrained model's confidence (pooled +0.9 to +1.2). Without the grammar the untrained model leaves the protocol on 1325 of 1354 questions."
7. **New control paragraph.** "With the same records and commit cap, a manager trained on one successful option drawn at random instead of the shortest one calls on 96 to 100% of questions and is 5.5 points more accurate pooled [+3.8, +7.3] (one training run): the shortest-path target buys fewer calls, not accuracy." The commit-only control is added here when it is finished.

Corrected advisor description.

8. **Setup, replaces "role-specific LoRA specialists".** "The Extractor, Reasoner and Verifier are the base model Qwen3.5-9B under role-specific prompts. Role adapters were trained but were not applied at serving time." The evidence (Verifier 20 of 20 outputs identical to the base model, Extractor 5 of 20, no Reasoner check) goes in the appendix.

New tables and figure.

9. Table R1 replaces the forced-Verifier baseline of Section 4.1 and gives the test-pool version of the fixed-role and random controls, with the random-subset Verifier and the confidence gate added, for four training seeds.
10. Table R2 becomes a new subsection on what training changed.
11. Table R3 reports the deployed evaluation of each training seed next to Table 1.
12. Figure R1 (accuracy against calls) is added next to Table R1.
13. **Table 4 caption.** "The GPQA row is measured on Diamond-100, which is also the evaluation set; on a disjoint 100-question pool learned calling reaches 53.0 at 0.62 calls and always-Verifier with the candidate 57.0." Source: `REB/verifier_grid/analysis/s1_gpqa_dev.md` (S_1 only; notes to Table R1).

Limitation statements.

14. "These comparisons use four training seeds (the original manager and three re-trainings of its label file with the logged recipe) and one test pool per benchmark. The intervals reflect the sampling of questions; the standard deviations reflect the four seeds, which share that pool. The seed and maximum sequence length of the original run were not logged; the re-trainings use fast attention kernels that the original run may not have used; their rendered training inputs differ from the original ones in the tool schemas (a null `tool_calls` field in each); the AQuA-RAT recipe is assumed from the other three benchmarks; and for AQuA-RAT two label files of 402 rows that differ in 52 commit rows exist, and which one the original manager was trained on is not established. The re-trainings may therefore differ from the original in more than the seed. GPQA has 100 questions and does not resolve the differences reported here (learned calling minus always-Verifier: −4.0 ± 1.8 points, significant in no seed)."
15. "The learned policy is not shown to select questions better than a confidence threshold, and its choice of advisor adds nothing measurable (pooled −0.5 to −0.8 points across seeds, significant in none)."
16. "Training does not reduce corruption: on same-candidate questions a Verifier call breaks 3.9 / 3.1 / 11.1 / 7.1% of right candidates for the original trained manager and 3.9 / 2.6 / 7.4 / 3.6% for the untrained one, and in no seed does the trained manager break a smaller share on any benchmark."
17. "Forced-all with other role orders and external learned routers were not evaluated. The Figure 3 configuration at about one call per question was not rerun: the comparison that replaces it sets first-call learned calling at 0.33 to 0.77 calls against always-Verifier at one call."
18. "The test pools were also used by our earlier runs."
19. "The MedQA manager evaluated here was trained with the paper's configuration but is not the paper's checkpoint, which was not archived."
20. "The development pools, the development-threshold confidence gates and the disjoint GPQA pool were evaluated for the original manager only; the success-distillation and no-cap controls are single training runs."

## Pending

- **Commit-only control (training now).** The same records with every commit row and no call row, so the manager is never shown a call. It answers sCpX's "equally trained manager-only revision control": it fills the bracketed placeholder in sCpX W1 (the only place where it stands), and then gets a note under Table R2 and a sentence in change 7. Until then the manager-only evidence is the empty-output revision turn at inference (C5); the no-cap sibling is not a commit-only control. sCpX has no word to spare (296 words without the heading line, 300 with it): the result sentence replaces a 3-word placeholder, so the section needs a cut of that sentence's length less 3 words. Candidates: the second sentence of W3 (random-subset Verifier, 21 words; its counts can join the W2 bullet on random questions as row m) and the W5 pointer line (5 words).
- **Deployed evaluations missing from `REB/verifier_grid/deploy_seeds.md`.** Seed 3 on all four benchmarks and seed 2 on GPQA. They fill five cells and the means of Table R3. Recheck afterwards: the range "0.35 to 0.78 calls" (eCWp W1; change 1), "lower accuracy on MedQA, MMLU-Pro and AQuA" (eCWp W1) and the note under Table R3. The bracketed seed-3 placeholder also stands in eCWp W1 (5 words, freed when it goes) and in change 1, which carries the seed-2 GPQA placeholder as well.
- **Done since the previous draft.** Equal-information grids and attribution tables for seeds 2 and 3 (four seeds per benchmark); success distillation and the no-cap sibling (one training run each).

## Notes for the author (not part of the response)

1. **Length.** Counting whitespace-separated tokens that contain a letter: cPCG 292, sCpX 296, A2jt 290, eCWp 296 words without the heading line (298 / 300 / 299 / 300 with it); common block 213 (223 with its heading; its source line is not counted). The common block is to be posted once: pasted above each response it would give posts of about 500 words. No section has room left. To stay inside the limit this revision shortened the heading lines of sCpX and A2jt (sCpX W4 is still answered in part only: the RSI contrasts are the co-authors'), wrote sCpX W2 as three bullets, and reduced the description of success distillation in sCpX W1 to a pointer to the Table R2 notes. No paragraph of a reviewer section holds more than three numbers; row letters, contrast labels (C1 to C5) and table numbers are not counted, and seed counts are written as words.
2. **What left the reviewer text and where it is.** Per-benchmark differences, intervals and p-values: Table R1b. The development-threshold confidence gate (0.80 / 0.68 / 0.84 / 0.93 calls, 5.5 points above learned calling for S_1): notes to Table R1 and change 5; sCpX W2 now cites it ("0.68 to 0.93 calls" is the range over the four benchmarks), and cPCG, A2jt and eCWp W2 say that the same-calls gate ranks within the test pool but do not repeat the 5.5 points, for which eCWp has no room. The gate with the threshold matched to the development call rate (+2.1 for learned calling, p = .043, at different calls) is in the notes to Table R1 only. The independent GPQA pool against random (.111): notes to Table R1. Corruption and call-rate numbers: notes to Table R2 (A2jt gives the AQuA call gap, 20.5 to 24.3 points). Left out in this revision: "three effects were mixed" and the description of the success-distillation target (sCpX W1), "not a better answer" (eCWp W1, which contradicted Table R2 rows 3, 6 and 7), "four training seeds confirm them" (eCWp opening, which did not hold for the deployed comparison). The comparison of the untrained manager + always-Verifier with learned calling (eCWp): Table R2 row 7 against Table R1 row c; note 4(e). The 1325 of 1354 count: Table R2 row 1 and change 6.
3. **Numbers that no listed result table prints.** (a) Means and SDs of rows i to l, of C5 and of the pooled rows of Table R1b: computed from the four full-precision per-seed values in `REB/verifier_grid/analysis/<seed>_<bench>_test.json` with the sample SD; the same computation reproduces every mean ± SD that `seeds_test.md` prints. Those files hold no pooled test, so a seed's pooled difference is the question-weighted mean of its four per-benchmark differences; before rounding the pooled rows are C1 −6.869 ± 0.405, C2 +0.369 ± 0.549, C3 +2.234 ± 0.547, C4 +22.323 ± 0.664, C5 −0.480 ± 0.128, and the per-seed pooled values agree with the pooled column of the four `*_test.md` files. The same holds for "pooled −0.5 ± 0.1" (change 3) and "pooled +0.4 ± 0.5" (change 5). Rounding of exact ties: `seeds_test.md` prints seven of them rounded down (41.2, 41.2, 51.2, 83.5, 0.35, −8.5, +10.6 are 41.25, 41.25, 51.25, 83.55, 0.355, −8.55, +10.65: rows a, b and f on GPQA, row g on MedQA, calls per question on MMLU-Pro, C1 on MMLU-Pro, C4 on MedQA) and its other exact ties away from zero (for example 64.65 → 64.7, −4.95 → −5.0). All are kept as that file prints them, and 83.5, 8.5 and 10.6 are reused in cPCG, sCpX W1 and changes 2 and 3. The cells computed for this response round ties half-up: the MMLU-Pro mean of C5 is exactly +0.35 and is written +0.4, row j on GPQA is 38.75 → 38.8, row k on MedQA 80.15 → 80.2. (b) Common block, item 3: recomputed again for this revision from `agent_routing/outputs/verifier_grid/s1/<bench>/test/grid_records.jsonl` against the S_1 test outputs in `agent_routing/docs/mcq_rsi_results/results/grpo_grid_locked_tests_v3g.tar.gz` and `grpo_grid_mmlu_pro_v3g_split.tar.gz`: right/wrong agrees on 500/500, 498/500, 100/100, 254/254. The result tables print a different check (first turn equal to unconstrained greedy decoding: 500/500, 499/500, 100/100, 254/254 for S_1). (c) Development pools and the independent GPQA pool: `REB/verifier_grid/analysis/s1_{medqa,mmlu_pro,gpqa,aqua}_dev.md`. (d) The sibling's learned-calling row: `REB/verifier_grid/allc_test.md`.
4. **Wording narrower than a one-line summary, because the tables are.** (a) C5: no seed shows a significant gain; two cells show a significant drop (AQuA for S_1, −3.5, p = .022; MedQA for seed 2, −1.6 [−3.0, −0.4], p = .021), so the text says "no significant gain" or "does not improve". (b) C3 on GPQA: the point estimate is negative in all four seeds (−5.0, −3.0, −3.0, −4.0) and significant in none; the seeds share one pool of 100 questions, so "unresolved" stays. (c) Use of evidence: not significant pooled in any seed by the exact McNemar test; for seed 3 on MedQA the bootstrap interval excludes zero (+1.0 [+0.2, +1.9], bootstrap p = .033, McNemar p = .125). This holds for the Verifier's output only: with the Extractor forced the trained manager is significantly lower on AQuA in seed 2 (−4.0, p = .031) and seed 3 (−7.1, p = .003) and higher on MedQA for S_1 (+2.5, p = .035), so the reviewer sections say "use of the Verifier's output", "not significantly different", and the Row 4 caveat lists the three cells. (d) Draft against the untrained letter-only accuracy: for seed 2 on MedQA the two are equal (74.6), so the text says "does not exceed". (e) Untrained manager + always-Verifier (83.6 / 70.4 / 52.0 / 81.9) against learned calling: above on MedQA, MMLU-Pro and AQuA in every seed; on GPQA below for S_1 (54.0), tied for seed 3 (52.0), above for seeds 1 and 2 (50.0, 51.0); no paired test exists, so eCWp W1 says "numerically" and names three benchmarks. (f) Table R2 row 7 (always Verifier, trained − untrained) is the least stable pooled estimate: +1.6, +1.7, +1.6, +2.4 with p from .044 to .001. (g) Always-Reasoner and always-Extractor against always-Verifier are descriptive; on the MedQA development pool always-Reasoner is 85.0 against 84.5 (S_1). On the AQuA test pool always-Reasoner against the no-call candidate is 37.0 / 36.2, 37.0 / 36.2, 36.2 / 35.8, 35.4 / 37.4 (above in three seeds, below in one; no paired test), so A2jt says "within 2 points". (h) "No call gap on GPQA in any seed": the call rate on wrong / right candidates is 50.8 / 53.7, 55.0 / 55.0, 43.9 / 44.2, 69.5 / 78.0. On AQuA it is 84.6 / 64.1, 82.1 / 59.8, 82.8 / 60.4, 81.1 / 56.8, a gap of 20.5, 22.3, 22.4 and 24.3 points. (i) "Significantly above random" (rows e and m) counts the seeds whose one-sided share of random draws is below .05: 4 / 4 / 2 / 1 of four for row e and 4 / 4 / 4 / 0 for row m (MMLU-Pro / AQuA / MedQA / GPQA). Numerically learned calling (row c against row d) and the learned selection sent to the Verifier (row k against row l) are above the mean of the random draws in all 16 seed-benchmark cells, so "ahead" without the criterion is not used for these rows. (j) "Always-Verifier is numerically highest" (sCpX W2): among the fixed-role rows g to j the highest is row g or row h in every seed on every benchmark; no paired test was run against the other roles.
5. **S_1 only.** Development pools, development-threshold gates, the independent GPQA pool, the row 6 decomposition and Figure R1. The letter-only row for the trained weights (Table R2, row 2) and the deployed reference (row 9) are identical in all four attribution files, so they are S_1 values, not re-measured per seed; row 2 is labelled "S_1 weights" and eCWp W1 says "S_1".
6. **Sources that disagree or are stale.** (a) The header of `seeds_test.md` calls `s1` "the paper's checkpoint"; for MedQA `REB/configuration.md` section 1 says it is a substitute trained with the same configuration, and the response follows the latter. (b) `REB/configuration.md` section 1 describes only the seed 1 and seed 2 re-trainings and says results exist for seed 1 on MedQA and GPQA only; it does not mention seed 3. (c) The attention kernels of the re-trainings are stated in the header of `seeds_test.md` only. (d) `deploy_seeds.md` has no seed 3 and no seed 2 on GPQA. (e) The attribution files label the deployed reference "Table 1"; eCWp quotes 79.8 / 60.2 / 76.8 from the submitted Table 1, and the run reproduced here is 78.8 / 63.6 / 54.0 / 77.2 (the S_1 row of the RSI table). The response says "re-evaluation" in the common block, in eCWp W1 and in the note "Rows 8 and 9". For AQuA (76.8 in the paper, 77.2 here) the files give no reason and hold no evaluation record of the paper run; eCWp W1 now says so. Still to fill in: the number of the paper table that the re-evaluated run corresponds to.
7. **Not rerun.** The paper's Figure 3 configuration at about one call per question (stated in limitation 17 and in the label of change 2; eCWp W2 therefore says "is not supported", not "does not hold"); forced-all with other role orders; an external learned router; a separately trained manager-only model (the commit-only control is pending).
8. **Advisor evidence.** `REB/configuration.md` section 2: the command lines of the MedQA, MMLU-Pro and GPQA paper-era runs are not recorded; the statement rests on the recorded decision, the parameter-name mechanism and the replays (Verifier 20 of 20 identical; Extractor 5 of 20, median similarity 0.924 against 0.392 once the names were corrected; no Reasoner entry). The same section has the comparison with correctly loaded adapters for whoever answers sCpX point 5.
9. **Selection protocol (sCpX W4).** `REB/configuration.md` section 3: how ρ was chosen is not recorded for any benchmark, and the run sheets state an intended protocol only. The response says "not recorded" for ρ and makes no claim about what was done. For checkpoints it states what the code does ("selects no checkpoint among epochs"; section 1, row "Checkpoint selection"); that each uploaded adapter is the end-of-training save is not recorded. The MMLU-Pro sentence of the common block comes from the same section: the paper's test set, as reconstructed, is 200 development questions, 68 label-collection roots, 47 / 50 / 31 questions of the three GRPO pools (128) and 104 clean questions. The replicate caveats of limitation 14 (tool-schema rendering, the two AQuA label files) come from section 1.
10. **Placeholders.** The commit-only placeholder stands once, in sCpX W1. The deployed-evaluation placeholder for seed 3 stands in four cells of Table R3, in eCWp W1 and in change 1; a second one, for seed 2 on GPQA, stands in one cell of Table R3 and in change 1. No seed-2 placeholder of the previous draft remains.
