# MARGENT author response: Verifier / calling policy (A) and what the trained manager contributes (B)

Draft for the author of topics A and B. GRPO details, RSI results and offline cost belong to the co-authors and are not answered here.

Conventions. Accuracy is in %. A difference is "first minus second" in accuracy points, followed by its 95% paired-bootstrap interval in brackets and the exact McNemar p. "S_1" is the registered single-pass manager of each benchmark. For MedQA it is not the paper's locked checkpoint, which was not archived, but a checkpoint trained with the same configuration; "second seed" is a retraining of the same label file. In the tables a cell "x / y" is S_1 / second seed unless the caption says otherwise. Evidence files are under `agent_routing/docs/mcq_rsi_results/rebuttal/` (below: `REB/`).

---

## Common block (post once, or paste above each response)

**Two corrections and one check apply to every answer below.**

1. **Advisors.** The paper calls the Extractor, Reasoner and Verifier "role-specific LoRA specialists". That description is wrong for every reported number. The adapters were trained and registered, but their parameter names matched no module of the served model, so the server answered with the base model (Qwen3.5-9B) under each role's prompt. For the Verifier this is confirmed output for output: on 20 prompts the adapter as served returned text identical to the base model on 20 of 20 (0 of 20 once the names were corrected). For the Extractor the outputs are close but not identical (5 of 20 identical, median similarity 0.924, against 0.392 once corrected), and no Reasoner check is recorded; for these two roles, and for the paper-era runs whose command lines were not kept, the statement rests on the loading mechanism. No reported number changes; the description will be corrected.

2. **Evidence base.** The new results are on test pools that share no question with the training pools or with the development pools used here (MedQA 500, MMLU-Pro 500, GPQA 100, AQuA 254; earlier runs of ours were also evaluated on these pools). For GPQA the test pool is Diamond-100, on which the paper's Table 4 development numbers were also measured and for which our records do not show which pool ρ was chosen on; it is disjoint from training but not an independent evaluation pool. The independent GPQA pool is in the notes to Table R1. For MMLU-Pro the paper's test set, as we reconstruct it, shares 200 questions with the development pool and 68 with the collection roots, so all MMLU-Pro numbers here use a new, disjoint 500-question pool. AQuA development and test are the official validation and test splits and share no question. For MedQA, S_1 is not the paper's locked checkpoint, which was not archived, but a checkpoint trained with the same configuration. Selection protocol: the single-pass training code makes no choice among epochs on a development pool, and on which pool ρ was chosen is not recorded for any of the four benchmarks. The results cover S_1 and one retraining of the same label file with training seed 1, using the recipe logged for MedQA, MMLU-Pro and GPQA (assumed for AQuA, which has no training log). The seed and maximum sequence length of S_1 were not logged, so the two runs may differ in more than the seed [seed 2 pending]. Every statement below rests on these two training seeds and on one test pool per benchmark; the bootstrap intervals cover questions, not seeds.

3. **Check of the measurement.** On 500/500, 498/500, 100/100 and 254/254 questions, the learned-calling row of the new comparison scores the question the same way (right or wrong) as our re-evaluation of the deployed S_1 on the same pool (78.8 / 63.6 / 54.0 / 77.2; not the submitted Table 1 run).

Sources: items 1 and 2, `REB/configuration.md`, sections 1 to 3, and the header of `REB/verifier_grid/seeds_test.md`. Item 3 is in none of the result tables: it is recomputed from the per-question records, `agent_routing/outputs/verifier_grid/s1/<bench>/test/grid_records.jsonl` against the S_1 test outputs in `agent_routing/docs/mcq_rsi_results/results/grpo_grid_locked_tests_v3g.tar.gz` and `grpo_grid_mmlu_pro_v3g_split.tar.gz`. The tables print a different check (first turn equal to unconstrained greedy decoding: 500/500, 499/500, 100/100, 254/254; `REB/verifier_grid/s1_test.md`).

---

## Table R1. Equal-information comparison on the test pools

Every rule starts from the same manager-written candidate and uses the same checkpoint, greedy decoding and forced-commit revision step, with at most one advisor call. Rows a to m: accuracy (%) unless the label says otherwise, S_1 / second seed. Lower block: paired difference in points with the exact McNemar p in parentheses, S_1 / second seed; the pooled column adds the 95% interval.

| | Rule | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) | Pooled (n=1354) |
|:--|:--|--:|--:|--:|--:|:--|
| a | Candidate only, no call | 73.4 / 74.4 | 50.8 / 51.4 | 41.0 / 40.0 | 36.2 / 36.2 | – |
| b | Revision turn on an empty Verifier output, no call | 72.8 / 73.2 | 51.4 / 51.6 | 41.0 / 40.0 | 32.7 / 36.2 | – |
| c | **Learned calling** (first call, then commit) | 78.8 / 78.0 | 63.2 / 65.2 | 54.0 / 50.0 | 77.2 / 76.0 | – |
| | Calls per question of row c | 0.37 / 0.35 | 0.33 / 0.35 | 0.52 / 0.55 | 0.77 / 0.74 | – |
| d | Random questions, same calls per advisor (2,000 draws) | 76.6 / 77.0 | 57.2 / 58.4 | 47.7 / 47.6 | 73.9 / 72.1 | – |
| e | Share of those draws at least as accurate as row c | .008 / .160 | <.001 / <.001 | .005 / .233 | .017 / .007 | – |
| f | Confidence-gated Verifier, same number of calls as row c | 79.6 / 79.0 | 61.6 / 63.8 | 52.0 / 50.0 | 76.8 / 73.2 | – |
| g | Always Verifier, with the candidate (1.00 call) | 83.4 / 82.8 | 72.4 / 73.6 | 56.0 / 56.0 | 85.0 / 84.6 | – |
| h | Always Verifier, candidate removed: the paper's baseline design, rerun under this protocol (1.00 call) | 84.2 / 84.4 | 69.2 / 67.4 | 61.0 / 59.0 | 78.3 / 79.9 | – |
| i | Always Reasoner (1.00 call) | 81.2 / 81.4 | 55.6 / 56.4 | 45.0 / 48.0 | 37.0 / 37.0 | – |
| j | Always Extractor (1.00 call) | 79.0 / 79.0 | 52.0 / 52.0 | 39.0 / 38.0 | 29.5 / 33.1 | – |
| k | Learned selection, all sent to the Verifier (same calls as row c) | 80.2 / 79.8 | 63.4 / 65.8 | 53.0 / 49.0 | 77.2 / 76.0 | – |
| l | Random subset, Verifier only, same number of calls (2,000 draws) | 77.1 / 77.3 | 58.0 / 59.2 | 48.8 / 48.8 | 73.9 / 72.1 | – |
| m | Share of those subsets at least as accurate as row k | <.001 / .002 | <.001 / <.001 | .052 / .555 | .017 / .007 | – |
| | c − g: learned − always Verifier with the candidate | −4.6 (.001) / −4.8 (.002) | −9.2 (<.001) / −8.4 (<.001) | −2.0 (.774) / −6.0 (.180) | −7.9 (<.001) / −8.7 (<.001) | −6.7 [−8.4, −5.0], <.001 / −6.9 [−8.7, −5.2], <.001 |
| | c − f: learned − confidence gate at the same calls | −0.8 (.672) / −1.0 (.576) | +1.6 (.416) / +1.4 (.470) | +2.0 (.754) / +0.0 (1.000) | +0.4 (1.000) / +2.8 (.401) | +0.5 [−1.3, +2.4], .648 / +0.7 [−1.3, +2.6], .554 |
| | g − h: always Verifier, with − without the candidate | −0.8 (.672) / −1.6 (.322) | +3.2 (.089) / +6.2 (<.001) | −5.0 (.332) / −3.0 (.607) | +6.7 (.012) / +4.7 (.073) | +1.8 [−0.2, +3.8], .091 / +2.4 [+0.4, +4.3], .021 |
| | g − b: Verifier output − empty Verifier output | +10.6 (<.001) / +9.6 (<.001) | +21.0 (<.001) / +22.0 (<.001) | +15.0 (.003) / +16.0 (.002) | +52.4 (<.001) / +48.4 (<.001) | +22.6 [+20.1, +25.3], <.001 / +21.9 [+19.4, +24.5], <.001 |
| | b − a: empty-output revision − candidate | −0.6 (.549) / −1.2 (.146) | +0.6 (.250) / +0.2 (1.000) | +0.0 (1.000) / +0.0 (1.000) | −3.5 (.022) / +0.0 (1.000) | −0.7 [−1.4, +0.1], .122 / −0.4 [−1.1, +0.3], .405 |

Sources: `REB/verifier_grid/s1_test.md` (S_1), `REB/verifier_grid/seed1_test.md` and `REB/verifier_grid/seeds_test.md` (second seed).

Notes to Table R1 (S_1 unless stated).

- **Per-benchmark intervals, S_1.** c − g: [−7.4, −1.8], [−12.2, −6.2], [−9.0, +5.0], [−11.4, −4.7]. g − h: [−3.6, +2.0], [−0.2, +6.6], [−13.0, +3.0], [+2.0, +11.4]. Second seed, g − h on MMLU-Pro: [+2.8, +9.6].
- **Rows d and l, 95% of draws.** Row d: [74.8, 78.4], [55.2, 59.2], [43.0, 52.0], [70.9, 76.8]. Row l: [75.4, 78.8], [56.0, 60.0], [44.0, 53.0], [70.9, 76.8].
- **Choice of advisor.** First calls by role (Extractor / Reasoner / Verifier): 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196. Sending the same selected questions to the Verifier only gives row k; learned minus this is −0.5 [−1.4, +0.4], p = .311 pooled (second seed −0.8 [−1.7, +0.1], p = .099).
- **Confidence gate.** Row f ranks the questions inside the test pool, so it is not a deployable gate. With a threshold fixed on the development pool: (i) Threshold chosen to match the learned call rate on development: 79.2 / 58.0 / 53.0 / 76.0 at 0.31 / 0.24 / 0.71 / 0.75 calls; learned minus gate is +2.1 [+0.1, +4.0], p = .043 pooled, but the calls differ from row c, so this is not an equal-budget result. (ii) Threshold chosen for best development accuracy: 83.0 / 71.4 / 55.0 / 81.9 at 0.80 / 0.68 / 0.84 / 0.93 calls; learned minus gate is −5.5 [−7.3, −3.8], p < .001 pooled.
- **Development pools** (n = 200 / 200 / 100 / 254). Share of random draws at least as accurate as learned calling: .024 / .185 / .111 / .197. c − g: −5.5 (.027), −11.5 (<.001), −4.0 (.388), −7.9 (<.001). c − f: −1.5 (.664), −3.0 (.345), −4.0 (.388), −2.8 (.296). Share of Verifier-only random subsets at least as accurate as the learned selection sent to the Verifier: .003 / .203 / .075 / .197. On the MedQA development pool always-Reasoner is 85.0 against 84.5 for always-Verifier with the candidate.
- **GPQA on an independent pool.** The GPQA development pool has 100 questions that are disjoint from Diamond-100 and from training. There learned calling scores 53.0 at 0.62 calls and always-Verifier with the candidate 57.0 (−4.0 [−11.0, +3.0], p = .388); with − without the candidate is +5.0 [−5.0, +15.0], p = .424. Learned calling is not above random there (.111 of random draws are at least as accurate) and is 4.0 points below the confidence gate at the same calls (p = .388).

Sources for the notes: `REB/verifier_grid/s1_test.md`, `REB/verifier_grid/seed1_test.md`, `REB/verifier_grid/analysis/s1_{medqa,mmlu_pro,gpqa,aqua}_dev.md`, `REB/configuration.md` section 3.

## Table R2. What training changed: S_1 against the untrained base model as manager

Same questions, advisors and advisor outputs for both managers; the untrained model is held to the same output grammar as the trained manager. Cells are untrained → trained (S_1), accuracy in %. Last column: trained − untrained, pooled.

| | Quantity | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) | Pooled, pt [95% CI], p |
|:--|:--|--:|--:|--:|--:|:--|
| 1 | Questions whose first turn follows the draft / commit / call format without the grammar | 0 → 500 of 500 | 2 → 499 of 500 | 0 → 100 of 100 | 27 → 254 of 254 | untrained: 29 of 1354 |
| 2 | Reference: letter only under a plain prompt (untrained weights; trained weights) | 74.6; 74.4 | 53.0; 53.8 | 45.0; 44.0 | 41.7; 42.1 | – |
| 3 | Draft under the manager format | 70.6 → 73.4 | 44.2 → 50.8 | 32.0 → 41.0 | 32.7 → 36.2 | +4.8 [+2.7, +6.9], <.001 |
| 4 | Use of evidence: accuracy after a forced Verifier call on same-candidate questions (n = 434 / 313 / 65 / 150) | 85.7 → 85.7 | 77.6 → 77.6 | 58.5 → 56.9 | 84.7 → 86.0 | +0.1 [−0.6, +0.8], 1.000 (n = 962) |
| 5 | Selection signal: the trained manager's Verifier outcomes on the questions chosen by the untrained model's confidence → by the learned policy, same number of calls | 79.8 → 80.2 | 62.0 → 63.4 | 52.0 → 53.0 | 75.6 → 77.2 | +1.0 [−0.7, +2.7], .265 |
| 6 | Whole system at equal calls (0.37 / 0.33 / 0.52 / 0.77): untrained manager + Verifier on its least confident candidates → MARGENT learned calling | 80.6 → 78.8 | 57.6 → 63.2 | 47.0 → 54.0 | 71.7 → 77.2 | +3.0 [+0.7, +5.2], .011 |
| | Row 6, difference per benchmark | −1.8 [−4.4, +0.8], .243 | +5.6 [+1.4, +9.8], .012 | +7.0 [+0.0, +14.0], .118 | +5.5 [−0.4, +11.0], .081 | – |
| 7 | Whole system at 1.00 call: always Verifier with the candidate | 83.6 → 83.4 | 70.4 → 72.4 | 52.0 → 56.0 | 81.9 → 85.0 | +1.6 [+0.1, +3.0], .044 |
| 8 | Reference, as deployed: untrained manager without the grammar, up to 3 calls: accuracy / calls (valid answers) | 82.8 / 2.98 (95%) | 69.0 / 2.51 (85%) | 39.0 / 2.53 (52%) | 78.3 / 1.62 (84%) | – |
| 9 | Reference, as deployed: S_1 (a second call is allowed): accuracy / calls | 78.8 / 0.37 | 63.6 / 0.35 | 54.0 / 0.52 | 77.2 / 0.78 | – |

Source: `REB/attribution/s1_vs_base_test.md` (and `.json`).

Notes to Table R2.

- **Row 4 caveat.** Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: 77.4 / 47.0, 61.0 / 33.7, 41.5 / 40.0, 37.3 / 34.6). On them a Verifier call breaks 3.9 / 3.1 / 11.1 / 7.1% of right candidates for the trained manager and 3.9 / 2.6 / 7.4 / 3.6% for the untrained one. Questions where the drafts differ cannot be compared this way. With another advisor forced on the same questions the pooled difference is +1.1 [−0.3, +2.6], p = .161 (Reasoner) and +0.8 [−0.7, +2.4], p = .366 (Extractor); per benchmark the Extractor on MedQA is +2.5 [+0.5, +4.8], p = .035.
- **Row 6, exact decomposition** = draft + what the same confidence-gated Verifier calls add (trained − untrained) + learned calling − the trained manager's own confidence gate. MedQA: −1.8 = +2.8 − 3.8 − 0.8. MMLU-Pro: +5.6 = +6.6 − 2.6 + 1.6. GPQA: +7.0 = +9.0 − 4.0 + 2.0. AQuA: +5.5 = +3.5 + 1.6 + 0.4.
- **Rows 8 and 9.** Both are runs on the test pools of this response. Row 9 is our re-evaluation of the deployed S_1, not the submitted Table 1 (79.8 / 60.2 / 76.8 on MedQA / MMLU-Pro / AQuA, as quoted by eCWp): the MedQA checkpoint is a substitute, the MMLU-Pro pool is new and disjoint, and AQuA is the official test split evaluated again. The untrained system of row 8 returns a valid answer on 95 / 85 / 52 / 84% of questions, so its GPQA accuracy (39.0) reflects invalid outputs as well as wrong answers.
- **Learned call rate on wrong / right candidates (%).** 63.2 / 27.0, 48.8 / 18.1, 50.8 / 53.7, 84.6 / 64.1 (second seed: 59.4 / 26.3, 51.0 / 20.2, 55.0 / 55.0, 82.1 / 59.8).
- **Second seed, pooled** (`REB/attribution/seed1_vs_base_test.md`) [seed 2 pending]. Draft +5.3 [+3.2, +7.5], p < .001. Evidence +0.2 [−0.4, +0.8], p = .754 (n = 956). Selection +1.0 [−0.6, +2.7], p = .265. Equal calls +2.7 [+0.5, +4.9], p = .017 (per benchmark: MedQA −2.4, p = .119; MMLU-Pro +6.8, p = .001; GPQA +2.0, p = .774; AQuA +5.1, p = .111). Always Verifier +1.7 [+0.3, +3.1], p = .022.
- **Sibling of S_1 trained with no commit cap** (same collection and recipe, every commit row kept; it is not a commit-only or manager-only control). S_1 − sibling, pooled: draft −0.3 [−1.4, +0.8], p = .699; evidence −0.1 [−0.6, +0.4], p = 1.000 (n = 1253); selection +0.1 [−1.6, +1.8], p = 1.000; equal calls −0.7 [−2.7, +1.3], p = .563; always Verifier −0.1 [−0.9, +0.7], p = 1.000. The sibling's learned calling scores 77.2 / 59.2 / 54.0 / 69.7 at 0.23 / 0.21 / 0.68 / 0.63 calls. Sources: `REB/attribution/s1_vs_allc_test.md`, `REB/verifier_grid/allc_test.md`.
- **Matched-data success distillation** (the same round-1 records and the same commit cap; each kept record trains one successful option drawn at random instead of the shortest one; same SFT recipe; the default selection of the same records reproduces the S_1 label file row for row). Its learned calling scores 81.6 / 73.2 / 50.0 / 83.1 at 0.98 / 1.00 / 0.99 / 0.96 calls, against 78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 for S_1. S_1 − success distillation: MedQA −2.8 [−5.4, −0.2], p = .044; MMLU-Pro −10.0 [−13.2, −6.8], p < .001; GPQA +4.0 [−4.0, +12.0], p = .454; AQuA −5.9 [−9.8, −2.0], p = .006; pooled −5.5 [−7.3, −3.8], p < .001. Accuracy gained per call: 14.8 / 37.3 / 25.0 / 53.1 points for S_1, 10.0 / 23.2 / 10.1 / 47.1 for success distillation. First calls to Extractor / Reasoner / Verifier: S_1 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196; success distillation 66 / 191 / 231, 73 / 59 / 367, 54 / 16 / 29, 0 / 0 / 244. Its draft accuracy is 71.8 / 50.0 / 40.0 / 37.8 (S_1: 73.4 / 50.8 / 41.0 / 36.2). Against always calling the Verifier with the candidate from its own draft it is not ahead (pooled −1.6 [−2.7, −0.4], p = .014). Sources: `REB/verifier_grid/s1_vs_sucd_policy.md`, `REB/verifier_grid/sucd_test.md`.

## Figure R1

File: `agent_routing/docs/mcq_rsi_results/paper_assets/fig/fig_rebuttal_budget_s1.pdf`.

Caption: "Accuracy against advisor calls per question for S_1 on the four test pools. Circle: candidate only. Diamond: learned calling (first call). Curve: Verifier on the least confident candidates. Dashed line: Verifier on a random subset, with the 95% interval at the learned number of calls. Squares: always Verifier with the candidate (filled) and with the candidate removed (open)."

---

## Reviewer cPCG: weakness 1 (the always-Verifier baseline)

Thank you. Weakness 1 is correct, and the rerun changes our conclusion.

**What each call receives.** A Verifier call made by the policy contains the question, options and context plus two lines naming the manager's draft letter as the candidate to audit. The paper's always-Verifier baseline omitted those two lines. The Extractor and Reasoner never receive the candidate.

**Equal-information rerun (Table R1).** All rules share the candidate, checkpoint, greedy decoding and revision step, and make at most one call. Always calling the Verifier with the candidate (row g) is more accurate than learned calling (row c) on MedQA (learned − always: −4.6 [−7.4, −1.8], p = .001), MMLU-Pro (−9.2 [−12.2, −6.2], p < .001) and AQuA (−7.9 [−11.4, −4.7], p < .001), and not distinguishable on GPQA (−2.0 [−9.0, +5.0], p = .774, n = 100). It makes 1.00 call per question against 0.37 / 0.33 / 0.52 / 0.77. The second seed gives −4.8 / −8.4 / −6.0 / −8.7 (p = .002, < .001, .180, < .001). The paper's baseline design, without the candidate (row h), is also numerically above learned calling on each test pool.

**Does the candidate matter (g − h)?** It helps on AQuA (+6.7 [+2.0, +11.4], p = .012; second seed +4.7, p = .073) and MMLU-Pro (+3.2, p = .089; second seed +6.2, p < .001), significant in one of the two seeds on each. It has no detectable effect on MedQA (−0.8, p = .672). On GPQA the interval [−13.0, +3.0] excludes neither direction.

**Other advisors.** On the four test pools, in both seeds, always-Reasoner and always-Extractor (rows i, j) are numerically below always-Verifier (no paired test was run; on the MedQA development pool always-Reasoner is 85.0 against 84.5).

**What survives.** Learned calling is a lower-call operating point, not a more accurate one. At its number of calls it is significantly above random questions with the same calls per advisor on MMLU-Pro and AQuA in both seeds (row e: < .001 / < .001 and .017 / .007), and on MedQA and GPQA only in the original seed (.008 / .160 and .005 / .233) [seed 2 pending]; on the development pools, only on MedQA (.024). It is not significantly different from calling the Verifier on the least confident candidates at the same number of calls (ranking within the pool; pooled +0.5 [−1.3, +2.4], p = .648). A gate whose threshold is fixed on the development pool for best development accuracy makes 0.80 / 0.68 / 0.84 / 0.93 calls and is 5.5 points above learned calling pooled ([+3.8, +7.3], p < .001).

We withdraw the claim that learned calling matches or exceeds always-Verifier, and will replace the baseline with Table R1 and Figure R1. These statements rest on two training seeds [seed 2 pending] and one test pool per benchmark.

---

## Reviewer sCpX: weaknesses 1, 2, 3 and the independence / seeds part of 4

Thank you. The always-Verifier, fixed-role, random and confidence-gate comparisons were rerun on the test pools; the matched-data success-distillation control is still training. The results do not support the paper's always-Verifier claim.

**W1 (isolating learned delegation).** Agreed: three effects are mixed.
- Manager-only revision (the same trained manager given one more revision turn on an empty Verifier output; we did not train a separate manager-only model): no gain over the candidate (Table R1, b − a: pooled −0.7 [−1.4, +0.1], p = .122); on AQuA a drop for S_1 (−3.5, p = .022; second seed +0.0, p = 1.000).
- Evidence: the Verifier's output adds 10.6 / 21.0 / 15.0 / 52.4 points over that turn (p < .001, < .001, = .003, < .001).
- SFT, against the untrained model under the same output grammar (Table R2): the draft rises by +4.8 [+2.7, +6.9] without exceeding that model's letter-only accuracy. On the 962 of 1354 questions where both managers wrote the same draft (the easier ones), accuracy after a forced Verifier call does not differ (+0.1 [−0.6, +0.8]); the other questions cannot be compared this way.

The gain is the Verifier's output; SFT supplies the protocol and the call rate.

**Matched-data success distillation (new control; notes to Table R2).** We retrained the manager on the same records with the same commit cap and recipe, changing only the target: one successful option drawn at random instead of the shortest one. That manager calls on almost every question (0.98 / 1.00 / 0.99 / 0.96 calls, against 0.37 / 0.33 / 0.52 / 0.77) and is more accurate than S_1 on MedQA (+2.8, p = .044), MMLU-Pro (+10.0, p < .001) and AQuA (+5.9, p = .006), not on GPQA (−4.0, p = .454); pooled it is 5.5 points ahead [+3.8, +7.3]. What the shortest-path target adds is therefore stopping: a call rate two to three times lower on three of four tests and more accuracy per call (14.8 / 37.3 / 25.0 / 53.1 points against 10.0 / 23.2 / 10.1 / 47.1). It does not add accuracy. A separately trained manager-only (commit-only) model was not run; the empty-output revision turn above is the manager-only control at inference time.

**W2 (controls on the test pools).** Fixed-role controls: Table R1, rows g to j. Learned calling is significantly above random questions with the same calls per advisor on MMLU-Pro and AQuA in both seeds, and on MedQA and GPQA only in the original seed (row e) [seed 2 pending]; on the development pools, only on MedQA. It is not significantly different from a confidence-gated Verifier at the same number of calls (ranking within the pool; pooled +0.5 [−1.3, +2.4], p = .648; development pools −1.5 / −3.0 / −4.0 / −2.8, none significant). A gate whose threshold is fixed on the development pool for best development accuracy reaches 83.0 / 71.4 / 55.0 / 81.9 at 0.80 / 0.68 / 0.84 / 0.93 calls, 5.5 points above learned calling pooled ([+3.8, +7.3], p < .001).

**W3 (always-Verifier).** Correct: the baseline withheld the candidate. With identical candidates, checkpoint, decoding and revision step, always-Verifier is more accurate than learned calling on MedQA, MMLU-Pro and AQuA (−4.6, −9.2, −7.9; p = .001, < .001, < .001) and not distinguishable on GPQA (−2.0 [−9.0, +5.0], p = .774), at 1.00 call against 0.37 / 0.33 / 0.52 / 0.77 (Figure R1). A random-subset Verifier at the same number of calls scores 77.1 / 58.0 / 48.8 / 73.9, against 80.2 / 63.4 / 53.0 / 77.2 for the learned selection sent to the Verifier (rows k to m). The share of 2,000 random subsets at least as accurate is < .001 / < .001 / .052 / .017 (second seed .002 / < .001 / .555 / .007): above random on MedQA, MMLU-Pro and AQuA in both seeds, on GPQA in neither.

**W4 (independence, seeds).** GPQA: conceded. Table 4 reports GPQA on Diamond-100, the evaluation set. On a 100-question pool disjoint from Diamond-100 and from training, learned calling scores 53.0 at 0.62 calls and always-Verifier 57.0 (−4.0 [−11.0, +3.0], p = .388). There it is not above random (.111 of random draws are at least as accurate), so the GPQA result against random holds only for S_1 on Diamond-100. AQuA development and test are the official validation and test splits and share no question. Selection protocol: on which pool ρ was chosen is not recorded for any of the four benchmarks. The rest (MMLU-Pro pool, training code, two seeds [seed 2 pending]) is in the common block, item 2.

W5 is answered by our co-authors; note the correction on advisors in the common block, item 1.

---

## Reviewer A2jt: "Uneven call selection and harmful roles" and "Confounded attribution and asymmetric baselines"

Thank you. Both weaknesses hold; the test-pool results make them more precise.

**Uneven call selection and harmful roles.** Agreed.
- Against random: on the test pools learned calling is significantly above random questions with the same calls per advisor on MMLU-Pro and AQuA in both seeds, and on MedQA and GPQA only in the original seed (Table R1, row e) [seed 2 pending]. On the development pools S_1 is above random only on MedQA.
- Against confidence: it is not significantly different from calling the Verifier on the manager's least confident candidates at the same number of calls (ranking within the pool; pooled +0.5 [−1.3, +2.4], p = .648). A gate whose threshold is fixed on the development pool for best development accuracy makes 0.80 / 0.68 / 0.84 / 0.93 calls and is 5.5 points above learned calling pooled ([+3.8, +7.3], p < .001).
- Call gap: on GPQA the policy calls on 50.8% of wrong and 53.7% of right candidates, so there is no gap. On AQuA it calls on 84.6% of wrong and 64.1% of right candidates, but makes 0.77 calls and is 7.9 points below always-Verifier (p < .001).
- Roles: always calling the Extractor is numerically below the no-call candidate on AQuA (29.5 against 36.2) and GPQA (39.0 against 41.0); always-Reasoner on AQuA is 37.0 against 36.2 for the candidate, no measurable gain. The policy's choice of advisor adds nothing measurable: learned calling minus the same questions all sent to the Verifier is −0.5 [−1.4, +0.4], p = .311.
- Corruption: training does not reduce it. On identical candidates a Verifier call breaks 3.9 / 3.1 / 11.1 / 7.1% of right answers for the trained manager and 3.9 / 2.6 / 7.4 / 3.6% for the untrained one.

**Confounded attribution and asymmetric baselines.** Correct on both counts.

With the candidate, always-Verifier is more accurate than learned calling on MedQA, MMLU-Pro and AQuA (−4.6, −9.2, −7.9; p = .001, < .001, < .001) and not distinguishable on GPQA (−2.0, p = .774), at 1.00 call against 0.37 / 0.33 / 0.52 / 0.77.

Table R2 separates what SFT changed, against the untrained model under the same output grammar:
- The draft rises (+4.8 [+2.7, +6.9]) but stays below that model's letter-only accuracy.
- The use of the Verifier's output on same-candidate questions is not different (+0.1 [−0.6, +0.8], 962 of 1354 questions).
- The learned choice of questions is not significantly better than the untrained model's own confidence (+1.0 [−0.7, +2.7]).

A gain from routing alone is not established. Training provides output that follows the protocol and a lower call rate. The matched-data success-distillation control agrees: with the same records and commit cap but a random successful option as the target, the manager calls on 96 to 100% of questions and is 5.5 points more accurate pooled ([+3.8, +7.3], p < .001), so the shortest-path target buys fewer calls (0.33 to 0.77), not accuracy.

Forced-all with other role orders and an external learned router were not run; both remain limitations. Two training seeds [seed 2 pending], one test pool per benchmark.

---

## Reviewer eCWp: weaknesses 1 and 2

Thank you. Your reading is correct on both points, and the new controls agree with it.

**W1 (reference for the gain).** Agreed. The trained manager's draft (73.4 / 50.8 / 41.0 / 36.2) is below the untrained model answering with a letter only (74.6 / 53.0 / 45.0 / 41.7) on every benchmark (Table R2).

The drop goes with the manager format, not the weights: under a plain letter-only prompt the trained weights score 74.4 / 53.8 / 44.0 / 42.1. Left to itself, the untrained model leaves the draft / commit / call grammar on 1325 of 1354 questions. Held to it, its draft falls 4.0 to 13.0 points below its own letter-only accuracy; training recovers 2.8 / 6.6 / 9.0 / 3.5 of those points (pooled +4.8 [+2.7, +6.9], p < .001; per benchmark p = .059, < .001, .064, .314). Training taught the protocol, not a better answer. Given the same candidate and Verifier output, the two managers revise equally well (+0.1 [−0.6, +0.8], 962 of 1354 questions).

Your comparison with the untrained system stands. On the same test pools the deployed S_1 makes 0.35 to 0.78 calls per question and the untrained system 1.62 to 2.98, with accuracy 78.8 / 63.6 / 54.0 / 77.2 against 82.8 / 69.0 / 39.0 / 78.3 (Table R2, rows 8 and 9). These are re-evaluations and differ from the submitted Table 1 values you quote (79.8 / 60.2 / 76.8 on MedQA / MMLU-Pro / AQuA): the MedQA checkpoint is a substitute for the paper's, which was not archived, and the paper's MMLU-Pro test set shared questions with the development pool and the collection roots, so MMLU-Pro is now reported on a disjoint pool (common block, item 2).

A stronger comparison exists: the untrained manager under the grammar with always-Verifier (83.6 / 70.4 / 52.0 / 81.9 at 1.00 call) is numerically above MARGENT's learned calling on three of four tests for S_1 (not GPQA, 54.0) and on all four for the second seed (78.0 / 65.2 / 50.0 / 76.0); no paired test was run for this contrast. Training buys valid output and fewer calls, not accuracy (equal calls: Table R2, row 6).

**W2 (learned call selection).** Agreed. The 0 to 4.5 point advantage of Figure 3 (development pools, about one call per question) does not hold in the new comparison and we no longer claim it. We did not rerun that configuration; on the test pools always-Verifier is numerically above learned calling even without the candidate (84.2 / 69.2 / 61.0 / 78.3 against 78.8 / 63.2 / 54.0 / 77.2). With the candidate it is ahead by 4.6 / 9.2 / 7.9 points on MedQA / MMLU-Pro / AQuA (p = .001, < .001, < .001; GPQA 2.0, p = .774, not significant).

The gain is the audit: the Verifier's output adds 10.6 / 21.0 / 15.0 / 52.4 points (p < .001, < .001, = .003, < .001) over a revision turn on an empty Verifier output, which itself does not improve the candidate (pooled −0.7 [−1.4, +0.1]).

Learned calling is significantly above random questions with the same calls per advisor on MMLU-Pro and AQuA in both seeds, and on MedQA and GPQA in one of two [seed 2 pending]; on the development pools only on MedQA, as you read Table 10. At the same number of calls it is not significantly different from ranking by the manager's draft confidence (+0.5 [−1.3, +2.4]) or by the untrained model's confidence (+1.0 [−0.7, +2.7]); a gate whose threshold is fixed on the development pool for best development accuracy makes 0.80 / 0.68 / 0.84 / 0.93 calls and is 5.5 points above learned calling ([+3.8, +7.3], p < .001).

We will restate the contribution as protocol-following output at 0.35 to 0.78 calls per question (deployed S_1), not an accuracy gain. These statements rest on two training seeds [seed 2 pending] and one test pool per benchmark.

---

## Changes we will make to the paper

Revised claim sentences.

1. **Abstract, headline claim.** "On four test pools the trained manager makes 0.35 to 0.78 advisor calls per question, against 1.62 to 2.98 for the untrained manager with the same advisors, which returns a valid answer on 95 / 85 / 52 / 84% of questions; accuracy is 78.8 / 63.6 / 54.0 / 77.2 against 82.8 / 69.0 / 39.0 / 78.3 (MedQA / MMLU-Pro / GPQA / AQuA-RAT). Against the untrained model held to the same output grammar with a confidence-gated Verifier at the same number of calls, the trained manager is ahead by 3.0 points pooled [+0.7, +5.2], per benchmark only on MMLU-Pro. Training buys output that follows the protocol and fewer calls, not higher accuracy."
2. **Abstract and Section 4.1, always-Verifier (replaces the Figure 3 claim).** "Under equal information (same candidate, checkpoint, decoding and revision step), calling the Verifier on every question reaches 83.4 / 72.4 / 56.0 / 85.0 at 1.00 call and learned calling 78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 calls. The difference is significant on MedQA, MMLU-Pro and AQuA-RAT (exact McNemar p = .001, < .001, < .001) and not on GPQA (n = 100, p = .774)."
3. **Section 4, the +17.1-point sentence.** "The gain over the manager's pre-call draft is produced by the Verifier's output: a revision turn on an empty Verifier output does not improve the draft (pooled −0.7 points [−1.4, +0.1], p = .122), and the Verifier's output adds 10.6 to 52.4 points over that turn."
4. **Section 4.4, candidate conditioning.** "Passing the candidate to the Verifier raises accuracy on AQuA-RAT (+6.7 [+2.0, +11.4], p = .012; +4.7, p = .073 in a second seed) and on MMLU-Pro (+3.2, p = .089; +6.2, p < .001 in a second seed), significant in one of the two seeds on each; it has no detectable effect on MedQA (−0.8, p = .672) and is unresolved on GPQA (−5.0 [−13.0, +3.0])."
5. **Selection (discussion of Table 10).** "On the test pools learned calling is significantly more accurate than calling on random questions with the same calls per advisor on MMLU-Pro and AQuA-RAT in both training seeds, and on MedQA and GPQA in one of two [seed 2 pending]; on the development pools, only on MedQA. Against random subsets sent to the Verifier only, the learned selection is above random on MedQA, MMLU-Pro and AQuA-RAT in both seeds and on GPQA in neither. It is not significantly different from ranking questions by the manager's draft confidence at the same number of calls (pooled +0.5 points [−1.3, +2.4]); a confidence gate whose threshold is fixed on the development pool for best development accuracy makes 0.80 / 0.68 / 0.84 / 0.93 calls and is 5.5 points more accurate [+3.8, +7.3]."
6. **New attribution paragraph.** "Relative to the untrained model under the same output grammar, training raises the draft by 4.8 points pooled [+2.7, +6.9] without exceeding that model's letter-only accuracy. The use of the Verifier's output on same-candidate questions is not different (pooled +0.1 [−0.6, +0.8], 962 of 1354 questions), and the learned policy does not select questions significantly better than the untrained model's confidence (pooled +1.0 [−0.7, +2.7]). Without the grammar the untrained model leaves the protocol on 1325 of 1354 questions."

Corrected advisor description.

7. **Setup, replaces "role-specific LoRA specialists".** "The Extractor, Reasoner and Verifier are the base model Qwen3.5-9B under role-specific prompts. Role adapters were trained but were not applied at serving time." The evidence (Verifier 20 of 20 outputs identical to the base model, Extractor 5 of 20, no Reasoner check) goes in the appendix.

New tables and figure.

8. Table R1 replaces the forced-Verifier baseline of Section 4.1 and gives the test-pool version of the fixed-role and random controls, with the random-subset Verifier and the confidence gate added.
9. Table R2 becomes a new subsection on what training changed.
10. Figure R1 (accuracy against calls) is added next to Table R1.
11. **Table 4 caption.** "The GPQA row is measured on Diamond-100, which is also the evaluation set; on a disjoint 100-question pool learned calling reaches 53.0 at 0.62 calls and always-Verifier with the candidate 57.0."

Limitation statements.

12. "These comparisons use two training seeds [seed 2 pending] and one test pool per benchmark. The intervals reflect the sampling of questions, not training-seed variation. The seed and maximum sequence length of the original run were not logged and the AQuA-RAT recipe is assumed from the other three benchmarks, so the two runs may differ in more than the seed. GPQA has 100 questions and does not resolve the differences reported here (for example −2.0 [−9.0, +5.0])."
13. "The learned policy is not shown to select questions better than a confidence threshold, and its choice of advisor adds nothing measurable (−0.5 [−1.4, +0.4])."
14. "Training does not reduce corruption: on same-candidate questions a Verifier call breaks 3.9 / 3.1 / 11.1 / 7.1% of right candidates for the trained manager and 3.9 / 2.6 / 7.4 / 3.6% for the untrained one."
15. "Forced-all with other role orders and external learned routers were not evaluated."
16. "The test pools were also used by our earlier runs."
17. "The MedQA manager evaluated here was trained with the paper's configuration but is not the paper's checkpoint, which was not archived."

## Pending

- **[seed 2 pending]** A third training seed (same label file and recipe, training seed 2) is running. It enters: the common block (item 2); the random-selection sentences (cPCG "What survives", sCpX W2, A2jt first bullet, eCWp W2); the seed statements (closing lines of cPCG, A2jt and eCWp, the last sentence of sCpX W4); the note "Second seed, pooled" under Table R2; change 5 and change 12. If it differs from the other two on MedQA or GPQA, "only in the original seed" becomes "in k of three seeds". The unmarked two-seed statements take a third value as well: rows a to m and the difference rows of Table R1, the random-subset sentence of sCpX W3, the candidate paragraph of cPCG, eCWp W1 ("on all four for the second seed") and change 4.
- **Success-distillation control: done (2026-11-26).** Filled in under Table R2 (note "Matched-data success distillation"), in sCpX W1 and in A2jt. It rests on one training run (seed 42). If space allows, its rows can be added next to S_1 in Table R1 from `REB/verifier_grid/sucd_test.md`.

## Notes for the author (not part of the response)

1. **The 498/500 check.** I recomputed it from the per-question records (`agent_routing/outputs/verifier_grid/s1/<bench>/test/grid_records.jsonl` against the S_1 locked-test outputs of the v3+GRPO runs in `agent_routing/docs/mcq_rsi_results/results/grpo_grid_locked_tests_v3g.tar.gz` and `grpo_grid_mmlu_pro_v3g_split.tar.gz`); a second recomputation for this revision gave the same counts. Right/wrong agrees on 500/500, 498/500, 100/100, 254/254. On MMLU-Pro the answer letter itself agrees on 495/500, so the response says "scores the question the same way (right or wrong)". No result table prints this count: the tables print a different check (first turn equal to unconstrained greedy decoding: 500/500, 499/500, 100/100, 254/254). The Sources line of the common block therefore cites the records and no longer `configuration.md` or `seeds_test.md` for item 3. Having the grid script print the count in `s1_test.md` would give it a table source.
2. **"Table 1".** The evidence files label the deployed run "Table 1", but eCWp quotes 79.8 / 60.2 / 76.8 from the submitted Table 1, and the run the grid reproduces is 78.8 / 63.6 / 54.0 / 77.2 (the S_1 row of the RSI table). The response now says this in three places: common block item 3 ("our re-evaluation ... not the submitted Table 1 run"), eCWp W1 and the note "Rows 8 and 9" under Table R2. The two reasons given are from `REB/configuration.md` sections 1 and 3: the registered MedQA S_1 substitutes for the paper's locked checkpoint (the registry says the locked one "is not on HF"; the response says "not archived"), and the paper's MMLU-Pro test set is the overlapping pool. For AQuA (76.8 in the paper, 77.2 here) the files give no reason and hold no evaluation record of the paper run, so the note says only that the official test split was evaluated again. Still to fill in: the number of the paper table that the re-evaluated run corresponds to.
3. **MMLU-Pro pool sentence.** It is not in the brief. It comes from `REB/configuration.md` section 3 (200 questions shared with the development pool, 68 with the collection roots; 104 clean). It now stands in the common block (item 2), because eCWp W1 needs it to explain the changed MMLU-Pro number; sCpX W4 points to it. Keep or move it to the co-author who answers the split question.
4. **Wording narrower than the brief, because the tables are.** (a) "Recovers most": on AQuA training recovers 3.5 of about 9 points (41.73 − 32.68 = 9.05 in the JSON), so the four numbers are given, with their p-values (only MMLU-Pro is significant per benchmark). (b) "Never improves ... lowers it on AQuA": no benchmark shows a significant gain in either seed (MMLU-Pro is +0.6, p = .250, for S_1), and the AQuA drop holds for S_1 (−3.5, p = .022), not for the second seed (+0.0, p = 1.000); the text says "does not improve" or "no gain", gives the pooled value, and names the seed for AQuA. (c) Untrained manager + always-Verifier against learned calling: no paired test in the files, so "numerically above", on three of four tests for S_1 and on four of four for the second seed (83.6 / 70.4 / 52.0 / 81.9 against 78.0 / 65.2 / 50.0 / 76.0); eCWp W1 now gives both. The equal-call comparison (row 6 of Table R2) is no longer written out in eCWp W1, which points to the row; it stays in change 1. (d) The Reasoner / Extractor comparisons are also descriptive only; cPCG now says that no paired test was run. (e) Use of evidence: the claim is limited to a forced Verifier call on the 962 of 1354 same-candidate questions; "unchanged" is no longer used.
5. **Not rerun.** The paper's Figure 3 configuration at about one call per question (eCWp W2 now says so); forced-all with other role orders; an external learned router; a separately trained manager-only model (sCpX W1 now says so).
6. **Advisor evidence.** `REB/configuration.md` section 2 notes that the command lines of the MedQA, MMLU-Pro and GPQA paper-era runs are not recorded; the statement rests on the recorded decision, the parameter-name mechanism and the replays. The common block now separates what is shown output for output (Verifier, 20 of 20) from what rests on the mechanism (Extractor, 5 of 20 identical; Reasoner, no check; the paper-era runs). The same section has the A/B with correctly loaded adapters on 754 development questions (base / paper adapter: Extractor 79.7 / 80.4, Reasoner 80.6 / 80.1, Verifier 81.4 / 79.8) for whoever answers sCpX point 5.
7. **Length.** Counting only tokens that contain letters (heading line excluded), the reviewer sections now have about 365 (cPCG), 490 (sCpX), 375 (A2jt) and 490 (eCWp) words; counting every number and symbol as a word, about 470, 625, 465 and 660. The same counter gives 281 / 339 / 302 / 341 for the previous draft, which this note reported as about 255 / 315 / 290 / 325, so cPCG and A2jt are at the upper end of the 250 to 350 target and sCpX and eCWp are above it. The excess is the set of statements this revision adds for those two reviewers (sCpX: the manager-only and success-distillation qualifications, the confidence gate with a development threshold, the random-subset Verifier, the independent GPQA pool against random, the selection protocol; eCWp: the letter-only reference for the trained weights and the per-benchmark p-values, the re-evaluation against the submitted Table 1, the second seed in the always-Verifier comparison, Figure 3). To make room, two things already moved to shared parts: the equal-call sentence of eCWp W1 (now a pointer to row 6 of Table R2) and the MMLU-Pro, training-code and bootstrap sentences of sCpX W4 (now in item 2 of the common block). Further cuts that lose no fact, because the content is in a shared part, in this order. sCpX: (i) the Evidence bullet of W1 and the same-draft sentence (Table R1, row g − b; Table R2, row 4 and its note), about 45 words; (ii) the development-threshold gate sentence of W2 (note "Confidence gate" under Table R1), about 30; (iii) the AQuA and selection-protocol sentences of W4 (common block, item 2), about 30. eCWp: (i) the paragraph "The gain is the audit ..." (Table R1, rows g − b and b − a), about 30 words; (ii) the reasons in the re-evaluation sentence (common block, items 2 and 3; note "Rows 8 and 9"), about 35; (iii) the development-threshold clause of the last paragraph, about 25; (iv) "Given the same candidate ..." (Table R2, row 4), about 20. With all of these, sCpX and eCWp come to about 385. The common block (about 500 words) and the two tables are shared and counted once.
8. **Placeholders.** "[seed 2 pending]": common block (item 2), Table R2 note "Second seed, pooled", cPCG "What survives" and closing line, sCpX W2 and W4 (last sentence), A2jt first bullet and closing line, eCWp W2 and closing line, changes 5 and 12. The success-distillation placeholders are resolved.
9. **Random-subset Verifier (Table R1, rows k to m; sCpX W3).** From the rows "Learned selection, Verifier only" and "Random subset, Verifier only (same budget)" and the line "Share of 2,000 random draws" of `REB/verifier_grid/s1_test.md` and `seed1_test.md`. On AQuA every learned call is a Verifier call, so rows d and l coincide there. GPQA for S_1 is .052 and is counted as not above random.
10. **Selection protocol (sCpX W4).** `REB/configuration.md` section 3, "Selection protocol": how ρ was chosen is not recorded for any benchmark, and the run sheets state an intended protocol only. The response says "not recorded" and makes no claim about what was done.
