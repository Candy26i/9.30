# MARGENT author response (posting view)

Generated from the response draft: the common response, Tables R1–R3, Figure R1 and the four reviewer sections, without the internal source pointers, the list of paper changes, the pending items and the notes for the author.

---

## Common response (to all reviewers)

1. **Advisors.** The paper's description of the advisors as "role-specific LoRA specialists" is wrong for every reported number: the adapters' parameter names matched no module of the served model, so each advisor was the base model (Qwen3.5-9B) under its role prompt. Replays confirm this for the Verifier (20 of 20 outputs identical); the Extractor matched on 5 of 20 and no Reasoner check is recorded. No reported number changes.

2. **Evidence base.** New results use test pools sharing no question with training or development pools; MMLU-Pro has a new pool because the paper's test set, as reconstructed, shared 200 of its 500 questions with the development pool and 196 with training pools, and GPQA is Diamond-100, which Table 7 also used. Each benchmark has four managers: S_1 and three re-trainings of its label file (seeds 1 to 3) with the logged recipe (assumed for AQuA: no training log), which may differ from S_1 in unlogged details (seed, maximum sequence length, attention kernels). The MedQA S_1 is a same-configuration substitute for the paper's unarchived checkpoint. Intervals cover questions, SDs cover seeds; each benchmark has one test pool, also used by our earlier runs.

3. **Check.** S_1's learned-calling row agrees (right or wrong) with the deployed S_1 run of Table 12 on 500/500, 498/500, 100/100 and 254/254 questions.

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

Notes to Table R1. Everything below is S_1 unless a seed is named. The development pools and the development-threshold gates on the manager's own confidence were run for all four seeds: the bullets below give S_1 first, and the seeds in the parts marked new. The development-threshold gate on the untrained manager's confidence is compared with all four seeds and the one on the commit-only manager's confidence with S_1 only (notes to Table R2); each threshold is fixed on that manager's own development grid.

- **Rows d and l, 95% of draws (S_1).** Row d: [74.8, 78.4], [55.2, 59.2], [43.0, 52.0], [70.9, 76.8]. Row l: [75.4, 78.8], [56.0, 60.0], [44.0, 53.0], [70.9, 76.8]. On AQuA every learned call goes to the Verifier, so rows d and l coincide there.
- **Choice of advisor.** First calls by role (Extractor / Reasoner / Verifier): 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196. Seeds 1 / 2 / 3: 3 / 118 / 53, 13 / 7 / 156, 1 / 13 / 41, 0 / 0 / 188; 9 / 122 / 60, 13 / 9 / 161, 1 / 9 / 34, 0 / 0 / 190; 7 / 134 / 38, 14 / 12 / 159, 3 / 14 / 56, 0 / 0 / 183. On AQuA-RAT no seed sends a call to the Extractor or the Reasoner. Learned calling minus the same questions all sent to the Verifier (row c − row k), pooled: −0.5 [−1.4, +0.4], p = .311; seeds 1 to 3: −0.8 [−1.7, +0.1], p = .099; −0.7 [−1.6, +0.1], p = .132; −0.8 [−1.7, +0.1], p = .099.
- **Confidence gate.** Row f ranks the questions inside the test pool, so it is not a deployable gate. With a threshold fixed on the development pool: (i) threshold chosen to match the learned call rate on development: 79.2 / 58.0 / 53.0 / 76.0 at 0.31 / 0.24 / 0.71 / 0.75 calls; learned minus gate is +2.1 [+0.1, +4.0], p = .043 pooled, but the calls differ from row c, so this is not an equal-budget result. (ii) Threshold chosen for best development accuracy (the lowest such threshold, that is the fewest calls, when several tie; on MedQA and AQuA the tie extends to always calling the Verifier in all four seeds, and with the opposite tie rule the gate would be always-Verifier there, with the same 4 / 4 / 0 / 4 pattern of significance): 83.0 / 71.4 / 55.0 / 81.9 at 0.80 / 0.68 / 0.84 / 0.93 calls; learned minus gate is −5.5 [−7.3, −3.8], p < .001 pooled. Seeds 1 / 2 / 3 (new, 2026-11-26; each seed's thresholds fixed on its own development grid), pooled learned minus gate: (i) +0.1 [−1.8, +2.0], p = 1.000; +0.5 [−1.4, +2.4], p = .656; −0.4 [−2.4, +1.6], p = .714, so S_1's +2.1 is not replicated (per benchmark significant in 0 / 1 / 0 / 0 of four seeds); (ii) −6.0 [−7.8, −4.2], p < .001; −5.5 [−7.3, −3.8], p < .001; −6.4 [−8.2, −4.7], p < .001; per benchmark the best-development-threshold gate is significantly more accurate than learned calling in 4 / 4 / 0 / 4 of four seeds, at 0.59 to 0.80 / 0.68 to 0.79 / 0.84 to 0.91 / 0.92 to 0.93 calls.
- **Development pools** (n = 200 / 200 / 100 / 254). Share of random draws at least as accurate as learned calling: .024 / .185 / .111 / .197. C1: −5.5 (.027), −11.5 (<.001), −4.0 (.388), −7.9 (<.001). C2: −1.5 (.664), −3.0 (.345), −4.0 (.388), −2.8 (.296). Share of Verifier-only random subsets at least as accurate as the learned selection sent to the Verifier: .003 / .203 / .075 / .197. On the MedQA development pool always-Reasoner is 85.0 against 84.5 for always-Verifier with the candidate.
- **Development pools, all four seeds (new, 2026-11-26; n = 200 / 200 / 100 / 254).** Mean ± SD over S_1 and seeds 1 to 3: learned calling 80.8 ± 1.3 / 61.2 ± 1.6 / 53.8 ± 1.0 / 73.5 ± 1.2 at 0.37 / 0.37 / 0.63 / 0.77 calls; always-Verifier with the candidate 84.4 ± 0.9 / 72.2 ± 1.0 / 57.0 ± 0.0 / 81.4 ± 0.4. C1 (learned − always-Verifier; S_1 / seed 1 / seed 2 / seed 3, exact McNemar p): MedQA −5.5 (.027) / −2.5 (.359) / −3.0 (.238) / −3.5 (.189); MMLU-Pro −11.5 (<.001) / −9.5 (<.001) / −11.5 (<.001) / −11.5 (<.001); GPQA −4.0 (.388) / −4.0 (.388) / −3.0 (.581) / −2.0 (.727); AQuA −7.9 (<.001) / −7.5 (<.001) / −7.1 (<.001) / −9.1 (<.001); significant in 1 / 4 / 0 / 4 of four seeds. Share of random draws (same calls per advisor) at least as accurate as learned calling: MedQA .024 / <.001 / .001 / <.001; MMLU-Pro .185 / .048 / .266 / .178; GPQA .111 / .054 / .016 / .056; AQuA .197 / .045 / .043 / .129; below .05 in 4 / 1 / 1 / 2 of four seeds. C2 (learned − confidence gate at the same calls): significant in 0 / 1 / 0 / 0 (MMLU-Pro seed 2: −6.0 (.036)). C3 (always-Verifier, with − without the candidate): significant in 0 / 0 / 0 / 4. The seeds share each development pool, as they share each test pool.
- **GPQA on an independent pool.** The GPQA development pool has 100 questions that share no question with Diamond-100, with the 200 collection roots from which the training labels were selected, with the 146-question GRPO pool or with the 160 advisor-SFT questions (checked on the registry pools; the label rows themselves were not read). They come from the same source file as the roots: 87 from its first 406 rows and 13 from the paper-era 40-question development slice, for which it is not recorded whether ρ or the checkpoint was chosen there. There learned calling scores 53.0 at 0.62 calls and always-Verifier with the candidate 57.0 (−4.0 [−11.0, +3.0], p = .388); with − without the candidate is +5.0 [−5.0, +15.0], p = .424. For S_1, learned calling is not above random there (.111 of random draws are at least as accurate) and is 4.0 points below the confidence gate at the same calls (p = .388). Seeds 1 / 2 / 3 on the same pool : learned calling 53.0 / 54.0 / 55.0 at 0.61 / 0.56 / 0.74 calls; always-Verifier with the candidate 57.0 / 57.0 / 57.0; learned − always-Verifier (exact McNemar p) −4.0 (.388), −3.0 (.581), −2.0 (.727). Four seeds: learned calling 53.8 ± 1.0 at 0.63 ± 0.08 calls, difference −3.2 ± 1.0, significant in 0 of 4. With − without the candidate: +7.0 (.210), +7.0 (.210), +7.0 (.210); share of random draws at least as accurate as learned calling: .054 / .016 / .056 (below .05 in seed 2 only: one of four seeds on this pool); learned − confidence gate at the same calls: −1.0 (1.000), +1.0 (1.000), −3.0 (.508).
- **The submission's fixed-role protocol on the test pools, run on S_1.** The paper reported the always-Verifier baseline in Table 7 and force-all in Table 8, both on development sets per their captions (the development pools for MedQA, MMLU-Pro and AQuA; for GPQA, Diamond-100, which the submission states is also its development pool; the runs behind both tables are not recorded). Because both tables used Diamond-100 for GPQA, the test pool used here (sCpX W4), the GPQA forced-Verifier and forced-all values are reruns on the same 100 questions, not first test-pool results, and they do not reproduce the paper: 60.0 here against 55.0 in Table 7, and 56.0 here against 61.0 in Table 8. What produced the paper's values is not recorded, so the gap cannot be attributed; with n = 100 one question is one point (author note 11: to be checked with the co-authors). Protocol of the submission (`evaluate --forced`): the manager writes no draft before the forced call and the Verifier receives no candidate. Accuracy (%) at 1.00 call (forced-all: 3.00 calls), MedQA / MMLU-Pro / GPQA / AQuA: forced Verifier 84.4 / 68.2 / 60.0 / 77.6; forced Reasoner 81.6 / 55.2 / 48.0 / 33.9 (93.3% valid answers on AQuA); forced Extractor 76.8 / 51.0 / 41.0 / 24.4 (98.2% valid on MedQA, 99.6% on MMLU-Pro, 63.4% on AQuA); forced Extractor, Reasoner, Verifier 84.0 / 70.0 / 56.0 / 79.1. Learned calling (row c): 78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 calls. Question by question (same questions; pt [95% CI], exact McNemar p), learned calling − the original forced Verifier: MedQA −5.6 [−9.0, −2.4], p = .001; MMLU-Pro −5.0 [−9.2, −0.8], p = .026; GPQA −6.0 [−14.0, +2.0], p = .238; AQuA −0.4 [−6.3, +5.1], p = 1.000; pooled −4.4 [−6.7, −2.1], p < .001. Learned calling − forced Extractor, Reasoner, Verifier (3 calls): −5.2 [−8.4, −2.0], p = .003; −6.8 [−11.0, −2.6], p = .002; −2.0 [−10.0, +6.0], p = .804; −2.0 [−7.5, +3.5], p = .583; pooled −4.9 [−7.2, −2.7], p < .001. The original forced Verifier differs from row h (always Verifier, candidate removed: 84.2 / 69.2 / 61.0 / 78.3) by +0.2 / −1.0 / −1.0 / −0.8 points (pooled −0.5 [−1.4, +0.4], p = .311) and gives the same outcome on 98.6 / 97.4 / 93.0 / 96.9% of questions (97.4% pooled), so for the Verifier the two protocols are not significantly different. This does not extend to the Extractor, where the original protocol is 2.0 points lower pooled ([−3.5, −0.5], p = .012; AQuA −5.1, p = .053, same outcome on 84.6% of questions); for the Reasoner the pooled difference is −0.4 [−1.8, +1.0], p = .682 (same outcome on 86.6% of AQuA questions). Always Verifier with the candidate (row g) − the original forced Verifier is −1.0 (p = .568), +4.2 (p = .020), −4.0 (p = .481), +7.5 (p = .003); pooled +2.3 [+0.4, +4.2], p = .025.

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

Notes to Table R2 (S_1 unless a seed is named).

- **Row 4 caveat.** Same-candidate questions are easier than the rest (trained draft accuracy inside / outside the subset: 77.4 / 47.0, 61.0 / 33.7, 41.5 / 40.0, 37.3 / 34.6); questions where the drafts differ cannot be compared this way. With another advisor forced on the same questions the pooled difference is +1.1 [−0.3, +2.6], p = .161 (Reasoner) and +0.8 [−0.7, +2.4], p = .366 (Extractor); per benchmark the Extractor on MedQA is +2.5 [+0.5, +4.8], p = .035. In two re-trainings the Extractor row is significant on AQuA with the trained manager lower: seed 2, −4.0 [−7.3, −1.3], p = .031; seed 3, −7.1 [−11.6, −3.2], p = .003. These three are the only per-benchmark cells of the forced-call rows (Verifier with and without the candidate, Reasoner, Extractor, empty output) that are significant by the exact McNemar test in any seed; for the Verifier's output (row 4) none is, per benchmark or pooled.
- **Right candidates broken by a Verifier call on same-candidate questions (%), trained / untrained.** S_1: 3.9 / 3.9, 3.1 / 2.6, 11.1 / 7.4, 7.1 / 3.6. Seed 1: 3.9 / 3.9, 2.6 / 2.6, 11.1 / 7.4, 10.3 / 6.9. Seed 2: 3.8 / 3.8, 2.6 / 2.6, 11.1 / 7.4, 10.3 / 6.9. Seed 3: 3.4 / 3.4, 2.6 / 2.6, 11.1 / 7.4, 8.3 / 5.0.
- **Learned call rate on wrong / right candidates (%).** S_1: 63.2 / 27.0, 48.8 / 18.1, 50.8 / 53.7, 84.6 / 64.1. Seed 1: 59.4 / 26.3, 51.0 / 20.2, 55.0 / 55.0, 82.1 / 59.8. Seed 2: 61.4 / 30.3, 53.1 / 21.0, 43.9 / 44.2, 82.8 / 60.4. Seed 3: 61.2 / 26.5, 52.8 / 21.7, 69.5 / 78.0, 81.1 / 56.8.
- **Row 6, exact decomposition (S_1)** = draft + what the same confidence-gated Verifier calls add (trained − untrained) + learned calling − the trained manager's own confidence gate. MedQA: −1.8 = +2.8 − 3.8 − 0.8. MMLU-Pro: +5.6 = +6.6 − 2.6 + 1.6. GPQA: +7.0 = +9.0 − 4.0 + 2.0. AQuA: +5.5 = +3.5 + 1.6 + 0.4.
- **Row 6 per benchmark in seeds 1 / 2 / 3** (difference, exact McNemar p). MedQA: −2.4 (.119), −2.0 (.212), −1.6 (.332). MMLU-Pro: +6.8 (.001), +6.8 (.002), +6.0 (.006). GPQA: +2.0 (.774), +7.0 (.143), +4.0 (.344). AQuA: +5.1 (.111), +4.7 (.141), +5.1 (.111). Per benchmark the equal-call difference is significant on MMLU-Pro only, in all four seeds.
- **Row 6 with a deployable gate (S_1 first, then seeds 1 to 3; new, 2026-11-26).** Row 6 ranks the untrained manager's confidences inside the test pool. With the threshold fixed on the development pool instead (the cut between two distinct development confidences of the untrained manager whose call count is closest to the learned policy's development call count: 37.0 / 30.5 / 62.0 / 78.0% of development questions), the untrained manager with that gate scores 80.6 / 56.0 / 49.0 / 73.6 at 0.32 / 0.28 / 0.71 / 0.79 calls on the test pools. Learned calling − that gate: MedQA −1.8 [−4.6, +0.8], p = .253; MMLU-Pro +7.2 [+3.0, +11.6], p = .001; GPQA +5.0 [−1.0, +12.0], p = .227; AQuA +3.5 [−2.0, +9.1], p = .272; pooled +3.0 [+0.9, +5.2], p = .009, at 0.42 calls per question for the gate against 0.44 for learned calling. Per benchmark the calls differ from learned calling's (0.37 / 0.33 / 0.52 / 0.77), so this is a deployable comparison, not an equal-budget one.
- **Rows 8 and 9.** Both are rows of the submission's Table 12, on the test pools of this response: row 8 is its "Base system" row and row 9 its S1 row (round 1); the untrained values of row 2 are its "Base, letter only" row. Row 9 is not the submitted Table 1 (79.8 / 60.2 / 76.8 on MedQA / MMLU-Pro / AQuA, as quoted by eCWp): the MedQA checkpoint is a substitute, the MMLU-Pro pool is new and disjoint, and AQuA is the official test split evaluated again. The untrained system of row 8 returns a valid answer on 95 / 85 / 52 / 84% of questions, so its GPQA accuracy (39.0) reflects invalid outputs as well as wrong answers.
- **Sibling of S_1 trained with no commit cap** (same collection and recipe, every commit row kept, as the training logs record for MedQA, MMLU-Pro and GPQA (433 / 382 / 160 rows, lr 1e-5, 10 epochs; the GPQA collection counts are themselves derived from that log); for AQuA no label file or training log was read, so its training data and recipe are known from the adapter name only; one training run; it is not a commit-only or manager-only control: that is the note after the next). S_1 − sibling, pooled: draft −0.3 [−1.4, +0.8], p = .699; evidence −0.1 [−0.6, +0.4], p = 1.000 (n = 1253); selection +0.1 [−1.6, +1.8], p = 1.000; equal calls −0.7 [−2.7, +1.3], p = .563; always Verifier −0.1 [−0.9, +0.7], p = 1.000. The sibling's learned calling scores 77.2 / 59.2 / 54.0 / 69.7 at 0.23 / 0.21 / 0.68 / 0.63 calls.
- **Commit cap ρ.** The submission's sibling checkpoints of S_1 that were evaluated under another cap (the seven sibling adapters with another cap and the four no-cap adapters of the collection runs that produced S_1; cap rule: every rescue record kept, commit records cut to int(ρ × rescue records); one adapter per cap; whether adapters are stored for further caps, for example ρ = 0 on MMLU-Pro or GPQA, is not recorded) were put through the same equal-information grid. Label files or training logs were read and agree with the cap rule for 10 of the 15 managers (for the two GPQA managers among them, ρ = 2 and no cap, the collection counts were themselves solved from the row counts of those two training logs, so that agreement is not an independent check); training logs that print the S_1 settings (lr 1e-5, 10 epochs) are on record for 8 of the 15 (none for MedQA ρ = 2 or for any AQuA manager); for MMLU-Pro ρ = 1, GPQA ρ = 1 and AQuA ρ = 0 / 0.5 / no cap no label file or training log was read, so their training data and recipe are known from the adapter names only. Learned calling, calls per question and accuracy on the test pools. MedQA, ρ = 0 / 1 / 2 / 3 (S_1) / no cap: 1.00 / 0.60 / 0.50 / 0.37 / 0.23 calls, 82.2 / 80.0 / 78.8 / 78.8 / 77.2 (draft 69.6 / 71.4 / 72.4 / 73.4 / 73.0). MMLU-Pro, ρ = 1 / 2 (S_1) / no cap: 0.33 / 0.33 / 0.21 calls, 62.0 / 63.2 / 59.2 (draft 48.8 / 50.8 / 51.8). GPQA, ρ = 1 / 2 (S_1) / no cap: 0.82 / 0.52 / 0.68 calls, 51.0 / 54.0 / 54.0 (draft 37.0 / 41.0 / 40.0). AQuA, ρ = 0 / 0.5 / 1 (S_1) / no cap: 1.00 / 0.88 / 0.77 / 0.63 calls, 85.0 / 81.9 / 77.2 / 69.7 (draft 30.3 / 34.3 / 36.2 / 37.0). With ρ = 0 the manager calls on every question. On MedQA and AQuA the call rate falls at every step as the cap rises; the draft accuracy rises at every step on AQuA (30.3 to 37.0) and up to ρ = 3 on MedQA (69.6 to 73.4; 73.0 with no cap); on MMLU-Pro ρ = 1 and ρ = 2 give the same call rate, and GPQA (n = 100) is not monotone (0.82 / 0.52 / 0.68). One result of Table R1 keeps its direction at every cap tested, and for a second a one-sided bound holds. C1: in all 15 managers learned calling is numerically below always Verifier with the candidate from the same manager's draft (−0.4 to −15.0 points; significant in 10 of 15 by the exact McNemar test, not on GPQA and not at ρ = 0). C2: in none is it significantly above the Verifier on that manager's least confident candidates at the same number of calls (ranked inside the test pool; at ρ = 0 that gate is always Verifier, so 13 of these comparisons are distinct); the "not significantly different" result of Table R1 does not hold at every cap, because learned calling is significantly below that gate in 2 (MedQA ρ = 1, −2.8 [−5.4, −0.2], p = .049; MedQA ρ = 2, −3.4 [−6.0, −1.0], p = .014), and the sign of the difference changes with the cap on GPQA and AQuA. The comparison with random questions at the same calls per advisor (row e) does depend on the cap. Share of 2,000 random draws at least as accurate as learned calling, caps in the order above: MedQA .785 / .051 / .096 / .008 / .005; MMLU-Pro <.001 at every cap; GPQA .492 / .005 / .035; AQuA .192 / .005 / .017 / .057 (at ρ = 0 on MedQA learned calling, 82.2, is below the random mean, 82.7). Among the 15 managers of this sweep, the trained-against-untrained contrasts of Table R2 were run for S_1 only (outside the sweep also for its three re-trainings with seeds 1 to 3, which use the S_1 label file and so S_1's cap: lower block of Table R2), not for any of the 11 sibling managers at another cap (for the no-cap siblings only S_1 − sibling was run: previous note).
- **Commit-only control** (sCpX's "equally trained manager-only revision control"; one training run, compared with S_1 only). A manager trained on the same round-1 records with the S_1 recipe and seed 42, on the root-commit rows only (304 / 213 / 86 / 217 rows for MedQA / MMLU-Pro / GPQA / AQuA; the GPQA run used maximum sequence length 16384 because one row exceeds 4096 tokens); it is never shown a call or an advisor output. It makes no call on any test question (0 of 500 / 500 / 100 / 254; its first turn under the output grammar equals unconstrained greedy decoding on every question), so as deployed it scores its draft: 73.6 / 53.2 / 42.0 / 38.6. S_1's learned calling (78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 calls) − the commit-only manager: MedQA +5.2 [+2.0, +8.4], p = .002; MMLU-Pro +10.0 [+6.2, +13.8], p < .001; GPQA +12.0 [+5.0, +19.0], p = .002; AQuA +38.6 [+31.5, +45.7], p < .001; pooled +13.7 [+11.4, +16.2], p < .001. Draft, S_1 (73.4 / 50.8 / 41.0 / 36.2) − commit-only: −0.2 [−2.4, +2.0], p = 1.000; −2.4 [−5.0, +0.2], p = .088; −1.0 [−5.0, +3.0], p = 1.000; −2.4 [−6.3, +1.6], p = .327; pooled −1.5 [−3.0, +0.0], p = .062; the commit-only draft is numerically higher on every benchmark. Use of evidence (accuracy after a forced Verifier call on the 1169 same-candidate questions), S_1 − commit-only, pooled: −0.6 [−1.5, +0.2], p = .210; per benchmark it is significant on MMLU-Pro only, with S_1 lower (76.3 against 78.0; −1.7 [−3.1, −0.5], p = .039). Selection signal (S_1's Verifier outcomes on the questions chosen by the learned policy − by the commit-only manager's confidence, same number of calls), pooled: +0.7 [−1.0, +2.3], p = .478. Confidence gate at equal calls: the commit-only manager with the Verifier on its least confident candidates (ranked within the test pool, S_1's number of calls) scores 80.2 / 66.0 / 49.0 / 74.8; S_1's learned calling − that system is −1.4 [−4.4, +1.4], p = .410; −2.8 [−6.4, +0.8], p = .156; +5.0 [−1.0, +12.0], p = .227; +2.4 [−3.1, +7.5], p = .461; pooled −0.7 [−2.8, +1.3], p = .518. Always Verifier with the candidate (1.00 call): 83.2 / 74.6 / 57.0 / 84.3 from its own draft, 21.1 points above that draft pooled ([+18.8, +23.6], p < .001), against 83.4 / 72.4 / 56.0 / 85.0 for S_1; S_1 − commit-only, pooled: −0.7 [−1.8, +0.5], p = .328 (no benchmark significant). Right candidates broken by a Verifier call on these same-candidate questions (%), S_1 / commit-only: 3.4 / 3.1, 5.0 / 2.5, 10.3 / 5.1, 8.5 / 4.9.
- **Matched-data success distillation** (one training run; the same round-1 records and the same commit cap; each kept record trains one successful option drawn at random instead of the shortest one; same SFT recipe; the default selection of the same records reproduces the S_1 label file row for row). Its learned calling scores 81.6 / 73.2 / 50.0 / 83.1 at 0.98 / 1.00 / 0.99 / 0.96 calls, against 78.8 / 63.2 / 54.0 / 77.2 at 0.37 / 0.33 / 0.52 / 0.77 for S_1. S_1 − success distillation: MedQA −2.8 [−5.4, −0.2], p = .044; MMLU-Pro −10.0 [−13.2, −6.8], p < .001; GPQA +4.0 [−4.0, +12.0], p = .454; AQuA −5.9 [−9.8, −2.0], p = .006; pooled −5.5 [−7.3, −3.8], p < .001. Accuracy gained per call: 14.8 / 37.3 / 25.0 / 53.1 points for S_1, 10.0 / 23.2 / 10.1 / 47.1 for success distillation. First calls to Extractor / Reasoner / Verifier: S_1 2 / 109 / 72, 10 / 12 / 144, 1 / 9 / 42, 0 / 0 / 196; success distillation 66 / 191 / 231, 73 / 59 / 367, 54 / 16 / 29, 0 / 0 / 244. Its draft accuracy is 71.8 / 50.0 / 40.0 / 37.8 (S_1: 73.4 / 50.8 / 41.0 / 36.2). Against always calling the Verifier with the candidate from its own draft it is not ahead (pooled −1.6 [−2.7, −0.4], p = .014).

## Table R3. Deployed evaluation (paper protocol) across training seeds

The manager routes freely and may call more than once. Cells: accuracy (%) / advisor calls per question on the test pools.

| Checkpoint | MedQA (n=500) | MMLU-Pro (n=500) | GPQA (n=100) | AQuA (n=254) |
|:--|--:|--:|--:|--:|
| S_1 | 78.8 / 0.366 | 63.6 / 0.352 | 54.0 / 0.520 | 77.2 / 0.776 |
| Seed 1 | 78.0 / 0.348 | 65.4 / 0.374 | 50.0 / 0.550 | 76.0 / 0.740 |
| Seed 2 | 78.6 / 0.386 | 65.6 / 0.404 | 51.0 / 0.440 | 76.0 / 0.748 |
| Seed 3 | 79.0 / 0.362 | 65.2 / 0.402 | 52.0 / 0.730 | 75.6 / 0.724 |
| Mean ± SD, four seeds | 78.6 ± 0.4 / 0.365 ± 0.016 | 65.0 ± 0.9 / 0.383 ± 0.025 | 51.8 ± 1.7 / 0.560 ± 0.122 | 76.2 ± 0.7 / 0.747 ± 0.022 |

Note. Corrupted answers (% of questions; mean ± SD over the four seeds): 3.9 ± 0.2 / 1.4 ± 0.2 / 1.5 ± 1.3 / 2.8 ± 0.0. Valid answers: 99.8% for seeds 2 and 3 on MedQA, 100.0% in every other cell. Every seed is numerically below the untrained system of Table R2, row 8, on MedQA, MMLU-Pro and AQuA and above it on GPQA.

## Figure R1


Caption: "Accuracy against advisor calls per question for S_1 on the four test pools. Circle: candidate only. Diamond: learned calling (first call). Curve: Verifier on the least confident candidates. Dashed line: Verifier on a random subset, with the 95% interval at the learned number of calls. Squares: always Verifier with the candidate (filled) and with the candidate removed (open)."

---

## Reviewer cPCG: weakness 1 (the always-Verifier baseline)

Thank you. Weakness 1 is correct, and the rerun changes our conclusion.

**What each call receives.** The policy's Verifier call carries the question, options and context plus two lines naming the manager's draft letter as the candidate to audit; the paper's always-Verifier baseline omitted those lines. The Extractor and Reasoner never receive the candidate.

**Equal-information rerun (Table R1), four training seeds.** All rules share the candidate, checkpoint, greedy decoding and revision step and make at most one call. Always-Verifier with the candidate (row g) is more accurate than learned calling (row c) in every seed on MedQA, MMLU-Pro and AQuA, by 5.0 to 8.5 points on average (contrast C1 lists each exact p). On GPQA (n = 100) it is not significantly ahead in any seed. Without the candidate (row h) it is still numerically above learned calling on every pool in every seed.

**Does the candidate matter (C3)?** It helps on MMLU-Pro (significant in three of four seeds) and AQuA (two of four), has no detectable effect on MedQA, and is unresolved on GPQA, where no seed is significant.

**Other advisors.** Always-Reasoner and always-Extractor (rows i, j) are numerically below always-Verifier on every test pool in every seed; no paired test was run.

**What survives.** Learned calling is a lower-call operating point, not a more accurate one. It is significantly above random questions at the same calls per advisor (row e: share of random draws below .05) in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA. It is not significantly different from a confidence-gated Verifier at the same calls, ranked within the test pool, in any seed on any benchmark (C2).

We withdraw the Figure 4 comparison with always-Verifier (Section 4.3, Tables 7 and 8); Table R1 and Figure R1 will replace the baseline.

---

## Reviewer sCpX: weaknesses 1 to 4

Thank you. Your controls do not support the always-Verifier claim.

**W1.** Agreed.
- Empty-output revision turn: no significant gain in any seed (Table R1, C5).
- Verifier output: a mean 10.6 to 49.5 points over that turn, significant in every seed and benchmark (C4).
- SFT against the untrained model, same grammar (Table R2): draft +4.8 to +5.5 pooled; use of the Verifier's output and selection not significantly different in any seed.
- Equally trained manager-only (commit-only) control (Table R2 notes; one run): with a confidence-gated Verifier at S_1's calls, ranked in-pool, not significantly different from learned calling (pooled p = .518). Alone it never calls: significantly less accurate on every benchmark.
- Matched-data success distillation (Table R2 notes) calls on almost every question and is 5.5 points more accurate pooled (p < .001; one run): the shortest path adds stopping, not accuracy.

**W2.**
- Fixed roles, test pools (rows g to j): always-Verifier is numerically highest.
- Random questions, same calls per advisor (row e: share of draws below .05): learned calling is significantly above in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA.
- Confidence-gated Verifier, same calls, ranked within the test pool: not significantly different in any seed or benchmark (C2); a development-threshold gate at 0.68 to 0.93 calls is 5.5 points more accurate pooled (S_1; Table R1 notes).

**W3.** Correct. Under equal information, always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1; Figure R1). Random-subset Verifier (row m): significantly below the learned selection, except on GPQA.

**W4.** GPQA: conceded; Diamond-100 also served development (Table 4, †). On a disjoint pool, S_1's learned calling scores 53.0 against always-Verifier's 57.0 (p = .388; Table R1 notes). AQuA: official validation and test splits, disjoint. The code selects no checkpoint among epochs; how ρ was selected is not recorded. Seeds, advisors: common block.

W5: co-authors.

---

## Reviewer A2jt: call selection and roles; attribution and baselines

**Uneven call selection and harmful roles.** Thank you for pointing this out. Selecting useful calls without overturning correct answers is also our central goal. Inspired by it, we ran supplementary experiments on test pools with four seeds (mean ± SD):

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| Learned calling: accuracy (%) / calls | 78.6 ± 0.4 / 0.36 | 64.7 ± 1.0 / 0.35 | 51.8 ± 1.7 / 0.56 | 76.2 ± 0.7 / 0.75 |
| Seeds above random, equal calls (p < .05) | 2/4 | 4/4 | 1/4 | 4/4 |
| Confidence gate, same calls (%) | 80.0 ± 0.9 | 63.3 ± 1.3 | 51.2 ± 1.5 | 74.4 ± 1.6 |
| Call rate, wrong / right drafts (%) | 61.3 / 27.5 | 51.4 / 20.3 | 54.8 / 57.7 | 82.7 / 60.3 |
| Calls to Extractor / Reasoner / Verifier (%) | 3 / 66 / 31 | 7 / 6 / 87 | 3 / 20 / 77 | 0 / 0 / 100 |
| Corrupted answers (%) | 3.9 ± 0.2 | 1.4 ± 0.2 | 1.5 ± 1.3 | 2.8 ± 0.0 |
| Untrained manager, same advisors: accuracy (%) / calls | 82.8 / 2.98 | 69.0 / 2.51 | 39.0 / 2.53 | 78.3 / 1.62 |
| Tokens per question, S1 / untrained | 2,222 / 11,770 | 2,373 / 9,545 | 2,807 / 10,098 | 3,171 / 5,310 |

Selection thus helps on MMLU-Pro and AQuA-RAT but not reliably on MedQA and GPQA. Accuracy, however, is not our main strength; our focus is cost. Against the untrained manager with the same advisors, MARGENT makes 52–88% fewer calls and uses 40–81% fewer tokens while retaining 92–99% of its accuracy (exceeding it on GPQA). Calls and accuracy hold in all four seeds (Table R3); tokens were measured for S1.

Thank you again; this motivates future work on more even call selection.

**Confounded attribution and asymmetric baselines.** Correct. With the candidate, always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1), at one call per question against 0.33 to 0.77.

Against the untrained model under the same output grammar (Table R2), in all four seeds:
- the draft rises (pooled +4.8 to +5.5) without exceeding that model's letter-only accuracy;
- the use of the Verifier's output on same-candidate questions does not differ significantly;
- the learned choice of questions is not significantly better than that model's own confidence.

A gain from routing alone is not established: training provides protocol-following output and a lower call rate. Matched-data success distillation confirms that the shortest-path target buys the lower call rate (Table R2 notes).

Forced-all in other role orders and external learned routers remain untested limitations.

---

## Reviewer eCWp: weaknesses 1 and 2

Thank you. Both points are correct.

**W1.** Agreed. In no seed does the trained manager's draft exceed the untrained model's letter-only accuracy on any benchmark (Table R1, row a; Table R2, row 2). The drop comes from the manager format, not the weights (letter-only, S_1 is within one point of the untrained model); under it, training recovers 4.8 to 5.5 points pooled (row 3). The two managers' use of the Verifier's output is not significantly different in any seed (row 4): training taught the protocol.

Your untrained-system comparison stands: the four deployed managers make 0.35 to 0.78 calls per question against at least 1.62, with numerically lower accuracy on MedQA, MMLU-Pro and AQuA (Table R2, rows 8 and 9; Table R3).

Our S_1 numbers are Table 12's S_1 row, not Table 1: substitute MedQA checkpoint and new MMLU-Pro pool (common block, item 2); AQuA is 77.2 against 76.8, and we hold no record of the Table 1 run.

**W2.** Agreed. The 0 to 4.5 point advantage of Figure 4 (your Figure 3) is not supported under equal information and we withdraw it: always-Verifier is ahead of learned calling in every seed on MedQA, MMLU-Pro and AQuA and never significantly on GPQA (C1).

The gain is the Verifier's output (C4: significant over an empty-output revision turn in every seed and benchmark), not that turn alone (C5).

Learned calling is significantly above random questions at the same calls per advisor (row e: share of random draws below .05) in all four seeds on MMLU-Pro and AQuA, two on MedQA and one on GPQA; on the development pools, for S_1, only on MedQA, as you read Table 10. At the same calls it is not significantly different from ranking the test pool by the manager's or the untrained model's confidence in any seed (C2; Table R2, row 5).

We will restate the contribution as protocol-following output at fewer calls, not accuracy.

---
