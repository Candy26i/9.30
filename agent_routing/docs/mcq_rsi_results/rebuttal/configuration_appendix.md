# Configuration appendix

Reviewer-facing configuration of every result in the submission and in this response: which configuration produced each result (A),
the manager's data collection, labels and SFT (B), the RSI rounds, FA-GRPO and the "Success" comparator (C), the exact prompts,
decoding settings and release plan (D), and the advisors (E). Values are given in the order MedQA / MMLU-Pro / GPQA / AQuA-RAT unless a
table has one column per benchmark. Every value comes from a stored record or from the code, and each section states how each value is
sourced: stored by the run itself; reproduced exactly from stored outputs; taken from the configuration or code only; stated in the
submission only; or not recorded. Gaps are stated as gaps. Table and figure numbers refer to the submission as reviewed.

Internal companion (not for reviewers): `configuration_appendix_sources.md` lists the source of every row and the points the authors
must decide before this appendix is posted.

## A. Which configuration produced each result

**Conventions.** Values for the four benchmarks are given in the order MedQA / MMLU-Pro / GPQA / AQuA-RAT. Each value carries a tag:

- **(R)**: recorded by the run itself, in a training log, evaluation record, run report, adapter digest or split manifest.
- **(C)**: set by a code default, a configuration file or a launch script, but not echoed in the run's own record.
- **(P)**: stated in the submission only; no run record exists.
- **n.r.**: not recorded.

Table and figure numbers are those of the submission. Pool names such as `dev`, `test`, `test_paper` and `collect_r1` are defined under *Data splits*. In every row the Extractor, Reasoner and Verifier are the base model (Qwen3.5-9B) answering under each role's system prompt. The trained advisor adapters were not applied in any of these results (see section E). The RSI and rebuttal evaluations record advisor mode "base", greedy decoding, at most 1,024 new tokens and thinking disabled (R).

### A.1 Results of the submission

| Result | Policy kind | Collection depth | Commit cap ρ | Training seed | Checkpoint | Data pool | Decoding / call limit |
|---|---|---|---|---|---|---|---|
| Fig. 3; Tables 3, 5, 6 (oracle by depth, root outcomes, net value of one forced call) | collection (no trained manager) | n.r. [1] / 2 (R) / 2 (R) [1] / 3 (R) | none (counts before balancing) | none; tie-break seed 42 (R) | base model as collecting manager (R) | `collect_r1`: 400 / 400 / 200 / 400 (R) | forced probes, greedy, ≤ 512 new tokens (C); each advisor at most once; expansion stops at the first depth with a success (C) |
| Table 4 (sizes, depth D) | description | D printed as 3 / 2 / 2 / 3 (P) for the oracle collections; for GPQA, D = 2 is also recorded by the matching collection (R). S_1's labels come from depth 2 / 2 / 1 / 3 (R) | – | – | – | see *Data splits* | – |
| Table 1 (collection-disjoint evaluation) | single-pass S_1 | 2 (P) / 2 / 1 / 3 (R) [2] | 3 (P) / 2 / 2 / 1 (R) [3] | n.r. / n.r. / 42 (R) / n.r. [4] | MedQA: the submission's checkpoint, not archived (n.r.); others: the S_1 adapter of the benchmark [5] | MedQA: 500 official test questions, which ones n.r.; `test_paper` 500 (reconstruction); `test` = Diamond-100; `test` 254 | submission-era loop [6]: greedy, ≤ 3 calls, ≤ 1,024 tokens per turn (C) |
| Table 7; Fig. 4, low-call points and always-Verifier | single-pass S_1; fixed-role baseline | 2 (P) / 2 / 1 / 3 (R) | 3 (P) / 2 / 2 / 1 (R) | MedQA: three seeds, integers n.r.; MMLU-Pro, AQuA n.r.; GPQA 42 (R) | MedQA: three checkpoints, none archived (n.r.); others: S_1 adapter [7]. Always-Verifier: "same manager" (P); run n.r. | `dev` 200 / `dev` 200 / Diamond-100 / `dev` 254 (P) [8] | [6]; always-Verifier: forced call, no draft, no candidate (P) |
| Table 8; Fig. 4, higher-call points and forced-all. Table 9; Fig. 5 | single-pass, other operating points; fixed-role baseline | 2 / n.r. / n.r. / 3 (P) [9] | 0 / n.r. / n.r. / 0 (P) [9] | n.r. | n.r. [9] | development sets (P); runs n.r. | [6]; forced-all: Extractor → Reasoner → Verifier, 3 calls, no draft, no candidate (P) |
| Table 10 (matched-budget replay) | replay | n.r. ("own development configurations", P) | n.r. [10] | n.r. | n.r. [10] | development sets (P) | role and number of calls fixed, called subset re-drawn 2,000 times (P) |
| Table 11 (commit cap) | control: single-pass siblings | MedQA 2 and 1; MMLU-Pro 1; AQuA 3 (P) | 0 vs 3; 1 vs 2; 1 vs 2; 0 vs all (P) | n.r. | [11] | development sets (P) | [6] |
| Table 14 A (three MedQA seeds; MedQA cell of Table 7) | single-pass S_1 configuration | 2, 400 roots (P) | 3 (P) | three seeds, integers n.r. | none archived; values typed into the table script (n.r.) [12] | MedQA development, 200 (P) | [6] |
| Table 12: "Base, letter only", "Base system" | baseline: untrained model | – | – | – | base model, no adapter (R) | locked `test` 500 / 500 / 100 / 254 (R) | letter only: plain prompt, no tools, ≤ 16 new tokens, letter forced (R). Base system: RSI harness [13], ≤ 3 calls, unparsed answers scored wrong; the evaluation gate fails (valid answers 95 / 85 / 52 / 84 %) and is reported, not enforced (R) |
| Table 12, S_1 row; Table 13, "S_1 (no extra training)"; Table 14 B, "S_1 twice" | RSI S_1 | 2 / 2 / 1 / 3 (R) | 3 / 2 / 2 / 1 (R) | n.r. / n.r. / 42 (R) / n.r. [4] | S_1 adapter (MedQA: same-configuration substitute) [5]; each evaluation records the adapter SHA-256, equal to the registered digest (R) | Tables 12 and 14 B: locked `test` (R); Table 13: `dev` 200 / 200 / 100 / 254 (R) | RSI harness [13] (R) |
| Table 12, "SFT + GRPO" (static, dynamic, success; round 3); Table 2, SFT + GRPO dynamic column | RSI round-3 arm | round 1: S_1's labels; rounds 2–3: re-collected at depth 2 by the arm's own model (dynamic, success) (C); static reuses S_1's label file | 3 / 2 / 2 / 1, cap seed 0 / 0 / 42 / 0 (C) [14] | collection 42, SFT 42, FA-GRPO 0 (C); evaluation 42 (R) | last round's resolved checkpoint, fixed before testing: G_3, or S_3 if GRPO was rejected; inside a GRPO stage, the last guard-passing step (R) [15] | locked `test` (R) | RSI harness [13] (R) |
| Table 12, "SFT only" (static, dynamic; rounds 3 and 5); Table 2, SFT-only columns (round 3); other Table 14 B contrasts | RSI round-3 or round-5 arm, no GRPO | as above; rounds 4–5 add `collect_r4`, `collect_r5`; no GPQA (C) | as above | collection 42, SFT 42 (C); evaluation 42 (R) | S_3, or S_5 continued from the same arm's S_3 (R); round 2 starts from S_1 (C) | locked `test` (R) | RSI harness [13] (R) |
| Table 13 A | control: one SFT from S_1 per label set | R1: S_1's labels; R2–R5: labels of the dynamic SFT-only arm, collected at depth 2 by S_1 to S_4 (C) | 3 / 2 / 2 / 1 (C) | SFT 42 (C) | end of one 3-epoch SFT from S_1 (lr 1e-5, no loss on draft tokens, max length 4,096; GPQA 16,384) (C), evaluated once | `dev` (R) | RSI harness [13] |
| Table 13 B | control | 2 (C) | as Table 13 A | as Table 13 A | as Table 13 A, trained on labels collected on the same questions by S_k and by G_k of an earlier RSI configuration [16] | `dev` (R) | RSI harness [13] |
| Fig. 7 (tokens) | baseline, RSI S_1, RSI round 3 | as in the rows above | as above | as above | base system; S_1 (from the SFT + GRPO runs for MedQA and GPQA, from the SFT-only runs for MMLU-Pro and AQuA); dynamic SFT + GRPO, round 3 (R) [17] | locked `test` (R) | RSI harness; input plus output tokens of manager and advisors per question (C) |

Notes to A.1:

1. **Collections behind MedQA and GPQA (Tables 3, 5, 6).**
   - MedQA: the depth-2 collection behind S_1 gives 304 / 57 / 5 roots solved at 0 / 1 / 2 calls, not Table 3's 304 / 63 / 2. Adapters trained on a depth-3 MedQA collection exist. That collection's records were not read, so Table 3's MedQA row is not confirmed.
   - GPQA: two collections exist on the same 200 roots, both collected by the base model (R). The depth-1 collection behind S_1 records 86 / 37 roots solved at 0 / 1 calls and 77 unsolved, and one-call net values 0.0 / +4.1 / +14.4 (Extractor / Reasoner / Verifier; R). Fig. 3 and the GPQA rows of Tables 3, 5 and 6 equal the report of a separate depth-2 collection (R): 86 / 44 / 9 roots solved at 0 / 1 / 2 calls and 61 unsolved; A0 / A1 / AD 43.00 / 65.00 / 69.50; one-call net values +0.5 / +4.1 / +15.5. Why the two runs record different one-call outcomes is n.r.
   - Table 6: the GPQA row matches the per-advisor values of the depth-2 GPQA collection report. Those values were not read for the MedQA, MMLU-Pro and AQuA collections, so the collection behind each of their rows is not confirmed.
2. **Depth.** This is the depth of the collection whose labels trained the checkpoint (collection reports, R). For MedQA, the submission places its checkpoint in the depth-2, 400-root, ρ = 3 configuration (P).
3. **Cap values.** The registered label files (315 / 331 / 148 / 402 rows) are reproduced row for row from the collection records by the cap rule, with these ρ and cap seeds 0 / 0 / 42 / 0 (R).
   - How ρ was chosen among {0, 0.5, 1, 2, 3, all} is n.r. The submission only states that ρ was chosen on development data (P).
   - AQuA: a second 402-row label file exists, with the same decision counts but 52 different commit rows. Which file trained the AQuA checkpoint is n.r.
4. **Training seed.** The GPQA S_1 run stores seed 42 in its saved trainer arguments (R), as do its ρ = 1 and no-cap siblings. The MedQA and MMLU-Pro logs print no seed, and AQuA has no training record. For these three, the code default 42 applies only if no seed was passed. The registry labels the MedQA substitute "seed 0"; no training record says which seed that refers to.
5. **S_1 adapters.**
   - MedQA: the ρ = 3 checkpoint of the depth-2, 400-root collection. It stands in, with the same configuration, for the submission's unarchived checkpoint, and its development result is the 82.0 % / 0.405 run of Table 11.
   - MMLU-Pro: the ρ = 2 checkpoint of the depth-2 collection. A depth-1 checkpoint with the same name also exists.
   - GPQA and AQuA: the checkpoints that the repository identifies as the submission's.
   - Recipe: LoRA r 16, α 32, dropout 0.05 on the query, key, value and output projections of the 8 full-attention layers and on the MLP (gate, up, down) projections of all 32 layers. The linear-attention projections of the other 24 layers are not adapted. These LoRA settings are stored with all four adapters. The training settings (lr 1e-5; 10 epochs; effective batch 8) are logged for MedQA, MMLU-Pro and GPQA and are n.r. for AQuA. The adapter saved after the last epoch is used; no choice is made among epochs.
   - The GPQA and AQuA values of Table 1 equal the registry's test targets. No evaluation record of Table 1 is in the repository.
6. **Submission-era evaluation loop.** Command defaults apply: greedy, ≤ 3 tool calls, ≤ 1,024 new tokens per turn. The 9B command lines and the tool-binding mode are n.r. An advisor failure reached the manager as an error tool message.
7. **Registry targets for these adapters.** MMLU-Pro development: 64.5 % at 0.36 calls per question; the printed 0.340 is the share of questions with a call. GPQA Diamond-100: 54.0 % at 0.51. AQuA development: 78.0 % at 0.894, taken from the submission text. No evaluation record is in the repository. The MedQA value 0.517 is the share of questions with a call (Table 14 A).
8. **Development pools.** GPQA's development diagnostics used Diamond-100, which is also the evaluation pool (Table 4, †). MedQA `dev` is the 200-question file of the registry targets. MMLU-Pro `dev` is a reconstruction.
9. **Tables 8 and 9.**
   - Table 8's MedQA "Selected" values equal Table 11's depth-2, ρ = 0 row. AQuA's accuracy equals Table 11's ρ = 0 row: 81.1 % at 1.031 calls in Table 8 against 1.000 in Table 11.
   - ρ = 0 adapters of both collections exist and are used in the commit-cap control of A.2. Their link to these rows is n.r.
   - The MMLU-Pro and GPQA configurations of Tables 8 and 9 are n.r. Table 9's GPQA row comes from a separate 1.110-call configuration (P).
   - Depth-2 GPQA adapters (ρ = 1 and no cap, B.3) exist. Their link to any row of Tables 8 or 9 is n.r.
10. **Table 10.** The MedQA "Selected" value of 82.0 % equals the development accuracy that the registry holds for the MedQA S_1 adapter (ρ = 3). The MMLU-Pro and GPQA configurations are n.r.
11. **Table 11.**
    - MedQA, depth 2: the ρ = 3 row equals the registry targets of the MedQA S_1 adapter (82.0 % / 0.405 / call gap 0.474). A submission-era development evaluation of that adapter is recorded, but its aggregate is not stored.
    - MedQA, depth 2, ρ = 0: an adapter of the same collection exists, with a training log (129 rows, lr 1e-5, 10 epochs). Its link to the 85.0 % row is n.r.
    - MedQA depth 1 and MMLU-Pro depth 1: for each benchmark a 400-root depth-1 collection exists (R: 302 commit and 59 rescue roots for MedQA; 213 and 73 for MMLU-Pro), with ρ = 1 and ρ = 2 adapters. The MedQA adapters have label files (177 / 236 rows) but no training logs. The MMLU-Pro adapters have training logs (219 / 292 rows, lr 1e-5, 10 epochs; see the seven sibling runs in B.3). Which runs produced Table 11's depth-1 rows is n.r.
    - AQuA: ρ = 0 and all-commit adapters of the S_1 collection exist, without training logs. Their link to these rows is n.r.
12. **Table 14 A.** For the MedQA checkpoint of Table 1, "the selected seed is the one closest to the replicated configuration's mean" (P). Whether that seed is one of these three is n.r.
13. **RSI harness**, recorded in each evaluation:
    - greedy decoding, ≤ 3 tool calls, ≤ 1,024 new tokens per turn, stage seed 42;
    - environment tool binding;
    - base-model advisors read from a shared cache; the run stops on any advisor failure;
    - gate: no malformed call and all answers valid, or a valid-answer rate ≥ 0.99 with a recorded acknowledgement.
14. **Cap check.** The recorded label counts agree with these caps. Example: MedQA, round 2, dynamic arm has 89 rescue and 267 = 3 × 89 commit decisions.
15. **RSI final checkpoints.**
    - MedQA: S_3 in all three arms (0 of 7 GRPO updates accepted). MMLU-Pro, GPQA and AQuA: G_3 in all three arms.
    - MedQA, GPQA and AQuA ran the three arms jointly, sharing round 1, and the dynamic and success arms shared a round-2 collection. MMLU-Pro ran one run per arm.
    - G_1 was rejected (MedQA, MMLU-Pro), accepted at step 0 and therefore equal to S_1 (GPQA), or accepted at step 32 of 40 (AQuA).
    - SFT in rounds ≥ 2 (C): 3 epochs from the previous checkpoint, on that round's rows only, lr 1e-5, no loss on draft tokens.
16. **Table 13 B.**
    - Round-1 pairs (GPQA, AQuA), on `collect_r2`: S_1-collected labels of the SFT-only arm against G_1-collected labels of an earlier SFT + GRPO configuration in which draft tokens were also supervised.
    - For the round-1 pairs the S_1 side is the same label set as Table 13 A's "R2 labels, from S_1" row (the dynamic SFT-only arm's round-2 labels), trained from S_1 a second time with the same recipe and seed. The two trainings give GPQA 54.0 (Panel B) against 53.0 (Panel A) and AQuA 72.4 against 72.8. The submission's statement that Panel B's S_k labels are separate from Panel A's collections therefore holds only for the round-2 pairs.
    - Round-2 pairs (MMLU-Pro, AQuA), on `collect_r3`: S_2 against G_2 of that earlier configuration.
    - These collecting checkpoints are not those of Table 12's SFT + GRPO rows.
17. **Fig. 7.** The MMLU-Pro and AQuA dynamic totals were copied from an earlier results page; their per-question files were not archived.

### A.2 Rebuttal results

| Result | Policy kind | Collection depth | Commit cap ρ | Training seed | Checkpoint | Data pool | Decoding / call limit |
|---|---|---|---|---|---|---|---|
| Equal-information grid: Table R1, Figure R1, development-pool notes | single-pass S_1 and three re-trainings of it | 2 / 2 / 1 / 3 (R) | 3 / 2 / 2 / 1 (R) | S_1: n.r. / n.r. / 42 (R) / n.r.; re-trainings: 1, 2, 3 (C) [18] | S_1 adapter; re-trainings of S_1's label file [18]; adapter digests recorded (R) | `test` 500 / 500 / 100 / 254 and `dev` 200 / 200 / 100 / 254 (R) | grid [19]: ≤ 1 call |
| Deployed evaluation: Table R3 | S_1 = RSI S_1 (its Table 12 evaluation); seeds 1–3 = the re-trainings | as above | as above | as above | as above (R) | `test` (R) | RSI harness [13], free routing, ≤ 3 calls (R); evaluation gate reported, not enforced (C) |
| Attribution against the untrained manager: Table R2, rows 1 and 3–7 | control: base model as manager, against S_1 and seeds 1–3 | – | – | – | base model, no adapter (R) | `test`; deployable-gate thresholds from both managers' `dev` grids (R) | grid [19]; base model held to the same output grammar [20] |
| Table R2, rows 2, 8, 9 | references | – | – | – | letter-only runs of the base model and of S_1; base system and RSI S_1 of Table 12 (R) | locked `test` (R) | as in the Table 12 rows (R) |
| Commit-only control | control: trained without any call row | S_1's round-1 records (R) | no cap, root-commit rows only: 304 / 213 / 86 / 217 rows (R, the commit records of the round-1 collections) [21] | 42 (C) | S_1 recipe [18]; max length 4,096 (GPQA 16,384) [21]; digest recorded (R) | `test` and `dev` (R) | grid [19]; makes no call on any test question (R) |
| Success-distillation control (matched data) | control: one SFT, not RSI | S_1's round-1 records (R) | 3 / 2 / 2 / 1 with S_1's cap seeds; target = one successful option drawn uniformly per kept record (C) [22] | 42 (C) | S_1 recipe [18]; digest recorded (R) | `test` (R) | grid [19] and RSI harness [13] |
| No-cap control | control: the submission's all-commit sibling of S_1 | S_1's collection | none: every commit row kept (433 / 382 / 160 rows, R; AQuA n.r.) | n.r. / n.r. / 42 (R) / n.r. | lr 1e-5 and 10 epochs logged for MedQA, MMLU-Pro and GPQA (R); no AQuA log; GPQA adapter fetched without a pinned revision; digests recorded (R) | `test` (R) | grid [19] |
| Commit-cap control | control: the submission's sibling adapters of S_1 | S_1's collection | MedQA 0, 1, 2, 3\*, none; MMLU-Pro 1, 2\*, none; GPQA 1, 2\*, none; AQuA 0, 0.5, 1\*, none (\* = S_1) [23] | GPQA ρ 1, 2\*, none: 42 (R); MedQA, MMLU-Pro, AQuA n.r. | one sibling adapter per cap [23]; digests recorded (R) | `test` (R) | grid [19] |
| Original fixed-role baselines (Table R1 notes) | baseline: the submission's protocol, run on S_1 | – | – | – | S_1 adapter (R) | `test` (R) | forced path: no draft before the call, Verifier gets no candidate; one role, or Extractor → Reasoner → Verifier (3 calls); greedy, ≤ 1,024 tokens, seed 42 (R) |

Notes to A.2:

18. **Re-trainings.** The launch script trains S_1's label file with:
    - seed K;
    - LoRA r 16, α 32 (code default);
    - lr 1e-5, 10 epochs, batch 1 × 8;
    - max length 4,096 (set by the launch script; 16,384 on a retry when a row is longer). No round-1 label row is longer, so nothing is truncated.

    The AQuA recipe is assumed from the other three benchmarks, because no AQuA S_1 log exists. The re-trainings use the fast linear-attention kernels; the kernels of the original runs are n.r. The training log stayed on the rebuttal machine.
19. **Grid protocol**, recorded in each grid's identity:
    - The manager's first turn is greedy decoding restricted to the legal actions (commit with a draft, or one call). For S_1 it equals unconstrained greedy decoding on 500/500, 499/500, 100/100 and 254/254 questions (R).
    - Every rule then starts from that candidate. It forces at most one advisor call, followed by a forced-commit greedy revision. The Verifier receives the candidate unless the rule removes it.
    - "Learned calling" is the policy's own first call, then commit.
    - Collection seed 42 (R).
    - Confidence-gate thresholds are fixed on the same manager's `dev` grid.
    - Random subsets use 2,000 draws. Paired tests use 10,000 bootstrap resamples and exact McNemar tests. Analysis seed 0 (C).
20. **Base-model grammar check.** The base model's first turn equals unconstrained greedy decoding on only 0/500, 2/500, 0/100 and 27/254 questions (R). Its grid therefore describes a grammar-forced system.
21. **Commit-only labels.** The row counts equal the commit records of the round-1 collections. One GPQA row exceeds 4,096 tokens, so the GPQA run is repeated at 16,384 (launch-script rule; the log is not in the repository).
22. **Success-distillation labels.** Before these labels were built, the shortest-success selection of the same records was checked to reproduce S_1's label file row for row (R, check on the rebuttal machine).
23. **Commit-cap siblings.**
    - Label files or training records agree with the cap rule for 11 of the 15 managers.
    - The S_1 recipe (lr 1e-5, 10 epochs) is recorded for 9 of them: MedQA ρ 0, 1, 3 and none; MMLU-Pro 2 and none; GPQA 1, 2 and none.
    - MMLU-Pro ρ 1 and AQuA ρ 0, 0.5 and none are known from the adapter names only. The GPQA ρ = 1 adapter was fetched without a pinned revision; its recorded digest equals the final checkpoint of a stored training run (111 rows, lr 1e-5, 10 epochs, batch 1 × 8, seed 42).
    - Whether adapters for other caps exist (for example ρ = 0 for MMLU-Pro or GPQA) is n.r.
    - These are not the depth-1 collections of Table 11's MedQA and MMLU-Pro depth-1 pairs.

### How to tell the configurations apart

- **Single-pass S_1 (submission): Tables 1, 7, 14 A.**
  - One SFT per benchmark, on labels selected from one collection.
  - Evaluated by the submission-era loop [6] on the submission's pools.
  - The MedQA checkpoint is not archived.
- **RSI S_1.**
  - The same registered S_1 adapters (for MedQA, the same-configuration substitute), re-evaluated by the RSI harness [13].
  - It uses the RSI pools: MedQA's first 500 official test rows, and a new MMLU-Pro `test` that is disjoint from all other pools.
  - It is the round-1 starting point of every RSI arm, not a separately trained model.
  - For these reasons its numbers differ from Table 1.
- **RSI round-k arms.**
  - From round 2 on, each round trains from the previous round's checkpoint. SFT-only arms start from S_1; SFT + GRPO arms start from the resolved G_1. On MedQA, GPQA and AQuA the three arms share it; on MMLU-Pro each arm has its own. Where G_1 was rejected (MedQA, MMLU-Pro), this is S_1 [15].
  - The **Success** arm re-collects with its own model every round. It trains each kept record on one uniformly drawn successful option instead of the shortest success; commit counts as an option when the root draft is correct. Gated FA-GRPO follows.
  - The rebuttal's success-distillation control applies the same target rule only once, to S_1's round-1 records, with S_1's recipe.
- **Replay (Table 10).**
  - A post-hoc analysis of development evaluations at a fixed role and number of calls. It trains nothing, and its configurations are n.r.
  - The repository's replay of RSI checkpoints is a different analysis.
- **Equal-information grid (rebuttal).**
  - One candidate per question, at most one call per rule, forced commit [19]. This is not the deployed protocol.
  - The deployed evaluation (Table R3) routes freely, with up to 3 calls.
  - "Always Verifier" in the grid passes the candidate to the Verifier. The submission's always-Verifier (Table 7) and the original fixed-role baselines do not.

### Data splits

| Benchmark | Source (identity) | Pool | Size | Definition |
|---|---|---|---|---|
| MedQA (US, 4 options) | normalized cache, 12,723 rows = official train 10,178 / validation 1,272 / test 1,273 (SHA-256 d5f81875…) | `collect_r1` | 400 | the submission's roots: the first 400 of `Random(42).shuffle(first 1,200 official train rows)` |
| | | `dev` | 200 | first 200 official validation rows (example ids 10178–10377); identical to the 200-question development file (SHA-256 498db4e0…) |
| | | `test` | 500 | first 500 of the 1,273 official test rows (ids 11450–11949) |
| | | `collect_r2`, `collect_r3` | 400 each | train rows after the first 1,200; never advisor-SFT questions |
| | | `grpo_r1`–`grpo_r3` | 256 each | from the other 800 of the first 1,200 train rows |
| MMLU-Pro | cache of 12,032 rows, all from the official test split (SHA-256 273747ca…); every pool is a slice of `Random(42).shuffle(rows)` | `collect_r1` | 400 (399 distinct stems) | the submission's roots, within shuffled positions 300–1,499 |
| | | `dev` | 200 | positions 0–199 (reconstruction) |
| | | `test_paper` | 500 | positions 0–499: the submission's test set as reconstructed (flagged) |
| | | `test_paper_clean` | 104 | the `test_paper` questions that are in no other pool and in no advisor-SFT file |
| | | `test` | 500 | first 500 eligible positions from 1,500 on (in fact 1,500–2,253): not advisor-SFT and no stem seen in an earlier pool |
| | | `collect_r2`, `collect_r3` | 400 each | positions from 2,000 on; never advisor-SFT |
| | | `grpo_r1`–`grpo_r3` | 256 each | positions 300–1,499 minus the roots (799 candidates) |
| GPQA | the 198 Diamond questions within the 546 extended questions, shuffled with seed 42: the first 100 form Diamond-100 (SHA-256 c2463316…); the other 98 Diamond plus 348 non-Diamond questions form the 446-row training file (SHA-256 cf1044b8…); the two files share no stem | `collect_r1` = `collect_r2` = `collect_r3` | 200 | the submission's roots in the first 406 rows of the training file (47 Diamond, 153 non-Diamond); reused every round |
| | | `dev` | 100 | new for the RSI work: all 51 leftover-Diamond rows that are not roots, plus 49 seeded others, with no advisor-SFT question. 87 rows come from the first 406 rows and 13 from the submission's 40-row development slice (rows 407–446) |
| | | `test` | 100 | Diamond-100, the whole file |
| | | `grpo_r1` = `grpo_r2` = `grpo_r3` | 146 | the rows left after `collect_r1` and `dev`; reused every round |
| AQuA-RAT | cache of 97,975 rows rebuilt from the official parquet files at a pinned dataset revision: train 97,467 / validation 254 / test 254 (SHA-256 f6bc0280…) | `collect_r1` | 400 (398 distinct stems) | the submission's roots: the first 400 of `Random(42).shuffle(first 1,200 official train rows)` |
| | | `dev` | 254 | the whole official validation split (ids 97467–97720) |
| | | `test` | 254 | the whole official test split (ids 97721–97974) |
| | | `collect_r2`, `collect_r3` | 400 each | train rows after the first 1,200; never advisor-SFT |
| | | `grpo_r1`–`grpo_r3` | 256 each | from the rest of the first 1,200 train rows (795 candidates) |

Rounds 4–5 (MedQA, MMLU-Pro, AQuA) add `collect_r4` and `collect_r5` (400 each) and `grpo_r4` and `grpo_r5` (256 each). They share no stem with any other pool.

- **Identity.**
  - A question is identified by the first 16 hex characters of the SHA-1 of its lower-cased, whitespace-collapsed text; the options are not hashed.
  - The split builder (seed 42) refuses any stem shared between pools, except the declared GPQA reuse and the flagged MMLU-Pro `test_paper`.
  - Root sampling reproduces the submission's roots on all four benchmarks (R).
  - Only the cache SHA-256 values are pinned. The raw dataset revisions behind the MedQA, MMLU-Pro and GPQA caches are n.r.
- **Disjoint.**
  - On every benchmark, `dev` and `test` share no stem with each other or with any training pool (R).
  - Advisor-SFT rows (600 / 600 / 160 / 600; 600 / 599 / 160 / 595 distinct stems) appear in no `dev` or `test` pool, except 3 same-stem matches in MMLU-Pro `dev`.
  - Advisor-SFT questions make up 248 / 87 / 160 / 400 of the `collect_r1` roots. The advisors were served as the base model, so this overlap did not act through trained advisor weights.
- **Known overlap, MMLU-Pro.** The reconstructed `test_paper` (the pool of Table 1) contains all 200 `dev` questions, 68 `collect_r1` roots and 47 / 50 / 31 questions of `grpo_r1`–`grpo_r3`. Four of these (three in `dev`, one in `grpo_r1`) also share a stem with an advisor-SFT question. The other 104 questions are clean. The RSI and rebuttal runs therefore use the new `test` pool.
- **Known overlap, GPQA.**
  - Diamond-100 is the evaluation pool of Table 1 and also the development pool of the submission's GPQA diagnostics (Table 4, †, and the GPQA rows of Tables 7–10).
  - It shares no question with the roots, with the advisor-SFT set or with any of the 446 training rows.
  - Whether ρ, the number of epochs or the checkpoint was chosen on Diamond-100 or on the 40-row slice is n.r.
  - RSI and rebuttal development use the new `dev` pool.
- **AQuA-RAT.**
  - `dev` and `test` are the two official 254-row files. They share no stem with each other, with any training pool or with the advisor-SFT set.
  - In the full official data, train shares 10 stems with validation and 12 with test; they occur in 18 and 18 train rows. None of these 36 train rows is in a training pool used here or in the advisor-SFT set.
- **MedQA.** The 500 test questions behind Table 1 are n.r. The RSI and rebuttal `test` pool takes the first 500 official test rows.

## B. Manager (S_1): data collection, labels and SFT

S_1 is one LoRA manager per benchmark. It is trained once, by SFT, on labels selected from one counterfactual ("marginal-value") collection. Every RSI run imports the same adapter unchanged as its round-1 manager. Three kinds of S_1 number are kept apart: **single-pass S_1** (the paper's own runs), **RSI S_1** (this adapter re-evaluated by the RSI harness) and the **replay** analyses. The SFT of later RSI rounds uses different settings and is described in C.1.

**Status key.**
- **Run**: stored by the run itself (collection report, label file, adapter configuration, training log, or the trainer's saved arguments and state).
- **Reproduced**: not stored, but re-deriving the recorded outputs with this value gives them exactly.
- **Code**: a default of the code or a documented intention; the run did not store it.
- **Not recorded**: no source holds it.

### B.1 Marginal-value collection

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT | Status |
|---|---|---|---|---|---|
| Root pool | first 1,200 rows of the official train split | positions 300–1,499 of a seed-42 shuffle of all 12,032 questions (MMLU-Pro has no train split) | first 406 of the 446 GPQA-extended questions outside Diamond-100 | first 1,200 rows of the official train split | reproduced |
| Root questions | 400 | 400 | 200 | 400 | run |
| Maximum depth (advisor calls per branch) | 2 | 2 | 1 | 3 | run |
| Commit roots (direct answer correct) | 304 | 213 | 86 | 217 | run |
| Rescued roots, shortest success with 1 / 2 / 3 calls | 57 / 5 / – | 74 / 7 / – | 37 / – / – | 126 / 6 / 0 | run |
| Roots with no label (no correct branch, or no parseable direct answer) | 34 | 106 | 77 | 51 | computed from the counts (GPQA: run) |

Settings shared by all four collections:

| Setting | Value | Status |
|---|---|---|
| Collecting manager | base Qwen3.5-9B, no adapter | run |
| Advisors | Extractor, Reasoner, Verifier; each usable at most once per branch | run (all three registered) |
| What answered as advisor | the base model under each role's system prompt; the server did not apply the trained advisor adapters (E.5) | established by a later audit; the collection command lines are not recorded for MedQA, MMLU-Pro and GPQA |
| Root sampling | first *n* questions of the pool after shuffling it with seed 42 | code; reproduced (the recorded roots) |
| Decoding of the manager's probes | greedy (temperature 0), at most 512 new tokens, thinking disabled | code (the reports do not store it; the documented AQuA command sets temperature 0) |
| Commit cap during collection | none (−1); the cap is applied afterwards (B.2) | run |
| Tie-break seed | 42 | reproduced (every stored label) |

Procedure (the collection code):
1. **Direct answer.** The manager receives its SFT prompt (B.4) and a probe message that asks for `DRAFT_ANSWER_<key>` and `ANSWER_<key>` without a tool. This answer is the shared draft. A root whose answer cannot be parsed is not expanded (GPQA: 6 of 200).
2. **Forced branches.** The search is breadth-first: from every node, every advisor not yet used on that branch is forced. The forced manager turn is `DRAFT_ANSWER_<current draft>` plus the tool call. The Verifier receives the current draft. After the advisor's output, a revision probe asks for a revised draft and answer. If the revision cannot be parsed, the previous draft is kept and the branch counts as wrong.
3. **Stopping.** A root answered correctly is expanded to depth 1 only, for diagnostics. Otherwise expansion stops at the first depth that contains a correct branch, or at the maximum depth.
4. **Target.** The target is "commit" if the direct answer is correct. Otherwise it is the shortest advisor sequence whose revised answer is correct; ties among equally short sequences are broken with `Random(42 + example_id)`. A root with no correct branch yields no row. Ground truth only selects a trajectory and never replaces a model answer. The probe messages are not part of the SFT rows.

**GPQA.** Two collections exist on the same 200 roots. S_1's labels come from the depth-1 collection above. The GPQA rows of the paper's oracle-by-depth table and per-advisor marginal-value table match a separate depth-2 collection (86 commit roots; 44 one-call and 9 two-call rescues). The two runs record different one-call outcomes on the same roots (37 against 44 one-call rescues). The cause is not recorded.

### B.2 Labels: commit cap and SFT rows

Rule:
- Keep every rescued record.
- Shuffle the commit records with `Random(s)` and keep the first ⌊ρ × rescued⌋.
- Shuffle the kept records with `Random(s + 1)`; this sets the row order.
- A commit record gives one row. A rescue with *k* calls gives *k* call rows and one commit-after-call row.

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT | Status |
|---|---|---|---|---|---|
| Commit cap ρ of S_1 | 3.0 | 2.0 | 2.0 | 1.0 | run (label-file name); reproduced |
| Commit rows kept / commit records | 186 / 304 | 162 / 213 | 74 / 86 | 132 / 217 | computed; GPQA counted in the file |
| Rescue rows (call + commit-after-call) | 129 | 169 | 74 | 270 | computed; GPQA counted (37 + 37) |
| SFT rows | 315 | 331 | 148 | 402 | run |
| Cap seed *s* | 0 | 0 | 42 | 0 | reproduced (row for row, same order) |

- The cap was applied after collection, by a script that is not recorded. The collection reports record no cap.
- AQuA: the same collection has a second 402-row file, with the same decision counts and 52 different commit rows. Which of the two files the AQuA S_1 was trained on is not established.

What a row looks like (chat messages, rendered with the Qwen3.5 chat template):

| Row type | Prompt | Response (supervised) |
|---|---|---|
| commit | system, user | `DRAFT_ANSWER_X` newline `ANSWER_X`; X is the correct direct answer |
| call | system, user, earlier call and tool turns | `DRAFT_ANSWER_Y` plus one native tool call. The call is `extractor_tool` or `reasoner_tool` without arguments, or `verifier_tool` with `current_draft = Y`. Y is the current draft; before the first call it is the wrong direct answer |
| commit-after-call | the rows above plus the call and the advisor's raw output as the tool message | `DRAFT_ANSWER_Z` newline `ANSWER_Z`; Z is the correct revised answer |

The loss is the causal-LM loss on every response token plus the end token. This includes the draft letter of the call rows. All prompt tokens are masked (code). Later RSI rounds remove the draft from the loss.

### B.3 Manager SFT

| Setting | Value | GPQA | MedQA, MMLU-Pro | AQuA-RAT |
|---|---|---|---|---|
| Base model | Qwen3.5-9B; the revision is not recorded | run | run | run |
| Initialisation | a new LoRA on the base model | run (log) | run (log) | not recorded |
| LoRA | r 16, alpha 32, dropout 0.05, no bias, no DoRA or rsLoRA; q, k, v, o, gate, up and down projections; PEFT 0.20.0 | run | run | run |
| Learning rate | 1e-5 | run | run (log) | not recorded |
| Schedule | linear decay to 0, no warm-up | run | code (library default; library version not recorded) | code (library default; library version not recorded) |
| Optimizer | AdamW (fused), β 0.9 / 0.999, ε 1e-8, weight decay 0, gradient-norm clip 1.0 | run | code (library default; library version not recorded) | code (library default; library version not recorded) |
| Epochs | 10 | run | run (log) | not recorded |
| Batch | 1 per device × gradient accumulation 8, one GPU | run | the log implies an effective batch of 8; the 1 × 8 split is code | not recorded |
| Training rows | 315 / 331 / 148 / 402 | run (148) | run (log) | label file only |
| Optimizer steps | GPQA: 190 (19 per epoch; the last step of each epoch accumulates 4 rows) | run | not recorded (the log prints only a rounded-down estimate, 390 / 410) | not recorded |
| Data order | random sampler, seed 42 | run | not recorded | not recorded |
| Training seed | 42 | run | not recorded | not recorded |
| Maximum sequence length | 4096 | code | code | code |
| Precision | bf16 mixed precision; the model is loaded in bf16 | run (mixed precision); code (loading in bf16) | code | code |
| Gradient checkpointing | on (non-reentrant) | run | not recorded | not recorded |
| Evaluation during training | none: no evaluation set, no early stopping | run | code | code |
| Saved checkpoints | one per epoch, plus the final adapter at the top level | run | code | code |

Notes on the table:
- **Schedule and optimizer.** The training code passes no optimizer, scheduler, warm-up, weight-decay or clipping setting, so both rows are library defaults. The GPQA run records what those defaults were. The library version of the MedQA, MMLU-Pro and AQuA-RAT runs is not recorded, so these values are the library defaults, not stored values.
- **Maximum length.** It is not a trainer argument and is not logged. The code default is 4096, and the trainer of that period truncated longer rows from the right. A repository test asserts that every S_1 label row fits in 4096 tokens, so this default would truncate nothing.
- **Seed.** The code default is 42 when no seed is passed. The documentation labels the registered MedQA S_1 "seed 0", and no training record says which seed that refers to.
- **Gradient checkpointing** affects memory only. The trainer hard-coded it only from 10 August 2026, and three MMLU-Pro depth-1 runs from that day record it off.
- **Seven sibling runs.** Seven further manager-SFT runs from the same period store their trainer arguments:
  - GPQA depth 1: ρ = 1 and no cap (111 / 160 rows);
  - GPQA depth 2: ρ = 1 and no cap (168 / 201 rows);
  - MMLU-Pro depth 1: ρ = 1, 2 and no cap (219 / 292 / 359 rows).

  All seven record lr 1e-5, 10 epochs, 1 × 8 on one GPU, seed 42, the optimizer and schedule above, and no evaluation. In each, the final adapter equals the epoch-10 checkpoint. This is consistent with the recipe of the MedQA, MMLU-Pro and AQuA S_1 runs, but it is not a record of it. The depth-1 MMLU-Pro ρ = 2 adapter is not S_1: S_1 is the depth-2 adapter, trained on 331 rows.

### B.4 Prompt and tool interface

The system prompt names the task and describes the three tools. It allows 0–3 calls, each tool at most once, and requires a `DRAFT_ANSWER_<key>` line before every call-or-stop decision and a final `ANSWER_<key>` line; think tags are not allowed. The user message holds the example ID, the question, the lettered choices and a note that the example is already bound to the tools.

The tools are three function schemas in Qwen3.5's native tool-calling format: `extractor_tool` and `reasoner_tool` without arguments, and `verifier_tool` with `current_draft`. Thinking is disabled. SFT and the paper-era collection use short tool descriptions; evaluation uses longer ones. The full prompts are given in D.1–D.2.

### B.5 Checkpoint selection

- **Within a training run there is no selection.** S_1 is the adapter the code saves after the last training epoch. No choice is made among epochs.
  - GPQA: run. The run trained for 10 epochs, the published adapter is byte-identical to the epoch-10 checkpoint, and the log has no evaluation entry.
  - MedQA, MMLU-Pro: 10 epochs (run, training log). That the published adapter is the end-of-training save is code only, because only the final adapter was published.
  - AQuA-RAT: end-of-training save (code). The number of epochs is not recorded: there is no training log or trainer state, and the code default is 1.
- **How ρ and the collection depth were chosen is not recorded.**
  - Adapters for other caps from the same collections exist: MedQA ρ = 0, 1, 2 and no cap; MMLU-Pro ρ = 1 and no cap; GPQA ρ = 1 and no cap; AQuA ρ = 0, 0.5 and no cap. MedQA also has adapters from a depth-3 collection (ρ = 1, 2, 3). Adapters from depth-1 collections (MedQA, MMLU-Pro) and a depth-2 collection (GPQA) also exist.
  - The pool and the criterion used to compare them are not recorded.
  - The run sheets state the intention: choose the ratio on the development split only, never on test.
  - For GPQA, the paper's development diagnostics were measured on Diamond-100, which is the evaluation pool.
- **MedQA substitute.** The paper's locked MedQA manager is not available. The registered MedQA S_1 is the ρ = 3 adapter of the 400-root depth-2 collection, trained on the 315-row label file above; the documentation describes it as "seed 0", trained with the same configuration. The paper's MedQA development number is a mean over three training seeds; their checkpoints and seed values are not recorded. Every RSI and rebuttal "S_1" for MedQA is this substitute.

## C. RSI rounds, FA-GRPO and the "Success" comparator

**How each value is sourced (Basis column).**
- **run**: the run wrote it itself, in its stage decision files, run reports, signed run configuration or evaluation signatures.
- **config**: it comes from the configuration file the run was started from. For the v3+GRPO runs we do not hold the run's own signed copy of the configuration. Their launch script starts them from the v3 file with only the GPU assignment changed, and names the arms on the command line: Dynamic, Static and Success, one arm per MMLU-Pro rerun. These replace the file's two SFT-only arms, so FA-GRPO runs every round. The runs record their arms in their reports.
- **code**: it is a default of the training code that no configuration overrides. The FA-GRPO, label-selection, collection and gate-metric code has not changed since before the first RSI run. The handling of acknowledged validity failures in the round controller (C.4) was added during the v1 runs. It was in place, unchanged, for every v3 and v3+GRPO run.
- **split manifest**: the frozen pool file (example ids and question hashes) the run reads its pools from. The run signature stores its SHA-256, so this counts as run where the signature is held (the v3 runs of MedQA, MMLU-Pro and AQuA-RAT, and their continuation). For v3+GRPO and GPQA v3 it counts as config.
- **registry**: the frozen per-benchmark registry in the code (pinned inputs, answer keys); no configuration overrides it, so this counts as code.
- **not recorded**: no record holds the value.

Every RSI cell is one run with one seed. In the RSI tables, "S_1" is the registered single-pass manager re-evaluated by the RSI harness. It is not the single-pass number reported in the paper.

### C.1 Run families and round structure

| Rows in the tables | Family | Arms | Draft letter in the SFT loss | FA-GRPO | Rounds | Benchmarks |
|---|---|---|---|---|---|---|
| "SFT + GRPO" | v3+GRPO | Dynamic, Static, Success | not trained | every round | 1–3 | all four (MMLU-Pro: three one-arm runs) |
| "SFT only", round 3 | v3 | Dynamic, Static | not trained | none | 1–3 | all four |
| "SFT only", round 5 | v3, continued | Dynamic, Static | not trained | none | 4–5, starting from each arm's round-3 checkpoint | MedQA, MMLU-Pro, AQuA-RAT |

- **Earlier families.** None of them feeds a main-table row. v1 trained every token, and its main runs ran FA-GRPO every round. It feeds only two appendix analyses (draft letters, and labels collected by v1 checkpoints). v2 trained drafts on commit rows only and feeds nothing.
- **MMLU-Pro under v3+GRPO.** The first MMLU-Pro run, with all three arms, stopped during its locked test when the compute credit ran out. The reported MMLU-Pro rows come from three one-arm runs trained again. Each MMLU-Pro arm therefore has its own round-1 FA-GRPO stage, and the Dynamic and Success arms each have their own round-2 collection (the Static arm collects in no round). Whether anyone looked at a locked-test output of the stopped run is not recorded.

| Round | Collection | Labels | Manager SFT | FA-GRPO | Development decision |
|---|---|---|---|---|---|
| 1 | none (S_1 and the registered round-1 label file are imported; for AQuA-RAT see C.3) | none | none | from S_1 on GRPO pool 1; one stage shared by all arms of a joint run | S_1: parity check. G_1: acceptance rule (C.4) |
| 2–3 | Dynamic and Success: new trees collected by the arm's own G_(k−1). Static: none | by arm (C.3) | from G_(k−1), on round-k rows only | from S_k on GRPO pool k | S_k: flags only, never selected. G_k: acceptance rule, otherwise G_k := S_k |
| 4–5 (SFT-only continuation) | Dynamic: collected by S_(k−1) | as in rounds 2–3 | from S_(k−1) | none | flags only |

- **Shared collection.** In a joint run, Dynamic and Success share the round-2 collection because both start from the shared G_1. From round 3 on, each arm collects with its own manager. SFT-only arms start round 2 from S_1. (code; run)
- **Collection procedure.** (config; code)
  - The root is the manager's own greedy deployment turn.
  - Unused advisors are forced breadth-first, up to depth 2.
  - A correct root is expanded to depth 1 only. A wrong root stops at the first depth that has a success.
  - The Verifier receives the current draft, and revisions are greedy.
  - Tie-break seed 42; batch size 1.
- **Advisors** in collection, FA-GRPO and evaluation: the base model with role prompts, greedy (temperature 0), at most 1,024 new tokens, outputs cached. (config; code)

| Per benchmark | MedQA | MMLU-Pro | GPQA | AQuA-RAT | Basis |
|---|---|---|---|---|---|
| Collection pool, each round k ≥ 2 | 400 new training questions | 400 new | the same 200 round-1 roots | 400 new | run; split manifest |
| GRPO pool, each of rounds 1–3 | 256 per round, disjoint across rounds, drawn from the 800 paper-pool training questions that are not roots | 256 of 799 | one pool of 146, reused | 256 of 795 | split manifest |
| Development pool (gates) | 200 | 200 | 100 | 254 | split manifest |
| Answer keys per question | 4 | the row's own, 3–10 | 4 | 5 | registry |
| Commit cap ρ; cap/draw seed | 3.0; 0 | 2.0; 0 | 2.0; 42 | 1.0; 0 | code (registry defaults; the configuration sets both to null) |
| Static label file | 315 rows | 331 | 148 | 402 | run |
| SFT and anchor maximum length (tokens) | 4,096 | 4,096 | 16,384 | 4,096 | config |

**Per-round manager SFT (all benchmarks).**
- Start: the LoRA of the initial checkpoint is continued with a new AdamW optimizer (trainer defaults).
- LoRA: r 16, α 32, dropout 0.05, on the q/k/v/o/gate/up/down projections. These are inherited from S_1, and the trainer checks that rank, α and target modules are unchanged.
- Schedule: 3 epochs; learning rate 1e-5 with linear decay to 0 and no warm-up; batch 1 × gradient accumulation 8; seed 42; bf16.
- Loss: only on the routing suffix (the tool call, or the ANSWER_ key). The prompt and the DRAFT_ANSWER_ letter are masked.
- Checkpoint: the end of epoch 3. No selection is made on any pool.

Basis: run for the v3 runs of MedQA, MMLU-Pro and AQuA-RAT (their signed configuration); config for the v3+GRPO runs and for GPQA v3; code for the optimizer and the checkpoint.

### C.2 FA-GRPO (v3+GRPO)

All benchmarks use the same settings, except the anchor length and the GPQA pool.

| Item | Setting | Basis |
|---|---|---|
| What is trained | The answer revision after one forced advisor call. The root draft and the commit/call decision get only KL terms, no reward gradient | code |
| Reward | r(y) = 1 if the revised answer key y is the gold key, otherwise 0. There is no call cost, no per-advisor or per-call penalty, and no length or format term (only legal keys are scored). The code gives the reason: the preference for fewer calls comes from the shortest-path labels, and under this reward always calling is reward-optimal, so routing is not trained by reward | code |
| Group | No sampled rollouts. The estimator is "exact", the infinite-group limit of GRPO: for each question and each of the 3 advisors, the group is the full softmax over the question's answer keys after the call. This gives 4 questions × 3 advisors = 12 revision states per step. The sampled estimator (16 generations) is not used | config; code |
| Revision state | The root draft X is the greedy key of the current policy. The call is forced and the advisor output comes from the cache (the Verifier receives X). Then the DRAFT_ANSWER_ prefix is forced | code |
| Advantage and surrogate | J = π(gold); A(y) = (r(y) − J) / (√(J(1−J)) + 10⁻⁴); L_pg = −Σ_y sg[π(y)·A(y)]·log π(y) | code |
| Loss of one step | Mean over the 4 questions of 0.1·KL_root + 0.1·KL_dec + the mean over the 3 revision states of (L_pg + 0.02·KL_rev). Added to this: 0.03 × the token-mean route-only cross-entropy on 4 anchor rows. Each KL is exact, KL(π_θ ‖ π_ref) over the finite support | config; code |
| KL reference | A frozen copy of the round's SFT checkpoint S_k (round 1: S_1), never the base model | code |
| Anchor rows | The arm's own label file for the round (round 1: the registered round-1 file; for AQuA-RAT it is not established that S_1 was trained on it, see C.3), in a seeded order, 4 per step, wrapping around | code |
| Data | GRPO pool k, shuffled once (seed 0) and read 4 questions per step with wrap-around. That is one pass over a 256-question pool in 64 steps; the 146-question GPQA pool is passed about 1.75 times | config; code |
| Steps | At most 64 optimizer steps. The guard can end a stage earlier (steps run: C.4) | config; run |
| Optimizer | AdamW (torch default betas and epsilon), constant learning rate 5e-6 with no schedule, weight decay 0, gradient-norm clip 1.0. The run reports print the learning rate rounded to 0.000. No v3+GRPO run record holds an unrounded value; the signed configuration of the v3 runs (MedQA, MMLU-Pro, AQuA-RAT), started from the same files, records 5e-6 | config; code |
| LoRA | The S_k adapter is continued (r 16, α 32), and only the policy adapter is trainable. Dropout is set to 0. Base model in bf16, LoRA weights in fp32, one GPU | code |
| Lengths | Root, decision and revision prompts are not truncated. Advisor outputs have at most 1,024 new tokens. Anchor rows have at most 4,096 tokens (GPQA 16,384); a longer row stops the stage | config; code |
| Sampling | None: the root draft is greedy and the cached advisors are greedy | code |
| Seed | 0, for the data order, the guard probe, the anchor order and the framework seed | config |
| Guard | Runs every 8 steps and at the last step, on a fixed probe of 64 pool questions. It fails if abs(greedy call rate − S_k's) > 0.10 or if the mean KL_dec > 0.05. The first failure ends the stage, and the candidate is the last guarded step that passed (step 0 = S_k) | config; code |
| Informativeness | The share of all revision states over the steps run, including steps after the rollback point, whose J lies strictly inside (0.02, 0.98). A stage is informative if this share is at least 0.15 | config; code |
| Not used | The decision-level policy gradient (an ablation switch, off). The learning-rate pilot sweep {2e-6, 5e-6, 1e-5} written in the configuration: the v3+GRPO runs were started in the main phase, which uses 5e-6, and whether a pilot was ever run is not recorded | config; code |

### C.3 Label arms and the "Success" comparator

**Dynamic.** In each round k ≥ 2, the arm's own manager collects new counterfactual trees (C.1). A root's target is "commit" if its draft is correct. Otherwise it is the shortest advisor sequence whose revised answer is correct, with ties broken at random using seed 42 + question index. A root with no success gets no target. At most int(ρ × number of rescue records) commit records are kept: if there are more, a random subset of that size is kept (cap seed); otherwise all are kept. All rescue records are kept. In the v3+GRPO collections the cap removed commit records only on MedQA (round 2: 282 to 267; round 3, the Dynamic arm's own collection: 303 to 213). This is the single-pass labelling rule, applied to the arm's own collections.

**Static.** There is no collection. Every round trains on an identical copy of the registered round-1 label file (the copy is checked by sha256 against the registry). For MedQA, MMLU-Pro and GPQA this is the file S_1 was trained on. For AQuA-RAT, a second 402-row round-1 file exists from the same collection, with the same decision counts and 52 different commit rows. Which of the two S_1 was trained on is not established, and Static uses the registered one. Only the starting checkpoint changes from round to round.

**Success.** Success uses the same collection and keeps the same records as Dynamic: the cap is applied to Dynamic's commit/rescue split. The difference is the target. Each kept record's target is replaced by one option drawn uniformly from all of that record's successful options: "commit" if the root draft is correct, plus every explored advisor sequence whose revised answer is correct. One random stream, seeded with the cap seed, makes all the draws, and each kept record still gives exactly one demonstration. A correct root is expanded only to depth 1, and a wrong root stops at its shallowest successful depth. So Success differs from Dynamic in two places:
- On correct-root records, "commit" is kept with probability 1/(m+1) when m one-call sequences keep the answer correct.
- On wrong-root records, only the random choice among equally short successes differs.

Success therefore removes the shortest-path preference and nothing else. In every arm, the gold answer only decides which explored trajectories count as successes; it is never written into a target.

| | Dynamic | Static | Success |
|---|---|---|---|
| Collection each round | by the arm's own G_(k−1) | none | by the arm's own G_(k−1) |
| Kept records | all rescues; commits capped at int(ρ × rescues) | the round-1 file | the records Dynamic keeps from the same collection |
| Target, correct root | commit | (round-1 file) | uniform over commit and every correct one-call sequence |
| Target, wrong root | shortest success, seeded tie-break | (round-1 file) | uniform over the shortest successes |
| Rows per target | commit: 1 row; call target: one row per call plus one commit-after-call row | same | same |
| SFT start; FA-GRPO | G_(k−1); from S_k | same | same |

Label files of the v3+GRPO runs, as recorded by the runs. Each cell gives total rows, then commit targets / call targets:

| Benchmark | Round | Dynamic | Success | Static |
|---|---|---|---|---|
| MedQA | 2 (shared collection) | 448; 267 / 89 | 634; 81 / 275 | 315 |
| | 3 (own collections) | 358; 213 / 71 | 525; 64 / 228 | 315 |
| MMLU-Pro | 2 (separate runs, identical collection statistics) | 432; 210 / 107 | 578; 64 / 253 | 331 |
| | 3 | 465; 183 / 137 | 602; 63 / 265 | 331 |
| GPQA | 2 (shared) | 210; 78 / 65 | 267; 21 / 122 | 148 |
| | 3 | 213; 79 / 65 | 268; 25 / 119 | 148 |
| AQuA-RAT | 2 (shared) | 547; 172 / 186 | 660; 59 / 299 | 402 |
| | 3 | 564; 149 / 206 | 670; 53 / 307 | 402 |

On a shared collection, both arms keep the same number of records (MedQA 356, MMLU-Pro 317, GPQA 143, AQuA-RAT 358). Success turns most commit targets into call targets.

Two related configurations are not RSI arms:
- The rebuttal's "success distillation" applies the Success rule once to the round-1 records, with the round-1 cap, and trains with the single-pass recipe. It is a single-pass control with no rounds.
- The code arm "Success, SFT only" feeds no reported table.

### C.4 Acceptance of an FA-GRPO update

G_k replaces S_k only if all five conditions below hold. Each model is evaluated once, and there is no statistical test. Otherwise G_k := S_k, and S_k goes on to the next round. The evaluation-gate, accuracy, calls and call-gap conditions are measured on the development pool. The informativeness condition is measured on the FA-GRPO stage's own revision states on the GRPO training pool (C.2), not on the development pool.

| Condition | Rule | Basis |
|---|---|---|
| Evaluation gate, G_k and S_k | No malformed tool call, no advisor error text in a tool output, and every answer parsed; an advisor failure stops the run instead. An S_k whose only failure is unparsed answers (valid-answer rate ≥ 0.99) passes on an acknowledgement. A candidate G_k gets no such allowance | config; code |
| Accuracy | acc(G_k) ≥ acc(S_k) − 0.01 | config |
| Calls | calls per question of G_k ≤ that of S_k + 0.15 | config |
| Call gap | gap(G_k) ≥ gap(S_k) − 0.5 × abs(gap(S_k)), where gap = P(at least one call given a wrong initial draft) − P(at least one call given a correct one) | config; code |
| Informative stage | as defined in C.2 | config |

Recorded decisions of the v3+GRPO runs (run):

| Benchmark | Accepted / proposed | Failed condition (counts may overlap) | Guard-stopped stages | Informative share |
|---|---|---|---|---|
| MedQA | 0 / 7 | informative 7; accuracy 3; candidate evaluation gate 1 | 3 | 0.036–0.141 |
| MMLU-Pro | 6 / 9 | informative 3 (all in round 1); candidate evaluation gate 1 | 4 | 0.094–0.115 (round 1); 0.169–0.305 (rounds 2–3) |
| GPQA | 6 / 7 | accuracy 1 | 1 (round 1, rolled back to step 0, so G_1 has the weights of S_1) | 0.285–0.488 |
| AQuA-RAT | 6 / 7 | accuracy 1 | 3 | 0.295–0.521 |

The calls condition and the call-gap condition failed in no decision. For the 11 guard-stopped stages, which guard criterion stopped each one is not recorded.

The submission says that GRPO updates are retained "only after development selection" (Sec. 4.5, Figure 6) and "only if they pass development selection" (Table 12). It does not name the informativeness condition. Of the 12 rejected updates (30 proposed, 18 accepted), this condition failed in 10. In 5 it was the only failed condition (MedQA: round 2 Static, round 2 Success, round 3 Dynamic; MMLU-Pro: round 1 Dynamic and Static). It alone is enough to reject all seven MedQA candidates and all three MMLU-Pro round-1 candidates.

MedQA, all seven decisions (run):

| Round | Arm | Steps run / candidate step | Informative share | Failed conditions |
|---|---|---|---|---|
| 1 | shared | 32 / 24 | 0.036 | informative; accuracy |
| 2 | Dynamic | 64 / 64 | 0.141 | informative; accuracy |
| 2 | Static | 56 / 48 | 0.122 | informative |
| 2 | Success | 64 / 64 | 0.139 | informative |
| 3 | Dynamic | 64 / 64 | 0.091 | informative |
| 3 | Static | 24 / 16 | 0.080 | informative; candidate evaluation gate (valid-answer rate 0.995) |
| 3 | Success | 64 / 64 | 0.098 | informative; accuracy |

**What the records say about MedQA.**
- The informativeness condition alone rejects all seven updates.
- In every MedQA stage, more than 85% of the revision states had a post-call probability of the gold key of at most 0.02 or at least 0.98.
- In round 3 Static, that round's S_3 had the same valid-answer rate (0.995) and passed on its acknowledgement; the candidate was rejected by rule.
- The records do not say why the MedQA revisions were this saturated. The per-step J values are in step files that are not in our records.
- As a result, every MedQA "SFT + GRPO" checkpoint is an SFT checkpoint.

**Other gates (run).**
- S_k flags (more than 1.5 calls per question, or accuracy down more than 0.03 from G_(k−1)) change nothing. In the v3+GRPO runs one was raised (MMLU-Pro Dynamic, round 2).
- The round-1 parity check compares S_1 with the registry targets (±0.02 accuracy, ±0.05 calls). It failed for MedQA and AQuA-RAT, passed for MMLU-Pro, and does not apply to GPQA.
- Those two failures passed on acknowledgements. The launch script registered them right after each run started, together with the validity acknowledgements for every S_k development evaluation and every locked test, all within about one second (at most 1.03 s). They were registered before any result of that run existed. The parity acknowledgement itself was decided from earlier runs of the same S_1 on the same development pool, whose development evaluations had already failed parity. Its text cites those results ("AQuA dev 0.73 vs the paper's 0.78") and states that the locked test of S_1 had reproduced the paper.

### C.5 Checkpoint selection per condition

- **Fixed by rule.** The tested checkpoints are fixed before the locked test: S_1, and for each arm the resolved checkpoint of the last round. That is G_R, or S_R when FA-GRPO was rejected or the arm has no FA-GRPO. The best development checkpoint is never used. Inside FA-GRPO the candidate is the last guarded step that passed; inside SFT it is the end of epoch 3. (code; run)
- **Locked test.** Each final is evaluated once per run on the locked test: greedy, at most 1,024 new tokens per turn, at most 3 calls, seed 42 (recorded in the evaluation signatures). Unparsed answers count as wrong. A locked test whose only failure is unparsed answers (valid ≥ 0.99) passes on the pre-registered acknowledgement. One reported row used it: SFT only, MedQA Static, round 3, where 1 of 500 answers was unparsed.
- **Earlier use of the locked pools.** The earlier v1, v2 and v3 families also used the locked pools. The draft-supervision change that came before the v3 configuration cites a locked-test statistic of the v1 runs. So the v3 configuration, which v3+GRPO also uses, was fixed after locked-test results of earlier versions had been seen.

| Condition | Round | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|---|---|---|---|---|---|
| S_1 (RSI re-evaluation) | 1 | e8ca757d0e2d | 72f44718b09c | cd889041acb6 | a25318ad635c |
| SFT + GRPO, Static | 3 | S_3 (FA-GRPO rejected), 5142927505dd | G_3, step 64 of 64, bb61e6de3b5f | G_3, 64 of 64, f5114cddbc5b | G_3, 16 of 24, 116fd680813e |
| SFT + GRPO, Dynamic | 3 | S_3, be9be6c808bd | G_3, 40 of 48, 734f87d3e723 | G_3, 64 of 64, 14352137f5e2 | G_3, 64 of 64, acf604aa519b |
| SFT + GRPO, Success | 3 | S_3, 7f59418a69a2 | G_3, 64 of 64, 93863556724e | G_3, 64 of 64, efad2426a3f5 | G_3, 64 of 64, 4cbc0cc88bc4 |
| SFT only, Static | 3 | S_3, 6dfac3d68aa6 | S_3, 63a2fa6a6d7a | S_3, digest not recorded | S_3, 774d6f011723 |
| SFT only, Dynamic | 3 | S_3, 9844e9f3e63a | S_3, d3ab8ef3c847 | S_3, digest not recorded | S_3, 49110ec3bdac |
| SFT only, Static | 5 | S_5, 61edaefdb71c | S_5, 635b5809f3be | not run | S_5, a289c16c85d0 |
| SFT only, Dynamic | 5 | S_5, d4e69f0f968a | S_5, 0187b0605490 | not run | S_5, 14c5b5f26fc5 |

Each cell gives the checkpoint; for G_3, the step selected out of the steps run in the round-3 FA-GRPO stage; and the first 12 hex digits of the adapter's sha256 (run). The MMLU-Pro appendix pool evaluates the same checkpoints. S_1 is evaluated in every v3 and v3+GRPO run, and the main table uses the v3+GRPO evaluation.

Two analyses are not RSI conditions:
- The matched-budget replay analyses development evaluations of S_1 and of the round-3 SFT-only checkpoints. It involves no training.
- The label-quality table trains one SFT from S_1 per label set, with the v3 SFT settings, and evaluates each once on the development pool.

## D. Prompts, decoding, advisors and release

**Status labels.** *Recorded*: stored by the run itself (evaluation signatures, collection records and label files, experiment manifests, training state). *Code*: fixed in the code, a code default or a documented intention, and not stored by the run. *Not recorded*: no source holds it.

**Scope.** The prompts and decoding settings below apply to three things: the single-pass manager S_1 as trained, S_1 and every later checkpoint re-evaluated in the RSI harness, and the replay and rebuttal analyses. "Paper era" means the original single-pass pipeline (collection, SFT and the evaluations reported in the submission). The tables say where the paper era differs or is not recorded.

### D.1 Manager prompt

| Part | Content | Status |
|---|---|---|
| Chat template | The base model's Qwen3.5 chat template, byte-identical to the file of the pinned base-model revision and shipped with every manager checkpoint. Thinking is disabled: the generation prefix and every assistant turn after the question open with an empty `<think>` block. Exception: in the paper-era collection's revision probes (the source of S_1's labels), the forced call turns come before the probe user turn and are rendered without a `<think>` block. | Template: recorded (pinned with each S_1 checkpoint; its digest is stored in every rebuttal-experiment manifest). Thinking off and the `<think>` rule: code (template rule and collection code) |
| System message | Block D.1a. Line 1 is the benchmark's task description (next table). The `DRAFT_ANSWER_` and `ANSWER_` lists have one line per choice key of the question. | Recorded: the round-1 collection records and S_1 label files contain these system messages, and the code reproduces them byte for byte. Every RSI evaluation stores the task description |
| User message | Block D.1a: example id, question, choices, binding instruction. | Recorded (environment binding) in the round-1 records and in every RSI evaluation. Paper-era evaluations: binding not recorded (note below) |
| Context block | Never present: the context of every question of the four benchmarks is empty. | Computed from the data |
| Tool list | The three schemas of D.2, placed by the template before the system text. | See D.2 |

| Benchmark | Line 1 of the system message | Keys listed |
|---|---|---|
| MedQA | You are a manager agent solving a medical multiple-choice question. | A–D |
| MMLU-Pro | You are a manager agent solving multiple-choice questions across diverse academic subjects. Each question has 10 options (A-J). | The question's own keys. The round-1 data has questions with 4, 6, 7, 8, 9 and 10 options, and line 1 is the same for all of them |
| GPQA | You are a manager agent solving expert-level graduate science multiple-choice questions. | A–D |
| AQuA-RAT | You are a manager agent solving five-choice algebraic word problems. | A–E |

<details><summary>D.1a Full first-turn input for one MedQA question (evaluation)</summary>

`{…}` marks a placeholder; everything else is verbatim. The input ends with two newlines after `</think>`, and generation starts there.

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "extractor_tool", "description": "Extract decision-relevant factual signals from the question and context.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "reasoner_tool", "description": "Produce a structured reasoning scaffold for the choices.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "verifier_tool", "description": "Identify relevant domain principles and audit the reasoning for logical or computational errors. Pass your current draft answer via current_draft.", "parameters": {"type": "object", "properties": {"current_draft": {"type": "string", "description": "Your current draft answer key (e.g. \"B\") to audit."}}, "required": []}}}
</tools>

{FIXED TOOL-CALLING INSTRUCTIONS OF THE BASE MODEL'S CHAT TEMPLATE, UNCHANGED}

You are a manager agent solving a medical multiple-choice question.

You have THREE cognitive specialist tools:
  - extractor_tool: extracts key signals and structures relevant facts from the question.
  - reasoner_tool: produces a structured reasoning scaffold (sub-questions, per-choice analysis).
  - verifier_tool: identifies domain principles and audits reasoning for errors. Pass your current draft answer key via the current_draft argument (e.g. current_draft="B") so it audits that specific hypothesis.

Deliberation policy:
  - You may call 0 to 3 tools total. Each tool may be used at most once.
  - Before every call-or-stop decision, state your current best answer on a new line:
  DRAFT_ANSWER_A
  DRAFT_ANSWER_B
  DRAFT_ANSWER_C
  DRAFT_ANSWER_D
  - Then decide: only call another tool if it might change your draft answer.
  - Stop when additional tools are unlikely to improve your answer.
  - Reserve all three tools for genuinely hard cases where each adds new signal.

Output rules:
  - Use the native tool-calling interface. Do NOT write tool calls as text, XML, or JSON.
  - In a turn where you call a tool, output DRAFT_ANSWER_ but NOT the final ANSWER_.
  - If you answer without a tool, output DRAFT_ANSWER_ immediately before the final ANSWER_.
  - When you are ready to submit your final answer (no more tools), end with exactly:
  ANSWER_A
  ANSWER_B
  ANSWER_C
  ANSWER_D
  - Brief reasoning above the ANSWER_ line is allowed; nothing after it.
  - Do not output <think> tags.<|im_end|>
<|im_start|>user
Example ID: {EXAMPLE_ID}

Question:
{QUESTION}

Choices:
  A. {CHOICE_A}
  B. {CHOICE_B}
  C. {CHOICE_C}
  D. {CHOICE_D}

If you call a tool, the current example is already bound — no example_id is needed. For verifier_tool, pass your current draft answer key as current_draft.

If you do not call any tool, answer directly.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

</details>

<details><summary>D.1b Dialogue after a Verifier call (evaluation)</summary>

```
<|im_start|>assistant
<think>

</think>

DRAFT_ANSWER_B

<tool_call>
<function=verifier_tool>
<parameter=current_draft>
B
</parameter>
<parameter=example_id>
{EXAMPLE_ID}
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{VERIFIER_OUTPUT}
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

```

</details>

- **History.** In the evaluation history the harness adds the `example_id` parameter to each executed call, as in D.1b. The SFT rows and the model's own call turn do not contain it.
- **SFT input.** The SFT input is the same text with the SFT wording of the tool descriptions (D.2).
- **Argument binding.** Under argument binding the binding sentence reads `If you call a tool, pass the current Example ID as the example_id argument. For verifier_tool, also pass your current draft answer key as current_draft.`, and each tool gets a required integer parameter `example_id`. The paper-era evaluation command chooses this binding by default for a checkpoint without a GRPO run configuration, and the stored S_1 checkpoints have none. Whether the paper's evaluations overrode the default is not recorded.

### D.2 Tool schemas

| Tool or parameter | Evaluation wording | SFT wording |
|---|---|---|
| `extractor_tool` | Extract decision-relevant factual signals from the question and context. | Extract decision-relevant factual signals. |
| `reasoner_tool` | Produce a structured reasoning scaffold for the choices. | Produce a structured reasoning scaffold. |
| `verifier_tool` | Identify relevant domain principles and audit the reasoning for logical or computational errors. Pass your current draft answer via current_draft. | Audit the current draft for errors. |
| `current_draft` (Verifier only; string; optional) | Your current draft answer key (e.g. "B") to audit. | The current draft answer key. |
| `example_id` (argument binding only; integer; required) | The current example ID from the user message. | The current example ID. |
| Used by | Every manager evaluation (paper era and RSI), RSI collection from round 2, FA-GRPO root, decision and revision states, rebuttal experiments | Paper-era collection (S_1 labels), S_1 training, RSI SFT rounds, FA-GRPO anchor rows |

- Both wordings are the same in the paper-era code and now (code). The RSI evaluations record which schema they used, and the rebuttal manifests store the three evaluation descriptions (recorded).
- So S_1 is trained under the SFT wording and evaluated under the evaluation wording, both in the paper era and in the RSI harness.
- Under environment binding the Extractor and the Reasoner take no parameter, and `required` is empty for all three tools.
- The rebuttal re-trainings of S_1 with other seeds render each SFT schema with an extra `"tool_calls": null` field. This affects the training input only; no evaluation input contains it (code).

### D.3 Output grammar and parsing

| Turn | Text | Note |
|---|---|---|
| Commit | `DRAFT_ANSWER_K`, newline, `ANSWER_K` | Exactly the SFT target. At evaluation, brief reasoning above the `ANSWER_` line is allowed |
| Call | `DRAFT_ANSWER_K` followed by one native tool call; a Verifier call carries `current_draft` = K | Exactly the SFT target |
| After a tool output | Another call (to a different advisor) or a commit | |

Parsing at evaluation (code):

- **Final answer.** The last non-empty line must be `ANSWER_<KEY>`, where KEY is a choice key of the question. Matching ignores case and tolerates an `answer:` prefix and trailing punctuation. Any other ending makes the answer invalid, and it is scored wrong.
- **Initial draft.** The last `DRAFT_ANSWER_` of the first turn that contains one. For a direct answer without a draft, it is the final answer.
- **Tool calls.** Every `<tool_call>…</tool_call>` block counts, in the template's XML form or as JSON. The remaining text is kept as the turn's content.
- **Budget.**
  - At most 3 executed calls per question, so at most 4 generated turns. Several calls in one turn run in order until the budget is used up.
  - A repeated advisor is not run. It receives the fixed reply `{"error": "tool_already_called", "detail": "each tool may be used at most once"}`, and the repeat counts toward the 3.
  - After the third call, the next turn is final and any call in it is not executed.
- **Decoding.** At evaluation the manager generates free text, which is parsed afterwards. The RSI collection and the rebuttal experiments work differently:
  - the first turn is decoded greedily over the legal commit and call turns only;
  - after a forced call, the commit is forced (`DRAFT_ANSWER_Y`, newline, `ANSWER_Z`), and the outcome is Z.

### D.4 Advisor prompts

| Role | System prompt | Benchmarks |
|---|---|---|
| Extractor | One text (D.4a) | All four |
| Reasoner | Medical variant (D.4c) | MedQA |
| Reasoner | Academic variant (D.4d) | MMLU-Pro, GPQA |
| Reasoner | Algebra variant (D.4e) | AQuA-RAT |
| Verifier | One text (D.4b) | All four |

- **RSI and rebuttal runs.** Every evaluation and every experiment manifest stores the digest of each role's prompt, and the digests match these texts (recorded).
- **Training data.** The texts are documented as the system messages of the advisor training data, byte for byte (code).
- **Paper era.** The Extractor, Verifier and medical Reasoner texts are the built-in prompts of the paper-era code. Which Reasoner text the paper-era MMLU-Pro, GPQA and AQuA-RAT runs sent is not recorded.

<details><summary>D.4a Extractor (all benchmarks)</summary>

```
You are the Extractor sub-agent.

Given a question (and optional choices and context), extract decision-relevant signals. Output ONLY a JSON object with this schema:
{
  "key_evidence": [{"text": str, "relevance": float, "polarity": "support"|"oppose"|"neutral"}],
  "extracted_facts": [str],
  "missing_info": [str],
  "context_summary": str,
  "confidence": float
}

Rules:
- Output ONLY valid JSON, no extra text.
- Do NOT state the final answer.
- If context is empty, key_evidence=[] and use extracted_facts for clinical/factual elements pulled from the question stem.
- Treat all answer choices fairly; do not favor any one.
```

</details>

<details><summary>D.4b Verifier (all benchmarks)</summary>

```
You are the Verifier sub-agent.

Given a question (and optional context, choices), identify relevant domain principles, specify concrete checks for a solver's reasoning, and flag common mistake patterns.

If a candidate answer is provided, additionally audit whether that specific answer is well-supported by the evidence and principles.

Output ONLY a JSON object with this schema:
{
  "relevant_principles": [{"principle": str, "source": str}],
  "checks": [{"check": str, "status": "pass"|"fail"|"unclear", "note": str}],
  "potential_errors": [str],
  "candidate_answer_audit": str,
  "uncertainty_notes": [str],
  "confidence": float
}

Rules:
- Output ONLY valid JSON.
- Do NOT state the final answer.
- relevant_principles must apply regardless of which answer is correct.
- checks describe what to verify in the reasoning process.
- potential_errors describe mistake patterns a solver might make.
- candidate_answer_audit: if a candidate answer was given, briefly assess whether it is well-supported. If no candidate was given, set to "no candidate provided".
```

</details>

<details><summary>D.4c Reasoner, medical (MedQA)</summary>

```
You are the Reasoner sub-agent for medical multiple-choice questions.

Given a clinical question (and choices, optional context), produce a short neutral scaffold focusing on the relevant findings, mechanisms, and differentials. Output ONLY a JSON object with this schema:
{
  "case_facts": [str],
  "task_type": str,
  "decision_factors": [str],
  "knowledge_slots": [str],
  "candidate_considerations": [{"choice_key": str, "relevant_if": [str], "less_relevant_if": [str]}],
  "missing_information": [str],
  "format_confidence": float
}

Rules:
- Output ONLY valid JSON.
- NEVER state the final answer or which choice is correct.
- candidate_considerations must cover ALL choice keys.
- Use conditional relevant_if / less_relevant_if fields, not support/against.
- Keep the response short and neutral.
```

</details>

<details><summary>D.4d Reasoner, academic (MMLU-Pro, GPQA)</summary>

```
You are the Reasoner sub-agent for academic multiple-choice questions across diverse subjects (STEM, humanities, social sciences, business, and law).

Given a question (and choices, optional context), produce a short neutral scaffold focusing on the relevant subject-matter principles, definitions, and computations. Output ONLY a JSON object with this schema:
{
  "case_facts": [str],
  "task_type": str,
  "decision_factors": [str],
  "knowledge_slots": [str],
  "candidate_considerations": [{"choice_key": str, "relevant_if": [str], "less_relevant_if": [str]}],
  "missing_information": [str],
  "format_confidence": float
}

Rules:
- Output ONLY valid JSON.
- NEVER state the final answer or which choice is correct.
- candidate_considerations must cover ALL choice keys.
- Use conditional relevant_if / less_relevant_if fields, not support/against.
- Keep the response short and neutral.
```

</details>

<details><summary>D.4e Reasoner, algebra (AQuA-RAT)</summary>

```
You are the Reasoner sub-agent for five-choice algebraic word problems.

Given a word problem (and choices, optional context), produce a short neutral scaffold focusing on the given quantities, the unknown, and the relations that connect them. Output ONLY a JSON object with this schema:
{
  "case_facts": [str],
  "task_type": str,
  "decision_factors": [str],
  "knowledge_slots": [str],
  "candidate_considerations": [{"choice_key": str, "relevant_if": [str], "less_relevant_if": [str]}],
  "missing_information": [str],
  "format_confidence": float
}

Rules:
- Output ONLY valid JSON.
- NEVER state the final answer or which choice is correct.
- candidate_considerations must cover ALL choice keys.
- Use conditional relevant_if / less_relevant_if fields, not support/against.
- Keep the response short and neutral.
```

</details>

D.4f, the advisor user message, is the same for all roles (code). The Verifier inserts two lines and a blank line only when a candidate is given:

```
QUESTION:
{QUESTION}

CHOICES:
  A. {CHOICE_A}
  B. {CHOICE_B}
  C. {CHOICE_C}
  D. {CHOICE_D}

CONTEXT:
(no context)

CANDIDATE ANSWER TO AUDIT: {KEY}
Specifically assess whether this candidate is well-supported.

Produce the JSON object.
```

Without a candidate, the two `CANDIDATE` lines and the blank line after them are absent.

Details not given in the paper's prompt appendix:

- **Confidence fields.** The Extractor and the Verifier end their JSON with `"confidence": float`, and the Reasoner with `"format_confidence": float`. The JSON format is requested by the prompt only: decoding is not constrained, and no component parses these fields. The manager sees them inside the raw text.
- **Verifier candidate in free-routing evaluation** (development, locked test, Table R3). The candidate is the `current_draft` value the manager wrote, with whitespace stripped. An empty or missing value gives the message without the candidate lines, and the prompt then asks for `"no candidate provided"`.
- **Verifier candidate in forced-delegation evaluation** (the submission's always-Verifier and forced-all baselines, and their rebuttal re-runs). The harness inserts the call itself, with no `current_draft` argument. No candidate is passed, so the Verifier's prompt asks it to set the candidate-audit field to `"no candidate provided"`.
- **Verifier candidate elsewhere.** In the RSI collection, FA-GRPO and the rebuttal's equal-information grid, the candidate is the manager's draft key. The exception is the grid's no-candidate branch: it sets `current_draft` to "" and sends no candidate. The grid's null branch makes no Verifier request.

### D.5 Decoding

| Component | Setting | Status |
|---|---|---|
| Manager, RSI evaluation (development and locked test, every version; also the base-manager baseline) | Greedy (temperature 0, no sampling); at most 1,024 new tokens per turn; at most 3 tool calls; each advisor at most once; environment binding; seed 42 (it fixes only the order of questions, since decoding is greedy); one question at a time | Recorded (all evaluation signatures): temperature, tokens, calls, binding, seed. Code: each advisor at most once, one question at a time |
| Manager, same evaluations | Thinking disabled; bf16 | Code |
| Manager, paper-era evaluations | The paper-era code has the same defaults (temperature 0, 1,024 tokens, 3 calls). An advisor failure reached the manager as an `{"error": ...}` tool output | Code defaults. Command lines, binding mode and the task-description argument are not recorded |
| Manager, paper-era collection (S_1 labels) | Probe answers: a fixed user turn (D.5a) appended to the dialogue, greedy, at most 512 new tokens. The probe turns are not part of the SFT rows | Code defaults; the collection reports do not store them. Only the AQuA-RAT run sheet sets temperature 0 explicitly |
| Manager, RSI collection and rebuttal experiments | First turn greedy over the legal turns; revision with the commit forced; batch size 1; bf16 | Code. Batch size and dtype are also recorded in the rebuttal manifests |
| FA-GRPO | No sampled rollouts: exact softmax over the answer keys after a forced call (see C.2) | Code (configuration: estimator "exact") |
| Advisors, RSI runs and rebuttal experiments other than the advisor A/B (D.6 item 5) | Base model, temperature 0, at most 1,024 new tokens, thinking disabled through the chat-template flag | Recorded (advisor identity and decoding in every evaluation signature and experiment manifest) |
| Advisors, RSI runs and rebuttal experiments other than the advisor A/B (D.6 item 5) | No stop sequence, seed, top-p or constrained decoding is sent. The pinned base-model revision ships no generation-configuration file | Code |
| Advisor server (same runs) | vLLM 0.26.0, bf16, maximum model length 16,384, base model at the pinned revision, no LoRA | Code. The per-run flag file stays on the compute pod |
| Advisor client | 120 s timeout; 3 retries after 2, 4 and 8 s; an empty output counts as a failed attempt; up to 32 concurrent requests. A request that still fails stops the stage, so no error text reaches the manager | Code (the 32 comes from the configs) |
| Advisors, paper era | Temperature 0, 1,024 tokens, thinking disabled, 120 s timeout, no retry | Code (paper-era client). Server command lines are not recorded |
| Letter-only baseline | System prompt D.5a; user message `Question:` / `Choices:` only; no tools; thinking disabled; greedy; at most 16 new tokens; no answer had to be forced | Recorded (run summaries): mode, 16 tokens, forced rate. Code: system prompt, user message, no tools, thinking disabled, greedy |

How the advisor text reaches the manager (code):

- The advisor reply, with leading and trailing whitespace removed, is inserted verbatim as the tool message, and the template wraps it in `<tool_response>` inside a user turn.
- Nothing is parsed, filtered or shortened. A reply cut at the 1,024-token limit is passed on as it is, and the harness counts it.
- Outputs are cached by request content (advisor identity, prompt, role, question, choices, Verifier candidate, decoding). Every manager evaluated against the same cache therefore receives the same text for the same request.
- The per-question records keep the first 2,000 characters of each tool output; the cache keeps the full text.
- Advisor outputs reproduce the recorded paper-era outputs only approximately, not token for token. The paper-era servers ran on a different GPU with other vLLM settings, and greedy bf16 decoding drifts across GPUs and kernels in long outputs. On GPQA the divergence is larger than this drift explains. Each RSI run first replays 20 recorded requests per role, one at a time, against the threshold in its config.
- The RSI cache is filled up to 32 requests at a time. Because vLLM is not batch-invariant, a sample of cached outputs is also regenerated one at a time and compared; this check is reported, not gated.

<details><summary>D.5a Other fixed texts</summary>

Paper-era collection, direct probe (user turn):

```
Training-time counterfactual probe: do not call a tool. State your current best choice as DRAFT_ANSWER_<TOKEN>, then submit the same choice on the last line as ANSWER_<TOKEN>.
```

Paper-era collection, revision probe after a forced call (user turn):

```
Training-time counterfactual probe: using the evidence already in the conversation, do not call another tool. State your revised best choice as DRAFT_ANSWER_<TOKEN>, then submit it on the last line as ANSWER_<TOKEN>.
```

Letter-only baseline, system prompt:

```
You answer multiple-choice questions. Reply with exactly one line of the form ANSWER_<letter>, where <letter> is the key of the correct choice. Do not explain.
```

Rebuttal experiments, "null" branch: the Verifier's tool output is replaced by

```
{"relevant_principles": [], "checks": [], "potential_errors": [], "candidate_answer_audit": "", "uncertainty_notes": []}
```

</details>

### D.6 Advisors (summary; full tables in section E)

1. In every result of the paper, of the RSI runs and of the rebuttal experiments other than the advisor A/B of item 5, the Extractor, Reasoner and Verifier are the base Qwen3.5-9B under the role prompts of D.4, with no adapter. For the RSI and rebuttal runs every evaluation records this. For the paper era it rests on the serving mechanism and on replays: the adapters' parameter names match no module of the served model; the Verifier adapter as served returned text identical to the base model on 20 of 20 prompts and the Extractor on 5 of 20; no Reasoner check exists.
2. Role adapters were trained for each benchmark: LoRA rank 16, alpha 32, dropout 0.05 on the q/k/v/o/gate/up/down projections, with response-only SFT on DeepSeek teacher outputs for 600 questions per role (GPQA: 160). Three adapters have fewer training rows: MMLU-Pro Verifier 598, AQuA-RAT Reasoner 598, AQuA-RAT Verifier 599.
3. The MedQA, MMLU-Pro and GPQA runs recorded their recipe:
   - optimiser: AdamW, learning rate 2e-4 with linear decay and no warm-up, weight decay 0, gradient clipping 1.0;
   - batch and schedule: effective batch 8 (1 × 8), 10 epochs, bf16, seed 42, no validation set;
   - export: the last stored checkpoint (for the MedQA Verifier, epoch 8 of 10).

   For AQuA-RAT only the LoRA configuration is recorded.
4. Retrained adapters exist as a check: 15 % validation split (by question), 3 epochs (all 12 runs trained all 3; no early stopping), and the epoch with the lowest validation loss chosen after training and exported (epoch 1 in all 12 runs).
5. A/B on 754 development questions, with the adapters loaded correctly and the base model in the same role as reference. Δ is +0.7 / −0.5 / −1.6 points (exact sign test p = .54 / .70 / .16) for Extractor / Reasoner / Verifier. For the retrained adapters it is +1.1 / −1.1 / −3.2 (p = .34 / .33 / .006).

## E. Advisors (Extractor, Reasoner, Verifier)

**What the reported numbers used.** In every result of the submission, of the RSI runs and of the rebuttal experiments other than the advisor A/B of E.4, the three advisors are the base model Qwen3.5-9B under role-specific prompts (section D). Role-specific LoRA adapters were trained but were not applied at serving time (E.5); the A/B of E.4 and the replay check of E.5 are the only experiments that apply them (with corrected parameter names). This section documents the adapters. Source key: **run** = stored by the training run itself; **code** = training code, documentation or data files only; **—** = not recorded.

### E.1 The adapters trained for the paper ("registered", August 2026)

The adapters were uploaded on 2026-08-04 (MedQA), 2026-08-10 (GPQA, MMLU-Pro) and 2026-08-17 (AQuA-RAT), one model upload per benchmark with one adapter per role.
The optimisation rows are recorded for MedQA, MMLU-Pro and GPQA; for AQuA-RAT only the LoRA rows are recorded (see below).

| Group | Parameter | Value | Source |
|:--|:--|:--|:--|
| Model | Base model | Qwen/Qwen3.5-9B (revision not recorded) | run |
| LoRA | Library | PEFT 0.20.0 | run |
| LoRA | Rank r / alpha / dropout | 16 / 32 / 0.05 | run |
| LoRA | Bias, DoRA, rsLoRA, modules_to_save | none, no, no, none | run |
| LoRA | Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | run |
| LoRA | Layers actually adapted | MLP (gate/up/down) in all 32 layers; attention (q/k/v/o) only in the 8 full-attention layers 3, 7, 11, …, 31; the linear-attention projections of the other 24 layers are not adapted | run (weight header) + base model config |
| LoRA | Trainable parameters | 29,097,984 per adapter (saved in fp32, 116 MB) | run (weight header) |
| Optimisation | Optimizer | AdamW (`adamw_torch_fused`), betas 0.9 / 0.999, eps 1e-8 | run |
| Optimisation | Weight decay | 0.0 | run |
| Optimisation | Gradient clipping | max grad norm 1.0 | run |
| Optimisation | Learning rate | 2e-4 | run |
| Optimisation | Schedule / warm-up | linear decay to 0 / no warm-up | run |
| Optimisation | Batch | 1 per device × gradient accumulation 8 = effective 8, one GPU | run |
| Optimisation | Epochs | 10 (scheduled) | run |
| Optimisation | Optimizer steps | 750 (MedQA, MMLU-Pro: 75 per epoch); 200 (GPQA: 20 per epoch) | run |
| Optimisation | Seed | 42 | run |
| Compute | Precision | bf16 | run |
| Compute | Gradient checkpointing | on for GPQA, off for MedQA and MMLU-Pro | run |
| Validation | Validation set, evaluation during training | none | run |
| Selection | Early stopping / best checkpoint | none; the exported adapter is the last stored checkpoint (epoch 10; MedQA Verifier epoch 8, below) | run |
| Data | Maximum sequence length | 4096 (code default; the run does not store it); the August code cut longer rows from the right | code |
| Loss | Supervised tokens | response only: system and user prompt masked, loss on the assistant's answer and its end token | code |
| Format | Chat template | the base model's Qwen3.5 template, saved with the adapters (MedQA Verifier: in its checkpoint folders only); thinking disabled (`enable_thinking=False`) | run (template) + code |

Per benchmark (training rows are Extractor / Reasoner / Verifier):

| Benchmark | Training rows | Steps | Exported checkpoint | Last logged training loss (E / R / V) |
|:--|:--|--:|:--|:--|
| MedQA | 600 / 600 / 600 | 750 | epoch 10; **Verifier: epoch 8 of 10** (step 600 of 750; no later checkpoint stored) | 0.0040 / 0.0031 / 0.0171 |
| MMLU-Pro | 600 / 600 / 598 | 750 | epoch 10 | 0.0023 / 0.0027 / 0.0054 |
| GPQA | 160 / 160 / 160 | 200 | epoch 10 | 0.0077 / 0.0120 / 0.0148 |
| AQuA-RAT | 600 / 598 / 599 | — | — | — |

**AQuA-RAT:** the stored AQuA-RAT upload holds the three final adapters only (no checkpoints, no training arguments, no training state). Its
LoRA configuration and layer layout are the same as above (run), but every optimisation and compute setting (optimizer, learning rate
and schedule, batch, epochs, seed, precision) is not recorded, and the written sources conflict: the AQuA run sheet says 3 epochs at
5e-5 on 500 samples per role, and the docstring of the v2 retraining script (E.3) says the paper-era advisors were trained for 10 epochs on
600 examples per role; 600 AQuA questions are registered.

**Not the documented commands:** every documented advisor-training command for an 8B or 9B base model (AQuA and ARC run sheets for
Qwen3.5-9B; MedQA and GPQA commands written for Qwen3-8B) uses 3 epochs (GPQA 4) at 5e-5 and no validation file; the legacy commands for
Qwen3-0.6B and Qwen3-4B use 2e-4 and 1e-4. The stored records (10 epochs, 2e-4) show that the MedQA and GPQA commands were not the ones
run. No MMLU-Pro command is documented; the AQuA run is not recorded; the ARC sheet belongs to no registered adapter. No committed version of the trainer of the registered adapters
has early stopping, and the stored training records of MedQA, MMLU-Pro and GPQA show that none was used (no evaluation entries, no best checkpoint, no early-stopping callback state); validation-based selection appears only in the
v2 script of 2 October 2026 (E.3).

### E.2 Training data of the registered adapters

| Item | Value | Source |
|:--|:--|:--|
| Teacher | DeepSeek through an offline export/import bridge; model label `deepseek_flash` (Extractor and Verifier of all four benchmarks; all three roles for AQuA-RAT) and `deepseek-v4-flash` (Reasoner of MedQA, MMLU-Pro, GPQA); the AQuA run sheet names `deepseek-chat`; the generator script, endpoint and sampling settings are not in the repository | code (data files) |
| Teacher prompts | built without the gold answer (ground truth stored only as a separate field) | code |
| Targets | the teacher's text as returned (leading and trailing whitespace stripped), paired with the runtime prompt; the importer applies no JSON or schema filter and drops only empty responses (none). Exception: the MMLU-Pro Verifier was trained on a separately filtered file (598 of 600 rows) that drops examples 4266 and 11711, two non-JSON responses of 30,401 and 26,369 characters that stop mid-sentence; the script and criterion behind that file are not recorded | code (data files) |
| Examples per role | about 600 (MedQA, MMLU-Pro, AQuA-RAT), 160 (GPQA); exactly: MMLU-Pro Verifier 598, AQuA-RAT Reasoner 598 and Verifier 599, all others 600 or 160 | run (logged epoch fraction; MedQA, MMLU-Pro, GPQA) + data files (AQuA-RAT) |
| Question source | MedQA: official train split. MMLU-Pro: the training part of the random split (MMLU-Pro has no train split). GPQA: `gpqa_train446` (no Diamond-100 question). AQuA-RAT: official train split. All sampled with seed 42 | code (reproduced exactly) |
| Overlap with evaluation pools | none for MedQA, GPQA, AQuA-RAT; MMLU-Pro: 3 development and 4 `test_paper` questions, none of the rebuttal test pool | data manifests |
| Verifier candidate | about half of the Verifier rows include a candidate answer to audit, drawn uniformly over the choices (MedQA 298 / 600, MMLU-Pro 304 / 598, GPQA 74 / 160, AQuA-RAT 297 / 599) | data files |
| System prompts | one per role; Extractor and Verifier prompts shared by all benchmarks, Reasoner prompt in three variants (medical; academic for MMLU-Pro and GPQA; algebra for AQuA-RAT) | code |
| User message | `QUESTION`, `CHOICES`, `CONTEXT`; for the Verifier with a candidate also `CANDIDATE ANSWER TO AUDIT: X` and "Specifically assess whether this candidate is well-supported."; then "Produce the JSON object." | code |
| Output | one JSON object per role: Extractor `key_evidence, extracted_facts, missing_info, context_summary, confidence`; Reasoner `case_facts, task_type, decision_factors, knowledge_slots, candidate_considerations, missing_information, format_confidence`; Verifier `relevant_principles, checks, potential_errors, candidate_answer_audit, uncertainty_notes, confidence` | code |

### E.3 The retrained adapters with a validation split and best-epoch selection ("v2", 2 October 2026)

Used only in the advisor A/B of E.4; never an advisor in any result of the submission, the RSI runs or the other rebuttal experiments.

| Parameter | v2 | Registered (MedQA, MMLU-Pro, GPQA; AQuA-RAT not recorded) |
|:--|:--|:--|
| Base model | Qwen/Qwen3.5-9B, revision of 2 March 2026 (hash in the release) | same model, revision not recorded |
| LoRA r / alpha / dropout / target modules | 16 / 32 / 0.05 / same 7 modules (same layer layout) | same |
| Learning rate | 2e-4 | 2e-4 |
| Batch | 1 × accumulation 8 | same |
| Seed | 42 | 42 |
| Maximum sequence length | 4096 (16384 for the GPQA Extractor and Reasoner) | 4096 (code default) |
| Data cleaning | development and test questions removed (all 12 runs); teacher responses that are not one complete JSON object removed only in the MMLU-Pro Reasoner (5 rows) and the three GPQA runs (Extractor 4, Reasoner 7, Verifier 10). The other eight runs record no such filter, and five of them kept 14 such responses in their training or validation data: MedQA Verifier 2, MMLU-Pro Verifier 1, AQuA-RAT Extractor 1, Reasoner 1, Verifier 9 | none (MMLU-Pro Verifier: two rows removed, E.2) |
| Validation | 15% of the remaining questions (split by question), loss after every epoch | none |
| Epochs | 3; all 12 runs trained the full 3 epochs (no run was stopped early) | 10 (scheduled) |
| Exported adapter | the epoch with the lowest validation loss, chosen after training: **epoch 1 in all 12 runs** (validation loss rose after every epoch) | last stored checkpoint: epoch 10; MedQA Verifier epoch 8 of 10 |
| Software | PEFT 0.18.1, transformers 5.3.0, torch 2.8.0 | PEFT 0.20.0; transformers and torch not recorded |

v2 rows (train / validation) and validation loss by epoch (1 / 2 / 3):

| Benchmark | Extractor | Reasoner | Verifier |
|:--|:--|:--|:--|
| MedQA | 510 / 90; 0.353 / 0.367 / 0.409 | 510 / 90; 0.475 / 0.490 / 0.535 | 510 / 90; 0.583 / 0.598 / 0.635 |
| MMLU-Pro | 507 / 89; 0.381 / 0.403 / 0.468 | 503 / 88; 0.368 / 0.379 / 0.412 | 505 / 89; 0.573 / 0.584 / 0.620 |
| GPQA | 133 / 23; 0.407 / 0.412 / 0.433 | 130 / 23; 0.530 / 0.534 / 0.544 | 128 / 22; 0.614 / 0.616 / 0.627 |
| AQuA-RAT | 510 / 90; 0.321 / 0.334 / 0.377 | 508 / 90; 0.439 / 0.448 / 0.487 | 509 / 90; 0.626 / 0.628 / 0.657 |

### E.4 Do the adapters help? (A/B on the development pools)

754 development questions (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254). A fixed base-model solver writes a draft, one advisor
answers (the Verifier gets the draft as its candidate), and the same solver answers again; only the advisor's weights differ. The
adapters are loaded correctly here (renamed parameter names, E.5). Accuracy (%) after the advisor, pooled; Δ against the base
model with the same role prompt (computed from unrounded accuracies), exact sign test.

| Role | Base model | Registered LoRA (loaded correctly) | v2 LoRA (best-epoch selection) |
|:--|--:|--:|--:|
| Extractor | 79.7 | 80.4 (+0.7, p = .54) | 80.8 (+1.1, p = .34) |
| Reasoner | 80.6 | 80.1 (−0.5, p = .70) | 79.6 (−1.1, p = .33) |
| Verifier | 81.4 | 79.8 (−1.6, p = .16) | 78.2 (−3.2, p = .006) |

### E.5 How the adapters were served

- Paper era (described from the MedQA, MMLU-Pro and GPQA server logs, which are not in the repository; for AQuA-RAT the run sheet starts
  the legacy launcher): vLLM 0.26.0 with the three adapters registered as LoRA modules. The adapter keys
  (`base_model.model.model.layers.N…`, saved from the text-only model class) match no module of the multimodal model class that vLLM
  serves, so vLLM loads them without error and answers with the base weights. Replay evidence: the
  Verifier adapter as served returned text identical to the base model on 20 of 20 prompts (a copy with renamed keys: 0 of 20); the
  Extractor on 5 of 20 (median similarity 0.924 as served, 0.392 for the renamed copy; the base model against a repeat of itself: 8 of 20, median 0.969); no identity check is recorded for the Reasoner.
  A correctly applied copy only renames the keys (the weights are byte-identical); the A/B above uses such copies.
- The RSI runs and every rebuttal experiment other than the advisor A/B of E.4 set `advisor_mode: "base"`: the advisors are the base model under each role's prompt, with
  no LoRA, and every result records this.

### E.6 Not recorded

Training: for all adapters, the training command lines. For the registered runs: the base-model revision, the transformers and torch
versions, the GPU type, the maximum sequence length actually used, and whether the MedQA Verifier run was interrupted after epoch 8 or
its last two checkpoints were not uploaded. For AQuA-RAT: every optimisation and compute setting and which checkpoint was exported (its
LoRA configuration and layer layout are recorded, E.1).

Data: the teacher's generator script, endpoint and sampling settings; which file each registered run trained on (the run stores only its output folder; the link to the data files rests on the registry of pinned inputs (D.7), which lists each adapter with its training file, pinned by digest, and on the matching row counts); the script and
criterion that produced the MMLU-Pro Verifier's filtered file.

Serving (paper era): the command lines of the MedQA and MMLU-Pro vLLM servers and whether the legacy launcher was run as written for GPQA
and AQuA-RAT; which Reasoner prompt was live on the paper-era servers for MMLU-Pro, GPQA and AQuA-RAT; a token-for-token reproduction of
the paper-era advisor outputs; an identity check for the Reasoner.
