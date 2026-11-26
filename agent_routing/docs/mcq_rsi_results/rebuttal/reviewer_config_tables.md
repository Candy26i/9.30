# Configuration tables for the author response

These tables summarise the full configuration appendix; inside a cell, values are in the order MedQA / MMLU-Pro / GPQA / AQuA-RAT.

## 1. Results and their configurations

| Result | Policy | Collection depth | Commit cap ρ | Training seed | Manager checkpoint | Data pool | Calls allowed |
|---|---|---|---|---|---|---|---|
| Fig. 3; Tables 3, 5, 6 | base-model collection | n.r. / 2 / 2 / 3 [b] | none | none; tie-break 42 | base model | `collect_r1` | forced probes, greedy, ≤ 512 tokens (c); each advisor at most once per branch; stops at the first depth with a success (c) |
| Table 4 | description | printed 3 / 2 / 2 / 3 (P); GPQA 2 also recorded; S1's labels from 2 / 2 / 1 / 3 | – | – | – | section 2 | – |
| Table 1 | single-pass S1 [a] | 2 (P) / 2 / 1 / 3 | 3 (P) / 2 / 2 / 1 | GPQA 42, others n.r. | MedQA: the submission's, n.r.; others: S1 | MedQA test 500 (which n.r.); `test_paper` (reconstruction); Diamond-100; `test` | ≤ 3 (c) |
| Table 7; Fig. 4 low-call points (reviews' "Figure 3"); Table 14 A | single-pass S1; always-Verifier | as Table 1 | as Table 1 | MedQA three seeds, n.r.; others as Table 1 | MedQA three, n.r.; others S1; always-Verifier: "same manager" (P), run n.r. | `dev` / `dev` / Diamond-100 / `dev` (P) | ≤ 3 (c); always-Verifier: one forced call, no draft or candidate (P) |
| Tables 8, 9, 11; Fig. 4 higher-call, forced-all; Fig. 5 | single-pass variants; forced-all | Table 8: 2 / n.r. / n.r. / 3 (P); Table 11: MedQA 2 and 1, MMLU-Pro 1, AQuA 3 (P) | Table 8: 0 / n.r. / n.r. / 0 (P); Table 11: 0 vs 3, 1 vs 2, 1 vs 2, 0 vs all (P) | n.r. | n.r. [d] | development sets (P) | ≤ 3 (c); forced-all: three calls (P) |
| Table 10 | replay, no training | n.r. | n.r. | n.r. | n.r. [d] | development sets (P) | fixed role and calls; 2,000 random subsets (P) |
| Table 12 base rows; Fig. 7; Table R2 | untrained base model; R2: as manager vs S1, seeds 1–3 | – | – | – | base model; R2 references: letter-only base model and S1, base system and RSI S1 of Table 12 | `test`; R2 gates from `dev` grids | letter-only: 0; base system: ≤ 3; R2: grid, same output grammar |
| Table 12 S1; Table 13 "S1"; Table 14 B "S1 twice"; Fig. 7 | RSI S1 [a] | 2 / 2 / 1 / 3 | 3 / 2 / 2 / 1 | as Table 1 | S1 (MedQA: substitute [c]) | `test`; Table 13: `dev` | ≤ 3 |
| Table 12 "SFT + GRPO", round 3; Table 2; Fig. 7 [e] | RSI arm (Static, Dynamic, Success) | rounds 2–3: 2, own model (c); Static: S1's label file | 3 / 2 / 2 / 1 (c) | collection 42, SFT 42, FA-GRPO 0 (c); evaluation 42 | G3, or S3 if FA-GRPO was rejected (MedQA: S3) | `test` | ≤ 3 |
| Table 12 "SFT only", rounds 3, 5; Table 2; Table 14 B | RSI arm (Static, Dynamic), no GRPO | as above; no GPQA in rounds 4–5 (c) | as above | as above, no FA-GRPO | S3; S5 continued from S3 | `test` | ≤ 3 |
| Table 13 | one SFT from S1 per label set (B: Sk vs Gk labels, earlier configuration) | R1: as S1; others 2 (c) | 3 / 2 / 2 / 1 (c) | 42 (c) | end of a 3-epoch SFT from S1 (c) | `dev` | ≤ 3 |
| Tables R1, R3; Figure R1; fixed-role notes | S1; three re-trainings of its labels | 2 / 2 / 1 / 3 | 3 / 2 / 2 / 1 | S1 as Table 1; 1, 2, 3 (c) | S1; re-trainings | `test`; R1 also `dev` | R1: grid [a]; R3: ≤ 3; fixed-role: one role or all three, no draft or candidate |
| Commit-only; success distillation | one SFT each, not RSI | S1's round-1 records | commit-only: root commits 304 / 213 / 86 / 217; success: 3 / 2 / 2 / 1, one successful option drawn uniformly per record (c) | 42 (c) | S1 recipe (c) | `test`; commit-only also `dev` | grid; commit-only: no call on any test question; success also ≤ 3 |
| Commit-cap sweep, incl. no cap [f] | S1's siblings | S1's collection | 0, 1, 2, 3\*, none / 1, 2\*, none / 1, 2\*, none / 0, 0.5, 1\*, none (\* = S1) | GPQA 42, others n.r. | one sibling per cap | `test` | grid |
| Advisor A/B | base-model solver | – | – | – | base model (solver); advisor adapters: section 5 | `dev`, 754 | one advisor |

n.r. = not recorded; (P) = stated in the submission only; (c) = set by code or configuration only (at the start of a cell: the whole cell); all other values are recorded by the runs or reproduced from their stored outputs.

- [a] Single-pass S1 is run by the submission's loop; RSI S1 is the same adapter (MedQA: [c]) run by the RSI harness on the RSI pools, the start of every arm. Grid: shared candidate, ≤ 1 call per rule, forced commit.
- [b] MedQA's depth-2 collection solves 304 / 57 / 5 roots at 0 / 1 / 2 calls, not Table 3's 304 / 63 / 2. GPQA's rows match a depth-2 collection, S1's labels a depth-1 one.
- [c] The ρ = 3 checkpoint of the depth-2, 400-root MedQA collection, with the submission's configuration.
- [d] Table 8's MedQA "Selected" values and AQuA accuracy equal Table 11's ρ = 0 rows; Table 10's MedQA value and Table 11's MedQA ρ = 3 row are S1's development values; the rest n.r.
- [e] Fig. 7: the MMLU-Pro and AQuA-RAT dynamic token totals were copied from an earlier results page; their per-question files are not archived.
- [f] The cap is confirmed by a label file or training record for 11 of the 15 managers; MMLU-Pro ρ 1 and AQuA-RAT ρ 0, 0.5 and none are known from the adapter names only.

## 2. Data splits

| Benchmark | Pool (size) | Definition | Overlap |
|---|---|---|---|
| MedQA | `collect_r1` (400) | first 400 of a seed-42 shuffle of the first 1,200 train rows | 248 advisor-SFT |
| | `grpo_r1`–`r3` (256 each) | the other 800 of those rows | 89 / 92 / 76 advisor-SFT |
| | `dev` (200) | first 200 validation rows | none |
| | `test` (500) | first 500 test rows | none; Table 1's 500 n.r. |
| MMLU-Pro (official test split) | `collect_r1` (400) | within positions 300–1,499 of a seed-42 shuffle | 87 advisor-SFT; 68 in `test_paper` |
| | `grpo_r1`–`r3` (256 each) | positions 300–1,499 minus roots | 47 / 50 / 31 in `test_paper`; 55 / 59 / 50 advisor-SFT |
| | `dev` (200) | positions 0–199 (reconstruction) | all in `test_paper` |
| | `test_paper` (500; Table 1) | positions 0–499 (reconstruction) | holds `dev`, 68 roots, 47 / 50 / 31 GRPO questions; 4 share a stem with advisor-SFT (3 in `dev`, 1 in `grpo_r1`); 104 clean |
| | `test` (500) | first 500 eligible positions from 1,500 | none |
| GPQA (198 Diamond in 546) | `collect_r1`–`r3` (200) | roots among the first 406 of 446 training rows; reused | 160 advisor-SFT |
| | `grpo_r1`–`r3` (146) | remaining training rows; reused | none |
| | `dev` (100) | 51 leftover Diamond and 49 other rows | none |
| | `test` (100) | Diamond-100 | the submission's GPQA development pool (Tables 4 †, 7–10); no other |
| AQuA-RAT | `collect_r1` (400) | as MedQA | 400 advisor-SFT |
| | `grpo_r1`–`r3` (256 each) | the rest of those 1,200 rows | 55 / 64 / 71 advisor-SFT |
| | `dev` (254) | official validation split | none |
| | `test` (254) | official test split | none (36 train rows share a stem with `dev` or `test`, none used) |

- [a] `dev` and `test` share no stem with each other, any training pool or advisor-SFT (except 3 MMLU-Pro `dev` stems); advisor-SFT overlap did not act through trained weights (section 5: the advisors were served as the base model).

## 3. Manager SFT (S1)

| Setting | MedQA [a] | MMLU-Pro | GPQA | AQuA-RAT [b] |
|---|---|---|---|---|
| Base model; LoRA; supervised tokens | Qwen3.5-9B (revision n.r.); r 16, α 32, dropout 0.05; q, k, v, o in the 8 full-attention layers, gate, up, down in all 32; response and end token, including the draft letter of call rows, prompt masked (c) | same | same | same |
| lr; epochs; batch; seed | 1e-5; 10; 8, as 1 × 8 (c); n.r. | same | 1e-5; 10; 1 × 8; 42 | n.r. |
| Schedule; optimizer | linear to 0, no warm-up; AdamW (fused), β 0.9 / 0.999, ε 1e-8, weight decay 0, clip 1.0 (c) | same (c) | same | same (c) |
| Precision; maximum length | bf16 mixed (c); 4,096 (c) | same | bf16 mixed; 4,096 (c) | as MedQA |
| Label rows (commit + rescue); ρ; cap seed; depth; roots | 315 (186 + 129); 3; 0; 2; 400 | 331 (162 + 169); 2; 0; 2; 400 | 148 (74 + 74); 2; 42; 1; 200 | 402 (132 + 270); 1; 0; 3; 400 |
| Checkpoint | after the last epoch (c) | same | epoch 10 | end of training (c) |

- [a] The same-configuration substitute, documented as "seed 0"; no record says which seed that is.
- [b] No training record; which of two 402-row label files trained it is n.r.
- [c] Cap rule: all rescues kept; commits cut to ⌊ρ × rescues⌋ with the cap seed. How ρ and depth were chosen is n.r.

## 4. RSI rounds and FA-GRPO

| Setting | Value |
|---|---|
| Rounds | 1: FA-GRPO on S1 gives G1. 2–3: Dynamic and Success collect with G(k−1); SFT from G(k−1) gives Sk, FA-GRPO Gk. SFT-only arms: no FA-GRPO; rounds 4–5 from S3 |
| Collection | (c) own greedy root; advisors forced to depth 2, stopping at the first success; 400 new questions (GPQA: same 200) |
| Round SFT | (c) 3 epochs from the previous checkpoint; lr 1e-5, linear; 1 × 8; seed 42; loss on the routing suffix only |
| Dynamic | shortest successful sequence, or commit if the draft is correct; commits capped at int(ρ × rescues) |
| Static | the registered round-1 label file every round; no collection |
| Success | Dynamic's records; target uniform over all successful options, commit included if the draft is correct |
| Reward | 1 if the answer revised after one forced call is gold, else 0; no call cost; routing gets KL terms only (c) |
| Group | no sampled rollouts: exact softmax over the answer keys; 4 questions × 3 advisors per step (c) |
| Loss, reference | A(y) = (r(y) − J) / (√(J(1−J)) + 10⁻⁴), J = π(gold); KL to frozen Sk: 0.1 root, 0.1 decision, 0.02 revision; 0.03 × label-row cross-entropy (c) |
| Pool, steps, optimizer | `grpo_rk`, 256 per round (GPQA 146, reused), 4 per step, ≤ 64 steps; AdamW, constant lr 5e-6, weight decay 0, gradient-norm clip 1.0; LoRA continued, dropout 0 (c) |
| Temperature; seeds | greedy; FA-GRPO 0, collection and SFT 42 (c); evaluation 42 |
| Guard | every 8 steps on 64 questions; fails if the call rate differs from Sk's by > 0.10 or KL_dec > 0.05; last passing step kept (c) |
| Acceptance: evaluation gate (`dev`) | no malformed call, advisor error or unparsed answer (Sk: ≥ 0.99 valid, acknowledged) (c) |
| Accuracy (`dev`) | acc(Gk) ≥ acc(Sk) − 0.01 (c) |
| Calls (`dev`) | calls(Gk) ≤ calls(Sk) + 0.15 (c) |
| Call gap (`dev`) | gap(Gk) ≥ gap(Sk) − 0.5 × abs(gap(Sk)); gap = call rate on wrong minus on correct drafts (c) |
| Informative (GRPO pool) | ≥ 15 % of revision states with J in (0.02, 0.98) (c) |
| Outcome | all five, else Gk := Sk; one evaluation, no statistical test. Accepted: 0/7, 6/9, 6/7, 6/7 |
| Checkpoint | fixed before testing. Round 3: MedQA S3; others G3 at step 64 of 64, except MMLU-Pro Dynamic 40 of 48 and AQuA-RAT Static 16 of 24. SFT only: epoch 3 |

- [a] One run, one seed per cell; MMLU-Pro ran each arm separately, the others jointly.
- [b] The submission does not name the informativeness condition; it failed in 10 of 12 rejections and alone rejects all seven MedQA candidates.

## 5. Advisors

| Setting | Registered adapters (August) | Retrained v2 (October) |
|---|---|---|
| Use | in every result of the submission and of the response other than the advisor A/B, the advisors were the base model under role prompts | advisor A/B only |
| Data | DeepSeek outputs, gold answer not in the prompt (c); rows (E / R / V) 600 each, 600 / 600 / 598, 160 each, 600 / 598 / 599 (AQuA-RAT (c)); response-only loss (c) | same, minus development and test questions; 15 % held out |
| LoRA | r 16, α 32, dropout 0.05; MLP in all 32 layers, attention in the 8 full-attention layers | same |
| Optimisation | AdamW, lr 2e-4, linear to 0; 1 × 8; 10 epochs; seed 42; bf16 | lr 2e-4; 1 × 8; 3 epochs; seed 42 |
| Exported | epoch 10 (MedQA Verifier: 8 of 10; AQuA-RAT n.r.) | lowest validation loss: epoch 1 in all 12 runs |
| A/B Δ (E / R / V) | +0.7 / −0.5 / −1.6 (p = .54 / .70 / .16) | +1.1 / −1.1 / −3.2 (p = .34 / .33 / .006) |

- [a] AQuA-RAT: only the LoRA configuration is recorded; its optimisation settings are n.r.
- [b] A/B: 754 development questions; base-model solver, one advisor, solver again; adapters loaded correctly; exact sign test.

## 6. Decoding

| Component | Temperature | Max new tokens | Call limit | Thinking | Other |
|---|---|---|---|---|---|
| Manager, RSI harness | 0 | 1,024 | 3, each advisor once (c) | off (c) | seed 42 |
| Manager, submission era | 0 (c) | 1,024 (c) | 3 (c) | off (c) | binding n.r. |
| Collecting manager | 0 (c) | 512 (c) | forced, to the depth | off (c) | probe turns |
| Manager, RSI collection and grid | greedy over legal turns (c) | – | grid: 1 | off (c) | forced commit |
| Letter-only | 0 (c) | 16 | none (c) | off (c) | – |
| Advisors (except A/B) | 0 | 1,024 | – | off | vLLM 0.26.0 (c) |
| Advisors, submission era | 0 (c) | 1,024 (c) | – | off (c) | no retry (c) |
| Advisor A/B solver | 0 (c) | 4,096 | one advisor | off (c) | base model |
| Advisors in the A/B | 0 (c) | 1,024 (c) | – | off (c) | adapters loaded correctly |

- [a] – = not applicable.
