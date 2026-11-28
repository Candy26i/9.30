Thank you for the careful review and the concrete controls. We ran them on the test pools, pooled over the 1,354 test questions. The main comparisons use four training runs (our checkpoint and three re-trainings with new seeds); the two new training controls use one run each.

**Table A. At most one call; every rule starts from the same first answer and uses the same checkpoint, decoding and revision prompt (mean ± SD, four runs). This replaces the development-set comparisons of Tables 7–8.**

| Rule | Accuracy (%) | Calls / question |
|:--|--:|--:|
| First answer, no call | 56.0 ± 0.3 | 0 |
| **MARGENT (learned calling)** | **71.0 ± 0.2** | **0.45** |
| Random questions, same calls | 67.0 ± 0.2 | 0.45 |
| Random Verifier subset, same calls | 67.6 ± 0.2 | 0.45 |
| Verifier on least-confident answers, same calls (ranked within the test pool) | 70.6 ± 0.6 | 0.45 |
| Same, threshold fixed on dev at the same call rate | 70.5 ± 1.2 | 0.44 |
| Same, threshold fixed on dev for best dev accuracy | 76.9 ± 0.5 | 0.77 |
| Always Verifier, given the first answer | 77.9 ± 0.4 | 1.00 |
| Always Verifier, first answer hidden (paper's design) | 75.6 ± 0.2 | 1.00 |
| Always Reasoner / always Extractor | 61.2 ± 0.3 / 57.1 ± 0.3 | 1.00 |

**W1. Isolating the benefit of learned delegation.** We agree that Table 1 mixes three effects. We added the closest versions of the two controls you asked for (one training run each, compared with our checkpoint):

| Control | Accuracy (%) | Calls / question |
|:--|--:|--:|
| Our checkpoint (MARGENT) | 70.9 | 0.44 |
| Manager trained without call examples | 57.2 | 0 |
| Same manager, Verifier on its least-confident answers | 71.6 | 0.44 |
| Success distillation on the same records | 76.4 | 0.98 |

- **The gain comes from the Verifier's output, not from an extra revision.** A revision turn on an empty Verifier output does not help (−0.5 points); the Verifier's output adds 22.3 points over that turn.
- **SFT changes the first answer, not the revision.** In the same format, the first answer is 4.8–5.5 points more accurate than the untrained model's; how the manager uses the Verifier's reply does not change significantly in any run.
- **Training teaches when to call.** MARGENT calls after 62% of wrong and 31% of right first answers, against 92% and 98% for the untrained model (no gap on GPQA).
- **Shortest-path labels teach the manager when not to call.** Success distillation calls on 98% of questions; MARGENT makes **55% fewer calls**, each worth more (37.3 vs. 23.2 points per call on MMLU-Pro), at 5.5 points lower accuracy.
- **Allocation on the test pools.** MARGENT is above random questions on every benchmark in all four runs, significantly in 11 of the 16 cases (all runs on MMLU-Pro and AQuA-RAT, two on MedQA, one on GPQA); on the development pools the pattern differs (4 / 1 / 1 / 2 runs), so selection is uneven, as you note. At the same calls it matches a confidence-gated Verifier (+0.4, not significant), and so does the manager trained without call examples once given that gate (p = .52). A gate tuned for best dev accuracy is 5.5–6.4 points higher at 1.7× the calls. We will state that the learned policy does not choose questions better than a confidence ranking.

**W2. Equal inputs for always-Verifier.** Thank you; this was a real asymmetry, and Table A removes it. Giving the Verifier the first answer helps on MMLU-Pro (+4.8) and AQuA-RAT (+5.5), not on MedQA or GPQA (+2.2 pooled), confirming your point. Sending MARGENT's chosen questions to the Verifier beats a random Verifier subset of the same size by 4.2 points (significant in every run on MedQA, MMLU-Pro and AQuA-RAT; in none on GPQA). With equal inputs, always-Verifier is 6.9 points more accurate at 2.2× the calls, so we withdraw the Figure 4 comparison with always-Verifier (your Figure 3; Section 4.3, Tables 7 and 8). MARGENT keeps **two thirds of always-Verifier's gain** over the first answer (15.0 of 21.8 points) **with 45% of the calls**. Table A gives the trade-off at 0, 0.45, 0.77 and 1.00 calls; the revision adds the full accuracy–call curve.

**W3. Independence and robustness.**

| Concern | What we did | Result |
|:--|:--|:--|
| GPQA: dev = test (Diamond-100) | Conceded. Added a 100-question pool sharing no question with Diamond-100 or training (13 from the paper-era dev slice) | MARGENT 53.8 ± 1.0 there vs. 51.8 ± 1.7 on Diamond-100; always-Verifier 57.0 (gap not significant in any run) |
| AQuA-RAT: 254 / 254 | Checked by question text | dev = official validation split, test = official test split; no shared question |
| MMLU-Pro (found by us) | As reconstructed, the paper's test pool shared 200 of 500 questions with dev and 196 with training | All new results use a new pool with no shared question |
| Single runs | Three re-trainings of the same labels with new seeds | MARGENT 71.0 ± 0.2; above random questions and below always-Verifier in all 16 run–benchmark cases (significantly in 11 and 12) |
| Checkpoint selection | Checked code and logs | The code keeps the last epoch and chooses none. How ρ and depth were chosen is not recorded. The paper's MedQA checkpoint was not archived; we use a re-training with the same configuration |

Across all 15 managers of a commit-cap sweep (3–5 caps per benchmark), learned calling stays below always-Verifier (significantly in 10), so that conclusion holds at every cap we tested; the margin over random questions does change with ρ. The RSI contrasts remain single runs, and we will mark them so.

**W4. Implementation details.** One correction first: the advisor LoRA adapters matched no module of the served model, so they were never applied. Every reported number uses the base Qwen3.5-9B with role prompts; no number changes (common response, item 1). We will correct the paper and still report the adapters' training. Key settings (MedQA / MMLU-Pro / GPQA / AQuA-RAT where four values are given):

| Component | Setting |
|:--|:--|
| Advisors | Qwen3.5-9B base model, role prompts; greedy, ≤ 1,024 tokens |
| Manager SFT | LoRA r 16, α 32, dropout 0.05 (attention in the 8 full-attention layers, MLP in all 32); lr 1e-5, linear decay, no warm-up; 10 epochs; batch 8; bf16; max length 4,096 (code default). Logged for MedQA, MMLU-Pro and GPQA; AQuA-RAT has no training log, so its values are assumed |
| Labels | 315 / 331 / 148 / 402 rows; collection depth 2 / 2 / 1 / 3 (Table 4 prints 3 / 2 / 2 / 3; we will correct it); commit cap ρ 3 / 2 / 2 / 1 |
| FA-GRPO (RSI) | reward 1 if the answer after one forced call is correct, no call cost; no sampled rollouts (exact probability of each answer letter), 4 questions × 3 advisors per step; KL to the SFT model 0.1 (first answer), 0.1 (call decision), 0.02 (revision); lr 5e-6; ≤ 64 steps |
| Checkpoint rules | SFT: last epoch. FA-GRPO: last step that passes the guard, kept only if four dev checks (valid outputs, accuracy, calls, call gap) and one informativeness check on the GRPO pool pass. All fixed before testing |
| Decoding | greedy; manager ≤ 1,024 tokens and ≤ 3 calls, each advisor at most once |

The appendix will also give the exact split definitions and the full prompts with confidence fields and wrappers. Some paper-era records are missing (the MedQA checkpoint, the runs behind Tables 8–11, how ρ and depth were chosen, three training seeds, the AQuA-RAT training log); we mark each as not recorded.

**Suggestions.**
1. We will add one row per result with its collection depth, commit cap, seed, checkpoint and split. Single-pass S1 and RSI S1 are the same adapter run by two evaluation loops; the replays are listed separately; entries we could not recover are marked not recorded.
2. Agreed: we will call the 91.7% oracle a macro-average over the measured pools and search depths, not a property of larger or interactive systems.
3. Tokens per question (all model input and output; our checkpoint, one run), MARGENT / untrained manager, MedQA / MMLU-Pro / GPQA / AQuA-RAT: 2,222 / 11,770, 2,373 / 9,545, 2,807 / 10,098, 3,171 / 5,310 (**40–81% fewer**), at 1–5 points lower accuracy on MedQA, MMLU-Pro and AQuA-RAT. Offline, label collection probes the advisors on 400 training questions per benchmark (GPQA 200) and SFT uses 148–402 rows; we have not measured this cost and will add it, an amortization estimate and a common-budget accuracy–token curve.
4. "Success" (Table 12) keeps the records Dynamic keeps, with the same commit cap, but draws each target uniformly from the record's successful options instead of the shortest one: after a right first answer, commit or any one-call sequence that keeps it right; after a wrong one, any shortest successful sequence. It mostly turns commit targets into calls. The W1 control applies this rule once to the round-1 records.

Thank you again; these experiments made the paper's claims clearer and more accurate.
