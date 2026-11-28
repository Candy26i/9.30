Thank you. We ran your controls on the test pools (1,354 questions; four training runs; new training controls one run each).

**Table A.** At most one call, same first answer, checkpoint, decoding and revision prompt (mean of four runs; SD ≤ 1.2).

| Rule | Acc. (%) | Calls |
|:--|--:|--:|
| First answer only | 56.0 | 0 |
| **MARGENT** | **71.0** | **0.45** |
| Random questions | 67.0 | 0.45 |
| Random Verifier subset | 67.6 | 0.45 |
| MARGENT's questions, all to Verifier | 71.7 | 0.45 |
| Confidence gate, ranked in test pool | 70.6 | 0.45 |
| Confidence gate, dev threshold, same rate | 70.5 | 0.44 |
| Confidence gate, best dev accuracy | 76.9 | 0.77 |
| Always Verifier, sees first answer | 77.9 | 1.00 |
| Always Verifier, paper's design | 75.6 | 1.00 |
| Always Reasoner / Extractor | 61.2 / 57.1 | 1.00 |

**W1.** Agreed; we separated the three effects.

| Effect (four runs) | Δ acc. (pts) |
|:--|--:|
| Revision turn, empty Verifier output | −0.5 |
| Verifier output over that turn | +22.3\* |
| SFT: first answer (same format) | +4.8 to +5.5\* |
| SFT: use of the Verifier's reply | +0.1 to +0.6 |

| Control (one run) | Acc. (%) | Calls |
|:--|--:|--:|
| MARGENT (our checkpoint) | 70.9 | 0.44 |
| Trained without call examples | 57.2 | 0 |
| Same + confidence gate | 71.6 | 0.44 |
| Success distillation, same records | 76.4 | 0.98 |

\* p < .05 in every run.

Shortest-path labels cut calls by 55% vs. success distillation (5.5 points lower accuracy). MARGENT calls after 62% of wrong vs. 31% of right first answers (untrained: 92% vs. 98%) and beats random questions in all 16 run–benchmark cases (11 significant; uneven on dev pools). It matches the confidence gate at equal calls; we will say it does not select better than confidence.

**W2.** Table A gives equal inputs. The first answer helps the Verifier on MMLU-Pro (+4.8) and AQuA-RAT (+5.5), not on MedQA or GPQA. MARGENT's questions beat a random Verifier subset (71.7 vs. 67.6; significant in every run except on GPQA). Always-Verifier is 6.9 points higher at 2.2× the calls, so we withdraw the Figure 4 comparison (your Figure 3); MARGENT gets two thirds of always-Verifier's gain over the first answer with 45% of the calls.

**W3.**

| Concern | Result |
|:--|:--|
| GPQA dev = test | Conceded. Disjoint 100-question pool: MARGENT 53.8 (Diamond-100: 51.8), always-Verifier 57.0 (n.s.) |
| AQuA-RAT 254 / 254 | Official validation vs. test split; no shared question |
| MMLU-Pro | Paper's pool (as reconstructed) shared 200 questions with dev, 196 with training; new disjoint pool used |
| Seeds | MARGENT 71.0 ± 0.2; vs. random and always-Verifier, same sign in all 16 cases |
| Checkpoints | Last epoch, none chosen; ρ and depth choice not recorded; MedQA retrained (original not archived) |
| ρ | Below always-Verifier for all 15 managers of a cap sweep (10 significant) |

RSI contrasts remain single runs.

**W4.** Correction: the advisor adapters were never applied (their parameter names matched no module); all numbers use base Qwen3.5-9B with role prompts, and none changes.

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| Label rows | 315 | 331 | 148 | 402 |
| Collection depth (Table 4: 3 / 2 / 2 / 3) | 2 | 2 | 1 | 3 |
| Commit cap ρ | 3 | 2 | 2 | 1 |
| Training log | yes | yes | yes | missing |

| Stage | Settings |
|:--|:--|
| Manager SFT | LoRA r 16, α 32, dropout 0.05; lr 1e-5, linear; 10 epochs; batch 8; bf16; last epoch |
| FA-GRPO | reward 1 if correct after one forced call; no sampled rollouts; 4 questions × 3 advisors per step; KL 0.1 / 0.1 / 0.02 (answer / decision / revision); lr 5e-6; ≤ 64 steps; kept only if 5 checks pass |
| Decoding | greedy; ≤ 1,024 tokens; ≤ 3 calls |

Full prompts, split definitions and per-condition rules go to the appendix; missing records are marked.

**Suggestions.** (1) We will add one row per result; single-pass S1 and RSI S1 are the same adapter. (2) Agreed. (3) Tokens per question (one run), MARGENT vs. untrained: 2.2k / 2.4k / 2.8k / 3.2k vs. 11.8k / 9.5k / 10.1k / 5.3k (40–81% fewer), at 1–5 points lower accuracy on three benchmarks; offline cost and a common-budget curve will be added. (4) Success = Dynamic's records, with each target drawn uniformly from the successes instead of the shortest.

Thank you again; this made the paper clearer and more accurate.
