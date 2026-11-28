Thank you for the detailed review. We ran the controls you suggested on the test pools (1,354 questions), averaging over four training runs (our checkpoint plus three re-trainings); the two new training controls were trained once.

**1. Isolating the benefit of learned delegation**

You are right that Table 1 mixes three effects. Below, every rule starts from the same first answer, checkpoint, decoding and revision prompt, with at most one call:

| Rule | Accuracy (%) | Calls per question |
|:--|--:|--:|
| First answer only | 56.0 | 0 |
| **MARGENT** | **71.0** | **0.45** |
| Random questions, same number of calls | 67.0 | 0.45 |
| Random Verifier subset, same number of calls | 67.6 | 0.45 |
| MARGENT's questions, all sent to the Verifier | 71.7 | 0.45 |
| Confidence gate, threshold set on dev | 70.5 | 0.44 |
| Confidence gate, tuned on dev for best accuracy | 76.9 | 0.77 |
| Always Verifier, given the first answer | 77.9 | 1.00 |
| Always Verifier, without it (as in the paper) | 75.6 | 1.00 |
| Always Reasoner / always Extractor | 61.2 / 57.1 | 1.00 |

The gain comes from the Verifier, not the extra revision: revising on an empty Verifier output changes accuracy by −0.5 points, the Verifier's output adds 22.3. SFT improves the first answer (+4.8 to +5.5 points over the untrained model) and teaches when to call: MARGENT calls after 62% of wrong first answers and 31% of right ones, the untrained model after 92% and 98%.

The two controls you asked for (one run each):

| Control | Accuracy (%) | Calls per question |
|:--|--:|--:|
| MARGENT (our checkpoint) | 70.9 | 0.44 |
| Manager trained without call examples | 57.2 | 0 |
| The same manager with a confidence gate | 71.4 | 0.45 |
| Success distillation on the same records | 76.4 | 0.98 |

Shortest-path labels teach the manager to stop: 55% fewer calls than success distillation, for 5.5 points less accuracy. MARGENT beats random questions on every benchmark in every run (significantly in 11 of 16 cases; less evenly on dev), but is about as good as a confidence gate, which we will state plainly.

**2. The always-Verifier baseline**

Thank you for catching this. The first table gives every rule the same input. Seeing the first answer helps the Verifier on MMLU-Pro (+4.8) and AQuA-RAT (+5.5), not on MedQA or GPQA. MARGENT's questions beat a random Verifier subset (71.7 vs. 67.6; significant in every run except on GPQA). With equal inputs, always-Verifier is 6.9 points more accurate at 2.2× the calls, so we withdraw the Figure 4 comparison (your Figure 3). MARGENT gets two thirds of always-Verifier's improvement over the first answer with 45% of the calls; we will present this as an accuracy–cost trade-off.

**3. Independence and robustness**

| Concern | What we found |
|:--|:--|
| GPQA: Diamond-100 used for dev and test | True. On a separate, non-overlapping pool: MARGENT 53.8 (51.8 on Diamond-100), always-Verifier 57.0, not significant |
| AQuA-RAT: 254 / 254 | Official validation vs. test split; no shared question |
| MMLU-Pro | Our reconstruction of the paper's test pool overlaps dev (200) and training (196); new results use a disjoint pool |
| Seeds | MARGENT 71.0 ± 0.2; above random and below always-Verifier in every run |
| Checkpoints | Last epoch kept. The choice of ρ and depth was not recorded; the original MedQA checkpoint was not archived, so we retrained it |

Across 15 managers of a commit-cap sweep, learned calling stays below always-Verifier (significantly for 10), so that result holds at every cap tested. RSI comparisons remain single runs, and we will mark them so.

**4. Implementation details**

First, a correction: the advisor LoRA adapters were never applied (their parameter names did not match the served model), so every number uses the base Qwen3.5-9B with role prompts; no number changes.

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| Label rows | 315 | 331 | 148 | 402 |
| Collection depth | 2 | 2 | 1 | 3 |
| Commit cap ρ | 3 | 2 | 2 | 1 |
| Training log available | yes | yes | yes | no |

(Table 4's depths, 3 / 2 / 2 / 3, will be corrected.) Manager SFT: LoRA r 16, α 32, dropout 0.05; lr 1e-5; 10 epochs; batch 8; last epoch. FA-GRPO: reward 1 if correct after one forced call; no sampling, 4 questions × 3 advisors per step; KL 0.1 / 0.1 / 0.02; lr 5e-6; ≤ 64 steps; five acceptance checks. Decoding: greedy, ≤ 1,024 tokens, ≤ 3 calls. The appendix adds full prompts, split definitions and per-condition rules, marking what we could not recover.

**Your suggestions**

1. *Configuration table.* We will add one, with a row per result. Single-pass S1 and RSI S1 are the same adapter run by two evaluation loops; the replays are listed separately.
2. *The 91.7% oracle.* Agreed. We will always describe it as a macro-average over the measured pools and search depths.
3. *Cost.* MARGENT uses 40–81% fewer tokens than the untrained manager (2.2k–3.2k vs. 5.3k–11.8k per question, one run), at 1–5 points lower accuracy on three benchmarks. We will add the offline collection and training cost and a common-budget accuracy–token curve.
4. *"Success" in Table 12.* It trains on the same records as Dynamic, but each target is drawn uniformly from all successful options instead of being the shortest one.

Thank you again; these experiments made the paper clearer and more accurate.
