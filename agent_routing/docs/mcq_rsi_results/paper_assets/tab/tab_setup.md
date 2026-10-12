**Table: Experimental setup. S_1 is the paper's one-round SFT manager; S_k and G_k are the round-k manager before and after FA-GRPO. The GPQA sequence-length override accommodates its longer depth-2 label rows. The MMLU-Pro *test_paper* pool overlaps training pools and is used only in the appendix.**

| Component | Setting |
|:--|:--|
| **Models** | |
| Base model | Qwen3.5-9B (non-thinking; enable_thinking=False everywhere) |
| Advisors | base model + role prompts (Extractor, Reasoner, Verifier), served with vLLM, outputs cached |
| Manager | LoRA on Qwen3.5-9B; S_1 = the paper's one-round SFT manager |
| **Rounds and data collection** | |
| Rounds | 3 (each: collect, SFT, FA-GRPO); rounds 4–5 for the no-GRPO v3 chain (MedQA, MMLU-Pro, AQuA) |
| Questions collected per round | 400 (GPQA reuses its 200 round-1 roots) |
| Max depth | 2 |
| Commit:rescue balancing ρ | MedQA 3.0, MMLU-Pro 2.0, GPQA 2.0, AQuA-RAT 1.0 |
| **SFT (each round)** | |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Batch | 1 per device × 8 gradient-accumulation steps (effective 8) |
| Max sequence length | 4,096 tokens (GPQA 16,384) |
| Seed / context / draft supervision | 42 / paper (tool schemas rendered as in the S_1 prompt) / none in v3 (route-only loss); the draft letter in v1 |
| **FA-GRPO (each round)** | |
| Steps | 64 steps × 4 questions per step (256 questions per round; GPQA pool 146) |
| Learning rate / max grad norm / seed | 5e-6 / 1.0 / 0 |
| β (root / dec / rev) | 0.1 / 0.1 / 0.02 |
| Anchor λ | 0.03 (4 anchor rows per step); estimator: exact |
| Guard | every 8 steps on a 64-question probe: call-rate delta ≤ 0.10 and KL_dec ≤ 0.05, otherwise roll back to the last good step |
| Informative fraction | ≥ 0.15 of revision states with J ∈ (0.02, 0.98) |
| **Acceptance and gates** | |
| GRPO acceptance | dev acc ≥ S_k − 1 pt; calls ≤ S_k + 0.15; call gap ≥ 0.5 × S_k gap; ≥ 15% informative revision states (J ∈ (0.02, 0.98)); else G_k := S_k |
| Gates | accuracy tolerance 0.01; extra calls 0.15; gap fraction 0.5; SFT max calls 1.5; SFT max accuracy drop 0.03 |
| **Pools** | |
| Development | MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254 |
| Locked test | MedQA 500, MMLU-Pro 500 (held-out split), GPQA 100, AQuA-RAT 254 |
| **Arms** | |
| dynamic | re-collect with G_{k-1} each round, marginal-value labels, SFT from G_{k-1}, then FA-GRPO |
| static | reuse S_1's labels every round, SFT from G_{k-1}, then FA-GRPO |
| success | like dynamic but any trajectory ending in a correct answer is a demonstration (shortest path not required) |
| **Versions** | |
| v1 | paper SFT target (draft letter supervised) + GRPO |
| v3 | draft letter never supervised (route-only loss), no GRPO |
| v3+GRPO (main) | v3 loss + FA-GRPO each round |
