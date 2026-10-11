**A. Original MARGENT, MedQA development, three training seeds**

| | Seed 1 | Seed 2 | Seed 3 | Mean ± SD |
|:--|--:|--:|--:|--:|
| Accuracy (%) | 80.5 | 81.5 | 79.5 | 80.5 ± 1.0 |
| Questions with a call (%) | 52.5 | 55.5 | 47.0 | 51.7 ± 4.3 |

**B. Paired contrasts on the locked tests: Δ accuracy in points (p)**

| Contrast | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| Dynamic − Static (R3, SFT only) | 0.0 (1.00) | +6.4 (<0.001) | −1.0 (0.83) | +3.1 (0.19) |
| Dynamic: with − without GRPO (R3) | 0.0 (1.00) | +4.2 (0.003) | +1.0 (0.78) | −0.4 (0.93) |
| Dynamic: R5 − R3 (SFT only) | −1.2 (0.33) | −3.0 (0.04) | — | −6.3 (<0.001) |
| S₁ evaluated twice (GRPO run − SFT run) | +0.4 (0.28) | −0.4 (0.28) | 0.0 (1.00) | −0.4 (0.74) |

A: depth 2, 400 questions, ρ = 3; sample SD over seeds. B: single RSI runs; two-sided paired bootstrap (10,000 resamples, unadjusted); – = no round 5.
