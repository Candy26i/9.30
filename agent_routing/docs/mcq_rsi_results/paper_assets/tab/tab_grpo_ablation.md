**Table: FA-GRPO ablation.** FA-GRPO ablation on the locked tests (MedQA 500, MMLU-Pro 500 test, GPQA 100, AQuA-RAT 254). Cells: accuracy in % (mean advisor calls per question) of the round-3 manager trained with SFT only (v3 chain) or with SFT + FA-GRPO in every round (v3+GRPO). Δ = with minus without GRPO in accuracy points, p from a two-sided paired bootstrap. Bold: best accuracy in the row. Accepted: FA-GRPO stage decisions accepted out of three rounds per arm (round 1 is shared by the three arms except on MMLU-Pro, which has one run per arm); the MMLU-Pro test_paper row evaluates the same managers on the appendix pool, which overlaps the training pools. Rejection reasons may co-occur; the accuracy gate is dev accuracy below S_k minus 1 point, the dev gate the valid-answer-rate check.

| Benchmark | Dynamic, SFT only | Dynamic, + FA-GRPO | Dynamic Δ (p) | Static, SFT only | Static, + FA-GRPO | Static Δ (p) | Accepted dyn. | Accepted static | Accepted succ. |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MedQA | 78.6 (0.15) | 78.6 (0.17) | +0.0 (1.000) | 78.6 (0.27) | **79.8** (0.28) | +1.2 (0.130) | 0/3 | 0/3 | 0/3 |
| MMLU-Pro (test) | 64.4 (0.27) | **68.6** (0.36) | +4.2 (0.003) | 58.0 (0.26) | 61.4 (0.26) | +3.4 (0.019) | 2/3 | 2/3 | 2/3 |
| MMLU-Pro (test_paper) | 59.8 (0.27) | **64.6** (0.39) | +4.8 (<0.001) | 59.0 (0.29) | 58.0 (0.28) | -1.0 (0.473) | -- | -- | -- |
| GPQA | 54.0 (0.58) | **55.0** (0.60) | +1.0 (0.780) | **55.0** (0.49) | 50.0 (0.47) | -5.0 (0.116) | 3/3 | 3/3 | 2/3 |
| AQuA-RAT | **78.0** (0.79) | 77.6 (0.77) | -0.4 (0.933) | 74.8 (0.73) | 73.6 (0.72) | -1.2 (0.516) | 2/3 | 3/3 | 3/3 |

Rejection reasons per benchmark:

- MedQA: 0 of 7 FA-GRPO stage decisions accepted; 7 rejected (uninformative stage 7, accuracy gate 3, dev gate 1).
- MMLU-Pro: 6 of 9 FA-GRPO stage decisions accepted; 3 rejected (uninformative stage 3, dev gate 1).
- GPQA: 6 of 7 FA-GRPO stage decisions accepted; 1 rejected (accuracy gate 1).
- AQuA-RAT: 6 of 7 FA-GRPO stage decisions accepted; 1 rejected (accuracy gate 1).
