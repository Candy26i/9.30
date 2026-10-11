**Table: no-GRPO chain (v3), locked test at rounds 3 and 5.** Locked-test accuracy (%) and calls per question for the no-GRPO chain (v3) at rounds 3 and 5, by arm. S₁ is round 1. Δ(r5−r3) compares round 5 with round 3 of the same arm; Δ vs S₁ compares round 5 with S₁; dyn−static compares the two arms at round 5. p: two-sided paired bootstrap (10k resamples). Locked tests: MedQA 500, MMLU-Pro 500 (test), GPQA 100, AQuA 254. GPQA ran three rounds only, so its round-5 cells are empty. MMLU-Pro test_paper overlaps the training pools and is shown for completeness. Bold marks the best accuracy per pool.

| Benchmark | Arm | S₁ acc | S₁ calls | r3 acc | r3 calls | r5 acc | r5 calls | Δacc (r5−r3) | p | Δacc vs S₁ (r5) | p | Δacc dyn−static (r5) | p |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MedQA | dynamic | 78.4 | 0.36 | 78.6 | 0.15 | 77.4 | 0.17 | −1.2 | 0.333 | −1.0 | 0.485 | −3.6 | 0.008 |
|  | static |  |  | 78.6 | 0.27 | **81.0** | 0.37 | +2.4 | 0.027 | +2.6 | 0.023 |  |  |
| MMLU-Pro | dynamic | 64.0 | 0.35 | **64.4** | 0.27 | 61.4 | 0.30 | −3.0 | 0.043 | −2.6 | 0.182 | +3.4 | 0.049 |
|  | static |  |  | 58.0 | 0.26 | 58.0 | 0.25 | +0.0 | 1.000 | −6.0 | <0.001 |  |  |
| GPQA | dynamic | 54.0 | 0.52 | 54.0 | 0.58 | – | – | – | – | – | – | – | – |
|  | static |  |  | **55.0** | 0.49 | – | – | – | – | – | – |  |  |
| AQuA-RAT | dynamic | 77.6 | 0.78 | **78.0** | 0.79 | 71.7 | 0.72 | −6.3 | <0.001 | −5.9 | 0.009 | −5.5 | 0.027 |
|  | static |  |  | 74.8 | 0.73 | 77.2 | 0.76 | +2.4 | 0.207 | −0.4 | 0.931 |  |  |
| MMLU-Pro (test_paper) | dynamic | 59.6 | 0.35 | **59.8** | 0.27 | 58.6 | 0.29 | −1.2 | 0.453 | −1.0 | 0.591 | +4.8 | 0.006 |
|  | static |  |  | 59.0 | 0.29 | 53.8 | 0.26 | −5.2 | <0.001 | −5.8 | <0.001 |  |  |
