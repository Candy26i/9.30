**Table: FA-GRPO acceptance log for the v3+GRPO chains, one row per decision; round 1 is shared by the three arms. Δ columns are the candidate G_k minus the round's SFT manager S_k on the development pools (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254): accuracy in points and calls per question. Inf.: share of revision states with J ∈ (0.02, 0.98); a stage is informative at ≥ 15%. Step: selected checkpoint / steps run (64 planned); a run with fewer steps was stopped by the guard (every 8 steps on a 64-question probe) and rolled back to the last good checkpoint, 0 being the SFT checkpoint. A candidate is rejected when its dev accuracy falls below S_k − 1 pt, when the stage is uninformative, or when its dev evaluation fails the valid-answer-rate gate; a rejected round keeps G_k := S_k. Per-arm counts include the shared round-1 decision.**

| Benchmark | Run | Round | Arm | Accepted | ΔAcc. (pt) | ΔCalls | Inf. (%) | Step | Reason |
|:--|:--|:-:|:--|:-:|--:|--:|--:|:-:|:--|
| MedQA | v3g | 1 | shared | no | −1.5 | −0.02 | 3.6 | 24/32 | acc < S_k − 1 pt; uninformative |
|  | v3g | 2 | dynamic | no | −2.5 | −0.07 | 14.1 | 64/64 | acc < S_k − 1 pt; uninformative |
|  | v3g | 2 | static | no | −0.5 | −0.04 | 12.2 | 48/56 | uninformative |
|  | v3g | 2 | success | no | +2.0 | −0.04 | 13.9 | 64/64 | uninformative |
|  | v3g | 3 | dynamic | no | −0.5 | −0.03 | 9.1 | 64/64 | uninformative |
|  | v3g | 3 | static | no | −0.5 | +0.00 | 8.0 | 16/24 | validity gate; uninformative |
|  | v3g | 3 | success | no | −1.5 | −0.01 | 9.8 | 64/64 | acc < S_k − 1 pt; uninformative |
| *MedQA: accepted 0 / 7 decisions (dynamic 0/3, static 0/3, success 0/3)* | | | | | | | | | |
| MMLU-Pro | v3g_dyn | 1 | shared | no | +2.5 | +0.07 | 10.4 | 16/24 | uninformative |
|  | v3g_dyn | 2 | dynamic | yes | +0.5 | +0.01 | 25.8 | 64/64 | accepted |
|  | v3g_dyn | 3 | dynamic | yes | +3.5 | +0.05 | 21.5 | 40/48 | accepted |
|  | v3g_sta | 1 | shared | no | +0.5 | +0.01 | 9.4 | 16/24 | uninformative |
|  | v3g_sta | 2 | static | yes | −1.0 | +0.02 | 24.6 | 64/64 | accepted |
|  | v3g_sta | 3 | static | yes | +1.5 | +0.10 | 20.2 | 64/64 | accepted |
|  | v3g_suc | 1 | shared | no | +1.5 | +0.05 | 11.5 | 8/16 | validity gate; uninformative |
|  | v3g_suc | 2 | success | yes | +0.0 | −0.02 | 30.5 | 64/64 | accepted |
|  | v3g_suc | 3 | success | yes | +1.0 | +0.00 | 16.9 | 64/64 | accepted |
| *MMLU-Pro: accepted 6 / 9 decisions (dynamic 2/3, static 2/3, success 2/3)* | | | | | | | | | |
| GPQA | v3g | 1 | shared | yes | +0.0 | +0.00 | 32.3 | 0/8 | accepted |
|  | v3g | 2 | dynamic | yes | +1.0 | −0.04 | 28.5 | 64/64 | accepted |
|  | v3g | 2 | static | yes | +0.0 | +0.03 | 31.5 | 64/64 | accepted |
|  | v3g | 2 | success | no | −2.0 | +0.00 | 30.1 | 64/64 | acc < S_k − 1 pt |
|  | v3g | 3 | dynamic | yes | +1.0 | +0.01 | 34.4 | 64/64 | accepted |
|  | v3g | 3 | static | yes | +5.0 | −0.06 | 39.5 | 64/64 | accepted |
|  | v3g | 3 | success | yes | +3.0 | +0.00 | 48.8 | 64/64 | accepted |
| *GPQA: accepted 6 / 7 decisions (dynamic 3/3, static 3/3, success 2/3)* | | | | | | | | | |
| AQuA-RAT | v3g | 1 | shared | yes | −0.8 | −0.02 | 43.1 | 32/40 | accepted |
|  | v3g | 2 | dynamic | no | −4.3 | −0.11 | 52.1 | 24/32 | acc < S_k − 1 pt |
|  | v3g | 2 | static | yes | +0.8 | −0.01 | 37.5 | 64/64 | accepted |
|  | v3g | 2 | success | yes | +0.4 | −0.01 | 34.9 | 64/64 | accepted |
|  | v3g | 3 | dynamic | yes | +0.0 | +0.00 | 34.8 | 64/64 | accepted |
|  | v3g | 3 | static | yes | +0.8 | +0.00 | 29.5 | 16/24 | accepted |
|  | v3g | 3 | success | yes | +1.2 | +0.00 | 40.9 | 64/64 | accepted |
| *AQuA-RAT: accepted 6 / 7 decisions (dynamic 2/3, static 3/3, success 3/3)* | | | | | | | | | |
