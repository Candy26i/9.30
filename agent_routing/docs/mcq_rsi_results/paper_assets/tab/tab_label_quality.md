**Table: tab_label_quality.** Label quality by round on the development pools. Each entry is a manager SFT'd from S_1 on that round's labels alone: dev accuracy (%) and advisor calls per question. Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254. r1 is the static arm's data; r_k is re-collected by S_{k-1}; GPQA labels stop at r3. Experiment D: labels collected on the same questions by S_k (before GRPO) and by G_k (after GRPO), both trained from S_1; S_k's labels are the ones used at chain round r_{k+1}. Δ is G_k minus S_k. Best chain-round accuracy per benchmark in bold; MedQA has no experiment-D runs.

| Labels | MedQA Acc. | MedQA Calls | MMLU-Pro Acc. | MMLU-Pro Calls | GPQA Acc. | GPQA Calls | AQuA-RAT Acc. | AQuA-RAT Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Chain rounds (one round's labels, SFT from S_1)* | | | | | | | | |
| S_1 | **79.0** | 0.37 | 62.5 | 0.32 | 53.0 | 0.62 | 73.2 | 0.79 |
| r1 | 78.0 | 0.25 | 61.5 | 0.26 | **55.0** | 0.69 | 72.4 | 0.74 |
| r2 | **79.0** | 0.28 | 56.5 | 0.11 | 53.0 | 0.62 | 72.8 | 0.78 |
| r3 | 76.5 | 0.09 | 63.0 | 0.23 | 52.0 | 0.56 | 73.6 | 0.81 |
| r4 | **79.0** | 0.18 | **64.0** | 0.23 | -- | -- | 77.2 | 0.80 |
| r5 | 76.0 | 0.20 | 63.5 | 0.21 | -- | -- | **78.7** | 0.90 |
| *Experiment D: labels collected by S_k (before GRPO) vs G_k (after GRPO)* | | | | | | | | |
| by S_1 (r2) | -- | -- | -- | -- | 54.0 | 0.61 | 72.4 | 0.77 |
| by G_1 (r2) | -- | -- | -- | -- | 58.0 | 0.51 | 69.3 | 0.66 |
| Δ (G_1 − S_1) | -- | -- | -- | -- | +4.0 | −0.10 | −3.1 | −0.11 |
| by S_2 (r3) | -- | -- | 65.0 | 0.31 | -- | -- | 75.6 | 0.81 |
| by G_2 (r3) | -- | -- | 65.0 | 0.28 | -- | -- | 74.4 | 0.85 |
| Δ (G_2 − S_2) | -- | -- | 0.0 | −0.03 | -- | -- | −1.2 | +0.03 |
