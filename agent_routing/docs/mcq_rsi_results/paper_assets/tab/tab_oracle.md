**Table: Oracle headroom on the development pools.** Dev accuracy (%) of the manager policy at three checkpoints (S_1, round 3, round 5) against the per-question oracle, the best of {answer directly, one advisor call} taken from the forced evaluations of the same checkpoint. Draft only answers every question directly; Always Verifier calls the Verifier on every question. Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA 254. R3/R5 are rounds 3 and 5 of the dynamic arm without GRPO; GPQA has no round 5. The advisors are frozen, so the oracle does not move. Bold marks the best policy checkpoint per benchmark.

| Benchmark | n | Draft only | Always Verifier | Policy S_1 | Policy R3 | Policy R5 | Oracle S_1 | Oracle R3 | Oracle R5 | Oracle − policy S_1 | Oracle − policy R3 | Oracle − policy R5 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MedQA | 200 | 70.5 | 86.5 | **79.0** | **79.0** | 76.0 | 92.0 | 92.0 | 92.5 | +13.0 | +13.0 | +16.5 |
| MMLU-Pro | 200 | 55.5 | 67.0 | **62.5** | **62.5** | 61.5 | 79.0 | 77.5 | 76.5 | +16.5 | +15.0 | +15.0 |
| GPQA | 100 | 41.0 | 52.0 | 52.0 | **53.0** | — | 61.0 | 63.0 | — | +9.0 | +10.0 | — |
| AQuA-RAT | 254 | 39.4 | 77.2 | 73.2 | **77.6** | 72.4 | 81.9 | 82.3 | 81.9 | +8.7 | +4.7 | +9.5 |
