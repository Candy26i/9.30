**A. One SFT from the same S₁: accuracy / calls**

| Training labels | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| S₁ (no extra training) | 79.0 / 0.37 | 62.5 / 0.32 | 53.0 / 0.62 | 73.2 / 0.79 |
| R1 labels (static) | 78.0 / 0.25 | 61.5 / 0.26 | 55.0 / 0.69 | 72.4 / 0.74 |
| R2 labels, from S₁ | 79.0 / 0.29 | 56.5 / 0.11 | 53.0 / 0.62 | 72.8 / 0.78 |
| R3 labels, from S₂ | 76.5 / 0.09 | 63.0 / 0.23 | 52.0 / 0.56 | 73.6 / 0.81 |
| R4 labels, from S₃ | 79.0 / 0.19 | 64.0 / 0.23 | — | 77.2 / 0.80 |
| R5 labels, from S₄ | 76.0 / 0.20 | 63.5 / 0.21 | — | 78.7 / 0.90 |

**B. Labels re-collected on the same questions before / after GRPO: accuracy**

| Collector | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| Round-1 model, S₁ / G₁ | — | — | 54.0 / 58.0 | 72.4 / 69.3 |
| Round-2 model, S₂ / G₂ | — | 65.0 / 65.0 | — | 75.6 / 74.4 |

Development accuracy (%) and calls per question of a manager trained once from S₁ on one label set (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254 questions; – = not collected). B uses the SFT + GRPO run, so its S_k labels are a separate collection from A's.
