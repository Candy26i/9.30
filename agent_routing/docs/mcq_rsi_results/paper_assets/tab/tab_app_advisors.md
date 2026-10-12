| Benchmark | Advisor | Base Acc. | Paper LoRA Acc. | Paper LoRA Δ (p) | Retrained LoRA Acc. | Retrained LoRA Δ (p) |
|:--|:--|--:|--:|--:|--:|--:|
| MedQA (200) | Extractor | 87.0 | 89.0 | +2.0 (0.39) | 89.0 | +2.0 (0.45) |
|  | Reasoner | 89.5 | 87.5 | −2.0 (0.48) | 87.0 | −2.5 (0.27) |
|  | Verifier | 90.5 | 87.0 | −3.5 (0.14) | 86.5 | −4.0 (0.10) |
| MMLU-Pro (200) | Extractor | 76.0 | 75.5 | −0.5 (1.00) | 75.5 | −0.5 (1.00) |
|  | Reasoner | 76.5 | 76.0 | −0.5 (1.00) | 75.0 | −1.5 (0.63) |
|  | Verifier | 76.0 | 74.5 | −1.5 (0.65) | 72.5 | −3.5 (0.21) |
| GPQA (100) | Extractor | 61.0 | 60.0 | −1.0 (1.00) | 62.0 | +1.0 (1.00) |
|  | Reasoner | 62.0 | 60.0 | −2.0 (0.77) | 61.0 | −1.0 (1.00) |
|  | Verifier | 61.0 | 64.0 | +3.0 (0.58) | 60.0 | −1.0 (1.00) |
| AQuA-RAT (254) | Extractor | 84.3 | 85.4 | +1.2 (0.45) | 85.8 | +1.6 (0.29) |
|  | Reasoner | 84.3 | 85.4 | +1.2 (0.55) | 84.6 | +0.4 (1.00) |
|  | Verifier | 86.6 | 84.6 | −2.0 (0.27) | 83.5 | −3.1 (0.08) |
| Pooled (754) | Extractor | 79.7 | 80.4 | +0.7 (0.54) | 80.8 | +1.1 (0.34) |
|  | Reasoner | 80.6 | 80.1 | −0.5 (0.70) | 79.6 | −1.1 (0.33) |
|  | Verifier | 81.4 | 79.8 | −1.6 (0.16) | 78.2 | −3.2 (0.006) |

Advisor A/B on the development pools: a fixed base-model solver re-answers after one advisor output; Δ is versus the base model with the same role prompt (exact sign test). Paper LoRA is the paper's adapter loaded correctly; as served in the paper (vLLM, unrenamed weight keys) the Verifier adapter reproduced the base model on 20/20 prompts.
