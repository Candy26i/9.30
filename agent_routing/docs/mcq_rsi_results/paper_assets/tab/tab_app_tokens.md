| Benchmark | Policy | Manager input | Manager output | Advisor input | Advisor output | Total | Calls |
|:--|:--|--:|--:|--:|--:|--:|--:|
| MedQA | Base system | 8,200 | 179 | 1,387 | 2,004 | 11,770 | 2.98 |
|  | S₁ | 1,741 | 27 | 179 | 274 | 2,222 | 0.37 |
|  | Dynamic R3 (+GRPO) | 1,359 | 18 | 77 | 115 | 1,568 | 0.17 |
| MMLU-Pro | Base system | 6,620 | 317 | 1,145 | 1,464 | 9,545 | 2.51 |
|  | S₁ | 1,862 | 28 | 193 | 290 | 2,373 | 0.35 |
|  | Dynamic R3 (+GRPO) | — | — | — | — | 2,355 | 0.36 |
| GPQA | Base system | 6,770 | 520 | 1,144 | 1,664 | 10,098 | 2.53 |
|  | S₁ | 2,064 | 36 | 247 | 460 | 2,807 | 0.52 |
|  | Dynamic R3 (+GRPO) | 2,212 | 38 | 281 | 521 | 3,053 | 0.60 |
| AQuA-RAT | Base system | 3,698 | 393 | 547 | 671 | 5,310 | 1.62 |
|  | S₁ | 2,278 | 51 | 315 | 536 | 3,180 | 0.78 |
|  | Dynamic R3 (+GRPO) | — | — | — | — | 3,127 | 0.77 |

**Advisor tokens per call (input + output)**

| Benchmark | Extractor | Reasoner | Verifier |
|:--|--:|--:|--:|
| MedQA | 950 | 1,102 | 1,365 |
| MMLU-Pro | 673 | 1,297 | 1,217 |
| GPQA | 859 | 1,170 | 1,406 |
| AQuA-RAT | 468 | 847 | 1,072 |

Mean tokens per question on the locked tests by who reads or writes them (Dynamic R3: only totals kept on MMLU-Pro and AQuA-RAT; S₁ there from its SFT-run evaluation). Per-call advisor tokens come from the base system, which makes 1.6—3.0 calls per question.
