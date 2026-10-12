| Policy | MedQA E | MedQA R | MedQA V | MMLU-Pro E | MMLU-Pro R | MMLU-Pro V | GPQA E | GPQA R | GPQA V | AQuA-RAT E | AQuA-RAT R | AQuA-RAT V |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S₁ | 0.2 | 21.2 | 14.2 | 1.8 | 3.6 | 30.0 | 1.0 | 9.0 | 42.0 | 0.0 | 0.4 | 77.6 |
| Dynamic R3 (SFT) | 0.0 | 14.8 | 0.0 | 0.0 | 1.2 | 25.8 | 1.0 | 21.0 | 36.0 | 0.0 | 0.0 | 79.1 |
| Dynamic R5 (SFT) | 7.2 | 9.6 | 0.4 | 0.2 | 7.6 | 22.0 | — | — | — | 0.0 | 0.4 | 71.7 |
| Static R3 (SFT) | 0.0 | 12.4 | 14.6 | 2.2 | 3.2 | 20.8 | 0.0 | 15.0 | 34.0 | 0.4 | 2.0 | 70.5 |
| Static R5 (SFT) | 1.0 | 10.8 | 25.0 | 3.8 | 4.6 | 16.8 | — | — | — | 0.4 | 2.8 | 73.2 |
| Dynamic R3 (+GRPO) | 0.0 | 16.8 | 0.2 | 0.0 | 4.0 | 31.8 | 0.0 | 23.0 | 37.0 | 0.0 | 0.0 | 76.8 |
| Static R3 (+GRPO) | 0.6 | 11.4 | 16.2 | 0.6 | 2.4 | 23.4 | 1.0 | 12.0 | 34.0 | 0.4 | 2.0 | 70.1 |
| Success R3 (+GRPO) | 5.8 | 90.0 | 3.6 | 9.4 | 40.2 | 50.0 | 0.0 | 55.0 | 45.0 | 0.0 | 0.0 | 100.0 |

Advisor calls per 100 locked-test questions by role (E = Extractor, R = Reasoner, V = Verifier).
