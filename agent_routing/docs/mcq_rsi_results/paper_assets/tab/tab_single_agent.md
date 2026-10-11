**Table: tab_single_agent.** Single-model baselines and the multi-agent systems on the locked tests (MedQA 500, MMLU-Pro 500 (test pool), GPQA 100, AQuA-RAT 254). Acc. and Valid are percentages; Budget = generation budget in tokens; Tokens = prefill + output per question (for systems: manager and advisors). Letter-only rows must answer with the choice letter only (no reasoning); a forced choice over the option letters backs up unparsed outputs and was never needed. Valid = share of questions with a parseable answer. No-tools rows run the manager prompt with the tools removed (DRAFT then ANSWER, no advisor calls). Multi-agent rows are the v3+GRPO round-3 systems. Where the v3+GRPO run has no token count, S₁ + advisors uses the v3-chain run of the same S₁ checkpoint (MMLU-Pro and AQuA-RAT) and RSI success uses the v1 success runs, which use the same labelling rule and call an advisor on 93–100% of questions (MedQA, MMLU-Pro and AQuA-RAT). Bold: best accuracy per benchmark within each block.

| Variant | Budget | MedQA Acc. | MedQA Valid | MedQA Tokens | MMLU-Pro Acc. | MMLU-Pro Valid | MMLU-Pro Tokens | GPQA Acc. | GPQA Valid | GPQA Tokens | AQuA-RAT Acc. | AQuA-RAT Valid | AQuA-RAT Tokens |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Single model* | | | | | | | | | | | | | |
| Base, letter only | 16 | **74.6** | 100.0 | 275 | 53.0 | 100.0 | 265 | **45.0** | 100.0 | 262 | 41.7 | 100.0 | 153 |
| S₁, letter only | 16 | 74.4 | 100.0 | 275 | 53.8 | 100.0 | 265 | 44.0 | 100.0 | 262 | 42.1 | 100.0 | 153 |
| Dyn. r3 (no GRPO), letter only | 16 | 74.2 | 100.0 | 275 | **54.6** | 100.0 | 265 | 43.0 | 100.0 | 262 | **42.9** | 100.0 | 153 |
| S₁, no tools | 512 | 73.8 | 100.0 | 674 | 52.4 | 100.0 | 740 | 44.0 | 100.0 | 660 | 37.0 | 100.0 | 564 |
| Dyn. r3 (no GRPO), no tools | 512 | **74.6** | 100.0 | 673 | 53.4 | 100.0 | 740 | 42.0 | 100.0 | 660 | 35.4 | 100.0 | 564 |
| *Multi-agent systems* | | | | | | | | | | | | | |
| S₁ + advisors | -- | 78.8 | 100.0 | 2,222 | 63.6 | 100.0 | 2,373 | 54.0 | 100.0 | 2,807 | 77.2 | 100.0 | 3,180 |
| RSI dynamic + GRPO (r3) | -- | 78.6 | 100.0 | 1,568 | 68.6 | 100.0 | 2,355 | 55.0 | 100.0 | 3,053 | 77.6 | 100.0 | 3,127 |
| RSI success (r3) | -- | 80.2 | 100.0 | 3,969 | **71.8** | 100.0 | 4,137 | **58.0** | 100.0 | 4,310 | **82.7** | 100.0 | 3,738 |
| Base system (untrained manager) | -- | **82.8** | 95.0 | 11,770 | 69.0 | 85.0 | 9,545 | 39.0 | 52.0 | 10,098 | 78.3 | 83.9 | 5,310 |

