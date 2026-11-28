**1. Call selection: MARGENT calls selectively, with far fewer calls.**

Thank you for raising uneven call selection and harmful roles; calling only when it helps is our central goal. We ran new test-set experiments with our checkpoint and three re-trainings with new seeds (mean ± SD over the four runs):

| | MedQA | MMLU-Pro | GPQA | AQuA-RAT |
|:--|--:|--:|--:|--:|
| MARGENT: accuracy (%) / calls | 78.6 ± 0.4 / 0.36 | 64.7 ± 1.0 / 0.35 | 51.8 ± 1.7 / 0.56 | 76.2 ± 0.7 / 0.75 |
| Runs above random questions, same calls (p < .05) | 2/4 | **4/4** | 1/4 | **4/4** |
| Verifier on least-confident answers, same calls (%) | 80.0 ± 0.9 | 63.3 ± 1.3 | 51.2 ± 1.5 | 74.4 ± 1.6 |
| Calls after a wrong / right first answer (%) | 61 / 28 | 51 / 20 | 55 / 58 | 83 / 60 |
| Calls to Extractor / Reasoner / Verifier (%) | 3 / 66 / 31 | 7 / 6 / 87 | 3 / 20 / 77 | 0 / 0 / 100 |
| Answers fixed / broken by calls (%) | 8.6 / 3.9 | 15.1 / 1.4 | 12.0 / 1.5 | 42.5 / 2.8 |
| Untrained manager: accuracy (%) / calls | 82.8 / 2.98 | 69.0 / 2.51 | 39.0 / 2.53 | 78.3 / 1.62 |
| Tokens per question, MARGENT / untrained | 2,222 / 11,770 | 2,373 / 9,545 | 2,807 / 10,098 | 3,171 / 5,310 |

*Rows 1–5: at most one call; rows 6–8: full system. Least-confident answers are ranked within the test set. The untrained manager is the base model with the same advisors, free format, up to 3 calls (valid answers 95 / 85 / 52 / 84%); it and the token counts are single runs.*

MARGENT is above random questions in all 16 runs, significantly in 11, including every run on **MMLU-Pro and AQuA-RAT**. Its calls fix **2 to 15 times** as many answers as they break, and on AQuA-RAT every call goes to the Verifier.

Its main strength is cost. Against the untrained manager it makes **54–88% fewer calls** and uses **40–81% fewer tokens**, while keeping **94–97% of its accuracy** on MedQA, MMLU-Pro and AQuA-RAT.

Selection is weaker on MedQA and absent on GPQA, calling on the least-confident answers is about as accurate, and outside AQuA-RAT the choice of role is no better than always using the Verifier. We will report these baselines; more even selection is future work.

**2. RSI: an exploratory extension; the core method needs one round of SFT.**

Thank you; we agree that RSI does not yet give steady gains across rounds. MedQA keeps no GRPO update, and later rounds can lower accuracy or raise cost.

**Our core method is single-round manager SFT and does not require RSI;** every result in this reply uses that round-1 manager. We will present RSI as an exploratory extension and remove any suggestion of steady gains across rounds.

**3. Attribution: SFT improves the first answer and teaches the manager to call less, and more after wrong answers.**

Thank you for asking whether the gains come from better first answers, better revision, or call selection. We ran controls with the same test questions, prompt, format and advisors (pooled over 1,354 questions; four runs):

| | Untrained | MARGENT |
|:--|--:|--:|
| First answer correct, same prompt (%) | 50.9 | 56.0 |
| Calls / question | 0.95 | 0.45 |
| Calls after a wrong first answer (%) | 92 | 62 |
| Calls after a right first answer (%) | 98 | 31 |

| MARGENT vs. | MARGENT − it (pts) | Its calls / question |
|:--|--:|--:|
| Untrained manager, Verifier on its least-confident answers | +2.9\* | 0.45 |
| Random questions, same calls | +4.0\* | 0.45 |
| Verifier on every question, given the manager's answer | −6.9\* | 1.00 |

\* p < .05 in every run.

The untrained manager calls on almost every question, right or wrong. MARGENT halves the calls and calls twice as often after a wrong answer. Training does not significantly change how the manager uses the Verifier's reply. At equal calls, most of the +2.9 comes from the better first answer, and MARGENT's choice of questions beats random but not a confidence ranking. We will report these parts separately rather than credit routing alone.

The paper's always-Verifier baseline did not see the manager's answer. With it, always-Verifier is 6.9 points more accurate at 2.2× the calls, so we withdraw the Figure 4 comparison; MARGENT is the lower-cost option. We did not test other role orders or external routers; we will list them as limitations.

Thank you; these experiments made the paper stronger.
