**Table: tab_selfcheck.** Self-check signals of the S_1 manager on the development pools (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254). AUROC: how well each draft-confidence signal ($1-p_{\max}$, $-$margin, entropy, $1-$agreement; higher means less confident in the draft letter) predicts that the draft is wrong; 0.5 is no signal. Router: call the Verifier when the signal exceeds a threshold chosen so that the call rate matches S_1's (Calls, advisor calls per question); accuracy in %. S_1 is the learned policy at that call rate; Always Verifier calls the Verifier on every question (1.00 calls). Bold: best accuracy at the matched call budget (routers and S_1); Always Verifier uses a larger budget and is not bolded.

| Benchmark | n | AUROC 1 − p_max | AUROC −margin | AUROC Entropy | AUROC 1 − agreement | Router 1 − p_max | Router −margin | Router Entropy | Router 1 − agreement | S_1 acc. | S_1 calls | Always Verifier |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MedQA | 200 | 0.717 | 0.716 | 0.719 | 0.580 | **82.5** | 82.0 | **82.5** | 77.5 | 79.0 | 0.37 | 86.5 |
| MMLU-Pro | 200 | 0.473 | 0.467 | 0.487 | 0.602 | 59.0 | 58.0 | 57.5 | 61.0 | **62.5** | 0.32 | 67.0 |
| GPQA | 100 | 0.647 | 0.640 | 0.659 | 0.660 | 49.0 | 48.0 | 49.0 | 51.0 | **52.0** | 0.62 | 52.0 |
| AQuA-RAT | 254 | 0.576 | 0.561 | 0.596 | 0.562 | 70.9 | 71.3 | 70.5 | 70.1 | **73.2** | 0.79 | 77.2 |
