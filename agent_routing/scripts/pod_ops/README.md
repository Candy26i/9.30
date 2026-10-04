# Pod operation scripts used for the 2026-10-03/04 MCQ RSI runs

Kept for the record; they hard-code the RunPod layout (`/workspace/mcq_rsi`, `/workspace/9.30`,
`/workspace/9.30-final2`) and run with the pod's experiment venv.

- `mcq_queue.py`: first overnight scheduler. It kept two main runs going, queued mmlu_pro → gpqa → aqua
  (`--arms dynamic,static`), auto-acknowledged a round-1 parity failure when the accuracy gap was ≤ 4 pt,
  resumed a failed run once, and ran queued `final:<bench>` locked tests.
- `mcq_supervisor.py`: second overnight supervisor. It resumed RSI runs once, started each finished run's
  final-test from a separate worktree with `--allow-code-change`, and retried forced (analysis) stages.
- `watch_runs.py`: exits as soon as any run or final-test changes state (an operator alert).

Operational lessons (see docs/MCQ_RSI_RUNBOOK.md for the supported procedure):
- Never run two inference managers next to vLLM on one 80 GB GPU. Give each run its own GPU via the
  operational `gpus` config key.
- GPQA needs `sft.max_seq_len` and `grpo.anchor_max_seq_len` of 16384, because one depth-2 label row is
  longer than 8k tokens.
- Never pull the main checkout while a run is active, because the run signature binds git HEAD. Run
  final-tests under newer code from a separate worktree with `final-test --allow-code-change REASON`.
