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

## 2026-10-07/08: v2, v3 and the v3 rounds 4–5 (second pod)

Same layout (`/workspace/mcq_rsi`, worktree `/workspace/9.30-v3`), advisors on `127.0.0.1:18002`, one public HF repo per run.

- `mcq_supervisor_v2.py` / `mcq_supervisor_v3.py`: run the four `<bench>_v2` / `<bench>_v3` runs two at a time (GPU per run in `RUNS`,
  `ENVLINE` passes HF_HOME inline because tmux sessions inherit the tmux server's environment), resume once, start each locked test with
  `--reuse-test`, and honour an operator hold list in the state file.
- `mcq_supervisor_v3r5.py`: the same for the rounds 4–5 continuations (`<bench>_v3r5`), whose config is the source run's
  `rsi_run.json` → `config_full` with `rounds=5`, `continue_from` and the `_splits_r5` manifest (runbook §17).
- `backup_loops.sh`: keeps one hourly `backup_mcq_rsi_hf.py --run <run>` loop per `runs/*_v[0-9]*` directory (tmux `bk_<run>`).
- `verify_runs_hf.py <run>...`: checks every staged file is on HF with the same size (run on the pod before stopping it).
- `state_now.py`: one JSON line with every run's controller state (what `poll_runs.py` polls over ssh from a laptop).
- `letters.py <run>...`: draft-letter distribution and E/R/V tool mix of every locked-test eval of a run.
- `rounds_any.py <run>...`: dev metrics of every `sft_dev` stage (accuracy, draft accuracy, call rate, advisor mix).

## 2026-10-09: success_sft (v3 setting) and the label-quality chain (third pod)

- `bootstrap_v3s.sh`: one-shot bootstrap of a fresh pod (clone, setup, import, advisor-cache restore from the mmlu_pro_v3r5
  backup, download of the v3/v3r5 dynamic_sft labels, advisors, preflight) that then starts `mcq_labelq` (GPU 1: `scripts/
  mcq_label_quality.py` per benchmark, marker `logs/labelq_done`), `mcq_supervisor_v3s` and the backup loops. Markers in
  `logs/bootstrap/` make reruns skip finished steps.
- `mcq_supervisor_v3s.py`: runs `<bench>_v3s` (arm `success_sft`) one per GPU, waits for the label-quality chain before
  using GPU 1, records the delegated acknowledgements itself once `rsi_run.json` exists, resumes once, final-test with
  `--reuse-test`.
