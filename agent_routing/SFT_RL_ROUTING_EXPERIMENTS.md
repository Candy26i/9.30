# SFT + GRPO routing-anchor experiments

Historical structured/MCQ documentation. For the current 9B math experiment, use the [math runbook](docs/MARGENT_END_TO_END_RUNBOOK.md) and [architecture](README.md). This document applies only to its stated benchmark/protocol.

This file covers the two experiments added for the MedQA 9B selective-routing study.  They are designed to reuse the existing marginal counterfactual data and the existing selective SFT checkpoint.  Do **not** recollect marginal branches, rerun the old pure-GRPO arm, or add a call-cost reward.

The task reward remains the existing binary terminal correctness reward in both experiments:

```text
R = 1[final answer is correct]
```

The only new term is an auxiliary supervised loss:

```text
L_total = L_GRPO + lambda * L_anchor
```

`--mgr_sft_anchor_coef 0` takes the untouched old GRPO path and instantiates the normal `GRPOTrainer`; the experimental trainer is used only when the coefficient is positive.

## What the two experiments test

### Experiment 1: full-turn SFT replay

Use:

```text
--mgr_sft_anchor_mode full
```

`L_anchor` replays the complete assistant target in `manager_sft_marginal.jsonl`.  For a CALL row this includes the current `DRAFT_ANSWER_*` and native tool call.  For a COMMIT row it includes `DRAFT_ANSWER_*` and `ANSWER_*`.

This is the lowest-risk implementation and tests whether continual replay of the counterfactual SFT behavior prevents GRPO from erasing selective routing and the tool protocol.  The limitation is that the auxiliary loss also anchors draft/answer generation, so it is not a routing-only intervention.

### Experiment 2: post-draft route-only replay

Use:

```text
--mgr_sft_anchor_mode route_only
```

No new `ROUTE_*` protocol is introduced.  The existing protocol already makes the routing boundary explicit:

```text
DRAFT_ANSWER_A
    -> native tool call       # CALL
    -> ANSWER_A               # COMMIT
```

For this experiment the prompt and `DRAFT_ANSWER_*` tokens are masked (`label=-100`).  Loss begins only after the draft, on the suffix that realizes CALL or COMMIT.  A CALL target therefore trains the native tool-call serialization/tool identity, while a COMMIT target trains the finalization suffix.  The model receives no auxiliary gradient for producing the initial draft answer itself.

This is the cleaner experiment for the paper: selective routing is supplied by counterfactual SFT, while outcome RL is still driven only by terminal correctness.

## Existing results that should NOT be rerun

Treat these as completed controls:

```text
8B LoRA GRPO, binary reward     -> all/near-all CALL
9B full-parameter GRPO         -> all/near-all CALL; protocol also degrades
fixed call-cost reward         -> opposite collapse toward no-CALL
selective marginal SFT         -> existing SFT accuracy/call-rate point
```

The new runs should start from the **same 9B SFT checkpoint**, use the **same GRPO prompt pool/split, seed, beta, batch size, generations, temperature, and number of steps** as the completed 9B pure-GRPO run, and differ only in the SFT anchor.  If the completed pure-GRPO command used `--exclude_sft_example_ids`, copy exactly the same exclusion flag into the new commands; do not change the RL prompt pool while testing the anchor.

## One-time setup and checks

The commands below assume the experimental code is in `/workspace/7.98_sft_rl/agent_routing`, while the already-generated SFT artifacts remain in `/workspace/7.98/agent_routing`.  If the SFT checkpoint has a different name, change only `SFT_CKPT`.

```bash
cd /workspace/7.98_sft_rl/agent_routing
source /workspace/research_7.41/agent_routing/.venv/bin/activate

export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1

export OLD_ROOT=/workspace/7.98/agent_routing
export SFT_CKPT=$OLD_ROOT/outputs/manager/medqa_marginal_9b_v2/sft_evolved
export ANCHOR_JSONL=$OLD_ROOT/outputs/manager/medqa_marginal_9b_v2/marginal_value/manager_sft_marginal.jsonl
export MEDQA_CACHE=$OLD_ROOT/outputs/data/medqa_us4_normalized.jsonl

test -e "$SFT_CKPT"
test -f "$ANCHOR_JSONL"
test -f "$MEDQA_CACHE"

python -m unittest tests.test_routing_anchor tests.test_marginal_value -v
python scripts/validate_sft_anchor.py \
  --tokenizer "$SFT_CKPT" \
  --anchor_jsonl "$ANCHOR_JSONL" \
  --mode full \
  --binding_mode environment \
  --max_seq_len 4096
python scripts/validate_sft_anchor.py \
  --tokenizer "$SFT_CKPT" \
  --anchor_jsonl "$ANCHOR_JSONL" \
  --mode route_only \
  --binding_mode environment \
  --max_seq_len 4096
python - <<'PY'
import inspect
import trl
from trl import GRPOTrainer
print("trl:", trl.__version__)
print("GRPOTrainer:", inspect.signature(GRPOTrainer.__init__))
print("compute_loss:", inspect.signature(GRPOTrainer.compute_loss))
PY
```

The two `validate_sft_anchor.py` calls load only the tokenizer, not the 9B model weights.  They should report `n_kept > 0`; ideally `n_dropped_no_target = 0`.  The existing repository uses `environment_factory` for the preferred stateful tool path.  Confirm that it appears in the installed `GRPOTrainer` signature, as it did for the environment used by the completed 9B run.  Do not upgrade TRL just for these experiments; use the same environment as the baseline.

Before training, keep the same vLLM subagent server used by the 9B baseline running at:

```text
http://localhost:8000
```

The examples below reserve GPU 0 for that server and GPUs 1,2,3 for full-parameter manager GRPO with the repository's existing Accelerate config.  If the completed 9B run used a different GPU layout or distributed config, keep its setup instead.

## Experiment 1 command: full-turn replay

First run only `lambda=0.03`.  This is a probe, not a claim that 0.03 is optimal.

```bash
cd /workspace/7.98_sft_rl/agent_routing
source /workspace/research_7.41/agent_routing/.venv/bin/activate
mkdir -p logs

HF_HOME=/workspace/hf_cache \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1,2,3 \
accelerate launch --config_file configs/accelerate_zero3.yaml \
  -m src.pipeline.cli train_manager_grpo \
  --base_model Qwen/Qwen3.5-9B \
  --teacher_id medqa_marginal_9b_v2_sftgrpo_full_l003 \
  --subagent_teacher_id ds_medqa_9B_sub \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size 1200 \
  --binding_mode environment \
  --mgr_init_adapter "$SFT_CKPT" \
  --mgr_full_parameter_rl \
  --mgr_bs 2 \
  --mgr_num_generations 6 \
  --mgr_max_completion_length 3072 \
  --mgr_temperature 1.0 \
  --mgr_grpo_beta 0.01 \
  --mgr_max_steps 200 \
  --subagent_server_url http://localhost:8000 \
  --mgr_sft_anchor_jsonl "$ANCHOR_JSONL" \
  --mgr_sft_anchor_mode full \
  --mgr_sft_anchor_coef 0.03 \
  --mgr_sft_anchor_batch_size 1 \
  --mgr_sft_anchor_max_seq_len 4096 \
  --mgr_output_dir outputs/manager/medqa_marginal_9b_v2_sftgrpo_full_l003/grpo \
  --mgr_use_wandb \
  --wandb_project agent_routing \
  --wandb_run_name medqa_9b_sftgrpo_full_l003 \
  --task_description 'You are a manager agent solving multiple-choice questions.' \
  2>&1 | tee logs/medqa_9b_sftgrpo_full_l003.log
```

If the old 9B pure-GRPO command did **not** use `3072 / 1.0 / beta=0.01 / bs=2 / G=6`, replace those values above with the exact old values.  The anchor must be the only experimental change.

## Experiment 2 command: route-only replay

Use the same initial coefficient and identical GRPO settings.  Only `mode`, run name, and output directory change.

```bash
cd /workspace/7.98_sft_rl/agent_routing
source /workspace/research_7.41/agent_routing/.venv/bin/activate
mkdir -p logs

HF_HOME=/workspace/hf_cache \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1,2,3 \
accelerate launch --config_file configs/accelerate_zero3.yaml \
  -m src.pipeline.cli train_manager_grpo \
  --base_model Qwen/Qwen3.5-9B \
  --teacher_id medqa_marginal_9b_v2_sftgrpo_route_l003 \
  --subagent_teacher_id ds_medqa_9B_sub \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size 1200 \
  --binding_mode environment \
  --mgr_init_adapter "$SFT_CKPT" \
  --mgr_full_parameter_rl \
  --mgr_bs 2 \
  --mgr_num_generations 6 \
  --mgr_max_completion_length 3072 \
  --mgr_temperature 1.0 \
  --mgr_grpo_beta 0.01 \
  --mgr_max_steps 200 \
  --subagent_server_url http://localhost:8000 \
  --mgr_sft_anchor_jsonl "$ANCHOR_JSONL" \
  --mgr_sft_anchor_mode route_only \
  --mgr_sft_anchor_coef 0.03 \
  --mgr_sft_anchor_batch_size 1 \
  --mgr_sft_anchor_max_seq_len 4096 \
  --mgr_output_dir outputs/manager/medqa_marginal_9b_v2_sftgrpo_route_l003/grpo \
  --mgr_use_wandb \
  --wandb_project agent_routing \
  --wandb_run_name medqa_9b_sftgrpo_route_l003 \
  --task_description 'You are a manager agent solving multiple-choice questions.' \
  2>&1 | tee logs/medqa_9b_sftgrpo_route_l003.log
```

At startup, route-only should print something like:

```text
[MANAGER_GRPO/SFT_ANCHOR] mode=route_only coef=0.03 rows=... kept=... dropped=... mean_supervised_tokens=...
```

A nonzero `dropped` count means truncation removed the action target for those rows.  If it is substantial, do not start the expensive run; increase `--mgr_sft_anchor_max_seq_len` or inspect the affected rows first.

## Evaluation commands

Use `eval_manager_tools`, not the no-tool `eval_manager`, because it already reports the routing metrics needed here: accuracy, valid-answer rate, tool-call rate, average calls, call rate conditioned on the initial draft being right/wrong, correction rate, and corruption rate.

Full-turn model:

```bash
CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli eval_manager_tools \
  --base_model Qwen/Qwen3.5-9B \
  --teacher_id medqa_marginal_9b_v2_sftgrpo_full_l003 \
  --subagent_teacher_id ds_medqa_9B_sub \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size 1200 \
  --test_size 200 \
  --binding_mode environment \
  --eval_manager_dir outputs/manager/medqa_marginal_9b_v2_sftgrpo_full_l003/grpo \
  --eval_n_samples 200 \
  --eval_temperature 0 \
  --eval_max_new_tokens 1024 \
  --eval_max_tool_calls 3 \
  --subagent_server_url http://localhost:8000 \
  --task_description 'You are a manager agent solving multiple-choice questions.' \
  2>&1 | tee logs/medqa_9b_sftgrpo_full_l003_eval.log
```

Route-only model:

```bash
CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli eval_manager_tools \
  --base_model Qwen/Qwen3.5-9B \
  --teacher_id medqa_marginal_9b_v2_sftgrpo_route_l003 \
  --subagent_teacher_id ds_medqa_9B_sub \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size 1200 \
  --test_size 200 \
  --binding_mode environment \
  --eval_manager_dir outputs/manager/medqa_marginal_9b_v2_sftgrpo_route_l003/grpo \
  --eval_n_samples 200 \
  --eval_temperature 0 \
  --eval_max_new_tokens 1024 \
  --eval_max_tool_calls 3 \
  --subagent_server_url http://localhost:8000 \
  --task_description 'You are a manager agent solving multiple-choice questions.' \
  2>&1 | tee logs/medqa_9b_sftgrpo_route_l003_eval.log
```

The reports are written under each experiment's `outputs/eval/<teacher_id>/manager_tool_eval_report.json`.

## Decision rule before spending on a lambda sweep

Run `lambda=0.03` for the two modes first.  Do not immediately launch six more jobs.

Interpret the first pair as follows:

```text
full works, route_only works
    -> strongest result: routing supervision itself can survive outcome RL.

full works, route_only fails
    -> full behavior/protocol replay is doing more than routing preservation;
       do not claim a routing-only result.

both collapse to all-CALL
    -> first inspect sft_anchor/loss and its weighted scale; lambda may simply
       be too weak. Do not change the reward.

route_only preserves selectivity but accuracy stays at the SFT point
    -> lambda may be too strong; reduce it before adding other RL tricks.
```

Only after the `0.03` pair establishes the direction should the same mode be swept at `0.01` and `0.10`.  The completed `lambda=0` pure-GRPO run is the zero-anchor endpoint and should not be rerun.  This gives the intended accuracy/call-cost curve without repeating the already-known all-CALL experiment.

## New CLI flags

```text
--mgr_sft_anchor_jsonl PATH
--mgr_sft_anchor_coef FLOAT
--mgr_sft_anchor_mode {full,route_only}
--mgr_sft_anchor_batch_size INT
--mgr_sft_anchor_max_seq_len INT
```

Training logs expose:

```text
sft_anchor/loss
sft_anchor/weighted_loss
```

The final `manager_run_config.json` also records the anchor path, mode, coefficient, batch size, maximum sequence length, and tokenization statistics so the two experiment arms remain auditable.
