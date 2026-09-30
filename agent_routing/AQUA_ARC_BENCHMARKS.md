# AQuA-RAT and ARC-Challenge pipeline

Historical structured/MCQ documentation. For the current 9B math experiment, use the [math runbook](docs/MARGENT_END_TO_END_RUNBOOK.md) and [architecture](README.md). This document applies only to its stated benchmark/protocol.

This guide adds two independent benchmark runs to the existing routing
pipeline. It does not mix domains: each benchmark gets its own normalized
cache, advisor checkpoints, marginal-value data, manager checkpoint, and
evaluation output.

The common path is:

```text
Hugging Face or local source
  -> benchmark loader
  -> StandardRow JSONL
  -> official train/dev/test splits
  -> advisor SFT
  -> marginal-value manager SFT
  -> optional binary-reward GRPO
  -> held-out evaluation
```

Downstream stages use the same code as MedQA. Only `--benchmark`, the
normalized-cache flag, run ids, and task description change.

## 1. Data contracts

### AQuA-RAT

- Source: `deepmind/aqua_rat`, config `raw`.
- Five answer choices are converted to `A` through `E`.
- Embedded prefixes such as `A)` and `(A)` are removed from option text.
- `validation` is normalized to `dev`.
- The gold `rationale` is never stored in `StandardRow`, `context`, or the
  normalized JSONL. Only a boolean `has_gold_rationale` is retained. This
  prevents answer leakage into manager and advisor prompts.

### ARC-Challenge

- Source: `allenai/ai2_arc`, config `ARC-Challenge`.
- `choices.label` and `choices.text` are paired in official order, then source
  labels are mapped to continuous `A/B/C/...` keys.
- Both numeric and alphabetic source labels are supported.
- Questions with three, four, or five choices are kept; the manager protocol
  already builds allowed answer tokens from each row's actual choice keys.
- `validation` is normalized to `dev`.

The official ARC-Challenge split sizes are 1,119 train, 299 validation, and
1,172 test. Therefore, `--train_size 1200` correctly yields 1,119 train rows.

## 2. Common environment

Run all commands from `agent_routing/`:

```bash
cd /workspace/7.98/agent_routing
source /workspace/research_7.41/agent_routing/.venv/bin/activate

export HF_HOME=/workspace/hf_cache
export BASE_MODEL=Qwen/Qwen3.5-9B
export SUBAGENT_SERVER_URL=http://localhost:8000
export PYTHONUTF8=1
mkdir -p logs
```

The first download must run with `HF_HUB_OFFLINE=0`. After the normalized
JSONL has been created, all later stages can use `HF_HUB_OFFLINE=1`; they read
the local normalized cache and do not contact Hugging Face.

## 3. AQuA-RAT: complete run

Set names and load the official splits:

```bash
export BENCHMARK=aqua_rat
export CACHE=outputs/data/aqua_rat_normalized.jsonl
export SUBAGENT_ID=ds_aqua_9B_sub
export MANAGER_ID=aqua_marginal_9b_v1
export TASK_DESC='You are a manager agent solving five-choice algebraic word problems.'
export SPLITS='--train_size 1200 --dev_size 254 --test_size 254'

HF_HUB_OFFLINE=0 python -u -X utf8 -m src.pipeline.cli load_aqua_rat \
  --benchmark "$BENCHMARK" \
  --aqua_rat_normalized_cache "$CACHE" \
  $SPLITS
```

If Hugging Face is unavailable, clone the official AQuA repository and load
its `train.json`, `dev.json`, and `test.json` files directly:

```bash
HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli load_aqua_rat \
  --benchmark aqua_rat \
  --aqua_rat_source local \
  --aqua_rat_local_path /path/to/AQuA \
  --aqua_rat_normalized_cache "$CACHE" \
  --aqua_rat_refresh_cache \
  $SPLITS
```

Create advisor SFT data with an online DeepSeek teacher. Skip this block if
the three adapters already exist under `outputs/adapters/$SUBAGENT_ID/`:

```bash
for KIND in extractor reasoner verifier; do
  HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli synth_subagent \
    --benchmark "$BENCHMARK" \
    --base_model "$BASE_MODEL" \
    --teacher_id "$SUBAGENT_ID" \
    --teacher_provider deepseek \
    --teacher_model deepseek-chat \
    --agent_kind "$KIND" \
    --n_samples 500 \
    --synth_symmetric_leakage \
    --aqua_rat_normalized_cache "$CACHE" \
    $SPLITS

  HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" \
    --teacher_id "$SUBAGENT_ID" \
    --agent_kind "$KIND" \
    --sft_epochs 3 --sft_lr 5e-5 --sft_bs 1 --sft_grad_accum 8
done
```

Start the frozen advisor server on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_subagent_server.sh \
  "$BASE_MODEL" "$SUBAGENT_ID" outputs 8000
```

In another terminal, run a 24-row one-step smoke test. This checks the loader,
answer protocol, advisor schemas, and tool server without repeating a full
depth-1 experiment:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli build_marginal_sft \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" \
  --teacher_id "${MANAGER_ID}_smoke" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  --aqua_rat_normalized_cache "$CACHE" \
  $SPLITS \
  --mv_n_samples 24 --mv_max_depth 1 --mv_temperature 0 \
  --mv_max_commit_rescue_ratio -1 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

If the smoke report is valid, collect the main depth-3 data once. The
collector stops expanding an example at its first successful depth, so this
tests the full 0-to-3-call policy without evaluating longer successful paths:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli build_marginal_sft \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" \
  --teacher_id "$MANAGER_ID" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  --aqua_rat_normalized_cache "$CACHE" \
  $SPLITS \
  --mv_n_samples 400 --mv_max_depth 3 --mv_temperature 0 \
  --mv_max_commit_rescue_ratio -1 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC" 2>&1 | tee logs/aqua_mv9b_400_d3.log
```

Train the manager on selected routing decisions:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli train_manager_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$MANAGER_ID" \
  --manager_sft_train_jsonl "outputs/manager/$MANAGER_ID/marginal_value/manager_sft_marginal.jsonl" \
  --manager_sft_output_dir "outputs/manager/$MANAGER_ID/sft_marginal" \
  --manager_sft_epochs 1 --manager_sft_lr 1e-5 \
  --sft_max_seq_len 4096 --sft_bs 1 --sft_grad_accum 8
```

Evaluate the SFT routing policy before adding RL:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli eval_manager_tools \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" \
  --teacher_id "$MANAGER_ID" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  --aqua_rat_normalized_cache "$CACHE" \
  $SPLITS --eval_n_samples 254 --eval_max_tool_calls 3 \
  --eval_manager_dir "outputs/manager/$MANAGER_ID/sft_marginal" \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

## 4. ARC-Challenge: complete run

ARC uses the same stages. Change only the data flag and experiment ids:

```bash
export BENCHMARK=arc_challenge
export CACHE=outputs/data/arc_challenge_normalized.jsonl
export SUBAGENT_ID=ds_arc_9B_sub
export MANAGER_ID=arc_marginal_9b_v1
export TASK_DESC='You are a manager agent solving grade-school science multiple-choice questions.'
export SPLITS='--train_size 1200 --dev_size 299 --test_size 1172'

HF_HUB_OFFLINE=0 python -u -X utf8 -m src.pipeline.cli load_arc_challenge \
  --benchmark "$BENCHMARK" \
  --arc_challenge_normalized_cache "$CACHE" \
  $SPLITS
```

Create and train the three ARC advisors:

```bash
for KIND in extractor reasoner verifier; do
  HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli synth_subagent \
    --benchmark "$BENCHMARK" \
    --base_model "$BASE_MODEL" --teacher_id "$SUBAGENT_ID" \
    --teacher_provider deepseek --teacher_model deepseek-chat \
    --agent_kind "$KIND" --n_samples 500 --synth_symmetric_leakage \
    --arc_challenge_normalized_cache "$CACHE" $SPLITS

  HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" --teacher_id "$SUBAGENT_ID" \
    --agent_kind "$KIND" \
    --sft_epochs 3 --sft_lr 5e-5 --sft_bs 1 --sft_grad_accum 8
done
```

Start the ARC advisor server on GPU 0, then use another terminal for the
manager collector:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_subagent_server.sh \
  "$BASE_MODEL" "$SUBAGENT_ID" outputs 8000
```

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -u -X utf8 -m src.pipeline.cli build_marginal_sft \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" \
  --teacher_id "$MANAGER_ID" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  --arc_challenge_normalized_cache "$CACHE" \
  $SPLITS \
  --mv_n_samples 400 --mv_max_depth 3 --mv_temperature 0 \
  --mv_max_commit_rescue_ratio -1 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC" 2>&1 | tee logs/arc_mv9b_400_d3.log
```

Train and evaluate the ARC manager:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli train_manager_sft \
  --base_model "$BASE_MODEL" --teacher_id "$MANAGER_ID" \
  --manager_sft_train_jsonl "outputs/manager/$MANAGER_ID/marginal_value/manager_sft_marginal.jsonl" \
  --manager_sft_output_dir "outputs/manager/$MANAGER_ID/sft_marginal" \
  --manager_sft_epochs 1 --manager_sft_lr 1e-5 \
  --sft_max_seq_len 4096 --sft_bs 1 --sft_grad_accum 8

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u -X utf8 -m src.pipeline.cli eval_manager_tools \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" --teacher_id "$MANAGER_ID" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  --arc_challenge_normalized_cache "$CACHE" \
  $SPLITS --eval_n_samples 1172 --eval_max_tool_calls 3 \
  --eval_manager_dir "outputs/manager/$MANAGER_ID/sft_marginal" \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

## 5. Optional SFT-anchored binary GRPO

Run this only after recording the SFT-only result. The terminal task reward
remains binary correctness; the route-only anchor preserves the routing action
learned from marginal SFT without adding a tool-cost reward.

The block works for either benchmark because it uses the environment variables
defined above:

```bash
export VENV_DIR=/workspace/research_7.41/agent_routing/.venv

DATA_FLAG="--aqua_rat_normalized_cache $CACHE"
if [ "$BENCHMARK" = "arc_challenge" ]; then
  DATA_FLAG="--arc_challenge_normalized_cache $CACHE"
fi

HF_HUB_OFFLINE=1 bash scripts/train_manager_grpo_multigpu.sh "$MANAGER_ID" \
  --benchmark "$BENCHMARK" \
  --base_model "$BASE_MODEL" \
  --subagent_teacher_id "$SUBAGENT_ID" \
  $DATA_FLAG $SPLITS \
  --mgr_init_adapter "outputs/manager/$MANAGER_ID/sft_marginal" \
  --mgr_output_dir "outputs/manager/$MANAGER_ID/grpo_binary_anchor" \
  --mgr_sft_anchor_jsonl "outputs/manager/$MANAGER_ID/marginal_value/manager_sft_marginal.jsonl" \
  --mgr_sft_anchor_coef 0.05 --mgr_sft_anchor_mode route_only \
  --exclude_sft_example_ids "outputs/sft_data/$SUBAGENT_ID/extractor_sft.jsonl" \
  --exclude_sft_example_ids "outputs/sft_data/$SUBAGENT_ID/reasoner_sft.jsonl" \
  --exclude_sft_example_ids "outputs/sft_data/$SUBAGENT_ID/verifier_sft.jsonl" \
  --exclude_sft_example_ids "outputs/manager/$MANAGER_ID/marginal_value/counterfactual_records.jsonl" \
  --mgr_routing_efficiency_bonus 0 --mgr_tool_use_bonus 0 \
  --mgr_grpo_beta 0.05 --mgr_clip_epsilon_high 0.28 --mgr_max_steps 200 \
  --task_description "$TASK_DESC"
```

Select the anchor coefficient and checkpoint on `dev`, then report one final
test result. Do not tune on the AQuA 254-row test set or the ARC 1,172-row test
set.

## 6. Safety checks

The CLI rejects ambiguous data selection. For example, passing both
`--aqua_rat_normalized_cache` and `--arc_challenge_normalized_cache` now raises
an error instead of silently choosing one. Old MedQA, LegalBench, GPQA, and
MMLU-Pro commands remain valid; when no benchmark signal exists, `auto`
defaults to MedQA.

Before a full run, inspect the normalized cache:

```bash
head -n 1 "$CACHE" | python -m json.tool
```

Verify that `ground_truth` is a key in `choices`, `context` is empty, split
labels are `train/dev/test`, and AQuA rows do not contain a `rationale` field.
