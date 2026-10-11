#!/usr/bin/env bash
# Second pass of the single-agent baseline: the plain-prompt variants with a 2048-token budget (the 512-token budget
# truncated 41-95% of the base model's chain-of-thought answers) and forced answers (every question parseable), for the
# base model and for S_1, plus the base at 512 tokens with forcing; then a backup.
# Waits for the first chain (logs/single_agent_done); marker logs/single_agent_long_done.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python; OUT=$WORK/runs/single_agent
until [ -f $WORK/logs/single_agent_done ]; do sleep 60; done
cd $TREE; BASE=$(cat $OUT/base_model_path.txt)
for bp in medqa:test gpqa:test aqua:test mmlu_pro:test mmlu_pro:test_paper; do
  b=${bp%%:*}; pool=${bp##*:}
  for spec in "base_plain_2048f:$BASE" "s1_plain_2048f:$WORK/import/$b/round1/sft" "base_plain_512f:$BASE"; do
    name=${spec%%:*}; ckpt=${spec#*:}; d=$OUT/$name/$b/$pool
    [ -f $d/summary.json ] && continue
    $PY scripts/mcq_single_agent.py --config configs/mcq_rsi_${b}_v3.json --pool $pool --mode plain --checkpoint $ckpt --out $d \
      --max-new-tokens $([ "${name%%_*}_${name##*_}" = "base_512f" ] && echo 512 || echo 2048) --force-answer --batch-size 16 2>&1 | grep -v "Warning\|warn(\|Loading weights" | tail -1 | tee -a $WORK/logs/single_agent_long.log
  done
done
$PY scripts/backup_mcq_rsi_hf.py --work $WORK --run single_agent --repo MaliDDD/margent-mcq-rsi-single_agent --public 2>&1 | tail -1 | cut -c1-160 | tee -a $WORK/logs/single_agent_long.log
touch $WORK/logs/single_agent_long_done
