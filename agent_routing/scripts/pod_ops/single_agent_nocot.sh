#!/usr/bin/env bash
# Third pass: the no-reasoning single-agent variant (the letter only, 16-token budget, forced choice) for the base model, S_1 and
# the v3 dynamic manager: the like-for-like counterpart of the manager's zero-reasoning draft. Waits for the long pass.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python; OUT=$WORK/runs/single_agent
until [ -f $WORK/logs/single_agent_long_done ]; do sleep 60; done
cd $TREE; BASE=$(cat $OUT/base_model_path.txt)
for bp in medqa:test gpqa:test aqua:test mmlu_pro:test mmlu_pro:test_paper; do
  b=${bp%%:*}; pool=${bp##*:}
  for spec in "base_nocot_f:$BASE" "s1_nocot_f:$WORK/import/$b/round1/sft" "dyn3_nocot_f:$OUT/adapters/${b}_dynamic_r3"; do
    name=${spec%%:*}; ckpt=${spec#*:}; d=$OUT/$name/$b/$pool
    [ -f $d/summary.json ] && continue
    $PY scripts/mcq_single_agent.py --config configs/mcq_rsi_${b}_v3.json --pool $pool --mode plain_nocot --checkpoint $ckpt --out $d \
      --max-new-tokens 16 --force-answer --batch-size 32 2>&1 | grep -v "Warning\|warn(\|Loading weights" | tail -1 | tee -a $WORK/logs/single_agent_nocot.log
  done
done
$PY scripts/backup_mcq_rsi_hf.py --work $WORK --run single_agent --repo MaliDDD/margent-mcq-rsi-single_agent --public 2>&1 | tail -1 | cut -c1-160 | tee -a $WORK/logs/single_agent_nocot.log
touch $WORK/logs/single_agent_nocot_done
