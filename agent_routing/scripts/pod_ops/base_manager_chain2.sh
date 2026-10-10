#!/usr/bin/env bash
# Untrained baseline (the base model as the manager, base advisors) on the four locked tests, then its token accounting.
# GPU: 0 next to the advisors on a 4-GPU pod; on a 2-GPU pod it waits until the success run on GPU 0 has finished.
# Waits for every benchmark's preflight (logs/bootstrap_preflight_all). Marker: logs/base_manager_done.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
W=/workspace/mcq_rsi; T=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
until [ -f $W/logs/bootstrap_preflight_all ]; do sleep 60; done
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
if [ "$NGPU" -lt 4 ]; then
  until python3 -c "import json,sys; d=json.load(open('$W/runs/mmlu_pro_v3g_suc/status.json')); sys.exit(0 if d.get('controller')=='final-test complete' else 1)" 2>/dev/null; do sleep 120; done
fi
export CUDA_VISIBLE_DEVICES=0
cd $T
BASE=$($PY -c "from src.manager.mcq_rsi.evaluate import resolve_base; print(resolve_base('Qwen/Qwen3.5-9B'))")
OUT=$W/runs/base_manager; mkdir -p $OUT; echo "$BASE" > $OUT/base_model_path.txt
for bp in medqa:test gpqa:test aqua:test mmlu_pro:test mmlu_pro:test_paper; do
  b=${bp%%:*}; pool=${bp##*:}
  [ -f $OUT/$b/$pool/mcq_rsi_eval.json ] && continue
  $PY -m src.manager.mcq_rsi evaluate --bench $b --pool $pool --checkpoint $BASE --out $OUT/$b/$pool \
    --advisor-url http://127.0.0.1:18002 --advisor-cache $W/advisor_cache --advisor-mode base --no-require-gate \
    2>&1 | grep -v "Warning\|warn(" | tail -2 | tee -a $W/logs/base_manager.log
done
export CUDA_VISIBLE_DEVICES=
mkdir -p $W/runs/base_manager/tokcost
for bp in medqa:test gpqa:test aqua:test mmlu_pro:test mmlu_pro:test_paper; do
  b=${bp%%:*}; pool=${bp##*:}
  $PY scripts/mcq_token_cost.py --config configs/mcq_rsi_${b}_v3.json --pool $pool --out $OUT/tokcost/${b}_${pool} \
    --eval base/manager=$OUT/$b/$pool/manager_tool_eval.jsonl 2>&1 | grep "^| base" | tee -a $W/logs/base_manager.log
done
touch $W/logs/base_manager_done
