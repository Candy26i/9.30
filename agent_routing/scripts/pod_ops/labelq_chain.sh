#!/usr/bin/env bash
# Label quality by round (scripts/mcq_label_quality.py) for the four benchmarks, GPU 1, after experiment D (logs/grpoq_done).
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
until [ -f $WORK/logs/grpoq_done ]; do sleep 60; done
cd $TREE
for b in medqa aqua mmlu_pro gpqa; do
  L=$WORK/labels/$b; args="r2=$L/r2.jsonl r3=$L/r3.jsonl"; [ -f $L/r4.jsonl ] && args="$args r4=$L/r4.jsonl r5=$L/r5.jsonl"
  $PY scripts/mcq_label_quality.py --config configs/mcq_rsi_${b}_v3.json --out $WORK/runs/${b}_v3lq --labels $args --round1 --eval-s1 \
    --advisor-url http://127.0.0.1:18002 2>&1 | tee -a $WORK/logs/${b}_v3lq.log || echo "labelq $b FAILED" | tee -a $WORK/logs/labelq.log
done
touch $WORK/logs/labelq_done
