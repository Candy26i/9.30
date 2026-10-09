#!/usr/bin/env bash
# Label quality by round (scripts/mcq_label_quality.py), one chain per GPU:
#   labelq_chain.sh <gpu> <done-marker> <wait-marker|none> <bench>...
# e.g. GPU 0 (next to the advisors): medqa aqua mmlu_pro right after the bootstrap; GPU 1: gpqa (16k-token SFT needs the
# card alone) after experiment D. Each chain writes logs/<done-marker> at the end.
set -o pipefail
GPU=$1; DONE=$2; WAIT=$3; shift 3
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
until grep -q "bootstrap complete" $WORK/logs/bootstrap.log 2>/dev/null; do sleep 60; done
if [ "$WAIT" != "none" ]; then until [ -f $WORK/logs/$WAIT ]; do sleep 60; done; fi
cd $TREE
for b in "$@"; do
  L=$WORK/labels/$b; args="r2=$L/r2.jsonl r3=$L/r3.jsonl"; [ -f $L/r4.jsonl ] && args="$args r4=$L/r4.jsonl r5=$L/r5.jsonl"
  $PY scripts/mcq_label_quality.py --config configs/mcq_rsi_${b}_v3.json --out $WORK/runs/${b}_v3lq --labels $args --round1 --eval-s1 \
    --advisor-url http://127.0.0.1:18002 2>&1 | tee -a $WORK/logs/${b}_v3lq.log || echo "labelq $b FAILED" | tee -a $WORK/logs/labelq.log
done
touch $WORK/logs/$DONE
