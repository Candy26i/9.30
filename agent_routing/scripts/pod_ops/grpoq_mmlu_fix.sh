#!/usr/bin/env bash
# Experiment D, MMLU-Pro units (missed by the first chain: variable-name bug), on GPU 0 after the label-quality chain there.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python; Q=$WORK/grpoq
until [ -f $WORK/logs/labelq_done_gpu0 ]; do sleep 60; done
cd $TREE
$PY scripts/mcq_label_quality.py --config configs/mcq_rsi_mmlu_pro_v3.json --out $WORK/runs/mmlu_pro_v3gq \
  --labels S2=$Q/mmlu_pro/labels_S2_r3.jsonl G2=$Q/mmlu_pro/labels_G2_r3.jsonl --advisor-url http://127.0.0.1:18002 \
  2>&1 | tee -a $WORK/logs/mmlu_pro_v3gq.log || echo "label quality mmlu_pro FAILED" | tee -a $WORK/logs/grpoq.log
touch $WORK/logs/grpoq_mmlu_done
