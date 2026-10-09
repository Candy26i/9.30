#!/usr/bin/env bash
# Self-check diagnostic (scripts/mcq_self_check.py) for S_1 on the four benchmarks' dev sets, GPU 1, in the gap after
# gpqa_v3g's locked test (the supervisor waits for logs/diag_done before starting aqua_v3g there). ~30 min.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python; DL=/workspace/tmp/oracle_dl/runs
until python3 -c "import json,sys; sys.exit(0 if json.load(open('$WORK/runs/gpqa_v3g/status.json')).get('controller') == 'final-test complete' else 1)" 2>/dev/null; do sleep 60; done
cd $TREE
# budgets: S_1's and v3 dynamic r3's dev call rates (table 8 of the v3 runs)
declare -A B=( [medqa]="0.37,0.14" [mmlu_pro]="0.32,0.22" [gpqa]="0.62,0.61" [aqua]="0.79,0.82" )
for b in medqa mmlu_pro gpqa aqua; do
  $PY scripts/mcq_self_check.py --config configs/mcq_rsi_${b}_v3.json --out $WORK/runs/${b}_v3sc \
    --policy-eval $DL/${b}_v3/r1/S1_dev/manager_tool_eval.jsonl \
    --forced-verifier $DL/${b}_v3/final/S_1/dev_forced_verifier/manager_forced_verifier.jsonl \
    --n-samples 8 --temperature 1.0 --budgets ${B[$b]} 2>&1 | grep -v "Warning\|warn(" | tee -a $WORK/logs/${b}_v3sc.log \
    || echo "self-check $b FAILED" | tee -a $WORK/logs/diag.log
done
touch $WORK/logs/diag_done
