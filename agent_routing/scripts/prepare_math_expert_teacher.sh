#!/usr/bin/env bash
# CPU/API teacher synthesis precedes the paid GPU SFT session.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TEACHER_PYTHON="${TEACHER_PYTHON:-/workspace/margent-venv/bin/python}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/math_expert_teacher.json}"
TEACHER_ROOT="${TEACHER_ROOT:-/workspace/margent-expert-teacher-01}"
EXPERT_MANAGER_DATA="${EXPERT_MANAGER_DATA:-/workspace/margent-data-restart-20260925}"
export WANDB_ENTITY="${WANDB_ENTITY:-yuningyangaillm}"
export WANDB_PROJECT="${WANDB_PROJECT:-MATH_rsi}"
export MARGENT_WANDB_MODE="${MARGENT_WANDB_MODE:-online}"
export MARGENT_WANDB_TEXT="${MARGENT_WANDB_TEXT:-1}"
operation="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$operation" in
  plan)
    "$TEACHER_PYTHON" - "$TEACHER_CONFIG" "$TEACHER_ROOT" <<'PY'
import json, sys
from src.verifiable.expert_synthesis import load_config
config = load_config(sys.argv[1])
questions = config['train_size'] + config['dev_size']
print(json.dumps({'teacher': {'provider': config['provider'], 'model': config['model']},
    'output': sys.argv[2], 'question_groups': questions, 'logical_requests': questions * 6,
    'attempt_cap': config['max_calls'], 'config': config,
    'note': 'Preview only. generate makes paid teacher API calls; prepare/finalize do not. No GPU model is loaded. API costs and GPU rental are separate.'}, indent=2))
PY
    ;;
  prepare)
    exec "$TEACHER_PYTHON" -m src.verifiable.expert_synthesis prepare --out "$TEACHER_ROOT" \
      --config "$TEACHER_CONFIG" --manager-data-dir "$EXPERT_MANAGER_DATA" "$@"
    ;;
  generate)
    exec "$TEACHER_PYTHON" -m src.verifiable.expert_synthesis generate --out "$TEACHER_ROOT" \
      --config "$TEACHER_CONFIG" "$@"
    ;;
  finalize)
    exec "$TEACHER_PYTHON" -m src.verifiable.expert_synthesis finalize --out "$TEACHER_ROOT" "$@"
    ;;
  status)
    if [[ -f "$TEACHER_ROOT/synthesis_status.json" ]]; then cat "$TEACHER_ROOT/synthesis_status.json"; fi
    ;;
  *) echo 'Usage: bash scripts/prepare_math_expert_teacher.sh [plan|prepare|generate|finalize|status] [options]' >&2; exit 2 ;;
esac
