#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_XET=1
export TMPDIR="${TMPDIR:-/workspace/margent-tmp}"
mkdir -p "$TMPDIR"
RSI_PYTHON="${RSI_PYTHON:-/workspace/margent-venv/bin/python}"
RSI_EXPERT_ROOT="${RSI_EXPERT_ROOT:-/workspace/margent-expert-teacher-sft-01}"
RSI_CONFIG="${RSI_CONFIG:-$RSI_EXPERT_ROOT/manager_config.json}"
RSI_DATA="${RSI_DATA:-/workspace/margent-data-restart-20260925}"
RSI_SUBSET="${RSI_SUBSET:-/workspace/margent-rsi-pilot-data-v1}"
RSI_OUTPUT="${RSI_OUTPUT:-/workspace/margent-rsi-pilot-v1}"
export WANDB_ENTITY="${WANDB_ENTITY:-yuningyangaillm}"
export WANDB_PROJECT="${WANDB_PROJECT:-MATH_rsi}"
export MARGENT_WANDB_MODE="${MARGENT_WANDB_MODE:-online}"
export MARGENT_WANDB_TEXT="${MARGENT_WANDB_TEXT:-1}"
case "${1:-}" in
  advisor)
    # Keep this terminal open; stop any previous advisor instance separately.
    CUDA_VISIBLE_DEVICES="${RSI_ADVISOR_GPU:-0}" "$RSI_PYTHON" -m src.verifiable.experts serve --config "$RSI_CONFIG"
    ;;
  plan|run)
    "$RSI_PYTHON" -m src.verifiable.rsi prepare --data-dir "$RSI_DATA" --out "$RSI_SUBSET" --train-n 16 --dev-n 16
    extra=()
    if [[ "$1" == plan ]]; then extra+=(--dry-run); fi
    CUDA_VISIBLE_DEVICES="${RSI_MANAGER_GPU:-1}" "$RSI_PYTHON" -m src.verifiable.rsi run \
      --config "$RSI_CONFIG" --data-dir "$RSI_SUBSET" --out "$RSI_OUTPUT" \
      --rounds 2 --hours 24 --arms dynamic static success "${extra[@]}"
    ;;
  report)
    "$RSI_PYTHON" -m src.verifiable.rsi report --out "$RSI_OUTPUT"
    ;;
  *) echo 'Usage: bash scripts/runpod_rsi_pilot.sh advisor|plan|run|report' >&2; exit 2 ;;
esac
