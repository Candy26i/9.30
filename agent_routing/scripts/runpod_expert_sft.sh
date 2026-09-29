#!/usr/bin/env bash
# Three independent math expert adapters; no Manager training or test evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXPERT_ROOT="${EXPERT_ROOT:-/workspace/margent-expert-sft-01}"
EXPERT_PYTHON="${EXPERT_PYTHON:-/workspace/margent-venv/bin/python}"
EXPERT_CONFIG="${EXPERT_CONFIG:-configs/math_expert_sft_pilot.json}"
EXPERT_SESSION="${EXPERT_SESSION:-margent-experts}"
EXPERT_GPU="${EXPERT_GPU:-0}"
EXPERT_MANAGER_DATA="${EXPERT_MANAGER_DATA:-/workspace/margent-data-restart-20260925}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_XET=1
export TMPDIR="${TMPDIR:-/workspace/margent-tmp}"
export WANDB_ENTITY="${WANDB_ENTITY:-yuningyangaillm}"
export WANDB_PROJECT="${WANDB_PROJECT:-MATH_rsi}"
export MARGENT_WANDB_MODE="${MARGENT_WANDB_MODE:-online}"
export MARGENT_WANDB_TEXT="${MARGENT_WANDB_TEXT:-1}"
operation="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
case "$operation" in
  status)
    for path in expert_report.json expert_status.json status.json; do
      if [[ -f "$EXPERT_ROOT/$path" ]]; then cat "$EXPERT_ROOT/$path"; fi
    done
    if [[ -f "$EXPERT_ROOT.log" ]]; then tail -n 20 "$EXPERT_ROOT.log"; fi
    ;;
  plan|run|start)
    test -x "$EXPERT_PYTHON"
    mkdir -p "$TMPDIR" "$(dirname "$EXPERT_ROOT")"
    action=run
    if [[ "$operation" == plan ]]; then action=plan; fi
    command=("$EXPERT_PYTHON" -u -m src.verifiable.experts "$action" --out "$EXPERT_ROOT"
      --config "$EXPERT_CONFIG" --manager-data-dir "$EXPERT_MANAGER_DATA" --gpu "$EXPERT_GPU" "$@")
    if [[ "$operation" != start ]]; then exec "${command[@]}"; fi
    command -v tmux >/dev/null
    if tmux has-session -t "$EXPERT_SESSION" 2>/dev/null; then
      echo "Expert session exists: $EXPERT_SESSION. Inspect it before restarting." >&2
      exit 1
    fi
    # Pass task environment explicitly: an existing tmux server may have stale values.
    launch=(env "HF_HOME=$HF_HOME" "HF_HUB_CACHE=$HF_HUB_CACHE" "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
      "HF_HUB_DISABLE_XET=1" "TMPDIR=$TMPDIR" "WANDB_ENTITY=$WANDB_ENTITY" "WANDB_PROJECT=$WANDB_PROJECT"
      "MARGENT_WANDB_MODE=$MARGENT_WANDB_MODE" "MARGENT_WANDB_TEXT=$MARGENT_WANDB_TEXT" "${command[@]}")
    printf -v line '%q ' "${launch[@]}"
    printf -v logfile '%q' "$EXPERT_ROOT.log"
    tmux new-session -d -s "$EXPERT_SESSION" -c "$PWD" "${line}>> ${logfile} 2>&1"
    echo "Expert SFT controller started in $EXPERT_SESSION; inspect status and W&B to confirm progress."
    echo "The two-hour controller limit does not stop Pod billing."
    ;;
  *) echo 'Usage: bash scripts/runpod_expert_sft.sh [plan|start|run|status] [controller options]' >&2; exit 2 ;;
esac
