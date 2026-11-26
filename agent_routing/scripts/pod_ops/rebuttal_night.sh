#!/usr/bin/env bash
# Rebuttal pod, night of 2026-11-26: keeps both GPUs busy without an operator until the queue is empty.
#   1. wait for the running training chain (seed 2) to end;
#   2. update the checkout; train `PLAN` (default "seed3 cmto") on GPU 1; restart the queue on GPU 0 with the new jobs
#      (seed 3, commit-only control, paper-protocol evaluations of the seed replicates);
#   3. when training has ended, restart the queue on both GPUs;
#   4. when the queue is empty, run one more HF backup pass.
# Replaces rebuttal_followup.sh (stop its tmux session first). Run from a copy outside the checkout:
#   tmux new -d -s night "bash /workspace/tmp/rebuttal_night.sh"
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
WORK=/workspace/mcq_rsi; CKPT=$WORK/ckpt/verifier_grid; REPO=/workspace/9.30; TREE=$REPO/agent_routing
PY=/workspace/mcq-venv/bin/python; LOG=$WORK/logs/verifier_grid/followup.log
BRANCH="${BRANCH:-feat/mcq-rsi-controller}"; PLAN="${PLAN:-seed3 cmto}"
log() { printf '[followup %s] night: %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
training() { pgrep -f '[s]eed_sft_chain\.sh' >/dev/null || pgrep -f '[t]rain_manager_sft' >/dev/null; }
queue_running() { pgrep -f '[v]erifier_grid_queue\.py' >/dev/null; }

log "waiting for the running training chain to end"
while training; do sleep 60; done
log "training chain ended ($(grep -c ' ok in ' $WORK/logs/verifier_grid/seed_sft.log) adapters so far)"
(cd $REPO && git pull -q --ff-only origin "$BRANCH" && log "checkout at $(git rev-parse --short HEAD)") || { log "git pull FAILED"; exit 1; }
touch $CKPT/TRAINING
tmux kill-session -t seed_sft 2>/dev/null
tmux new-session -d -s seed_sft "PLAN='$PLAN' SFT_GPU=1 bash $TREE/scripts/pod_ops/seed_sft_chain.sh; exec bash"
log "training chain started: $PLAN"
sleep 30
RESTART_QUEUE=1 QUEUE_ARGS="--slots 0:2" bash $TREE/scripts/pod_ops/verifier_grid_chain.sh >> $LOG 2>&1 || log "queue restart (GPU 0) FAILED"
log "queue restarted on GPU 0"
while training; do sleep 60; done
log "training ended ($(grep -c ' ok in ' $WORK/logs/verifier_grid/seed_sft.log) adapters in total); restarting the queue on both GPUs"
RESTART_QUEUE=1 QUEUE_ARGS="--slots 0:2,1:3" bash $TREE/scripts/pod_ops/verifier_grid_chain.sh >> $LOG 2>&1 || log "queue restart (both GPUs) FAILED"
sleep 120
while queue_running; do sleep 120; done
log "queue ended: $(tail -200 $WORK/logs/verifier_grid/queue.log | grep -E 'queue empty|not finished|tables still' | tail -2 | tr '\n' ' ')"
(cd $TREE && $PY scripts/backup_mcq_rsi_hf.py --work $WORK --run verifier_grid --repo MaliDDD/margent-mcq-rsi-verifier_grid --public 2>&1 | tail -1 | cut -c1-200 | tee -a $LOG)
log "done"
