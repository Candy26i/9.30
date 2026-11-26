#!/usr/bin/env bash
# Waits for the running grid queue to end, updates the checkout and starts the queue again on both GPUs, so that jobs
# added to scripts/pod_ops/verifier_grid_queue.py meanwhile run without interrupting the ones in flight.
#   tmux new -d -s after "bash /workspace/tmp/after_queue.sh"     (run from a copy outside the checkout)
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
WORK=/workspace/mcq_rsi; REPO=/workspace/9.30; TREE=$REPO/agent_routing; PY=/workspace/mcq-venv/bin/python
LOG=$WORK/logs/verifier_grid/followup.log; BRANCH="${BRANCH:-feat/mcq-rsi-controller}"
log() { printf '[followup %s] after: %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
queue_running() { pgrep -f '[v]erifier_grid_queue\.py' >/dev/null; }
log "waiting for the running queue to end"
while queue_running; do sleep 30; done
(cd $REPO && git pull -q --ff-only origin "$BRANCH" && log "checkout at $(git rev-parse --short HEAD)") || { log "git pull FAILED"; exit 1; }
RESTART_QUEUE=1 QUEUE_ARGS="${QUEUE_ARGS:---slots 0:2,1:3}" bash $TREE/scripts/pod_ops/verifier_grid_chain.sh >> $LOG 2>&1 || log "queue start FAILED"
log "queue started again: $(tail -50 $WORK/logs/verifier_grid/queue.log | grep 'jobs to run' | tail -1)"
sleep 120
while queue_running; do sleep 120; done
log "queue ended: $(tail -200 $WORK/logs/verifier_grid/queue.log | grep -E 'queue empty|not finished|tables still' | tail -2 | tr '\n' ' ')"
(cd $TREE && $PY scripts/backup_mcq_rsi_hf.py --work $WORK --run verifier_grid --repo MaliDDD/margent-mcq-rsi-verifier_grid --public 2>&1 | tail -1 | cut -c1-200 | tee -a $LOG)
log "done"
