#!/usr/bin/env bash
# Rebuttal pod, 2026-11-26: hands the training GPU from the seed-1 chain to "success distillation, then seed 2" and
# keeps the grid queue in step, without an operator:
#   1. wait until seed 1 exists for the four benchmarks;
#   2. stop the old training chain, update the checkout, start `PLAN="sucd seed2" seed_sft_chain.sh` on GPU 1;
#   3. restart the queue (GPU 0 only) so that it knows the new jobs; running grid jobs resume from their shards;
#   4. when training has ended, restart the queue on both GPUs.
# Run from a copy outside the checkout (step 2 pulls):  tmux new -d -s followup "bash /workspace/tmp/rebuttal_followup.sh"
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
WORK=/workspace/mcq_rsi; CKPT=$WORK/ckpt/verifier_grid; REPO=/workspace/9.30; TREE=$REPO/agent_routing
LOG=$WORK/logs/verifier_grid/followup.log; BRANCH="${BRANCH:-feat/mcq-rsi-controller}"
log() { printf '[followup %s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
have() { [ -f $CKPT/$1/adapter_model.safetensors ] && [ -f $CKPT/$1/adapter_config.json ]; }
training() { pgrep -f '[s]eed_sft_chain\.sh' >/dev/null || pgrep -f '[t]rain_manager_sft' >/dev/null; }

log "waiting for seed 1 (gpqa, medqa, mmlu_pro, aqua)"
until have gpqa_seed1 && have medqa_seed1 && have mmlu_pro_seed1 && have aqua_seed1; do
  training || { log "the training chain stopped before seed 1 was complete; nothing is switched"; exit 1; }
  sleep 30
done
log "seed 1 complete: stopping the old chain"
tmux kill-session -t seed_sft 2>/dev/null; pkill -f '[s]eed_sft_chain\.sh' 2>/dev/null; pkill -f '[t]rain_manager_sft' 2>/dev/null
for i in $(seq 1 30); do training || break; sleep 2; done
(cd $REPO && git pull -q --ff-only origin "$BRANCH" && log "checkout at $(git rev-parse --short HEAD)") || { log "git pull FAILED"; exit 1; }
touch $CKPT/TRAINING
tmux new-session -d -s seed_sft "PLAN='sucd seed2' SFT_GPU=1 bash $TREE/scripts/pod_ops/seed_sft_chain.sh; exec bash"
log "training chain restarted: sucd, then seed2"
sleep 20
RESTART_QUEUE=1 QUEUE_ARGS="--slots 0:2" bash $TREE/scripts/pod_ops/verifier_grid_chain.sh >> $LOG 2>&1 || log "queue restart (GPU 0) FAILED"
log "queue restarted on GPU 0"
while training; do sleep 60; done
log "training ended: $(grep -c ' ok in ' $WORK/logs/verifier_grid/seed_sft.log) adapters trained in total; restarting the queue on both GPUs"
RESTART_QUEUE=1 QUEUE_ARGS="--slots 0:2,1:3" bash $TREE/scripts/pod_ops/verifier_grid_chain.sh >> $LOG 2>&1 || log "queue restart (both GPUs) FAILED"
log "done"
