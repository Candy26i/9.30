#!/usr/bin/env bash
# Bootstrap the 2026-10-10 pod (the previous one stopped on credit) and run what is missing:
#   the MMLU-Pro v3-with-GRPO grid as three parallel runs (scripts/pod_ops/mcq_supervisor_v3g2.py) and the untrained
#   base-manager baseline on the four locked tests (GPU 0 next to the advisors), with 30-minute HF backups whose repo
#   creation is checked before the runs start.
# Steps leave markers in $WORK/logs/bootstrap/ (idempotent). Credentials are the operator's (HF_HOME=/workspace/hf-cache,
# W&B netrc); nothing here reads a token: the backup loops wait for the token file to exist. Log: $WORK/logs/bootstrap.log
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
BRANCH="${BRANCH:-feat/mcq-rsi-controller}"
WORK=/workspace/mcq_rsi
REPO=/workspace/9.30
TREE=$REPO/agent_routing
PY=/workspace/mcq-venv/bin/python
M=$WORK/logs/bootstrap
mkdir -p $WORK/logs $M /workspace/tmp
log() { printf '[bootstrap %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
step() { local name=$1; shift; if [ -f "$M/$name" ]; then log "$name: done earlier"; return 0; fi; log "$name: start"
  if "$@"; then touch "$M/$name"; log "$name: ok"; else log "$name: FAILED (rc=$?)"; return 1; fi; }

clone() { if [ ! -d $REPO/.git ]; then git clone https://github.com/Candy26i/9.30.git $REPO || return 1; fi
  cd $REPO && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH" && git rev-parse HEAD; }
setup() { cd $TREE && bash scripts/setup_mcq_rsi_pod.sh; }
import_assets() { cd $TREE && $PY -m src.manager.mcq_rsi import --bench all --out $WORK/import --cache-dir $HF_HOME/hub \
    && $PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir $WORK/import --hf-cache-dir $HF_HOME/hub; }
restore_cache() { cd $TREE && $PY - <<'PYEOF'
from huggingface_hub import hf_hub_download
import tarfile
p = hf_hub_download("MaliDDD/margent-mcq-rsi-mmlu_pro_v3r5", "advisor_cache.tar.gz", local_dir="/workspace/tmp/restore")
with tarfile.open(p) as tar:
    n = len(tar.getmembers()); tar.extractall("/workspace/mcq_rsi")
print("advisor cache restored:", n, "members")
PYEOF
}
advisors() { cd $TREE && MCQ_CONFIG_SUFFIX=_v3 BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg advisors
  for i in $(seq 1 120); do curl -fsS http://127.0.0.1:18002/health >/dev/null 2>&1 && return 0; sleep 10; done
  log "advisors: not healthy after 20 min"; return 1; }
preflight_one() { cd $TREE && MCQ_CONFIG_SUFFIX=_v3 BENCH=$1 PREFLIGHT_BENCHES=$1 bash scripts/runpod_mcq_rsi.sh preflight 2>&1 | tail -2 \
    && touch $WORK/logs/bootstrap_preflight_$1; }
supervisor() {
  tmux set-environment -g HF_HOME /workspace/hf-cache; tmux set-environment -g HF_HUB_DISABLE_XET 1; tmux set-environment -g TMPDIR /workspace/tmp
  cp $TREE/scripts/pod_ops/mcq_supervisor_v3g2.py $TREE/scripts/pod_ops/backup_loops2.sh $TREE/scripts/pod_ops/base_manager_chain2.sh \
     $TREE/scripts/pod_ops/state_now.py $TREE/scripts/pod_ops/verify_runs_hf.py $TREE/scripts/pod_ops/letters.py $TREE/scripts/pod_ops/rounds_any.py /workspace/tmp/
  chmod +x /workspace/tmp/*.sh
  tmux kill-session -t mcq_supervisor_v3g2 2>/dev/null
  tmux new-session -d -s mcq_supervisor_v3g2 "cd /workspace/tmp && HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1 python3 mcq_supervisor_v3g2.py 2>&1 | tee -a $WORK/logs/supervisor_v3g2.log; exec bash"
  tmux kill-session -t mcq_backups 2>/dev/null
  tmux new-session -d -s mcq_backups "bash /workspace/tmp/backup_loops2.sh; exec bash"
  tmux kill-session -t mcq_base_manager 2>/dev/null
  tmux new-session -d -s mcq_base_manager "bash /workspace/tmp/base_manager_chain2.sh; exec bash"
}

{
  step clone clone || exit 1
  step setup setup || exit 1
  step import import_assets || exit 1
  step restore_cache restore_cache || exit 1
  step advisors advisors || exit 1
  step preflight_mmlu_pro preflight_one mmlu_pro || log "preflight mmlu_pro FAILED: the runs will be refused"
  step supervisor supervisor || exit 1
  for b in medqa gpqa aqua; do step preflight_$b preflight_one $b || log "preflight $b FAILED"; done
  touch $WORK/logs/bootstrap_preflight_all
  log "bootstrap complete: tmux mcq_supervisor_v3g2, mcq_backups, mcq_base_manager, mcq_advisors_medqa"
} 2>&1 | tee -a $WORK/logs/bootstrap.log
