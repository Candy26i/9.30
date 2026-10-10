#!/usr/bin/env bash
# One HF backup loop per run directory with an rsi_run.json (every 30 min), plus a loop for the base-manager baseline
# directory. Waits for the operator's HF login (token file) and, for each run, runs one immediate pass and checks the
# repo exists before trusting the loop (the 2026-10-09 pod's loops never created their repos). tmux "mcq_backups".
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp
W=/workspace/mcq_rsi; T=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
L=$W/logs/backup_loops.log
until [ -f $HF_HOME/token ]; do echo "[backups $(date -u +%FT%TZ)] waiting for the HF login ($HF_HOME/token)" | tee -a $L; sleep 120; done
while true; do
  for d in $W/runs/*; do
    r=$(basename "$d")
    case "$r" in smoke*|*.*) continue;; esac
    [ -f "$d/rsi_run.json" ] || [ "$r" = "base_manager" ] || continue
    tmux has-session -t "bk_$r" 2>/dev/null && continue
    # first pass now, then check the repo answers
    (cd $T && $PY scripts/backup_mcq_rsi_hf.py --work $W --run $r --repo MaliDDD/margent-mcq-rsi-$r --public 2>&1 | tail -1 | cut -c1-200) | tee -a $L
    if $PY - "$r" <<'PYEOF' 2>>$L
import sys
from huggingface_hub import HfApi
run = sys.argv[1]
info = HfApi().repo_info(f"MaliDDD/margent-mcq-rsi-{run}")
print(f"[backups] repo MaliDDD/margent-mcq-rsi-{run} exists, private={info.private}")
PYEOF
    then
      tmux new-session -d -s "bk_$r" "cd $T && $PY scripts/backup_mcq_rsi_hf.py --work $W --run $r --repo MaliDDD/margent-mcq-rsi-$r --public --every-minutes 30 2>&1 | tee -a $W/logs/backup_$r.log; exec bash"
      echo "[backups $(date -u +%FT%TZ)] loop started for $r" | tee -a $L
    else
      echo "[backups $(date -u +%FT%TZ)] REPO CHECK FAILED for $r (will retry)" | tee -a $L
    fi
  done
  sleep 300
done
