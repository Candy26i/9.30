#!/usr/bin/env bash
# Keeps one hourly HF backup loop per versioned run directory (runs/*_v[0-9]*), one public repo per run. tmux "mcq_backups".
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp
while true; do
  for d in /workspace/mcq_rsi/runs/*_v[0-9]*; do
    r=$(basename "$d")
    case "$r" in smoke*|*.*) continue;; esac
    [ -f "$d/rsi_run.json" ] || continue
    tmux has-session -t "bk_$r" 2>/dev/null && continue
    tmux new-session -d -s "bk_$r" "cd /workspace/9.30/agent_routing && /workspace/mcq-venv/bin/python scripts/backup_mcq_rsi_hf.py --work /workspace/mcq_rsi --run $r --repo MaliDDD/margent-mcq-rsi-$r --public --every-minutes 60 2>&1 | tee -a /workspace/mcq_rsi/logs/backup_$r.log; exec bash"
    echo "[backups $(date -u +%FT%TZ)] loop started for $r" | tee -a /workspace/mcq_rsi/logs/backup_loops.log
  done
  sleep 300
done
