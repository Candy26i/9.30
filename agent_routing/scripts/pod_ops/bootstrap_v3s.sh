#!/usr/bin/env bash
# Bootstrap a fresh 2xA100 RunPod (no volume) for the 2026-10-09 experiments and start them:
#   1. clone Candy26i/9.30 (branch $BRANCH) into /workspace/9.30, run the pod setup (venvs, base model, CPU tests);
#   2. import + prepare-splits; 3. restore the advisor-output cache from HF (per-run backup of mmlu_pro_v3r5);
#   4. download the v3 / v3r5 dynamic_sft labels of every round; 5. start the base advisors; 6. preflight (not gating);
#   7. start the label-quality chain on GPU 1 (tmux mcq_labelq) and the success_sft supervisor (tmux mcq_supervisor_v3s)
#      plus the hourly per-run backup loops (tmux mcq_backups).
# Idempotent: each step leaves a marker in $WORK/logs/bootstrap/. Credentials: HF/W&B logins are the operator's
# (HF_HOME=/workspace/hf-cache); nothing here reads or stores a token. Log: $WORK/logs/bootstrap.log
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
step() { # step NAME cmd...   (skipped when $M/NAME exists; the marker is written only on success)
  local name=$1; shift
  if [ -f "$M/$name" ]; then log "$name: done earlier"; return 0; fi
  log "$name: start"
  if "$@"; then touch "$M/$name"; log "$name: ok"; else log "$name: FAILED (rc=$?)"; return 1; fi
}

clone() {
  if [ ! -d $REPO/.git ]; then git clone https://github.com/Candy26i/9.30.git $REPO || return 1; fi
  cd $REPO && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH" && git rev-parse HEAD
}
setup() { cd $TREE && bash scripts/setup_mcq_rsi_pod.sh; }
import_assets() {
  cd $TREE && $PY -m src.manager.mcq_rsi import --bench all --out $WORK/import --cache-dir $HF_HOME/hub \
    && $PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir $WORK/import --hf-cache-dir $HF_HOME/hub
}
restore_cache() {
  cd $TREE && $PY - <<'PYEOF'
from huggingface_hub import hf_hub_download
import tarfile
p = hf_hub_download("MaliDDD/margent-mcq-rsi-mmlu_pro_v3r5", "advisor_cache.tar.gz", local_dir="/workspace/tmp/restore")
with tarfile.open(p) as tar:
    n = len(tar.getmembers()); tar.extractall("/workspace/mcq_rsi")
print("advisor cache restored:", n, "members")
PYEOF
}
fetch_labels() {
  cd $TREE && $PY - <<'PYEOF'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
out = Path("/workspace/mcq_rsi/labels")
for b in ("medqa", "mmlu_pro", "gpqa", "aqua"):
    rounds = {2: "v3", 3: "v3"} if b == "gpqa" else {2: "v3", 3: "v3", 4: "v3r5", 5: "v3r5"}
    for k, ver in rounds.items():
        run = f"{b}_{ver}"
        for name in ("labels.jsonl", "labels.report.json"):
            p = hf_hub_download(f"MaliDDD/margent-mcq-rsi-{run}", f"runs/{run}/r{k}/dynamic_sft/select/{name}",
                                local_dir="/workspace/tmp/labels_dl")
            dst = out / b / (f"r{k}.jsonl" if name == "labels.jsonl" else f"r{k}.report.json")
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
        print(b, f"r{k}", "ok")
PYEOF
}
advisors() {
  cd $TREE && MCQ_CONFIG_SUFFIX=_v3 BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg advisors
  for i in $(seq 1 120); do curl -fsS http://127.0.0.1:18002/health >/dev/null 2>&1 && return 0; sleep 10; done
  log "advisors: not healthy after 20 min"; return 1
}
preflight() { # every benchmark; a failure is logged, not fatal (the run itself refuses a missing/failed report)
  cd $TREE; local rc=0
  for b in medqa mmlu_pro gpqa aqua; do
    MCQ_CONFIG_SUFFIX=_v3 BENCH=$b PREFLIGHT_BENCHES=$b bash scripts/runpod_mcq_rsi.sh preflight 2>&1 | tail -3 || { log "preflight $b FAILED"; rc=1; }
  done
  return $rc
}
labelq_chain() { # GPU 1, sequential; the marker logs/labelq_done releases GPU 1 to the supervisor
  local sh=/workspace/tmp/labelq_chain.sh
  cat > $sh <<EOF
#!/usr/bin/env bash
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
cd $TREE
for b in medqa aqua mmlu_pro gpqa; do
  L=$WORK/labels/\$b; args="r2=\$L/r2.jsonl r3=\$L/r3.jsonl"; [ -f \$L/r4.jsonl ] && args="\$args r4=\$L/r4.jsonl r5=\$L/r5.jsonl"
  $PY scripts/mcq_label_quality.py --config configs/mcq_rsi_\${b}_v3.json --out $WORK/runs/\${b}_v3lq --labels \$args --round1 --eval-s1 \\
    --advisor-url http://127.0.0.1:18002 2>&1 | tee -a $WORK/logs/\${b}_v3lq.log || echo "labelq \$b FAILED" | tee -a $WORK/logs/labelq.log
done
touch $WORK/logs/labelq_done
EOF
  chmod +x $sh; tmux kill-session -t mcq_labelq 2>/dev/null; tmux new-session -d -s mcq_labelq "bash $sh; exec bash"
}
supervisor() {
  tmux set-environment -g HF_HOME /workspace/hf-cache; tmux set-environment -g HF_HUB_DISABLE_XET 1; tmux set-environment -g TMPDIR /workspace/tmp
  cp $TREE/scripts/pod_ops/mcq_supervisor_v3s.py $TREE/scripts/pod_ops/backup_loops.sh $TREE/scripts/pod_ops/state_now.py \
     $TREE/scripts/pod_ops/verify_runs_hf.py $TREE/scripts/pod_ops/letters.py $TREE/scripts/pod_ops/rounds_any.py /workspace/tmp/
  tmux kill-session -t mcq_supervisor_v3s 2>/dev/null
  tmux new-session -d -s mcq_supervisor_v3s "cd /workspace/tmp && HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1 python3 mcq_supervisor_v3s.py 2>&1 | tee -a $WORK/logs/supervisor_v3s.log; exec bash"
  tmux kill-session -t mcq_backups 2>/dev/null
  tmux new-session -d -s mcq_backups "bash /workspace/tmp/backup_loops.sh; exec bash"
}

{
  step clone clone || exit 1
  step setup setup || exit 1
  step import import_assets || exit 1
  step restore_cache restore_cache || exit 1
  step fetch_labels fetch_labels || exit 1
  step advisors advisors || exit 1
  step preflight preflight || log "preflight: at least one benchmark failed; the supervisor will show which run is refused"
  step labelq_chain labelq_chain || exit 1
  step supervisor supervisor || exit 1
  log "bootstrap complete: tmux mcq_labelq (GPU 1), mcq_supervisor_v3s, mcq_backups, mcq_advisors_medqa"
} 2>&1 | tee -a $WORK/logs/bootstrap.log
